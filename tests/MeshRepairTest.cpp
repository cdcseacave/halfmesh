/*
* MeshRepairTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for MeshRepair.cpp:
//   RemoveDuplicateFaces, RemoveDegenerateFaces, RemoveSmallComponents,
//   RemoveFacesOutside, FixNonManifold, ListHalfEdgesSafe
//   + sanity run on tests/data/mesh.ply

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/OrientedBoundingBox.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// Helper: path to tests/data/mesh.ply (repo root / data / mesh.ply)
// ---------------------------------------------------------------------------
static std::string TestMeshPath()
{
	return (std::filesystem::path(__FILE__).parent_path()
	        / "data" / "mesh.ply")
	    .string();
}

// ---------------------------------------------------------------------------
// Helper: build a simple tetrahedron (4 vertices, 4 faces, manifold).
// Winding is chosen so that each pair of adjacent faces has OPPOSITE orientation
// for their shared edge (manifold requirement for FEdgeAdjacentFace).
//
// Vertices:
//   v0=(0,0,0), v1=(1,0,0), v2=(0.5,1,0), v3=(0.5,0.5,1)
//
// Faces (outward-normal consistent winding):
//   f0 = (0,2,1)   bottom  (seen from above: CCW)
//   f1 = (0,1,3)   front
//   f2 = (1,2,3)   right
//   f3 = (0,3,2)   left
// Each shared edge appears in opposite winding order in the two faces:
//   edge 0-1: f0 has 1→0 (via 2,1), f1 has 0→1  ✓
//   edge 1-2: f0 has 2→1, f2 has 1→2           ✓
//   edge 0-2: f0 has 0→2, f3 has 2→0 (via 3,2) ✓
//   edge 0-3: f1 has 0→...→3, f3 has 3→0        ✓
//   edge 1-3: f1 has 1→3, f2 has 3→1 (via 3)    ✓
//   edge 2-3: f2 has 2→3, f3 has 3→2            ✓
// ---------------------------------------------------------------------------
static Mesh MakeTetra()
{
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // v0
	    {1.f, 0.f, 0.f}, // v1
	    {0.5f, 1.f, 0.f}, // v2
	    {0.5f, 0.5f, 1.f} // v3
	};
	m.faces = {
	    {0, 2, 1}, // bottom
	    {0, 1, 3}, // front
	    {1, 2, 3}, // right
	    {0, 3, 2}, // left
	};
	return m;
}

// ---------------------------------------------------------------------------
// RemoveDuplicateFaces
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveDuplicateFacesRemovesBoth)
{
	Mesh m = MakeTetra();
	// Add a duplicate of face 0 (same 3 vertices, same winding) and
	// a reversed duplicate of face 2 — both share same sorted vertices,
	// so they are "duplicates" in the sorted-vertex sense.
	m.faces.push_back({0, 1, 2}); // exact dup of face 0
	m.faces.push_back({1, 3, 2}); // exact dup of face 2 (sorted same as {1,2,3})

	const Mesh::FIndex removed = m.RemoveDuplicateFaces(/*removeBothFaces=*/true);

	// Two pairs → 4 faces removed
	EXPECT_EQ(removed, 4u);
	// No duplicates remain: verify by sorting and checking uniqueness
	std::vector<std::array<Mesh::VIndex, 3>> sortedFaces;
	for (const auto& f : m.faces) {
		std::array<Mesh::VIndex, 3> sv = {f[0], f[1], f[2]};
		std::sort(sv.begin(), sv.end());
		sortedFaces.push_back(sv);
	}
	std::sort(sortedFaces.begin(), sortedFaces.end());
	for (size_t i = 0; i + 1 < sortedFaces.size(); ++i) {
		EXPECT_NE(sortedFaces[i], sortedFaces[i + 1]) << "Duplicate remains at index " << i;
	}
}

TEST(MeshRepairTest, RemoveDuplicateFacesRemovesOnlyOne)
{
	Mesh m = MakeTetra();
	m.faces.push_back({0, 1, 2}); // dup of face 0
	const Mesh::FIndex removed = m.RemoveDuplicateFaces(/*removeBothFaces=*/false);
	EXPECT_EQ(removed, 1u);
}

TEST(MeshRepairTest, RemoveDuplicateFacesNoDuplicates)
{
	Mesh m = MakeTetra();
	const Mesh::FIndex removed = m.RemoveDuplicateFaces();
	EXPECT_EQ(removed, 0u);
	EXPECT_EQ(m.faces.size(), 4u);
}

// ---------------------------------------------------------------------------
// RemoveDegenerateFaces
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveDegenerateFacesZeroAreaRemoved)
{
	Mesh m;
	// Three collinear vertices → zero area
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {2.f, 0.f, 0.f},
	    // add a non-degenerate triangle
	    {0.f, 1.f, 0.f}};
	m.faces = {
	    {0, 1, 2}, // collinear → area == 0
	    {0, 1, 3}, // valid triangle
	};

	const Mesh::FIndex removed = m.RemoveDegenerateFaces(/*thArea=*/1e-5f);
	EXPECT_EQ(removed, 1u);
	// The valid face must still be present; the degenerate one gone
	EXPECT_GE(m.faces.size(), 1u);
	// Verify the zero-area face is not in the remaining set
	for (const auto& f : m.faces) {
		// sorted vertices of a degenerate face: 0,1,2
		std::array<Mesh::VIndex, 3> sv = {f[0], f[1], f[2]};
		std::sort(sv.begin(), sv.end());
		EXPECT_FALSE(sv[0] == 0 && sv[1] == 1 && sv[2] == 2)
		    << "Degenerate face {0,1,2} still present";
	}
}

TEST(MeshRepairTest, RemoveDegenerateFacesDuplicateVertexIndices)
{
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.f, 1.f, 0.f},
	};
	// face with repeated vertex index 0 → immediately degenerate
	m.faces = {
	    {0, 0, 1}, // degenerate (v[0]==v[1])
	    {0, 1, 2}, // valid
	};
	const Mesh::FIndex removed = m.RemoveDegenerateFaces(1e-5f);
	EXPECT_GE(removed, 1u);
}

TEST(MeshRepairTest, RemoveDegenerateFacesHealthyFaceKept)
{
	Mesh m = MakeTetra();
	const Mesh::FIndex removed = m.RemoveDegenerateFaces(1e-5f);
	EXPECT_EQ(removed, 0u);
	EXPECT_EQ(m.faces.size(), 4u);
}

// The iterated overload must actually iterate: removing the nearly-collinear
// (0,1,2) vertex-merges P2 onto P1, silently making the VALID neighbor
// (2,3,4) -> (1,3,4) newly collinear with 3 distinct indices -- invisible to
// the single pass's index-duplicate-only bonus cleanup, so only a genuine
// second pass removes it. The old fixture was exhausted in pass 1 and could
// not distinguish iteration from a single pass.
TEST(MeshRepairTest, RemoveDegenerateFacesIteratedVersion)
{
	const std::vector<Mesh::Vertex> verts = {
	    Mesh::Vertex(0.f, 0.f, 0.f), Mesh::Vertex(2.f, 0.f, 0.f),
	    Mesh::Vertex(4.f, 0.f, 0.f), Mesh::Vertex(2.f, 1.f, 0.f),
	    Mesh::Vertex(2.f, 2.f, 0.f)};
	const std::vector<Mesh::Face> tris = {Mesh::Face(0, 1, 2), Mesh::Face(2, 3, 4)};

	// Single pass: removes only (0,1,2); the merge leaves (1,3,4) collinear.
	Mesh single;
	single.vertices = verts;
	single.faces = tris;
	EXPECT_EQ(single.RemoveDegenerateFaces(1e-5f), 1u);
	EXPECT_EQ(single.faces.size(), 1u);

	// Iterated: pass 2 must find and remove the newly-collinear neighbor.
	Mesh iterated;
	iterated.vertices = verts;
	iterated.faces = tris;
	EXPECT_EQ(iterated.RemoveDegenerateFaces(3u, 1e-5f), 2u);
	EXPECT_EQ(iterated.faces.size(), 0u);
}

// ---------------------------------------------------------------------------
// RemoveSmallComponents
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveSmallComponentsRemovesIsolatedTriangle)
{
	// Big component: 2 triangles sharing an edge (4 vertices)
	// Small component: 1 isolated triangle (3 new vertices, far away)
	Mesh m;
	m.vertices = {
	    // big component
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.5f, 1.f, 0.f},
	    {1.f, 1.f, 0.f},
	    // small isolated component (far away)
	    {10.f, 0.f, 0.f},
	    {11.f, 0.f, 0.f},
	    {10.5f, 1.f, 0.f},
	};
	m.faces = {
	    {0, 1, 2}, // big
	    {1, 3, 2}, // big
	    {4, 5, 6}, // small isolated
	};

	// minComponentSize=2 → the 1-face component (size=1) is removed
	const unsigned removed = m.RemoveSmallComponents(/*minComponentSize=*/2);
	EXPECT_EQ(removed, 1u);
	// The big component (2 faces) must remain
	EXPECT_EQ(m.faces.size(), 2u);
}

TEST(MeshRepairTest, RemoveSmallComponentsAllSmallRemovesEverything)
{
	// Three isolated triangles: 3 components of 1 face each — ALL small.
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.5f, 1.f, 0.f}, // component 0
	    {10.f, 0.f, 0.f},
	    {11.f, 0.f, 0.f},
	    {10.5f, 1.f, 0.f}, // component 1
	    {20.f, 0.f, 0.f},
	    {21.f, 0.f, 0.f},
	    {20.5f, 1.f, 0.f}, // component 2
	};
	m.faces = {
	    {0, 1, 2},
	    {3, 4, 5},
	    {6, 7, 8},
	};

	// Every component (1 face) is below minComponentSize=2, so all three must
	// be removed and the removed count (3) returned. The inverted count bug
	// counts LARGE components (size >= min), gets 0 here, hits the
	// "nothing small" early-out, and cleans nothing.
	const unsigned removed = m.RemoveSmallComponents(/*minComponentSize=*/2);
	EXPECT_EQ(removed, 3u);
	EXPECT_TRUE(m.faces.empty());
}

TEST(MeshRepairTest, RemoveSmallComponentsReturnsRemovedCount)
{
	// One large component (2 faces) + TWO small isolated triangles: the
	// large-component count (1) differs from the small-component count (2), so
	// the inverted return value cannot pass by coincidence.
	Mesh m;
	m.vertices = {
	    // large component: 2 triangles sharing edge (1,2)
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.5f, 1.f, 0.f},
	    {1.f, 1.f, 0.f},
	    // small component A (far away)
	    {10.f, 0.f, 0.f},
	    {11.f, 0.f, 0.f},
	    {10.5f, 1.f, 0.f},
	    // small component B (farther away)
	    {20.f, 0.f, 0.f},
	    {21.f, 0.f, 0.f},
	    {20.5f, 1.f, 0.f},
	};
	m.faces = {
	    {0, 1, 2}, // large
	    {1, 3, 2}, // large
	    {4, 5, 6}, // small A
	    {7, 8, 9}, // small B
	};

	const unsigned removed = m.RemoveSmallComponents(/*minComponentSize=*/2);
	EXPECT_EQ(removed, 2u); // bug returns 1 (the LARGE-component count)
	EXPECT_EQ(m.faces.size(), 2u); // only the 2-face component survives
}

TEST(MeshRepairTest, RemoveSmallComponentsSingleComponent)
{
	// A well-connected tetrahedron is a single component of 4 faces.
	// With minComponentSize=2, the component is large enough to keep (4 >= 2),
	// so RemoveSmallComponents finds no small component and removes no faces
	// (the returned count is the number of small components removed, here 0).
	Mesh m = MakeTetra();
	const size_t faceCountBefore = m.faces.size();
	m.RemoveSmallComponents(/*minComponentSize=*/2);
	// Face count must not decrease — the single big component is preserved
	EXPECT_EQ(m.faces.size(), faceCountBefore);
}

// ---------------------------------------------------------------------------
// RemoveFacesOutside (OBB)
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveFacesOutsideOBB)
{
	// Build a mesh: 3 faces, 2 inside the OBB, 1 outside.
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // v0 – inside
	    {0.5f, 0.f, 0.f}, // v1 – inside
	    {0.25f, 0.5f, 0.f}, // v2 – inside
	    {5.f, 0.f, 0.f}, // v3 – outside
	    {5.5f, 0.f, 0.f}, // v4 – outside
	    {5.25f, 0.5f, 0.f}, // v5 – outside
	    {0.f, 1.f, 0.f}, // v6 – inside
	};
	m.faces = {
	    {0, 1, 2}, // all inside
	    {0, 1, 6}, // all inside
	    {3, 4, 5}, // all outside
	};

	// OBB: axis-aligned box [−1,2] × [−1,2] × [−1,1]  (identity rotation)
	halfmesh::OBB obb;
	const halfmesh::Matrix3 rot = halfmesh::Matrix3::Identity();
	const halfmesh::Vector3 mn(-1.f, -1.f, -1.f);
	const halfmesh::Vector3 mx(2.f, 2.f, 1.f);
	obb = halfmesh::OBB(mn, mx, rot);

	const unsigned removed = m.RemoveFacesOutside(obb);
	EXPECT_EQ(removed, 1u);
	EXPECT_EQ(m.faces.size(), 2u);
	// All remaining face vertices must be inside the OBB
	for (const auto& f : m.faces) {
		for (int k = 0; k < 3; ++k) {
			const halfmesh::Vector3 vp = m.vertices[f[k]].cast<halfmesh::real>();
			EXPECT_TRUE(obb.Contains(vp))
			    << "Remaining face references vertex outside OBB: v=" << f[k];
		}
	}
}

TEST(MeshRepairTest, RemoveFacesOutsideNoFacesRemoved)
{
	Mesh m = MakeTetra();
	// Huge OBB that contains everything
	halfmesh::OBB obb;
	const halfmesh::Matrix3 rot = halfmesh::Matrix3::Identity();
	obb = halfmesh::OBB(halfmesh::Vector3(-100.f, -100.f, -100.f),
	                    halfmesh::Vector3(100.f, 100.f, 100.f), rot);
	const unsigned removed = m.RemoveFacesOutside(obb);
	EXPECT_EQ(removed, 0u);
	EXPECT_EQ(m.faces.size(), 4u);
}

// ---------------------------------------------------------------------------
// FixNonManifold
// ---------------------------------------------------------------------------
// Bowtie configuration: two triangles sharing only vertex 0 (not an edge).
// v0 is the bowtie vertex. After FixNonManifold, v0 should be split so that
// the HalfMesh can be built without error.
TEST(MeshRepairTest, FixNonManifoldBowtieSplitsVertex)
{
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // v0: bowtie centre
	    {1.f, 1.f, 0.f}, // v1
	    {-1.f, 1.f, 0.f}, // v2
	    {1.f, -1.f, 0.f}, // v3
	    {-1.f, -1.f, 0.f}, // v4
	};
	// Two triangles touching only at v0 (no shared edge → bowtie)
	m.faces = {
	    {0, 1, 2}, // top fan
	    {0, 3, 4}, // bottom fan, only connected at v0
	};

	const unsigned vertexCountBefore = m.vertices.size();
	const unsigned fixed = m.FixNonManifold(/*thMoveDuplicate=*/0.01f);

	// Non-manifold vertex 0 must have been split → at least 1 fix reported
	EXPECT_GE(fixed, 1u);
	// A new vertex should have been added
	EXPECT_GT(m.vertices.size(), vertexCountBefore);
	// After fixing, HalfMesh should build successfully
	EXPECT_TRUE(m.halfMesh.Build(m));
}

TEST(MeshRepairTest, FixNonManifoldManifoldMeshUnchanged)
{
	Mesh m = MakeTetra();
	const unsigned fixed = m.FixNonManifold();
	EXPECT_EQ(fixed, 0u);
	EXPECT_EQ(m.vertices.size(), 4u);
	EXPECT_EQ(m.faces.size(), 4u);
}

// ---------------------------------------------------------------------------
// ListHalfEdgesSafe
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, ListHalfEdgesSafeBuildsHalfMesh)
{
	Mesh m = MakeTetra();
	m.ListHalfEdgesSafe();
	// vHalfedges should now have one entry per vertex
	EXPECT_EQ(m.halfMesh.vHalfedges.size(), m.vertices.size());
}

// ---------------------------------------------------------------------------
// RemoveDuplicateVertices — weld coincident vertices (glTF seam / soup case)
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveDuplicateVerticesWeldsCoincident)
{
	// A quad as two triangles, but the shared-edge vertices are DUPLICATED, as
	// glTF stores them: 6 vertices at only 4 distinct positions.
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, // tri 0
	    {0.f, 0.f, 0.f},
	    {1.f, 1.f, 0.f},
	    {0.f, 1.f, 0.f}, // tri 1 (v3,v4 dup v0,v2)
	};
	m.faces = {{0, 1, 2}, {3, 4, 5}};
	const auto sharedVerts = [&] {
		int n = 0;
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
				if (m.faces[0][a] == m.faces[1][b])
					++n;
		return n;
	};
	EXPECT_EQ(sharedVerts(), 0); // before weld: disconnected sub-meshes

	const Mesh::VIndex removed = m.RemoveDuplicateVertices();
	EXPECT_EQ(removed, 2u);
	EXPECT_EQ(m.vertices.size(), 4u);
	EXPECT_EQ(m.faces.size(), 2u);
	// after weld the two triangles share exactly two vertices (a real edge)
	EXPECT_EQ(sharedVerts(), 2);
	EXPECT_TRUE(m.IsManifold());
}

TEST(MeshRepairTest, RemoveDuplicateVerticesNoOpWhenUnique)
{
	Mesh m = MakeTetra();
	const Mesh::VIndex before = static_cast<Mesh::VIndex>(m.vertices.size());
	EXPECT_EQ(m.RemoveDuplicateVertices(), 0u);
	EXPECT_EQ(m.vertices.size(), before);
}

// Epsilon-weld: a near pair straddling a grid-cell boundary must still weld.
// 0.14 -> cell 1, 0.16 -> cell 2 at eps=0.1, but they are only 0.02 apart
// (< eps): the old single-cell snap misses them; the neighbor-cell probe welds.
TEST(MeshRepairTest, RemoveDuplicateVerticesEpsilonWeldsAcrossCellBoundary)
{
	Mesh m;
	m.vertices = {
	    {0.14f, 0.f, 0.f}, // A: cell 1
	    {0.16f, 0.f, 0.f}, // B: cell 2, but 0.02 (< eps) from A
	    {5.f, 0.f, 0.f}, // far, distinct
	    {5.f, 5.f, 0.f}, // far, distinct
	};
	m.faces = {{0, 2, 3}, {1, 3, 2}};
	const Mesh::VIndex removed = m.RemoveDuplicateVertices(0.1f);
	EXPECT_EQ(removed, 1u); // A and B collapse to one vertex
	EXPECT_EQ(m.vertices.size(), 3u);
}

// Epsilon-weld: a far pair at a large coordinate must NOT weld. At ~1.15e6 the
// float grid key (coord*inv done in float) has an ulp of ~64 cells, so the old
// code buckets vertices up to ~0.06 apart together. These two are 0.125 apart
// (125*eps) and must stay distinct — the double-precision cell index keeps them
// apart (and the exact distance check rejects the neighbor probe).
TEST(MeshRepairTest, RemoveDuplicateVerticesEpsilonKeepsFarPairAtLargeCoord)
{
	Mesh m;
	m.vertices = {
	    {1153413.0f, 0.f, 0.f}, // exactly representable (ulp 0.125 here)
	    {1153413.125f, 0.f, 0.f}, // 0.125 away -> SAME float grid key, must NOT weld
	    {2.f, 0.f, 0.f},
	    {2.f, 2.f, 0.f},
	};
	m.faces = {{0, 2, 3}, {1, 3, 2}};
	const Mesh::VIndex removed = m.RemoveDuplicateVertices(1e-3f);
	EXPECT_EQ(removed, 0u); // nothing within eps=1e-3
	EXPECT_EQ(m.vertices.size(), 4u);
}

// ---------------------------------------------------------------------------
// IsManifold — detect the exact conditions the half-edge build assumes away
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, IsManifoldDetectsConditions)
{
	EXPECT_TRUE(MakeTetra().IsManifold());

	Mesh selfEdge;
	selfEdge.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
	selfEdge.faces = {{0, 1, 1}}; // a==b corner
	EXPECT_FALSE(selfEdge.IsManifold());

	Mesh edgeIn3; // edge (0,1) shared by three faces
	edgeIn3.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}};
	edgeIn3.faces = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}};
	EXPECT_FALSE(edgeIn3.IsManifold());
}

// ---------------------------------------------------------------------------
// HalfMesh::Build must REJECT (not silently corrupt) non-manifold input.
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, HalfMeshBuildRejectsNonManifold)
{
	Mesh dupEdge; // directed edge (0,1) appears twice
	dupEdge.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	dupEdge.faces = {{0, 1, 2}, {0, 1, 3}};
	EXPECT_FALSE(dupEdge.halfMesh.Build(dupEdge));

	Mesh selfEdge;
	selfEdge.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
	selfEdge.faces = {{0, 0, 1}};
	EXPECT_FALSE(selfEdge.halfMesh.Build(selfEdge));

	// a clean manifold mesh still builds
	Mesh tetra = MakeTetra();
	EXPECT_TRUE(tetra.halfMesh.Build(tetra));
}

// ---------------------------------------------------------------------------
// Regression: ListHalfEdges on a NON-manifold mesh must NOT hang; it must
// detect the corruption, repair to manifold, and rebuild.  (Before the fix the
// manifold-only build produced a corrupt structure and the adjacency walk
// looped forever — mis-diagnosed as a "RAM exhaustion".)
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, ListHalfEdgesRepairsNonManifoldNoHang)
{
	Mesh m; // edge (0,1) shared by three faces => non-manifold
	m.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}};
	m.faces = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}};

	m.ListHalfEdges(); // must terminate (auto-routes through the safe path)

	// a valid half-edge structure was produced (one entry per vertex) and the
	// repaired mesh is manifold
	EXPECT_EQ(m.halfMesh.vHalfedges.size(), m.vertices.size());
	EXPECT_TRUE(m.IsManifold());
}

// ---------------------------------------------------------------------------
// Regression: RemoveFaces must keep per-corner faceTexcoords aligned (it used
// to index the face id instead of the corner slot, corrupting textured meshes).
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveFacesKeepsFaceTexcoordsAligned)
{
	Mesh m;
	m.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}};
	m.faces = {{0, 1, 2}, {1, 3, 2}, {0, 2, 3}};
	// distinct per-corner UVs (x = global corner index) so misalignment shows
	m.faceTexcoords = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
	// faceTexblobs intentionally EMPTY (single-texture convention) — this is the
	// case that used to crash RemoveFaces by indexing an empty texblob array.

	std::vector<Mesh::FIndex> remove{0}; // drop face 0 (swap-with-back = face 2)
	m.RemoveFaces(remove, /*updateLists=*/false);

	ASSERT_EQ(m.faces.size(), 2u);
	ASSERT_EQ(m.faceTexcoords.size(), 6u);
	// slot 0 now holds the former last face's UVs (6,7,8), in order
	EXPECT_FLOAT_EQ(m.faceTexcoords[0].x(), 6.f);
	EXPECT_FLOAT_EQ(m.faceTexcoords[1].x(), 7.f);
	EXPECT_FLOAT_EQ(m.faceTexcoords[2].x(), 8.f);
	// the surviving face 1 keeps its UVs (3,4,5)
	EXPECT_FLOAT_EQ(m.faceTexcoords[3].x(), 3.f);
	EXPECT_FLOAT_EQ(m.faceTexcoords[4].x(), 4.f);
	EXPECT_FLOAT_EQ(m.faceTexcoords[5].x(), 5.f);
}

// ---------------------------------------------------------------------------
// Sanity: tests/data/mesh.ply
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, SanityMeshPly)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path)) {
		GTEST_SKIP() << "tests/data/mesh.ply not found at: " << path;
	}

	Mesh m;
	ASSERT_TRUE(m.Load(path));
	ASSERT_FALSE(m.Empty());
	const size_t origFaces = m.faces.size();
	const size_t origVerts = m.vertices.size();

	// RemoveDuplicateFaces
	m.RemoveDuplicateFaces();
	EXPECT_FALSE(m.Empty());
	EXPECT_LE(m.faces.size(), origFaces);

	// RemoveDegenerateFaces
	m.RemoveDegenerateFaces(1e-5f);
	EXPECT_FALSE(m.Empty());

	// FixNonManifold
	m.FixNonManifold();
	EXPECT_FALSE(m.Empty());

	// RemoveSmallComponents (keep components of at least 3 faces)
	m.RemoveSmallComponents(3);
	EXPECT_FALSE(m.Empty());

	// HalfMesh should build successfully
	EXPECT_TRUE(m.halfMesh.Build(m));
	EXPECT_EQ(m.halfMesh.vHalfedges.size(), m.vertices.size());

	// RemoveFacesOutside with a big OBB (should remove nothing)
	const halfmesh::Matrix3 rot = halfmesh::Matrix3::Identity();
	halfmesh::OBB obb(halfmesh::Vector3(-1e6f, -1e6f, -1e6f),
	                  halfmesh::Vector3(1e6f, 1e6f, 1e6f), rot);
	const unsigned outside = m.RemoveFacesOutside(obb);
	EXPECT_EQ(outside, 0u);
	EXPECT_FALSE(m.Empty());
}

// Auto-repair (ListHalfEdgesSafe) must KEEP one copy of a duplicated face:
// the old default removed both copies, deleting valid surface and punching a
// hole exactly where the input had a redundant triangle. Keep-one leaves
// unique directed edges, so the
// manifold half-edge build is still satisfied.
TEST(MeshRepairTest, AutoRepairKeepsOneCopyOfDuplicateFaces)
{
	Mesh mesh;
	mesh.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0),
	                 Mesh::Vertex(1, 1, 0), Mesh::Vertex(0, 1, 0)};
	// unit quad = 2 triangles, with the first triangle duplicated (same winding)
	mesh.faces = {Mesh::Face(0, 1, 2), Mesh::Face(0, 2, 3), Mesh::Face(0, 1, 2)};
	mesh.ListHalfEdgesSafe();
	EXPECT_EQ(mesh.faces.size(), 2u) << "duplicate removal must keep one copy";
	// both quad halves must survive: total area 1.0, not 0.5 (hole punched)
	EXPECT_NEAR(mesh.ComputeArea(), 1.0, 1e-6);
}

// RemoveDuplicateFaces on an empty mesh must be a safe no-op: FIndex is
// unsigned, so the pre-fix `i < numFaces - 1` loop bound wrapped to
// 0xFFFFFFFF and indexed empty vectors — an out-of-bounds read on a direct
// public call (2026-07-17 final batch review). A single-face mesh is the
// other no-duplicate boundary.
TEST(MeshRepairTest, RemoveDuplicateFacesEmptyAndSingleFaceSafe)
{
	Mesh empty;
	EXPECT_EQ(empty.RemoveDuplicateFaces(), 0u);
	EXPECT_TRUE(empty.faces.empty());

	Mesh single;
	single.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0),
	                   Mesh::Vertex(0, 1, 0)};
	single.faces = {Mesh::Face(0, 1, 2)};
	EXPECT_EQ(single.RemoveDuplicateFaces(), 0u);
	EXPECT_EQ(single.faces.size(), 1u);
}

// Contract: FixNonManifold on an empty mesh is a graceful no-op in every build
// mode — previously a Debug-only ASSERT aborted while Release proceeded.
TEST(MeshRepairTest, FixNonManifoldEmptyMeshIsNoOp)
{
	Mesh mesh;
	EXPECT_EQ(mesh.FixNonManifold(), 0u);
	EXPECT_TRUE(mesh.Empty());
}

} // namespace
} // namespace halfmesh
