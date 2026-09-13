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
//   RemoveLongEdgeFaces, RemoveLongEdgeFacesLocal, RemoveLongEdgeFacesCapped,
//   RemoveSpuriousComponents, RemoveSpikes,
//   RemoveFacesOutside, FixNonManifold, ListHalfEdgesSafe
//   + sanity run on tests/data/mesh.ply

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/OrientedBoundingBox.h>

#include "Corpus.h"
#include "Metrics.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

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

static unsigned RemoveSmallComponentsArraysReference(Mesh& mesh, unsigned minComponentSize)
{
	mesh.ListHalfEdges();
	std::vector<Mesh::FIndex> components;
	const Mesh::FIndex numComponents = mesh.halfMesh.ConnectedComponents(components);
	std::vector<unsigned> sizes(numComponents, 0);
	for (Mesh::FIndex component : components)
		++sizes[component];
	const unsigned removedComponents = std::accumulate(sizes.begin(), sizes.end(), 0u,
	                                                   [minComponentSize](unsigned count, unsigned size) { return count + (size < minComponentSize); });
	std::vector<Mesh::FIndex> removes;
	for (Mesh::FIndex face = 0; face < mesh.faces.size(); ++face)
		if (sizes[components[face]] < minComponentSize)
			removes.emplace_back(face);
	mesh.RemoveFaces(removes);
	mesh.RemoveUnreferencedVertices();
	return removedComponents;
}

static std::vector<float> EdgeLengthsReference(Mesh& mesh)
{
	mesh.ListHalfEdges();
	std::vector<float> edgeLengths;
	for (Mesh::EIndex edge = 0; edge < mesh.halfMesh.ESize(); ++edge) {
		const auto vertices = mesh.halfMesh.EVertices(edge);
		edgeLengths.emplace_back((mesh.vertices[vertices.first] - mesh.vertices[vertices.second]).norm());
	}
	return edgeLengths;
}

static Mesh::FIndex RemoveLongEdgeFacesArraysReference(Mesh& mesh, float factor)
{
	const Mesh::FIndex initialFaces = static_cast<Mesh::FIndex>(mesh.faces.size());
	std::vector<float> edgeLengths = EdgeLengthsReference(mesh);
	const size_t idx95 = edgeLengths.size() * 95 / 100;
	std::nth_element(edgeLengths.begin(), edgeLengths.begin() + idx95, edgeLengths.end());
	const float maxEdgeLength = edgeLengths[idx95] * factor;
	std::vector<Mesh::FIndex> removes;
	for (Mesh::FIndex face = 0; face < mesh.faces.size(); ++face)
		for (int edge = 0; edge < 3; ++edge)
			if ((mesh.vertices[mesh.faces[face][edge]] - mesh.vertices[mesh.faces[face][(edge + 1) % 3]]).norm() > maxEdgeLength) {
				removes.emplace_back(face);
				break;
			}
	if (!removes.empty()) {
		mesh.RemoveFaces(removes);
		mesh.RemoveUnreferencedVertices();
	}
	return initialFaces - static_cast<Mesh::FIndex>(mesh.faces.size());
}

static Mesh::FIndex RemoveSpuriousComponentsArraysReference(Mesh& mesh, float factor)
{
	const Mesh::FIndex initialFaces = static_cast<Mesh::FIndex>(mesh.faces.size());
	std::vector<float> edgeLengths = EdgeLengthsReference(mesh);
	const size_t idx55 = edgeLengths.size() * 55 / 100;
	std::nth_element(edgeLengths.begin(), edgeLengths.begin() + idx55, edgeLengths.end());
	const float minComponentDiameter = edgeLengths[idx55] * factor;
	std::vector<Mesh::FIndex> components;
	const Mesh::FIndex numComponents = mesh.halfMesh.ConnectedComponents(components);
	if (numComponents > 1) {
		std::vector<Eigen::AlignedBox<float, 3>> bounds(numComponents);
		for (Mesh::FIndex face = 0; face < mesh.faces.size(); ++face)
			for (int corner = 0; corner < 3; ++corner)
				bounds[components[face]].extend(mesh.vertices[mesh.faces[face][corner]]);
		std::vector<Mesh::FIndex> removes;
		for (Mesh::FIndex face = 0; face < mesh.faces.size(); ++face)
			if (bounds[components[face]].diagonal().norm() < minComponentDiameter)
				removes.emplace_back(face);
		if (!removes.empty()) {
			mesh.RemoveFaces(removes);
			mesh.RemoveUnreferencedVertices();
		}
	}
	return initialFaces - static_cast<Mesh::FIndex>(mesh.faces.size());
}

// Helper: a 10x1 strip of unit cells (22 vertices, 20 faces) with a detached
// triangle of the given size at x=20; spurious-debris fixtures build on it.
static Mesh MakeStripWithDetachedTriangle(float size)
{
	Mesh mesh;
	for (unsigned x = 0; x <= 10; ++x) {
		mesh.vertices.emplace_back(static_cast<float>(x), 0.f, 0.f);
		mesh.vertices.emplace_back(static_cast<float>(x), 1.f, 0.f);
	}
	for (unsigned x = 0; x < 10; ++x) {
		const Mesh::VIndex lower = 2 * x;
		mesh.faces.emplace_back(lower, lower + 1, lower + 3);
		mesh.faces.emplace_back(lower, lower + 3, lower + 2);
	}
	const Mesh::VIndex tri = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.emplace_back(20.f, 0.f, 0.f);
	mesh.vertices.emplace_back(20.f + size, 0.f, 0.f);
	mesh.vertices.emplace_back(20.f, size, 0.f);
	mesh.faces.emplace_back(tri, tri + 1, tri + 2);
	return mesh;
}

// Helper: a hexagonal fan of 6 faces around the origin whose ring holds two
// edges of length 2; dropping those two faces leaves a pinch at the center.
static Mesh MakePinchFan()
{
	Mesh mesh;
	mesh.vertices = {
	    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {-0.5f, 0.8660254f, 0.f}, {0.f, 1.f, 0.f}, {0.f, -1.f, 0.f}, {0.8660254f, -0.5f, 0.f}};
	for (Mesh::VIndex ring = 1; ring <= 6; ++ring)
		mesh.faces.emplace_back(0, ring, ring == 6 ? 1 : ring + 1);
	return mesh;
}

// Helper: append a regular (cols x rows)-cell grid of the given spacing at the
// given origin and return its base index (vertex (x, y) is base + y * (cols + 1) + x).
// Cells are split (a,b,c),(a,c,d) with a=(x,y), b=(x+1,y), c=(x+1,y+1), d=(x,y+1),
// so every axis-aligned edge is shared by exactly two cells (or is a border) and
// the diagonals are unique to their cell.
static Mesh::VIndex AppendGrid(Mesh& mesh, unsigned cols, unsigned rows, float spacing, float x0, float y0, float z = 0.f)
{
	const Mesh::VIndex base = static_cast<Mesh::VIndex>(mesh.vertices.size());
	const Mesh::VIndex stride = cols + 1;
	for (unsigned y = 0; y <= rows; ++y)
		for (unsigned x = 0; x <= cols; ++x)
			mesh.vertices.emplace_back(x0 + spacing * x, y0 + spacing * y, z);
	for (unsigned y = 0; y < rows; ++y)
		for (unsigned x = 0; x < cols; ++x) {
			const Mesh::VIndex a = base + y * stride + x;
			mesh.faces.emplace_back(a, a + 1, a + stride + 1);
			mesh.faces.emplace_back(a, a + stride + 1, a + stride);
		}
	return base;
}

static bool HasFace(const Mesh& mesh, Mesh::VIndex a, Mesh::VIndex b, Mesh::VIndex c)
{
	std::vector<Mesh::VIndex> want = {a, b, c};
	std::sort(want.begin(), want.end());
	for (const Mesh::Face& face : mesh.faces) {
		std::vector<Mesh::VIndex> have = {face[0], face[1], face[2]};
		std::sort(have.begin(), have.end());
		if (have == want)
			return true;
	}
	return false;
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

TEST(MeshRepairTest, RemoveDegenerateFacesHalfEdgeCollapsesNeedleWithoutBuild)
{
	Mesh m = hmtest::corpus::GridPlane(8);
	m.ListHalfEdges();
	Mesh::EIndex shortEdge = math::NO_ID;
	Mesh::Vertex midpoint;
	for (Mesh::EIndex edge = 0; edge < m.halfMesh.ESize(); ++edge) {
		if (m.halfMesh.EIsBoundary(edge) || !m.halfMesh.EIsCollapseValidTopologically(edge))
			continue;
		const auto edgeVertices = m.halfMesh.EVertices(edge);
		const Mesh::Vertex candidate = (m.vertices[edgeVertices.first] + m.vertices[edgeVertices.second]) * 0.5f;
		if (m.halfMesh.EIsCollapseValidGeometrically(edge, candidate, m.vertices)) {
			shortEdge = edge;
			midpoint = candidate;
			break;
		}
	}
	ASSERT_NE(shortEdge, math::NO_ID);
	const auto edgeVertices = m.halfMesh.EVertices(shortEdge);
	const Mesh::Vertex direction = (m.vertices[edgeVertices.second] - m.vertices[edgeVertices.first]).normalized();
	m.vertices[edgeVertices.first] = midpoint - direction * 5e-7f;
	m.vertices[edgeVertices.second] = midpoint + direction * 5e-7f;
	ASSERT_TRUE(m.halfMesh.EIsCollapseValidGeometrically(shortEdge, midpoint, m.vertices));
	const Mesh input = m;
	Mesh arrays = input;
	arrays.InvalidateHalfMesh();
	EXPECT_GT(arrays.RemoveDegenerateFacesArrays(1e-5f), 0u);
	arrays.RemoveUnreferencedVerticesArrays();
	const std::size_t initialFaces = m.faces.size();
	const std::size_t initialVertices = m.vertices.size();

	HalfMesh::ResetBuildCount();
	EXPECT_EQ(m.RemoveDegenerateFaces(1e-5f), 2u);
	EXPECT_EQ(HalfMesh::BuildCount(), 0u);
	EXPECT_EQ(m.faces.size(), initialFaces - 2u);
	EXPECT_EQ(m.vertices.size(), initialVertices - 1u);
	EXPECT_TRUE(m.ValidateHalfMesh());
	EXPECT_LT(hmtest::metrics::ComputeDistanceKdTree(input, arrays).hausdorffSymmetric, 2e-6);
	EXPECT_LT(hmtest::metrics::ComputeDistanceKdTree(input, m).hausdorffSymmetric, 2e-6);
	for (const Mesh::Face& face : m.faces) {
		const auto cross = (m.vertices[face[1]] - m.vertices[face[0]]).cross(m.vertices[face[2]] - m.vertices[face[0]]);
		EXPECT_GT(cross.squaredNorm(), 4e-10f);
	}
}

TEST(MeshRepairTest, RemoveDegenerateFacesHalfEdgeFlipsCapWithoutBuild)
{
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {2.f, 0.f, 0.f},
	    {1.f, 1e-6f, 0.f},
	    {1.f, -1.f, 0.f},
	};
	m.faces = {{0, 1, 2}, {1, 0, 3}};
	m.ListHalfEdges();
	ASSERT_NE(m.halfMesh.EEdge(0, 1), math::NO_ID);

	HalfMesh::ResetBuildCount();
	// A flip repairs both triangles without removing either face.
	EXPECT_EQ(m.RemoveDegenerateFaces(1e-5f), 0u);
	EXPECT_EQ(HalfMesh::BuildCount(), 0u);
	EXPECT_EQ(m.faces.size(), 2u);
	EXPECT_EQ(m.vertices.size(), 4u);
	EXPECT_EQ(m.halfMesh.EEdge(0, 1), math::NO_ID);
	EXPECT_NE(m.halfMesh.EEdge(2, 3), math::NO_ID);
	EXPECT_TRUE(m.ValidateHalfMesh());
	for (const Mesh::Face& face : m.faces) {
		const auto cross = (m.vertices[face[1]] - m.vertices[face[0]]).cross(m.vertices[face[2]] - m.vertices[face[0]]);
		EXPECT_GT(cross.squaredNorm(), 4e-10f);
	}
}

// The needle/cap split is an edge-length RATIO, so the same cap must still be
// flipped (never collapsed) when the mesh is expressed in different units --
// comparing the shortest edge against the area threshold directly would flip
// the classification as soon as the model is scaled.
TEST(MeshRepairTest, RemoveDegenerateFacesHalfEdgeCapClassificationIsScaleInvariant)
{
	for (const float scale : {1e-2f, 1.f, 1e3f}) {
		SCOPED_TRACE(scale);
		Mesh m;
		m.vertices = {
		    {0.f, 0.f, 0.f},
		    {2.f * scale, 0.f, 0.f},
		    {1.f * scale, 1e-6f * scale, 0.f},
		    {1.f * scale, -1.f * scale, 0.f},
		};
		m.faces = {{0, 1, 2}, {1, 0, 3}};
		m.ListHalfEdges();
		// threshold scaled with the model, as an area threshold must be
		const float thArea = 1e-5f * scale * scale;
		EXPECT_EQ(m.RemoveDegenerateFaces(thArea), 0u); // flipped, not collapsed
		EXPECT_EQ(m.faces.size(), 2u);
		EXPECT_EQ(m.vertices.size(), 4u);
		EXPECT_EQ(m.halfMesh.EEdge(0, 1), math::NO_ID);
		EXPECT_NE(m.halfMesh.EEdge(2, 3), math::NO_ID);
		EXPECT_TRUE(m.ValidateHalfMesh());
	}
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
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.5f, 1.f, 0.f}, // component 0
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

TEST(MeshRepairTest, RemoveSmallComponentsNativeMatchesArrayReference)
{
	Mesh arrays;
	arrays.vertices = {
	    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}, {10.f, 0.f, 0.f}, {11.f, 0.f, 0.f}, {10.f, 1.f, 0.f}};
	arrays.faces = {{0, 1, 2}, {1, 3, 2}, {4, 5, 6}};
	Mesh native = arrays;
	native.ListHalfEdges();

	EXPECT_EQ(RemoveSmallComponentsArraysReference(arrays, 2), native.RemoveSmallComponents(2));
	EXPECT_EQ(native.vertices, arrays.vertices);
	EXPECT_EQ(native.faces, arrays.faces);
	EXPECT_FALSE(native.halfMesh.Empty());
	EXPECT_TRUE(native.ValidateHalfMesh());
}

// ---------------------------------------------------------------------------
// RemoveLongEdgeFaces
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveLongEdgeFacesDisabledIsNoOp)
{
	Mesh mesh = MakeTetra();
	const std::vector<Mesh::Vertex> vertices = mesh.vertices;
	const std::vector<Mesh::Face> faces = mesh.faces;
	EXPECT_EQ(mesh.RemoveLongEdgeFaces(0.f), 0u);
	EXPECT_EQ(mesh.vertices, vertices);
	EXPECT_EQ(mesh.faces, faces);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesCountsAutoRepairRemovals)
{
	Mesh mesh;
	mesh.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
	mesh.faces = {{0, 1, 2}, {0, 1, 2}};
	EXPECT_EQ(mesh.RemoveLongEdgeFaces(100.f), 1u);
	EXPECT_EQ(mesh.faces.size(), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(MeshRepairTest, RemoveLongEdgeFacesDropsLongEdgeFace)
{
	// strip edges are 1 / sqrt(2), the detached triangle has edges 10, 10, 14.1:
	// percentile95 is 10, so at factor 1 only the 14.1 edge exceeds it
	Mesh mesh = MakeStripWithDetachedTriangle(10.f);
	EXPECT_EQ(mesh.RemoveLongEdgeFaces(1.f), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 20u);
	EXPECT_EQ(mesh.vertices.size(), 22u);
	for (const Mesh::Vertex& vertex : mesh.vertices)
		EXPECT_LT(vertex.x(), 20.f);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesSplitsCreatedPinch)
{
	Mesh mesh = MakePinchFan();
	EXPECT_EQ(mesh.RemoveLongEdgeFaces(0.9f), 2u);
	EXPECT_EQ(mesh.faces.size(), 4u);
	EXPECT_EQ(mesh.vertices.size(), 8u) << "the center pinch must duplicate its source vertex";
	EXPECT_FALSE(mesh.halfMesh.Empty());
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	HalfMesh rebuilt;
	EXPECT_TRUE(rebuilt.Build(mesh));
}

TEST(MeshRepairTest, RemoveLongEdgeFacesNativeMatchesArrayReference)
{
	Mesh arrays = MakeStripWithDetachedTriangle(10.f);
	Mesh native = arrays;
	native.ListHalfEdges();

	EXPECT_EQ(RemoveLongEdgeFacesArraysReference(arrays, 1.f), native.RemoveLongEdgeFaces(1.f));
	EXPECT_EQ(native.vertices, arrays.vertices);
	EXPECT_EQ(native.faces, arrays.faces);
	EXPECT_FALSE(native.halfMesh.Empty());
	EXPECT_TRUE(native.ValidateHalfMesh());
}

// ---------------------------------------------------------------------------
// RemoveLongEdgeFacesLocal
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveLongEdgeFacesLocalKeepsUniformlySparseSurface)
{
	// a spacing-1 grid and a far-away spacing-8 grid: each face's longest edge is
	// sqrt(2) x its own scale, so nothing exceeds 4 x the local scale
	Mesh mesh;
	AppendGrid(mesh, 5, 5, 1.f, 0.f, 0.f);
	AppendGrid(mesh, 5, 5, 8.f, 100.f, 0.f);
	const size_t numFaces = mesh.faces.size();
	EXPECT_EQ(mesh.RemoveLongEdgeFacesLocal(4.f, 1), 0u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), numFaces);
	EXPECT_EQ(mesh.vertices.size(), 72u);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesLocalDropsBridgesBetweenDenseRegions)
{
	// two spacing-1 grids 8 units apart bridged by two faces (edges 8, 8.06):
	// every bridge vertex sits in a dense grid, so the face scale stays ~1.4 and
	// the bridge exceeds 4 x scale; a second bridge from the right grid to a
	// spacing-8 grid (edges 6.1-8.5) has one vertex of scale 8 and survives
	Mesh mesh;
	const Mesh::VIndex left = AppendGrid(mesh, 3, 3, 1.f, 0.f, 0.f);
	const Mesh::VIndex right = AppendGrid(mesh, 3, 3, 1.f, 11.f, 0.f);
	const Mesh::VIndex sparse = AppendGrid(mesh, 3, 3, 8.f, 20.f, 0.f);
	const auto at = [](Mesh::VIndex base, unsigned x, unsigned y) { return base + y * 4 + x; };
	// dense-dense bridge: left border edge (3,1)-(3,2) to right border edge (0,1)-(0,2)
	mesh.faces.emplace_back(at(left, 3, 2), at(left, 3, 1), at(right, 0, 1));
	mesh.faces.emplace_back(at(left, 3, 2), at(right, 0, 1), at(right, 0, 2));
	// dense-sparse bridge: right border edge (3,1)-(3,2) to sparse border edge (0,0)-(0,1)
	mesh.faces.emplace_back(at(right, 3, 2), at(right, 3, 1), at(sparse, 0, 0));
	mesh.faces.emplace_back(at(right, 3, 2), at(sparse, 0, 0), at(sparse, 0, 1));
	const size_t numFaces = mesh.faces.size();
	const size_t numVertices = mesh.vertices.size();

	EXPECT_EQ(mesh.RemoveLongEdgeFacesLocal(4.f, 1), 2u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), numFaces - 2);
	EXPECT_EQ(mesh.vertices.size(), numVertices) << "bridge vertices stay referenced by their grids";
	EXPECT_FALSE(HasFace(mesh, at(left, 3, 2), at(left, 3, 1), at(right, 0, 1)));
	EXPECT_FALSE(HasFace(mesh, at(left, 3, 2), at(right, 0, 1), at(right, 0, 2)));
	EXPECT_TRUE(HasFace(mesh, at(right, 3, 2), at(right, 3, 1), at(sparse, 0, 0)));
	EXPECT_TRUE(HasFace(mesh, at(right, 3, 2), at(sparse, 0, 0), at(sparse, 0, 1)));
}

TEST(MeshRepairTest, RemoveLongEdgeFacesLocalRingsWidenTheScale)
{
	// an 8x8 spacing-1 grid whose center vertex is lifted 10 units and fanned to
	// its 8 ring neighbours (the 4 cells around it become 8 tent faces): every
	// edge at the apex is ~10, so with rings=1 the apex scale is ~10 and the tent
	// survives, while with rings=2 the apex sees the ring's short grid edges, its
	// median drops to ~1 and the tent goes
	const auto makeTent = []() {
		Mesh mesh;
		for (unsigned y = 0; y <= 8; ++y)
			for (unsigned x = 0; x <= 8; ++x)
				mesh.vertices.emplace_back(static_cast<float>(x), static_cast<float>(y), (x == 4 && y == 4) ? 10.f : 0.f);
		const auto at = [](unsigned x, unsigned y) { return static_cast<Mesh::VIndex>(y * 9 + x); };
		for (unsigned y = 0; y < 8; ++y)
			for (unsigned x = 0; x < 8; ++x) {
				if (x >= 3 && x <= 4 && y >= 3 && y <= 4)
					continue;
				mesh.faces.emplace_back(at(x, y), at(x + 1, y), at(x + 1, y + 1));
				mesh.faces.emplace_back(at(x, y), at(x + 1, y + 1), at(x, y + 1));
			}
		const Mesh::VIndex ring[8] = {at(3, 3), at(4, 3), at(5, 3), at(5, 4), at(5, 5), at(4, 5), at(3, 5), at(3, 4)};
		for (unsigned i = 0; i < 8; ++i)
			mesh.faces.emplace_back(at(4, 4), ring[i], ring[(i + 1) % 8]);
		return mesh;
	};

	Mesh oneRing = makeTent();
	EXPECT_EQ(oneRing.RemoveLongEdgeFacesLocal(4.f, 1), 0u);
	EXPECT_TRUE(oneRing.ValidateHalfMesh());
	EXPECT_EQ(oneRing.faces.size(), 128u);

	Mesh twoRings = makeTent();
	EXPECT_EQ(twoRings.RemoveLongEdgeFacesLocal(4.f, 2), 8u);
	EXPECT_TRUE(twoRings.ValidateHalfMesh());
	EXPECT_EQ(twoRings.faces.size(), 120u);
	EXPECT_EQ(twoRings.vertices.size(), 80u) << "the unreferenced apex is dropped";
	for (const Mesh::Vertex& vertex : twoRings.vertices)
		EXPECT_EQ(vertex.z(), 0.f);
}

// ---------------------------------------------------------------------------
// RemoveLongEdgeFacesCapped
// ---------------------------------------------------------------------------
namespace {

// a 9x9 spacing-1 floor (128 faces, longest edge sqrt(2)) and a single 8x8 cell
// (2 faces, longest edge 8*sqrt(2)) floating `height` above it, inside its footprint:
// the cell is the only candidate (median longest edge stays sqrt(2))
Mesh MakeFloorAndLid(float height)
{
	Mesh mesh;
	AppendGrid(mesh, 8, 8, 1.f, 0.f, 0.f);
	const Mesh::VIndex lid = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.emplace_back(0.f, 0.f, height);
	mesh.vertices.emplace_back(8.f, 0.f, height);
	mesh.vertices.emplace_back(8.f, 8.f, height);
	mesh.vertices.emplace_back(0.f, 8.f, height);
	mesh.faces.emplace_back(lid, lid + 1, lid + 2);
	mesh.faces.emplace_back(lid, lid + 2, lid + 3);
	return mesh;
}

} // namespace

TEST(MeshRepairTest, RemoveLongEdgeFacesCappedDisabledIsNoOp)
{
	Mesh mesh = MakeFloorAndLid(6.f);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(0.f), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 0.f), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 4.f, 0.f), 0u);
	EXPECT_EQ(mesh.faces.size(), 130u);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesCappedDropsLidOverSurface)
{
	// lid 6 above the floor: the first probe (0.5 x 11.3 = 5.66 below the centroid)
	// lands 0.34 from the floor, inside its 0.35 x 5.66 = 1.98 cone radius
	Mesh mesh = MakeFloorAndLid(6.f);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(), 2u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 128u);
	EXPECT_EQ(mesh.vertices.size(), 81u) << "the unreferenced lid corners are dropped";
	for (const Mesh::Vertex& vertex : mesh.vertices)
		EXPECT_EQ(vertex.z(), 0.f);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesCappedKeepsCoarseSurfaceWithNothingBehind)
{
	// the same lid 100 above the floor: the farthest probe (4 x 11.3 = 45.3) stops
	// 54.7 short of the floor, so the lid is just a coarsely sampled surface
	Mesh mesh = MakeFloorAndLid(100.f);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(), 0u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 130u);
	// reach 6 still misses (probe 6 x 11.3 = 67.9, 32.1 short, radius 23.8); reach 7
	// brings the floor into range (probe 79.2, 20.8 short, radius 27.7)
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 6.f), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 7.f), 2u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 128u);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesCappedInsidePipeline)
{
	// the probe BVH reads the face array, which a pipeline scope has cleared: the
	// filter must harvest it rather than probe an empty tree and remove nothing
	Mesh mesh = MakeFloorAndLid(6.f);
	mesh.BeginHalfEdgePipeline();
	ASSERT_TRUE(mesh.faces.empty());
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(), 2u);
	mesh.EndHalfEdgePipeline();
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 128u);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesCappedRejectsInvalidParams)
{
	// cone >= 1 would let a probe find the face's own plane (exactly one probe
	// distance away) and cap every candidate; a non-finite reach never ends the probes
	Mesh mesh = MakeFloorAndLid(100.f);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 4.f, 1.f), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 4.f, 1.5f), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, std::numeric_limits<float>::infinity()), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(std::numeric_limits<float>::quiet_NaN()), 0u);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 4.f, std::numeric_limits<float>::quiet_NaN()), 0u);
	EXPECT_EQ(mesh.faces.size(), 130u);
}

TEST(MeshRepairTest, RemoveLongEdgeFacesCappedHugeReachStopsAtMeshExtent)
{
	// probes farther from the centroid than diagonal / (1 - cone) cannot reach the
	// mesh, so a reach far beyond the float-step limit (2^24) ends at the extent
	Mesh mesh = MakeFloorAndLid(100.f);
	EXPECT_EQ(mesh.RemoveLongEdgeFacesCapped(2.f, 1e30f), 2u);
	EXPECT_EQ(mesh.faces.size(), 128u);
}

// ---------------------------------------------------------------------------
// RemoveSpuriousComponents
// ---------------------------------------------------------------------------
TEST(MeshRepairTest, RemoveSpuriousComponentsDisabledIsNoOp)
{
	Mesh mesh = MakeTetra();
	const std::vector<Mesh::Vertex> vertices = mesh.vertices;
	const std::vector<Mesh::Face> faces = mesh.faces;
	EXPECT_EQ(mesh.RemoveSpuriousComponents(0.f), 0u);
	EXPECT_EQ(mesh.vertices, vertices);
	EXPECT_EQ(mesh.faces, faces);
}

TEST(MeshRepairTest, RemoveSpuriousComponentsCountsAutoRepairRemovals)
{
	Mesh mesh;
	mesh.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
	mesh.faces = {{0, 1, 2}, {0, 1, 2}};
	EXPECT_EQ(mesh.RemoveSpuriousComponents(100.f), 1u);
	EXPECT_EQ(mesh.faces.size(), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(MeshRepairTest, RemoveSpuriousComponentsDropsSmallDisconnectedSurface)
{
	Mesh mesh = MakeStripWithDetachedTriangle(0.01f);
	EXPECT_EQ(mesh.RemoveSpuriousComponents(), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 20u);
	EXPECT_EQ(mesh.vertices.size(), 22u);
	for (const Mesh::Vertex& vertex : mesh.vertices)
		EXPECT_LT(vertex.x(), 20.f);
}

TEST(MeshRepairTest, RemoveSpuriousComponentsKeepsLongEdgeFaces)
{
	// the component pass alone: the detached triangle spans 14 units, far above
	// percentile55 x factor, so it stays even though its edges are the longest
	Mesh mesh = MakeStripWithDetachedTriangle(10.f);
	EXPECT_EQ(mesh.RemoveSpuriousComponents(1.f), 0u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 21u);
	EXPECT_EQ(mesh.RemoveLongEdgeFaces(1.f), 1u) << "the edge pass is what drops it";
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_EQ(mesh.faces.size(), 20u);
}

TEST(MeshRepairTest, RemoveSpuriousComponentsNativeMatchesArrayReference)
{
	Mesh arrays = MakeStripWithDetachedTriangle(0.01f);
	Mesh native = arrays;
	native.ListHalfEdges();

	EXPECT_EQ(RemoveSpuriousComponentsArraysReference(arrays, 2.f), native.RemoveSpuriousComponents(2.f));
	EXPECT_EQ(native.vertices, arrays.vertices);
	EXPECT_EQ(native.faces, arrays.faces);
	EXPECT_FALSE(native.halfMesh.Empty());
	EXPECT_TRUE(native.ValidateHalfMesh());
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

// Splitting the bowtie vertex appends a duplicate of it; vertexColors is
// parallel to `vertices` (Mesh::vertexColors), so the copy has to carry the
// source colour -- exactly as RemoveFacesHalfEdgeImpl does for pinch splits.
TEST(MeshRepairTest, FixNonManifoldKeepsVertexColorsInLockstep)
{
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // v0: bowtie centre
	    {1.f, 1.f, 0.f},
	    {-1.f, 1.f, 0.f},
	    {1.f, -1.f, 0.f},
	    {-1.f, -1.f, 0.f},
	};
	m.faces = {{0, 1, 2}, {0, 3, 4}};
	for (uint8_t i = 0; i < 5; ++i)
		m.vertexColors.emplace_back(i, i, i); // vertex i tagged colour i

	ASSERT_GE(m.FixNonManifold(/*thMoveDuplicate=*/0.01f), 1u);

	ASSERT_EQ(m.vertexColors.size(), m.vertices.size());
	EXPECT_TRUE(m.ValidateInvariants());
	// the originals keep their tags, and the split copies inherit v0's
	for (uint8_t i = 0; i < 5; ++i)
		EXPECT_EQ(static_cast<int>(m.vertexColors[i].x()), i) << "original at slot " << unsigned(i);
	for (size_t i = 5; i < m.vertexColors.size(); ++i)
		EXPECT_EQ(static_cast<int>(m.vertexColors[i].x()), 0) << "split copy of v0 at slot " << i;
}

TEST(MeshRepairTest, FixNonManifoldManifoldMeshUnchanged)
{
	Mesh m = MakeTetra();
	const unsigned fixed = m.FixNonManifold();
	EXPECT_EQ(fixed, 0u);
	EXPECT_EQ(m.vertices.size(), 4u);
	EXPECT_EQ(m.faces.size(), 4u);
}

TEST(MeshRepairTest, FixNonManifoldPrebuiltMeshIsUntouchedWithoutBuild)
{
	Mesh m = MakeTetra();
	m.ListHalfEdges();
	const auto vertices = m.vertices;
	const auto faces = m.faces;
	const auto heNexts = m.halfMesh.heNexts;
	const auto heVertices = m.halfMesh.heVertices;
	const auto heFaces = m.halfMesh.heFaces;
	const auto vHalfedges = m.halfMesh.vHalfedges;
	const auto fHalfedges = m.halfMesh.fHalfedges;

	HalfMesh::ResetBuildCount();
	EXPECT_EQ(m.FixNonManifold(), 0u);
	EXPECT_EQ(HalfMesh::BuildCount(), 0u);
	EXPECT_EQ(m.vertices, vertices);
	EXPECT_EQ(m.faces, faces);
	EXPECT_EQ(m.halfMesh.heNexts, heNexts);
	EXPECT_EQ(m.halfMesh.heVertices, heVertices);
	EXPECT_EQ(m.halfMesh.heFaces, heFaces);
	EXPECT_EQ(m.halfMesh.vHalfedges, vHalfedges);
	EXPECT_EQ(m.halfMesh.fHalfedges, fHalfedges);
	EXPECT_TRUE(m.ValidateHalfMesh());
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
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {1.f, 1.f, 0.f}, // tri 0
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

// ---------------------------------------------------------------------------
// RemoveSpikes
// ---------------------------------------------------------------------------

// A closed surface has no vertex below valence 2, so nothing is a spike.
TEST(MeshRepairTest, RemoveSpikesClosedSurfaceIsNoOp)
{
	Mesh mesh = MakeTetra();
	EXPECT_EQ(mesh.RemoveSpikes(), 0u);
	EXPECT_EQ(mesh.vertices.size(), 4u);
	EXPECT_EQ(mesh.faces.size(), 4u);
}

TEST(MeshRepairTest, RemoveSpikesEmptyMeshIsNoOp)
{
	Mesh mesh;
	EXPECT_EQ(mesh.RemoveSpikes(), 0u);
	EXPECT_TRUE(mesh.Empty());
}

// An isolated vertex is incident to zero faces, so it is a spike and no face
// is harmed removing it.
TEST(MeshRepairTest, RemoveSpikesDropsIsolatedVertex)
{
	Mesh mesh = MakeTetra();
	mesh.vertices.push_back(Mesh::Vertex(5.f, 5.f, 5.f));
	EXPECT_EQ(mesh.RemoveSpikes(), 1u);
	EXPECT_EQ(mesh.vertices.size(), 4u);
	EXPECT_EQ(mesh.faces.size(), 4u);
}

// A triangle hanging off the surface by a single edge: its free tip is incident
// to exactly one face, so tip and triangle both go.
TEST(MeshRepairTest, RemoveSpikesDropsDanglingTriangle)
{
	Mesh mesh = MakeTetra();
	const Mesh::VIndex tip = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back(Mesh::Vertex(2.f, 0.f, 0.f));
	mesh.faces.push_back(Mesh::Face(0, 1, tip));
	EXPECT_EQ(mesh.RemoveSpikes(), 1u);
	EXPECT_EQ(mesh.vertices.size(), 4u);
	EXPECT_EQ(mesh.faces.size(), 4u);
}

// Removing a spike can starve its neighbour down to a single face, so the sweep
// has to iterate: here the chain unwinds one triangle per round.
TEST(MeshRepairTest, RemoveSpikesUnwindsChainOverIterations)
{
	Mesh mesh = MakeTetra();
	const Mesh::VIndex mid = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back(Mesh::Vertex(2.f, 0.f, 0.f));
	const Mesh::VIndex tip = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back(Mesh::Vertex(3.f, 0.f, 0.f));
	mesh.faces.push_back(Mesh::Face(0, 1, mid)); // attached to the tetra by edge 0-1
	mesh.faces.push_back(Mesh::Face(mid, 1, tip)); // attached to the previous by edge mid-1
	// `mid` starts with two incident faces and only becomes a spike after `tip`
	// takes the outer triangle with it.
	EXPECT_EQ(mesh.RemoveSpikes(), 2u);
	EXPECT_EQ(mesh.vertices.size(), 4u);
	EXPECT_EQ(mesh.faces.size(), 4u);
}

// maxIterations bounds the sweep: one round peels exactly one link of the chain.
TEST(MeshRepairTest, RemoveSpikesHonorsIterationLimit)
{
	Mesh mesh = MakeTetra();
	const Mesh::VIndex mid = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back(Mesh::Vertex(2.f, 0.f, 0.f));
	const Mesh::VIndex tip = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back(Mesh::Vertex(3.f, 0.f, 0.f));
	mesh.faces.push_back(Mesh::Face(0, 1, mid));
	mesh.faces.push_back(Mesh::Face(mid, 1, tip));
	EXPECT_EQ(mesh.RemoveSpikes(1), 1u);
	EXPECT_EQ(mesh.vertices.size(), 5u);
	EXPECT_EQ(mesh.faces.size(), 5u);
}

TEST(MeshRepairTest, RemoveSpikesNativeMatchesArrayCascade)
{
	Mesh arrays = MakeTetra();
	std::vector<Mesh::FIndex> openFace{0};
	arrays.RemoveFaces(openFace);
	const Mesh::VIndex mid = static_cast<Mesh::VIndex>(arrays.vertices.size());
	arrays.vertices.push_back(Mesh::Vertex(2.f, 0.f, 0.f));
	const Mesh::VIndex tip = static_cast<Mesh::VIndex>(arrays.vertices.size());
	arrays.vertices.push_back(Mesh::Vertex(3.f, 0.f, 0.f));
	arrays.faces.push_back(Mesh::Face(1, 0, mid));
	arrays.faces.push_back(Mesh::Face(mid, 0, tip));
	Mesh native = arrays;
	native.ListHalfEdges();

	EXPECT_EQ(arrays.RemoveSpikesArrays(), native.RemoveSpikes());
	EXPECT_EQ(native.vertices, arrays.vertices);
	EXPECT_EQ(native.faces, arrays.faces);
	EXPECT_FALSE(native.halfMesh.Empty());
	EXPECT_TRUE(native.ValidateHalfMesh());
}

TEST(MeshRepairTest, RemoveSpikesNativeMatchesArrayIterationLimit)
{
	Mesh arrays = MakeTetra();
	std::vector<Mesh::FIndex> openFace{0};
	arrays.RemoveFaces(openFace);
	const Mesh::VIndex mid = static_cast<Mesh::VIndex>(arrays.vertices.size());
	arrays.vertices.push_back(Mesh::Vertex(2.f, 0.f, 0.f));
	const Mesh::VIndex tip = static_cast<Mesh::VIndex>(arrays.vertices.size());
	arrays.vertices.push_back(Mesh::Vertex(3.f, 0.f, 0.f));
	arrays.faces.push_back(Mesh::Face(1, 0, mid));
	arrays.faces.push_back(Mesh::Face(mid, 0, tip));
	Mesh native = arrays;
	native.ListHalfEdges();

	EXPECT_EQ(arrays.RemoveSpikesArrays(1), native.RemoveSpikes(1));
	EXPECT_EQ(native.vertices, arrays.vertices);
	EXPECT_EQ(native.faces, arrays.faces);
	EXPECT_TRUE(native.ValidateHalfMesh());
}

TEST(MeshRepairTest, RemoveSpikesDispatchAttributePolicy)
{
	Mesh arrays = MakeTetra();
	const Mesh::VIndex tip = static_cast<Mesh::VIndex>(arrays.vertices.size());
	arrays.vertices.push_back(Mesh::Vertex(2.f, 0.f, 0.f));
	arrays.faces.push_back(Mesh::Face(0, 1, tip));
	arrays.faceTexcoords.resize(arrays.faces.size() * 3, Mesh::TexCoord::Zero());
	Mesh native = arrays;
	native.ListHalfEdgesSafe();

	arrays.RemoveSpikes();
	EXPECT_EQ(arrays.faceTexcoords.size(), arrays.faces.size() * 3);
	native.RemoveSpikes();
	EXPECT_TRUE(native.faceTexcoords.empty());
}

// vertexNormals rides the same per-vertex lockstep as vertexColors: the bowtie
// split copy inherits the source normal (Mesh::VertexAttributesAppendFrom), so
// the array stays parallel to `vertices`.
TEST(MeshRepairTest, FixNonManifoldKeepsVertexNormalsInLockstep)
{
	Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // v0: bowtie centre
	    {1.f, 1.f, 0.f},
	    {-1.f, 1.f, 0.f},
	    {1.f, -1.f, 0.f},
	    {-1.f, -1.f, 0.f},
	};
	m.faces = {{0, 1, 2}, {0, 3, 4}};
	for (int i = 0; i < 5; ++i)
		m.vertexNormals.emplace_back(float(i), 0.f, 1.f); // vertex i tagged by x

	ASSERT_GE(m.FixNonManifold(/*thMoveDuplicate=*/0.01f), 1u);

	ASSERT_EQ(m.vertexNormals.size(), m.vertices.size());
	EXPECT_TRUE(m.ValidateInvariants());
	// the originals keep their tags, and the split copies inherit v0's
	for (int i = 0; i < 5; ++i)
		EXPECT_FLOAT_EQ(m.vertexNormals[i].x(), float(i)) << "original at slot " << i;
	for (size_t i = 5; i < m.vertexNormals.size(); ++i)
		EXPECT_FLOAT_EQ(m.vertexNormals[i].x(), 0.f) << "split copy of v0 at slot " << i;
}

// Welding renumbers vertices but never moves them, so an authored normal stays
// valid for every survivor and must be carried through the remap rebuild.
TEST(MeshRepairTest, RemoveDuplicateVerticesKeepsVertexNormals)
{
	Mesh m;
	// a quad as two triangles that do NOT share vertices: v2/v3 duplicate v1/v0
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.f, 1.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.f, 1.f, 0.f},
	    {1.f, 1.f, 0.f},
	};
	m.faces = {{0, 1, 2}, {3, 5, 4}};
	// tag every vertex by its position, so the survivor's normal is recognizable
	for (const Mesh::Vertex& v : m.vertices)
		m.vertexNormals.emplace_back(v.x(), v.y(), 1.f);

	ASSERT_EQ(m.RemoveDuplicateVertices(), 2u);

	ASSERT_EQ(m.vertexNormals.size(), m.vertices.size());
	EXPECT_TRUE(m.ValidateInvariants());
	// every survivor still carries the tag matching its own position
	for (size_t i = 0; i < m.vertices.size(); ++i) {
		EXPECT_FLOAT_EQ(m.vertexNormals[i].x(), m.vertices[i].x()) << "slot " << i;
		EXPECT_FLOAT_EQ(m.vertexNormals[i].y(), m.vertices[i].y()) << "slot " << i;
	}
}

} // namespace
} // namespace halfmesh
