/*
* MeshCoreTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Mesh core tests: construction, normals, area, AABB, adjacency, edit primitives.
// All meshes are built inline; no I/O used.
#include <gtest/gtest.h>

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <set>

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// Helper: build a Mesh from a vertex list and face list.
// ---------------------------------------------------------------------------
static Mesh BuildMesh(std::vector<Mesh::Vertex> verts,
                      std::vector<std::array<uint32_t, 3>> tris)
{
	Mesh m;
	m.vertices = std::move(verts);
	for (auto& t : tris) {
		Mesh::Face f;
		f[0] = t[0];
		f[1] = t[1];
		f[2] = t[2];
		m.faces.push_back(f);
	}
	return m;
}

// ---------------------------------------------------------------------------
// ComputeFaceNormals — single triangle in the XY plane (z=0),
// CCW winding → normal should point +Z.
// Triangle: (0,0,0), (1,0,0), (0,1,0)
// edge01 = (1,0,0), edge02 = (0,1,0)
// cross = edge01 × edge02 = (0,0,1) → normalized = (0,0,1)
// ---------------------------------------------------------------------------
TEST(MeshCore, ComputeFaceNormals_SingleTriangle)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	m.ComputeFaceNormals();
	ASSERT_EQ(m.faceNormals.size(), 1u);
	const auto& n = m.faceNormals[0];
	EXPECT_NEAR(n.x(), 0.f, 1e-6f);
	EXPECT_NEAR(n.y(), 0.f, 1e-6f);
	EXPECT_NEAR(n.z(), 1.f, 1e-6f);
}

// CW winding → normal points -Z.
TEST(MeshCore, ComputeFaceNormals_CWWinding)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 0.f, 0.f}},
	    {{0, 1, 2}});
	m.ComputeFaceNormals();
	ASSERT_EQ(m.faceNormals.size(), 1u);
	const auto& n = m.faceNormals[0];
	EXPECT_NEAR(n.x(), 0.f, 1e-6f);
	EXPECT_NEAR(n.y(), 0.f, 1e-6f);
	EXPECT_NEAR(n.z(), -1.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// ComputeArea — right triangle with legs (1,1) → area 0.5.
// ---------------------------------------------------------------------------
TEST(MeshCore, ComputeArea_RightTriangle)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	const double area = m.ComputeArea();
	EXPECT_NEAR(area, 0.5, 1e-10);
}

// Unit square made of two right triangles → area 1.0.
// (0,0,0), (1,0,0), (1,1,0), (0,1,0)
// face0: 0,1,2 — area 0.5
// face1: 0,2,3 — area 0.5
TEST(MeshCore, ComputeArea_UnitSquare)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	const double area = m.ComputeArea();
	EXPECT_NEAR(area, 1.0, 1e-10);
}

// ComputeArea with indices — sum only face 0 of a two-face mesh → 0.5.
TEST(MeshCore, ComputeArea_WithIndices)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	const std::vector<Mesh::FIndex> idx = {0};
	EXPECT_NEAR(m.ComputeArea(idx), 0.5, 1e-10);
}

// ---------------------------------------------------------------------------
// ComputeAABBox — known point set.
// Vertices: (0,0,0), (3,0,0), (3,2,0), (0,2,1)
// Expected min=(0,0,0), max=(3,2,1)
// ---------------------------------------------------------------------------
TEST(MeshCore, ComputeAABBox)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {3.f, 0.f, 0.f}, {3.f, 2.f, 0.f}, {0.f, 2.f, 1.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	auto bbox = m.ComputeAABBox();
	EXPECT_NEAR(bbox.min().x(), 0.f, 1e-6f);
	EXPECT_NEAR(bbox.min().y(), 0.f, 1e-6f);
	EXPECT_NEAR(bbox.min().z(), 0.f, 1e-6f);
	EXPECT_NEAR(bbox.max().x(), 3.f, 1e-6f);
	EXPECT_NEAR(bbox.max().y(), 2.f, 1e-6f);
	EXPECT_NEAR(bbox.max().z(), 1.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// ListVertexFaces — two-triangle quad sharing vertices 0,2.
// Mesh: vertices 0,1,2,3; face0=(0,1,2), face1=(0,2,3)
// vertex 0 → faces {0,1}, vertex 1 → faces {0},
// vertex 2 → faces {0,1}, vertex 3 → faces {1}
// ---------------------------------------------------------------------------
TEST(MeshCore, ListVertexFaces_TwoTriangles)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	ASSERT_EQ(m.vertexFaces.size(), 4u);
	EXPECT_EQ(m.vertexFaces[0], (Mesh::VertexFaces{0, 1}));
	EXPECT_EQ(m.vertexFaces[1], (Mesh::VertexFaces{0}));
	EXPECT_EQ(m.vertexFaces[2], (Mesh::VertexFaces{0, 1}));
	EXPECT_EQ(m.vertexFaces[3], (Mesh::VertexFaces{1}));
}

// ValidateVertexFaces returns true when vertexFaces is already correct.
TEST(MeshCore, ValidateVertexFaces)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	EXPECT_TRUE(m.ValidateVertexFaces());
}

// ---------------------------------------------------------------------------
// VAdjacentVertices — vertex 0 in two-triangle quad adjacent to {1,2,3}.
// ---------------------------------------------------------------------------
TEST(MeshCore, VAdjacentVertices)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	auto adj = m.VAdjacentVertices(0);
	std::sort(adj.begin(), adj.end());
	const std::vector<Mesh::VIndex> expected = {1, 2, 3};
	EXPECT_EQ(adj, expected);
}

// Vertex 1 in the two-triangle quad is only adjacent to {0,2}.
TEST(MeshCore, VAdjacentVertices_Boundary)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	auto adj = m.VAdjacentVertices(1);
	std::sort(adj.begin(), adj.end());
	const std::vector<Mesh::VIndex> expected = {0, 2};
	EXPECT_EQ(adj, expected);
}

// ---------------------------------------------------------------------------
// FVertexIdx — locate a vertex index within a face.
// ---------------------------------------------------------------------------
TEST(MeshCore, FVertexIdx)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	EXPECT_EQ(m.FVertexIdx(0, 0), 0u);
	EXPECT_EQ(m.FVertexIdx(0, 1), 1u);
	EXPECT_EQ(m.FVertexIdx(0, 2), 2u);
	EXPECT_EQ(m.FVertexIdx(0, 99), math::NO_ID);
}

// ---------------------------------------------------------------------------
// FSameVertices — two faces with the same vertex set (different winding).
// ---------------------------------------------------------------------------
TEST(MeshCore, FSameVertices)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 1}} // same vertices, reverse winding
	);
	EXPECT_TRUE(m.FSameVertices(0, 1));
	EXPECT_TRUE(m.FSameVertices(1, 0));
}

// ---------------------------------------------------------------------------
// FEdgeOrientation — CCW face (0,1,2): edge 0→1 is forward, edge 0→2 backward.
// ---------------------------------------------------------------------------
TEST(MeshCore, FEdgeOrientation)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	// forward: (i0+1)%3 == iV1
	EXPECT_TRUE(m.FEdgeOrientation(0, 0, 1)); // 0→1: forward
	EXPECT_FALSE(m.FEdgeOrientation(0, 0, 2)); // 0→2: backward (2 is at (i0+2)%3)
}

// ---------------------------------------------------------------------------
// FEdgeAdjacentFace — two triangles sharing edge (0,2):
// face0=(0,1,2), face1=(0,2,3).
// Edge 0→2 in face0 (backward) should be adjacent to face1 (forward 0→2).
// ---------------------------------------------------------------------------
TEST(MeshCore, FEdgeAdjacentFace)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	// Edge shared between face0 and face1 is (0,2).
	// In face0: 0 is at slot 0, 2 is at slot 2 → backward (FEdgeOrientation=false).
	// In face1: 0 is at slot 0, 2 is at slot 1 → forward (FEdgeOrientation=true).
	// They have opposite orientation → valid manifold edge.
	const Mesh::FIndex adj = m.FEdgeAdjacentFace(0, 0, 2);
	EXPECT_EQ(adj, 1u);
	// Reverse query: adjacent to face1 over edge (0,2) is face0.
	const Mesh::FIndex adj2 = m.FEdgeAdjacentFace(1, 0, 2);
	EXPECT_EQ(adj2, 0u);
	// Non-shared edge (0,1) in face0 has no adjacent face.
	const Mesh::FIndex noAdj = m.FEdgeAdjacentFace(0, 0, 1);
	EXPECT_EQ(noAdj, math::NO_ID);
}

// ---------------------------------------------------------------------------
// ListHalfEdges — builds halfMesh; spot-check vHalfedges populated.
// ---------------------------------------------------------------------------
TEST(MeshCore, ListHalfEdges_PopulatesHalfMesh)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	EXPECT_TRUE(m.halfMesh.Empty());
	m.ListHalfEdges();
	EXPECT_FALSE(m.halfMesh.Empty());
	// 4 vertices → 4 entries in vHalfedges
	EXPECT_EQ(m.halfMesh.vHalfedges.size(), 4u);
	// 2 faces → 2 entries in fHalfedges
	EXPECT_EQ(m.halfMesh.fHalfedges.size(), 2u);
	// Calling again should be a no-op (already built)
	const auto oldSize = m.halfMesh.vHalfedges.size();
	m.ListHalfEdges();
	EXPECT_EQ(m.halfMesh.vHalfedges.size(), oldSize);
}

// Adjacency via HalfMesh after ListHalfEdges: face 0 adjacent to face 1 over shared edge.
TEST(MeshCore, ListHalfEdges_FAdjacentFaces)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListHalfEdges();
	// FAdjacentFaces(0) should include face 1 (they share edge 0-2)
	std::vector<Mesh::FIndex> adjFaces;
	for (Mesh::FIndex f : m.halfMesh.FAdjacentFaces(0))
		adjFaces.push_back(f);
	const bool hasFace1 = std::find(adjFaces.begin(), adjFaces.end(), 1u) != adjFaces.end();
	EXPECT_TRUE(hasFace1);
}

TEST(MeshCore, HalfEdgeGateRequiresExplicitInvalidationAfterDirectFaceEdit)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListHalfEdges();
	m.faces = {Mesh::Face(0, 1, 3), Mesh::Face(1, 2, 3)};
	m.ListHalfEdges();
	std::vector<Mesh::Face> harvested;
	m.halfMesh.FFaces(harvested);
	EXPECT_EQ(harvested[0][2], 2u) << "a live half-edge is trusted until explicitly invalidated";

	m.InvalidateHalfMesh();
	m.ListHalfEdges();
	harvested.clear();
	m.halfMesh.FFaces(harvested);
	ASSERT_EQ(harvested.size(), m.faces.size());
	for (size_t i = 0; i < harvested.size(); ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_EQ(harvested[i][j], m.faces[i][j]);
}

TEST(MeshCore, RepresentationStateRoundTrip)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	const std::vector<Mesh::Face> originalFaces = m.faces;
	EXPECT_TRUE(m.halfMesh.Empty());
	m.ListHalfEdges();
	EXPECT_TRUE(m.ValidateInvariants());
	m.InvalidateFaces();
	EXPECT_TRUE(m.faces.empty());
	EXPECT_FALSE(m.halfMesh.Empty());
	EXPECT_TRUE(m.ValidateInvariants());
	m.SyncFaces();
	m.SyncFaces();
	ASSERT_EQ(m.faces.size(), originalFaces.size()) << "SyncFaces must not append twice";
	for (size_t i = 0; i < originalFaces.size(); ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_EQ(m.faces[i][j], originalFaces[i][j]);
}

TEST(MeshCore, RepresentationInvariantAndHalfMeshValidation)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListHalfEdges();
	EXPECT_TRUE(m.ValidateInvariants());
	EXPECT_TRUE(m.ValidateHalfMesh());
	m.vertices.emplace_back(2.f, 2.f, 0.f);
	EXPECT_FALSE(m.ValidateInvariants());
	m.vertices.pop_back();
	EXPECT_TRUE(m.ValidateInvariants());
}

TEST(MeshCore, InvalidateFacesDropsAttributesAndWarnsOnce)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	m.ListHalfEdges();
	m.faceNormals.emplace_back(0.f, 0.f, 1.f);
	m.faceTexcoords.resize(3);
	m.faceTexblobs.emplace_back(0);
	m.texturesDiffuse.emplace_back(1, 1);
	m.ListVertexFaces();
	testing::internal::CaptureStderr();
	m.InvalidateFaces();
	const std::string firstWarning = testing::internal::GetCapturedStderr();
	EXPECT_NE(firstWarning.find("face attributes dropped: processing methods expect untextured meshes"), std::string::npos);
	EXPECT_TRUE(m.faces.empty());
	EXPECT_TRUE(m.faceNormals.empty());
	EXPECT_TRUE(m.faceTexcoords.empty());
	EXPECT_TRUE(m.faceTexblobs.empty());
	EXPECT_TRUE(m.texturesDiffuse.empty());
	EXPECT_TRUE(m.vertexFaces.empty());

	m.faceNormals.emplace_back(0.f, 0.f, 1.f);
	testing::internal::CaptureStderr();
	m.InvalidateFaces();
	EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

// ---------------------------------------------------------------------------
// RemoveFaces — remove face 0 from a two-face mesh; expect 1 face left.
// updateLists=false → vertexFaces cleared.
// ---------------------------------------------------------------------------
TEST(MeshCore, RemoveFaces_Simple)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	std::vector<Mesh::FIndex> removes = {0};
	m.RemoveFaces(removes, /*updateLists=*/false);
	EXPECT_EQ(m.faces.size(), 1u);
	EXPECT_EQ(m.vertices.size(), 4u); // vertices untouched
}

// RemoveFaces with updateLists=true: vertexFaces kept consistent.
TEST(MeshCore, RemoveFaces_UpdateLists)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	std::vector<Mesh::FIndex> removes = {0};
	m.RemoveFaces(removes, /*updateLists=*/true);
	ASSERT_EQ(m.faces.size(), 1u);
	ASSERT_EQ(m.vertexFaces.size(), 4u);
	// vertex 1 was only in face 0; it must now have empty face list
	EXPECT_TRUE(m.vertexFaces[1].empty());
}

// Build an N x N quad grid (2*N*N triangles, consistent CCW winding).
static Mesh BuildGrid(int N)
{
	Mesh m;
	for (int y = 0; y <= N; ++y)
		for (int x = 0; x <= N; ++x)
			m.vertices.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.f);
	const auto vid = [N](int x, int y) { return static_cast<Mesh::VIndex>(y * (N + 1) + x); };
	for (int y = 0; y < N; ++y)
		for (int x = 0; x < N; ++x) {
			Mesh::Face f0, f1;
			f0[0] = vid(x, y);
			f0[1] = vid(x + 1, y);
			f0[2] = vid(x + 1, y + 1);
			f1[0] = vid(x, y);
			f1[1] = vid(x + 1, y + 1);
			f1[2] = vid(x, y + 1);
			m.faces.push_back(f0);
			m.faces.push_back(f1);
		}
	return m;
}

// RemoveFaces incremental path (small removal fraction, updateLists=true): the
// O(d) pop-tail + lower_bound reinsertion must keep vertexFaces byte-identical
// to a fresh ListVertexFaces() rebuild (ValidateVertexFaces recomputes+compares).
TEST(MeshCore, RemoveFaces_UpdateListsIncrementalConsistent)
{
	Mesh m = BuildGrid(6); // 72 faces
	m.ListVertexFaces();
	ASSERT_GT(m.faces.size(), 20u);
	std::vector<Mesh::FIndex> removes = {3, 17}; // small fraction -> incremental path
	m.RemoveFaces(removes, /*updateLists=*/true);
	EXPECT_EQ(m.faces.size(), 70u);
	EXPECT_TRUE(m.ValidateVertexFaces())
	    << "incremental vertex_faces diverged from a fresh ListVertexFaces()";
}

// RemoveFaces bulk-rebuild path (large removal fraction): compact then one
// ListVertexFaces() rebuild; result must be consistent as well.
TEST(MeshCore, RemoveFaces_UpdateListsBulkRebuildConsistent)
{
	Mesh m = BuildGrid(6); // 72 faces
	m.ListVertexFaces();
	std::vector<Mesh::FIndex> removes;
	for (Mesh::FIndex i = 0; i < 40; ++i) // >10% -> bulk rebuild path
		removes.push_back(i);
	m.RemoveFaces(removes, /*updateLists=*/true);
	EXPECT_EQ(m.faces.size(), 32u);
	EXPECT_TRUE(m.ValidateVertexFaces());
}

// ---------------------------------------------------------------------------
// RemoveUnreferencedVertices — after removing face 0 (which referenced vertex 1
// exclusively), vertex 1 is unreferenced and should be removed.
// ---------------------------------------------------------------------------
TEST(MeshCore, RemoveUnreferencedVertices)
{
	// vertex 4 is isolated (not referenced by any face)
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {9.f, 9.f, 9.f}}, // vertex 4: unreferenced
	    {{0, 1, 2}, {0, 2, 3}});
	m.ListVertexFaces();
	const Mesh::VIndex removed = m.RemoveUnreferencedVertices();
	EXPECT_EQ(removed, 1u); // one vertex removed
	EXPECT_EQ(m.vertices.size(), 4u); // 5-1=4 vertices remain
	// all face vertices must still be in range [0,4)
	for (const auto& face : m.faces)
		for (int i = 0; i < 3; ++i)
			EXPECT_LT(face[i], static_cast<Mesh::VIndex>(m.vertices.size()));
}

// RemoveUnreferencedVertices must keep vertexColors in lockstep with the
// swap-compaction (sibling RemoveVertices/RemoveFaces already do): pre-fix the
// colors array was left untouched -- wrong size AND stale entries -- silently
// desynchronizing color from vertex on any colored mesh. Vertex i is tagged
// x==i and color r==i so any mismatch
// is visible.
TEST(MeshCore, RemoveUnreferencedVerticesRemapsVertexColors)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {3.f, 0.f, 0.f}, {4.f, 0.f, 0.f}},
	    {{1, 2, 3}, {2, 3, 4}}); // vertex 0 is unreferenced
	for (uint8_t i = 0; i < 5; ++i)
		m.vertexColors.emplace_back(i, i, i);
	m.ListVertexFaces();
	EXPECT_EQ(m.RemoveUnreferencedVertices(), 1u);
	ASSERT_EQ(m.vertexColors.size(), m.vertices.size());
	// vertex 4 was swapped into slot 0; its color must follow
	for (size_t i = 0; i < m.vertices.size(); ++i)
		EXPECT_EQ(static_cast<float>(m.vertexColors[i].x()), m.vertices[i].x())
		    << "color/vertex desync at slot " << i;
}

// RemoveVertices' documented contract is "remove specified vertices along with
// all faces referencing them"; the old default (updateLists=false) silently
// skipped the face removal, leaving faces pointing at whatever vertex was
// swapped into the freed slot. The
// default must honor the doc; the fast path stays available as an explicit
// opt-out.
TEST(MeshCore, RemoveVerticesDefaultRemovesReferencingFaces)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}, {0, 2, 3}}); // vertex 1 is referenced only by face 0
	m.ListVertexFaces();
	std::vector<Mesh::VIndex> removes = {1};
	m.RemoveVertices(removes); // default args -- must remove referencing faces
	EXPECT_EQ(m.vertices.size(), 3u);
	ASSERT_EQ(m.faces.size(), 1u) << "the face referencing removed vertex 1 must be gone";
	for (int i = 0; i < 3; ++i)
		EXPECT_LT(m.faces[0][i], static_cast<Mesh::VIndex>(m.vertices.size()));
}

// Mesh::ECollapse swap-pops the removed vertex's position slot but left
// vertexColors untouched -- wrong size AND a stale entry at the swapped slot
// on any colored mesh. Colors are tagged so the color channel == x + 3*y of the paired
// position; the pairing must survive the collapse regardless of which
// endpoint ERemove removes or which slot the tail vertex is swapped into.
TEST(MeshCore, ECollapseRemapsVertexColors)
{
	// 3x3 vertex grid (positions (x,y,0), id = y*3+x), 8 triangles.
	std::vector<Mesh::Vertex> verts;
	for (int y = 0; y < 3; ++y)
		for (int x = 0; x < 3; ++x)
			verts.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.f);
	std::vector<std::array<uint32_t, 3>> tris;
	for (uint32_t y = 0; y < 2; ++y)
		for (uint32_t x = 0; x < 2; ++x) {
			const uint32_t v00 = y * 3 + x, v10 = v00 + 1, v01 = v00 + 3, v11 = v01 + 1;
			tris.push_back({v00, v10, v11});
			tris.push_back({v00, v11, v01});
		}
	Mesh m = BuildMesh(std::move(verts), std::move(tris));
	for (uint8_t i = 0; i < 9; ++i)
		m.vertexColors.emplace_back(i, i, i);
	m.ListHalfEdges();
	ASSERT_FALSE(m.halfMesh.Empty());
	const Mesh::EIndex iE = m.halfMesh.EEdge(4, 5); // centre vertex to its right neighbour
	ASSERT_NE(iE, math::NO_ID);
	m.ECollapse(iE);
	ASSERT_EQ(m.vertexColors.size(), m.vertices.size());
	for (size_t s = 0; s < m.vertices.size(); ++s) {
		const int expect = static_cast<int>(m.vertices[s].x() + 3.f * m.vertices[s].y() + 0.5f);
		EXPECT_EQ(static_cast<int>(m.vertexColors[s].x()), expect)
		    << "color/vertex desync at slot " << s;
	}
}

// ---------------------------------------------------------------------------
// ComputeVertexNormals — single triangle; all 3 vertex normals equal face normal.
// ---------------------------------------------------------------------------
TEST(MeshCore, ComputeVertexNormals)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	const auto vnormals = m.ComputeVertexNormals();
	ASSERT_EQ(vnormals.size(), 3u);
	for (const auto& vn : vnormals) {
		EXPECT_NEAR(vn.x(), 0.f, 1e-6f);
		EXPECT_NEAR(vn.y(), 0.f, 1e-6f);
		EXPECT_NEAR(vn.z(), 1.f, 1e-6f);
	}
}

// ---------------------------------------------------------------------------
// ComputeVertexNormals — angle-weighted (Thurmer-Wuthrich) normals are
// tessellation-independent.  A corner vertex shared by three mutually-
// perpendicular faces (normals +X,+Y,+Z, each subtending a 90 deg corner) has
// ground-truth normal (1,1,1)/sqrt(3).  Subdividing ONE of the three faces into
// K coplanar slivers must NOT move the vertex normal: the sub-triangles' corner
// angles still sum to 90 deg.  Uniform averaging of unit face normals instead
// counts the subdivided face K times and skews badly toward its axis.
// ---------------------------------------------------------------------------
TEST(MeshCore, ComputeVertexNormals_AngleWeightedTessellationIndependent)
{
	static constexpr int K = 12; // slivers subdividing the +Z face over a 90 deg fan

	Mesh m;
	// vertex 0 is the shared corner V at the origin
	m.vertices.emplace_back(0.f, 0.f, 0.f);
	// arc points p_0..p_K in the XY plane (z=0): a 90 deg fan from +X to +Y,
	// all sub-triangles (V, p_i, p_{i+1}) share the +Z face normal.
	const Mesh::VIndex arc0 = static_cast<Mesh::VIndex>(m.vertices.size());
	for (int i = 0; i <= K; ++i) {
		const float t = static_cast<float>(i) * (static_cast<float>(M_PI) / 2.f) / static_cast<float>(K);
		m.vertices.emplace_back(std::cos(t), std::sin(t), 0.f);
	}
	// face 1 (YZ plane, +X normal): V, (0,1,0), (0,0,1)
	const Mesh::VIndex c = static_cast<Mesh::VIndex>(m.vertices.size());
	m.vertices.emplace_back(0.f, 1.f, 0.f);
	const Mesh::VIndex d = static_cast<Mesh::VIndex>(m.vertices.size());
	m.vertices.emplace_back(0.f, 0.f, 1.f);
	// face 2 (ZX plane, +Y normal): V, (0,0,1), (1,0,0)
	const Mesh::VIndex e = static_cast<Mesh::VIndex>(m.vertices.size());
	m.vertices.emplace_back(0.f, 0.f, 1.f);
	const Mesh::VIndex f = static_cast<Mesh::VIndex>(m.vertices.size());
	m.vertices.emplace_back(1.f, 0.f, 0.f);

	Mesh::Face face;
	for (int i = 0; i < K; ++i) {
		face[0] = 0;
		face[1] = arc0 + static_cast<Mesh::VIndex>(i);
		face[2] = arc0 + static_cast<Mesh::VIndex>(i) + 1;
		m.faces.push_back(face);
	}
	face[0] = 0;
	face[1] = c;
	face[2] = d;
	m.faces.push_back(face);
	face[0] = 0;
	face[1] = e;
	face[2] = f;
	m.faces.push_back(face);

	const std::vector<Mesh::Normal> vnormals = m.ComputeVertexNormals();
	ASSERT_EQ(vnormals.size(), m.vertices.size());
	const Mesh::Normal truth = Mesh::Normal(1.f, 1.f, 1.f).normalized();
	const Mesh::Normal n = vnormals[0].normalized();
	const float cosErr = std::clamp(n.dot(truth), -1.f, 1.f);
	const float errDeg = std::acos(cosErr) * 180.f / static_cast<float>(M_PI);
	// Angle weighting keeps V's normal on the true diagonal (~0 deg error);
	// the old uniform averaging skews ~48 deg toward +Z with K=12.
	EXPECT_LT(errDeg, 1.0f)
	    << "vertex normal should be tessellation-independent; angular error "
	    << errDeg << " deg";
}

// ---------------------------------------------------------------------------
// ReleaseOptional — clears all optional data.
// ---------------------------------------------------------------------------
TEST(MeshCore, ReleaseOptional)
{
	Mesh m = BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{0, 1, 2}});
	m.ComputeFaceNormals();
	m.ListVertexFaces();
	ASSERT_FALSE(m.faceNormals.empty());
	ASSERT_FALSE(m.vertexFaces.empty());
	m.ReleaseOptional();
	EXPECT_TRUE(m.faceNormals.empty());
	EXPECT_TRUE(m.vertexFaces.empty());
	EXPECT_TRUE(m.vertexColors.empty());
	EXPECT_TRUE(m.faceTexcoords.empty());
	EXPECT_TRUE(m.faceTexblobs.empty());
	EXPECT_TRUE(m.texturesDiffuse.empty());
}

// Public array surgery must invalidate the half-edge so the exact validity gate
// never trusts stale connectivity.
TEST(MeshCore, HalfEdgesRebuiltAfterRemoveFaces)
{
	Mesh mesh;
	mesh.vertices.push_back(Mesh::Vertex(0, 0, 0));
	mesh.vertices.push_back(Mesh::Vertex(1, 0, 0));
	mesh.vertices.push_back(Mesh::Vertex(0, 1, 0));
	mesh.vertices.push_back(Mesh::Vertex(0, 0, 1));
	// closed tetrahedron: every directed edge once, its reverse once
	Mesh::Face f0, f1, f2, f3;
	f0[0] = 0;
	f0[1] = 1;
	f0[2] = 2;
	f1[0] = 1;
	f1[1] = 0;
	f1[2] = 3;
	f2[0] = 2;
	f2[1] = 1;
	f2[2] = 3;
	f3[0] = 0;
	f3[1] = 2;
	f3[2] = 3;
	mesh.faces.push_back(f0);
	mesh.faces.push_back(f1);
	mesh.faces.push_back(f2);
	mesh.faces.push_back(f3);
	mesh.ListHalfEdges();
	ASSERT_EQ(mesh.halfMesh.FSize(), 4u);
	std::vector<Mesh::FIndex> removes{3}; // all 4 vertices stay referenced
	mesh.RemoveFaces(removes);
	ASSERT_EQ(mesh.faces.size(), 3u);
	mesh.ListHalfEdges();
	EXPECT_EQ(mesh.halfMesh.FSize(), mesh.faces.size());
	// the motivating consumer must now be safe to call as-is
	mesh.ComputeSmoothFaceNormals();
	ASSERT_EQ(mesh.faceNormals.size(), mesh.faces.size());
	for (const Mesh::Normal& n : mesh.faceNormals)
		EXPECT_TRUE(n.allFinite());
}

// An exactly-collinear (zero-area) face used to produce a NaN face normal
// (normalized zero cross product), which ComputeVertexNormals then silently
// accumulated into every touching vertex normal (2026-08 review). Degenerate
// faces must yield a zero normal and finite vertex normals.
TEST(MeshCoreTest, DegenerateFaceNormalsStayFinite)
{
	Mesh m;
	m.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0), Mesh::Vertex(2, 0, 0),
	              Mesh::Vertex(0, 1, 0)};
	m.faces = {Mesh::Face(0, 1, 2), // exactly collinear: zero-area
	           Mesh::Face(1, 0, 3)}; // valid
	m.ComputeFaceNormals();
	ASSERT_EQ(m.faceNormals.size(), 2u);
	EXPECT_EQ(m.faceNormals[0], Mesh::Normal(Mesh::Normal::Zero()));
	EXPECT_TRUE(m.faceNormals[1].allFinite());
	EXPECT_NEAR(m.faceNormals[1].norm(), 1.f, 1e-6f);
	for (const Mesh::Normal& n : m.ComputeVertexNormals())
		EXPECT_TRUE(n.allFinite());
}

// ComputeSmoothFaceNormals: a face with no admissible neighbor (isolated, or
// all neighbors beyond maxAngle) left the neighbor average at zero, whose
// normalization injected NaN into the blend. It must keep its own normal.
TEST(MeshCoreTest, SmoothFaceNormalsIsolatedFaceStaysFinite)
{
	Mesh m;
	m.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0), Mesh::Vertex(0, 1, 0)};
	m.faces = {Mesh::Face(0, 1, 2)};
	m.ComputeSmoothFaceNormals();
	ASSERT_EQ(m.faceNormals.size(), 1u);
	EXPECT_TRUE(m.faceNormals[0].allFinite());
	EXPECT_NEAR(m.faceNormals[0].norm(), 1.f, 1e-6f);
}

} // namespace
} // namespace halfmesh
