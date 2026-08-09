/*
* HalfMeshTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// HalfMesh half-edge core topology tests.
// Hand-verified topology on small inline meshes (no I/O).
#include <gtest/gtest.h>

#include <halfmesh/HalfMesh.h>
#include <halfmesh/Mesh.h>

#include <algorithm>
#include <set>
#include <vector>

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// Helper: collect vertex indices adjacent to a given vertex
// ---------------------------------------------------------------------------
static std::vector<HalfMesh::VIndex> AdjacentVertices(
    const HalfMesh& hm, HalfMesh::VIndex v)
{
	std::vector<HalfMesh::VIndex> result;
	for (HalfMesh::VIndex u : hm.VAdjacentVertices(v))
		result.push_back(u);
	std::sort(result.begin(), result.end());
	return result;
}

// Helper: collect face indices adjacent to a given vertex
static std::vector<HalfMesh::FIndex> AdjacentFaces(
    const HalfMesh& hm, HalfMesh::VIndex v)
{
	std::vector<HalfMesh::FIndex> result;
	for (HalfMesh::FIndex f : hm.VAdjacentFaces(v))
		result.push_back(f);
	std::sort(result.begin(), result.end());
	return result;
}

// Helper: build a Mesh inline (vertices are zero-position dummies — topology only)
static Mesh BuildMesh(uint32_t numVerts,
                      std::vector<std::array<uint32_t, 3>> triList)
{
	Mesh m;
	m.vertices.resize(numVerts, HalfMesh::Vertex::Zero());
	for (auto& t : triList) {
		HalfMesh::Face f;
		f[0] = t[0];
		f[1] = t[1];
		f[2] = t[2];
		m.faces.push_back(f);
	}
	return m;
}

// ---------------------------------------------------------------------------
// Fixture 1: Tetrahedron — 4 vertices, 4 faces, closed manifold (no boundary)
//   faces: (0,1,2), (0,2,3), (0,3,1), (1,3,2)
// Hand-verified expected topology:
//   - 4 vertices, 4 faces, 6 edges, 12 half-edges
//   - No boundary vertices
//   - Every vertex is adjacent to exactly 3 faces and 3 other vertices
//   - Specifically: v0 adj {1,2,3}, v1 adj {0,2,3}, v2 adj {0,1,3}, v3 adj {0,1,2}
// ---------------------------------------------------------------------------
TEST(HalfMeshTest, Tetrahedron_Counts)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	EXPECT_EQ(hm.VSize(), 4u);
	EXPECT_EQ(hm.FSize(), 4u);
	EXPECT_EQ(hm.ESize(), 6u);
	EXPECT_EQ(hm.HeSize(), 12u);
}

TEST(HalfMeshTest, Tetrahedron_NoBoundary)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	for (HalfMesh::VIndex v = 0; v < 4; ++v)
		EXPECT_FALSE(hm.VIsBoundary(v)) << "vertex " << v << " should not be boundary";
}

TEST(HalfMeshTest, Tetrahedron_VertexFaceDegree)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	for (HalfMesh::VIndex v = 0; v < 4; ++v)
		EXPECT_EQ(hm.VFaceDegree(v), 3u) << "vertex " << v << " should be incident to 3 faces";
}

TEST(HalfMeshTest, Tetrahedron_VertexAdjacency)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	// Every vertex should be adjacent to the other 3
	const std::vector<std::vector<HalfMesh::VIndex>> expected = {
	    {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
	for (HalfMesh::VIndex v = 0; v < 4; ++v) {
		EXPECT_EQ(AdjacentVertices(hm, v), expected[v])
		    << "vertex " << v << " adjacency mismatch";
	}
}

TEST(HalfMeshTest, Tetrahedron_NoHoles)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	EXPECT_TRUE(holes.empty()) << "tetrahedron should have no holes";
}

TEST(HalfMeshTest, Tetrahedron_ConnectedComponents)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	std::vector<HalfMesh::FIndex> components;
	const HalfMesh::FIndex num = hm.ConnectedComponents(components);
	EXPECT_EQ(num, 1u) << "tetrahedron should be one connected component";
	EXPECT_EQ(components.size(), 4u);
	for (auto c : components)
		EXPECT_EQ(c, 0u);
}

TEST(HalfMeshTest, Tetrahedron_AllEdgesInterior)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}});
	HalfMesh hm(m);

	for (HalfMesh::EIndex e = 0; e < hm.ESize(); ++e)
		EXPECT_FALSE(hm.EIsBoundary(e)) << "edge " << e << " should be interior";
}

// ---------------------------------------------------------------------------
// Fixture 2: Single triangle — 3 vertices, 1 face (0,1,2)
// Hand-verified expected topology:
//   - 3 vertices, 1 face, 3 edges, 6 half-edges
//   - All 3 vertices are on the boundary
//   - 3 boundary edges
//   - 1 boundary loop of length 3
//   - Each vertex is adjacent to exactly 1 face and 2 other vertices
// ---------------------------------------------------------------------------
TEST(HalfMeshTest, SingleTriangle_Counts)
{
	Mesh m = BuildMesh(3, {{0, 1, 2}});
	HalfMesh hm(m);

	EXPECT_EQ(hm.VSize(), 3u);
	EXPECT_EQ(hm.FSize(), 1u);
	EXPECT_EQ(hm.ESize(), 3u);
	EXPECT_EQ(hm.HeSize(), 6u);
}

TEST(HalfMeshTest, SingleTriangle_AllVerticesBoundary)
{
	Mesh m = BuildMesh(3, {{0, 1, 2}});
	HalfMesh hm(m);

	for (HalfMesh::VIndex v = 0; v < 3; ++v)
		EXPECT_TRUE(hm.VIsBoundary(v)) << "vertex " << v << " should be boundary";
}

TEST(HalfMeshTest, SingleTriangle_AllEdgesBoundary)
{
	Mesh m = BuildMesh(3, {{0, 1, 2}});
	HalfMesh hm(m);

	unsigned boundaryCount = 0;
	for (HalfMesh::EIndex e = 0; e < hm.ESize(); ++e)
		if (hm.EIsBoundary(e))
			++boundaryCount;
	EXPECT_EQ(boundaryCount, 3u) << "all 3 edges of a single triangle should be boundary";
}

TEST(HalfMeshTest, SingleTriangle_OneBoundaryLoopOfLength3)
{
	Mesh m = BuildMesh(3, {{0, 1, 2}});
	HalfMesh hm(m);

	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	ASSERT_EQ(holes.size(), 1u) << "single triangle should have exactly 1 boundary loop";
	EXPECT_EQ(holes[0].size(), 3u) << "boundary loop should contain all 3 vertices";
	// The loop contains {0,1,2} in some order
	std::vector<HalfMesh::VIndex> loop = holes[0];
	std::sort(loop.begin(), loop.end());
	EXPECT_EQ(loop, (std::vector<HalfMesh::VIndex>{0, 1, 2}));
}

TEST(HalfMeshTest, SingleTriangle_FaceVertices)
{
	Mesh m = BuildMesh(3, {{0, 1, 2}});
	HalfMesh hm(m);

	std::vector<HalfMesh::VIndex> faceVerts;
	for (HalfMesh::VIndex v : hm.FAdjacentVertices(0))
		faceVerts.push_back(v);
	std::sort(faceVerts.begin(), faceVerts.end());
	EXPECT_EQ(faceVerts, (std::vector<HalfMesh::VIndex>{0, 1, 2}));
}

TEST(HalfMeshTest, SingleTriangle_VertexFaceDegree)
{
	Mesh m = BuildMesh(3, {{0, 1, 2}});
	HalfMesh hm(m);

	for (HalfMesh::VIndex v = 0; v < 3; ++v)
		EXPECT_EQ(hm.VFaceDegree(v), 1u) << "vertex " << v << " should be incident to 1 face";
}

// ---------------------------------------------------------------------------
// Fixture 3: Two triangles sharing an edge — 4 vertices, 2 faces
//   faces: (0,1,2), (0,2,3)  — shared edge is 0-2
// Hand-verified expected topology:
//   - 4 vertices, 2 faces, 5 edges (4 boundary + 1 interior), 10 half-edges
//   - All 4 vertices are on the boundary
//   - Edge 0-2 is interior
//   - Vertex 0 and 2 each incident to 2 faces; vertices 1 and 3 each to 1 face
//   - 1 boundary loop of length 4: {0,1,2,3} or {0,3,2,1} depending on orientation
// ---------------------------------------------------------------------------
TEST(HalfMeshTest, TwoTriangles_Counts)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	EXPECT_EQ(hm.VSize(), 4u);
	EXPECT_EQ(hm.FSize(), 2u);
	EXPECT_EQ(hm.ESize(), 5u);
	EXPECT_EQ(hm.HeSize(), 10u);
}

TEST(HalfMeshTest, TwoTriangles_AllVerticesBoundary)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	for (HalfMesh::VIndex v = 0; v < 4; ++v)
		EXPECT_TRUE(hm.VIsBoundary(v)) << "vertex " << v << " should be boundary";
}

TEST(HalfMeshTest, TwoTriangles_BoundaryEdgeCount)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	unsigned boundaryCount = 0;
	for (HalfMesh::EIndex e = 0; e < hm.ESize(); ++e)
		if (hm.EIsBoundary(e))
			++boundaryCount;
	EXPECT_EQ(boundaryCount, 4u) << "two-triangle mesh should have 4 boundary edges";
}

TEST(HalfMeshTest, TwoTriangles_SharedEdgeIsInterior)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	// Find the edge between vertex 0 and vertex 2
	const HalfMesh::EIndex iE = hm.EEdge(0, 2);
	ASSERT_NE(iE, math::NO_ID) << "edge 0-2 must exist";
	EXPECT_FALSE(hm.EIsBoundary(iE)) << "edge 0-2 should be interior";
}

TEST(HalfMeshTest, TwoTriangles_VertexFaceDegrees)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	EXPECT_EQ(hm.VFaceDegree(0), 2u) << "v0 should be incident to 2 faces";
	EXPECT_EQ(hm.VFaceDegree(2), 2u) << "v2 should be incident to 2 faces";
	EXPECT_EQ(hm.VFaceDegree(1), 1u) << "v1 should be incident to 1 face";
	EXPECT_EQ(hm.VFaceDegree(3), 1u) << "v3 should be incident to 1 face";
}

TEST(HalfMeshTest, TwoTriangles_OneBoundaryLoopOfLength4)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	ASSERT_EQ(holes.size(), 1u) << "two-triangle mesh should have exactly 1 boundary loop";
	EXPECT_EQ(holes[0].size(), 4u) << "boundary loop should contain all 4 vertices";
	// The loop should contain all vertices {0,1,2,3}
	std::vector<HalfMesh::VIndex> loop = holes[0];
	std::sort(loop.begin(), loop.end());
	EXPECT_EQ(loop, (std::vector<HalfMesh::VIndex>{0, 1, 2, 3}));
}

TEST(HalfMeshTest, TwoTriangles_ConnectedComponents)
{
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm(m);

	std::vector<HalfMesh::FIndex> components;
	const HalfMesh::FIndex num = hm.ConnectedComponents(components);
	EXPECT_EQ(num, 1u) << "two triangles sharing an edge should be one component";
	EXPECT_EQ(components.size(), 2u);
	EXPECT_EQ(components[0], components[1]) << "both faces should have the same component ID";
}

TEST(HalfMeshTest, TwoTriangles_EdgeFlip)
{
	// Build with unit-distance vertices so EIsFlipValid can check geometry
	Mesh m;
	m.vertices.resize(4);
	m.vertices[0] = HalfMesh::Vertex(0.f, 0.f, 0.f);
	m.vertices[1] = HalfMesh::Vertex(1.f, 0.f, 0.f);
	m.vertices[2] = HalfMesh::Vertex(0.5f, 1.f, 0.f);
	m.vertices[3] = HalfMesh::Vertex(0.5f, -1.f, 0.f);
	{
		HalfMesh::Face f0;
		f0[0] = 0;
		f0[1] = 1;
		f0[2] = 2;
		HalfMesh::Face f1;
		f1[0] = 0;
		f1[1] = 3;
		f1[2] = 1;
		m.faces.push_back(f0);
		m.faces.push_back(f1);
	}
	HalfMesh hm(m);

	// Sanity: 4 vertices, 2 faces, 5 edges
	ASSERT_EQ(hm.VSize(), 4u);
	ASSERT_EQ(hm.FSize(), 2u);
	ASSERT_EQ(hm.ESize(), 5u);

	// Find the interior edge (between v0 and v1)
	const HalfMesh::EIndex iE = hm.EEdge(0, 1);
	ASSERT_NE(iE, math::NO_ID);
	ASSERT_FALSE(hm.EIsBoundary(iE));

	// Check flip is valid
	EXPECT_TRUE(hm.EIsFlipValid(iE, m.vertices));

	// Perform the flip — edge v0-v1 becomes v2-v3
	hm.EFlip(iE);

	// After flip: still 4 verts, 2 faces, 5 edges
	EXPECT_EQ(hm.VSize(), 4u);
	EXPECT_EQ(hm.FSize(), 2u);
	EXPECT_EQ(hm.ESize(), 5u);

	// The flipped edge should connect v2 and v3
	const HalfMesh::EIndex iEFlipped = hm.EEdge(2, 3);
	EXPECT_NE(iEFlipped, math::NO_ID) << "edge v2-v3 should exist after flip";
	EXPECT_FALSE(hm.EIsBoundary(iEFlipped)) << "flipped edge should be interior";

	// The old edge (v0-v1) should no longer exist as an interior edge
	// (they may still exist as boundary edges or separate boundary)
	// More specifically: neither v0-v1 nor v1-v0 should be interior
	const HalfMesh::EIndex iEOld = hm.EEdge(0, 1);
	if (iEOld != math::NO_ID) {
		EXPECT_TRUE(hm.EIsBoundary(iEOld)) << "old edge v0-v1 should be boundary after flip";
	}

	// Boundary loop should still be size 4
	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	ASSERT_EQ(holes.size(), 1u);
	EXPECT_EQ(holes[0].size(), 4u);
}

// ---------------------------------------------------------------------------
// Build non-manifold rejection: the single-emplace half-edge construction must
// reject the same inputs the reference two-key probe did. A self-edge and an
// edge shared by >2 faces (same directed edge seen twice) cannot be represented
// by a manifold half-edge structure. (Bow-tie vertex rejection is covered in
// HalfMeshInvariantsTest.)
// ---------------------------------------------------------------------------
TEST(HalfMeshTest, Build_RejectsSelfEdge)
{
	// A face with a repeated corner has a zero-length (self) edge.
	Mesh m = BuildMesh(3, {{0, 1, 1}});
	HalfMesh hm;
	EXPECT_FALSE(hm.Build(m)) << "Build must reject a self-edge (face[i]==face[i+1])";
	EXPECT_TRUE(hm.Empty()) << "rejected Build must leave the mesh cleared";
}

TEST(HalfMeshTest, Build_RejectsEdgeSharedByThreeFaces)
{
	// Three triangles all using the SAME directed edge (0->1): the directed edge
	// is seen more than twice, so the edge is shared by >2 faces (non-manifold).
	Mesh m = BuildMesh(5, {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}});
	HalfMesh hm;
	EXPECT_FALSE(hm.Build(m)) << "Build must reject an edge shared by >2 faces";
	EXPECT_TRUE(hm.Empty()) << "rejected Build must leave the mesh cleared";
}

TEST(HalfMeshTest, Build_RejectsDuplicateFace)
{
	// The identical directed edges of a duplicated face collide (edge shared by
	// two faces in the SAME orientation).
	Mesh m = BuildMesh(3, {{0, 1, 2}, {0, 1, 2}});
	HalfMesh hm;
	EXPECT_FALSE(hm.Build(m)) << "Build must reject a duplicated face";
	EXPECT_TRUE(hm.Empty()) << "rejected Build must leave the mesh cleared";
}

TEST(HalfMeshTest, Build_AcceptsSharedEdgeOppositeOrientation)
{
	// Two triangles sharing edge 0-2 in OPPOSITE directions form a valid manifold
	// (the twin case) — the reject path must not trip on a legitimate shared edge.
	Mesh m = BuildMesh(4, {{0, 1, 2}, {0, 2, 3}});
	HalfMesh hm;
	EXPECT_TRUE(hm.Build(m)) << "Build must accept a manifold shared edge";
	EXPECT_FALSE(hm.EIsBoundary(hm.EEdge(0, 2))) << "the shared edge is interior";
}

} // namespace
} // namespace halfmesh
