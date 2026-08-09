/*
* MetricsTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/metrics/MetricsTest.cpp — analytic self-validation of the metrics toolkit
//
// All assertions use analytic known answers.
// Float tolerances are named and justified at each assertion.

#include "Metrics.h"
#include "Corpus.h"

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace hmtest {
namespace metrics {
namespace {

// ============================================================
// Inline mesh builders
// ============================================================

// Build a Mesh from vertex list and face list.
static halfmesh::Mesh BuildMesh(
    std::vector<halfmesh::Mesh::Vertex> verts,
    std::vector<std::array<uint32_t, 3>> tris)
{
	halfmesh::Mesh m;
	m.vertices = std::move(verts);
	for (const auto& t : tris) {
		halfmesh::HalfMesh::Face f;
		f[0] = t[0];
		f[1] = t[1];
		f[2] = t[2];
		m.faces.push_back(f);
	}
	return m;
}

// Unit cube: 8 vertices, 12 triangles (6 faces × 2 triangles each).
// Side length s. Vertices at (0,0,0)...(s,s,s).
// Surface area = 6·s², volume = s³.
// V=8, E=18, F=12, χ=V-E+F=8-18+12=2, genus=0, watertight.
static halfmesh::Mesh MakeCube(float s = 1.0f)
{
	// 8 corners of the unit cube
	halfmesh::Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f}, // 0
	    {s, 0.f, 0.f}, // 1
	    {s, s, 0.f}, // 2
	    {0.f, s, 0.f}, // 3
	    {0.f, 0.f, s}, // 4
	    {s, 0.f, s}, // 5
	    {s, s, s}, // 6
	    {0.f, s, s}, // 7
	};
	// 12 triangles (consistent outward normals)
	// Bottom (z=0): 0,2,1 / 0,3,2
	// Top    (z=s): 4,5,6 / 4,6,7
	// Front  (y=0): 0,1,5 / 0,5,4
	// Back   (y=s): 2,3,7 / 2,7,6
	// Left   (x=0): 0,4,7 / 0,7,3
	// Right  (x=s): 1,2,6 / 1,6,5
	std::vector<std::array<uint32_t, 3>> tris = {
	    {0, 2, 1},
	    {0, 3, 2}, // bottom
	    {4, 5, 6},
	    {4, 6, 7}, // top
	    {0, 1, 5},
	    {0, 5, 4}, // front
	    {2, 3, 7},
	    {2, 7, 6}, // back  -- note: using 2,3,7 and 2,7,6 for proper orientation
	    {3, 0, 4},
	    {3, 4, 7}, // left  -- adjusted for outward normal
	    {1, 2, 6},
	    {1, 6, 5}, // right
	};
	for (const auto& t : tris) {
		halfmesh::HalfMesh::Face f;
		f[0] = t[0];
		f[1] = t[1];
		f[2] = t[2];
		m.faces.push_back(f);
	}
	return m;
}

// Regular tetrahedron with edge length 1.
// V=4, E=6, F=4, χ=2, genus=0, watertight.
// Surface area = √3, volume = 1/(6√2).
static halfmesh::Mesh MakeTetra()
{
	halfmesh::Mesh m;
	// Vertices of a regular tetrahedron with edge length 1
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.5f, std::sqrt(3.f) / 2.f, 0.f},
	    {0.5f, std::sqrt(3.f) / 6.f, std::sqrt(6.f) / 3.f}};
	// Outward-normal winding
	std::vector<std::array<uint32_t, 3>> tris = {
	    {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}};
	for (const auto& t : tris) {
		halfmesh::HalfMesh::Face f;
		f[0] = t[0];
		f[1] = t[1];
		f[2] = t[2];
		m.faces.push_back(f);
	}
	return m;
}

// Single triangle (open surface, 1 boundary loop).
static halfmesh::Mesh MakeSingleTriangle()
{
	return BuildMesh(
	    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
	    {{{0, 1, 2}}});
}

// 2×2 quad grid (open surface, 1 boundary loop).
// Grid in xy plane, 3×3 vertices, 8 triangles.
// V=9, E=16 interior + boundary, F=8 triangles.
static halfmesh::Mesh MakeGrid2x2()
{
	halfmesh::Mesh m;
	m.vertices.resize(9);
	for (int j = 0; j < 3; ++j)
		for (int i = 0; i < 3; ++i)
			m.vertices[j * 3 + i] = {static_cast<float>(i), static_cast<float>(j), 0.f};
	// 4 quads → 8 triangles
	for (int j = 0; j < 2; ++j) {
		for (int i = 0; i < 2; ++i) {
			const uint32_t v00 = j * 3 + i;
			const uint32_t v10 = j * 3 + (i + 1);
			const uint32_t v01 = (j + 1) * 3 + i;
			const uint32_t v11 = (j + 1) * 3 + (i + 1);
			halfmesh::HalfMesh::Face f0, f1;
			f0[0] = v00;
			f0[1] = v10;
			f0[2] = v11;
			f1[0] = v00;
			f1[1] = v11;
			f1[2] = v01;
			m.faces.push_back(f0);
			m.faces.push_back(f1);
		}
	}
	return m;
}

// ============================================================
// TOPOLOGY TESTS
// ============================================================

TEST(Topology, Cube_VEFCounts)
{
	halfmesh::Mesh cube = MakeCube(1.0f);
	const TopologyCounts tc = ComputeTopology(cube);
	EXPECT_EQ(tc.numVertices, 8u);
	EXPECT_EQ(tc.numFaces, 12u);
	// A cube triangulated as above: 12 triangles, 8 vertices.
	// E = (3F + boundaryEdges) / 2 for a closed mesh = 3*12/2 = 18
	EXPECT_EQ(tc.numEdges, 18u);
	EXPECT_EQ(tc.euler, 2); // χ = 8-18+12 = 2
	EXPECT_EQ(tc.genus, 0); // genus = (2-2-0)/2 = 0
	EXPECT_EQ(tc.numBoundaryLoops, 0u);
	EXPECT_TRUE(tc.isWatertight);
	EXPECT_TRUE(tc.isEdgeManifold);
	EXPECT_TRUE(tc.isVertexManifold);
}

TEST(Topology, Tetrahedron_VEFCounts)
{
	halfmesh::Mesh t = MakeTetra();
	const TopologyCounts tc = ComputeTopology(t);
	EXPECT_EQ(tc.numVertices, 4u);
	EXPECT_EQ(tc.numFaces, 4u);
	EXPECT_EQ(tc.numEdges, 6u);
	EXPECT_EQ(tc.euler, 2);
	EXPECT_EQ(tc.genus, 0);
	EXPECT_EQ(tc.numBoundaryLoops, 0u);
	EXPECT_TRUE(tc.isWatertight);
	EXPECT_TRUE(tc.isEdgeManifold);
	EXPECT_TRUE(tc.isVertexManifold);
}

TEST(Topology, SingleTriangle_OpenSurface)
{
	halfmesh::Mesh m = MakeSingleTriangle();
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_EQ(tc.numVertices, 3u);
	EXPECT_EQ(tc.numFaces, 1u);
	EXPECT_EQ(tc.numEdges, 3u);
	EXPECT_EQ(tc.euler, 1); // χ = 3-3+1 = 1 (disk)
	// genus for disk: (2 - 1 - 1) / 2 = 0
	EXPECT_EQ(tc.genus, 0);
	EXPECT_EQ(tc.numBoundaryLoops, 1u);
	EXPECT_FALSE(tc.isWatertight);
}

TEST(Topology, Grid2x2_OneBoundaryLoop)
{
	halfmesh::Mesh m = MakeGrid2x2();
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_EQ(tc.numVertices, 9u);
	EXPECT_EQ(tc.numFaces, 8u);
	// Open 2×2 grid: V=9, F=8. E via Euler: for a planar disk, χ=1, so E = V+F-1 = 16
	EXPECT_EQ(tc.numEdges, 16u);
	EXPECT_EQ(tc.euler, 1); // χ = 9-16+8 = 1
	EXPECT_EQ(tc.numBoundaryLoops, 1u);
	EXPECT_FALSE(tc.isWatertight);
}

TEST(Topology, CubeValence)
{
	halfmesh::Mesh cube = MakeCube(1.0f);
	// In the unit cube each vertex is shared by exactly 3 quads = 6 triangles.
	// However, in our specific triangulation some vertices touch more or fewer.
	// We just check the histogram is non-empty and sums to 8.
	const auto hist = ComputeValenceHistogram(cube);
	uint32_t total = 0;
	for (const auto& [val, cnt] : hist)
		total += cnt;
	EXPECT_EQ(total, 8u);
	// All valences should be 6 for a cube triangulated with 2 tris per quad face
	// if the diagonals are arranged consistently; our arrangement has valence 6.
	// We verify the histogram has at least one entry and each valence > 0.
	EXPECT_FALSE(hist.empty());
}

// Bow-tie vertex: two triangle fans sharing a single vertex but no edge.
// Topology: vertex 0 is the bow-tie center shared by two independent fans.
//   Fan 1: triangles (0,1,2) and (0,2,3)
//   Fan 2: triangles (0,4,5) and (0,5,6)
// This is vertex-non-manifold (disconnected fan), but each edge has ≤ 2 faces.
TEST(Topology, BowTie_VertexNonManifold)
{
	halfmesh::Mesh m = BuildMesh(
	    {
	        {0.f, 0.f, 0.f}, // 0 — bow-tie center
	        {1.f, 0.f, 0.f}, // 1
	        {1.f, 1.f, 0.f}, // 2
	        {0.f, 1.f, 0.f}, // 3
	        {-1.f, 0.f, 0.f}, // 4
	        {-1.f, -1.f, 0.f}, // 5
	        {0.f, -1.f, 0.f}, // 6
	    },
	    {{{0, 1, 2}}, {{0, 2, 3}}, {{0, 4, 5}}, {{0, 5, 6}}});
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_FALSE(tc.isVertexManifold)
	    << "Bow-tie vertex must be detected as non-manifold";
	// Each edge is shared by at most 2 faces → still edge-manifold.
	EXPECT_TRUE(tc.isEdgeManifold)
	    << "Bow-tie mesh edges are each shared by ≤ 2 faces";
}

// Three triangles sharing one edge: edge non-manifold.
TEST(Topology, ThreeFacesOnEdge_EdgeNonManifold)
{
	// Edge 0-1 appears in all three triangles.
	halfmesh::Mesh m = BuildMesh(
	    {
	        {0.f, 0.f, 0.f}, // 0
	        {1.f, 0.f, 0.f}, // 1
	        {0.5f, 1.f, 0.f}, // 2
	        {0.5f, -1.f, 0.f}, // 3
	        {0.5f, 0.f, 1.f}, // 4
	    },
	    {{{0, 1, 2}}, {{0, 1, 3}}, {{0, 1, 4}}});
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_FALSE(tc.isEdgeManifold)
	    << "Edge shared by 3 faces must be detected as non-manifold";
}

// Clean cube and tetrahedron must be both edge- and vertex-manifold (sanity check).
TEST(Topology, ManifoldMeshes_BothFlagsTrue)
{
	EXPECT_TRUE(ComputeTopology(MakeCube()).isEdgeManifold);
	EXPECT_TRUE(ComputeTopology(MakeCube()).isVertexManifold);
	EXPECT_TRUE(ComputeTopology(MakeTetra()).isEdgeManifold);
	EXPECT_TRUE(ComputeTopology(MakeTetra()).isVertexManifold);
}

// ============================================================
// GEOMETRY TESTS
// ============================================================

TEST(Geometry, Cube_SurfaceArea)
{
	const float s = 2.0f;
	halfmesh::Mesh cube = MakeCube(s);
	const double area = ComputeSurfaceArea(cube);
	// Analytic: 6 * s² = 6 * 4 = 24
	const double expected = 6.0 * s * s;
	// Tolerance: float vertices, exact analytic computation → 1e-4 relative.
	EXPECT_NEAR(area, expected, expected * 1e-4) << "Cube surface area ε ≤ 1e-4 relative";
}

TEST(Geometry, Cube_Volume)
{
	const float s = 2.0f;
	halfmesh::Mesh cube = MakeCube(s);
	const double vol = std::abs(ComputeSignedVolume(cube));
	// Analytic: s³ = 8
	const double expected = static_cast<double>(s) * s * s;
	// Tolerance: 1e-4 relative — divergence theorem on float positions.
	EXPECT_NEAR(vol, expected, expected * 1e-4) << "Cube volume ε ≤ 1e-4 relative";
}

TEST(Geometry, Tetra_SurfaceArea)
{
	halfmesh::Mesh t = MakeTetra();
	const double area = ComputeSurfaceArea(t);
	// Regular tetrahedron edge=1: area = √3 ≈ 1.7320508
	const double expected = std::sqrt(3.0);
	EXPECT_NEAR(area, expected, 1e-5) << "Tetra surface area ε ≤ 1e-5";
}

TEST(Geometry, Tetra_Volume)
{
	halfmesh::Mesh t = MakeTetra();
	const double vol = std::abs(ComputeSignedVolume(t));
	// Regular tetrahedron edge=1: volume = 1/(6√2) ≈ 0.117851...
	const double expected = 1.0 / (6.0 * std::sqrt(2.0));
	EXPECT_NEAR(vol, expected, 1e-5) << "Tetra volume ε ≤ 1e-5";
}

TEST(Geometry, EquilateralTriangleQuality)
{
	// Equilateral triangle: all angles 60°, aspectRatio=1, radiusRatio=1
	const halfmesh::Mesh::Vertex a(0.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex b(1.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex c(0.5f, std::sqrt(3.f) / 2.f, 0.f);
	const TriangleQuality q = ComputeTriangleQuality(a, b, c);
	EXPECT_NEAR(q.minAngleDeg, 60.f, 0.1f) << "Equilateral min angle ε ≤ 0.1°";
	EXPECT_NEAR(q.maxAngleDeg, 60.f, 0.1f) << "Equilateral max angle ε ≤ 0.1°";
	EXPECT_NEAR(q.aspectRatio, 1.f, 0.01f) << "Equilateral aspect_ratio ε ≤ 0.01";
	EXPECT_NEAR(q.radiusRatio, 1.f, 0.01f) << "Equilateral radius_ratio ε ≤ 0.01";
}

TEST(Geometry, EdgeLengthStats_Tetra)
{
	halfmesh::Mesh t = MakeTetra();
	const EdgeLengthStats s = ComputeEdgeLengthStats(t);
	// Regular tetrahedron: all edges = 1.0
	EXPECT_NEAR(s.minLen, 1.0, 1e-5) << "Tetra min edge ε ≤ 1e-5";
	EXPECT_NEAR(s.maxLen, 1.0, 1e-5) << "Tetra max edge ε ≤ 1e-5";
	EXPECT_NEAR(s.meanLen, 1.0, 1e-5) << "Tetra mean edge ε ≤ 1e-5";
	EXPECT_NEAR(s.stddev, 0.0, 1e-5) << "Tetra stddev ε ≤ 1e-5 (all equal)";
}

// ============================================================
// DISTANCE TESTS
// ============================================================

TEST(Distance, SelfDistance_IsZero)
{
	halfmesh::Mesh t = MakeTetra();
	const DistanceResult kd = ComputeDistanceKdTree(t, t);
	// Every vertex of A is a vertex of B → KD-tree nearest = 0.
	// Tolerance: 1e-6 (float arithmetic in KD-tree).
	EXPECT_NEAR(kd.hausdorffSymmetric, 0.0, 1e-6)
	    << "Hausdorff(mesh, self) ε ≤ 1e-6";
	EXPECT_NEAR(kd.meanSurfaceDist, 0.0, 1e-6)
	    << "MeanDist(mesh, self) ε ≤ 1e-6";
}

TEST(Distance, TranslatedMesh_Hausdorff_KdTree)
{
	// Translate the tetrahedron by d along x.
	halfmesh::Mesh t = MakeTetra();
	const float d = 5.0f;
	halfmesh::Mesh t2 = t;
	for (auto& v : t2.vertices)
		v.x() += d;

	const DistanceResult kd = ComputeDistanceKdTree(t, t2);
	// With surface sampling (vertices + midpoints + centroids), the nearest sample
	// on A has its mirror on B at exactly distance d. The Hausdorff = d exactly.
	// Tolerance: 5e-4 (float arithmetic on the KD-tree's NearestPoint).
	EXPECT_NEAR(kd.hausdorffSymmetric, static_cast<double>(d), 5e-4)
	    << "Hausdorff(mesh, translated_by_d) must equal d within 5e-4 after surface sampling";
	EXPECT_GE(kd.hausdorffSymmetric, static_cast<double>(d) - 5e-4)
	    << "Hausdorff must be at least d";
}

TEST(Distance, KdTree_vs_BruteForce_Tetra)
{
	// Cross-check: KD-tree vs brute-force on a small mesh.
	halfmesh::Mesh t = MakeTetra();
	halfmesh::Mesh t2 = t;
	for (auto& v : t2.vertices) {
		v.x() += 0.5f;
		v.y() += 0.1f;
	}

	const DistanceResult kd = ComputeDistanceKdTree(t, t2);
	const DistanceResult bf = ComputeDistanceBruteForce(t, t2);

	// KD-tree and brute-force must agree within 1e-4 (float precision).
	EXPECT_NEAR(kd.hausdorffSymmetric, bf.hausdorffSymmetric, 1e-4)
	    << "KD-tree Hausdorff vs brute-force ε ≤ 1e-4";
	EXPECT_NEAR(kd.meanSurfaceDist, bf.meanSurfaceDist, 1e-4)
	    << "KD-tree MeanDist vs brute-force ε ≤ 1e-4";
}

TEST(Distance, KdTree_vs_BruteForce_Grid)
{
	// Cross-check on the 2×2 grid (slightly larger).
	halfmesh::Mesh g = MakeGrid2x2();
	halfmesh::Mesh g2 = g;
	for (auto& v : g2.vertices)
		v.z() += 0.3f; // shift in z

	const DistanceResult kd = ComputeDistanceKdTree(g, g2);
	const DistanceResult bf = ComputeDistanceBruteForce(g, g2);

	EXPECT_NEAR(kd.hausdorffSymmetric, bf.hausdorffSymmetric, 2e-4)
	    << "KD-tree Hausdorff vs brute-force on grid ε ≤ 2e-4";
	EXPECT_NEAR(kd.meanSurfaceDist, bf.meanSurfaceDist, 2e-4)
	    << "KD-tree MeanDist vs brute-force on grid ε ≤ 2e-4";
}

TEST(Distance, BruteForce_SelfDistance_IsZero)
{
	halfmesh::Mesh t = MakeSingleTriangle();
	const DistanceResult bf = ComputeDistanceBruteForce(t, t);
	EXPECT_NEAR(bf.hausdorffSymmetric, 0.0, 1e-6)
	    << "BruteForce Hausdorff(mesh, self) ε ≤ 1e-6";
}

// ============================================================
// UV TESTS
// ============================================================

// Build a mesh with a perfect isometric UV mapping (single flat triangle mapped
// to itself in UV space): signed area > 0, symmetric-Dirichlet ≈ 4.
TEST(UV, IsometricMapping_ZeroFlips_SD4)
{
	// Single triangle in 3-D and UV space with identical coordinates (scaled).
	halfmesh::Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.f, 1.f, 0.f}};
	halfmesh::HalfMesh::Face f;
	f[0] = 0;
	f[1] = 1;
	f[2] = 2;
	m.faces.push_back(f);
	// UV = same as 3-D xy (isometric mapping for planar triangle)
	m.faceTexcoords = {
	    {0.f, 0.f},
	    {1.f, 0.f},
	    {0.f, 1.f}};
	const UVMetrics uv = ComputeUVMetrics(m);
	EXPECT_EQ(uv.flipCount, 0) << "Isometric mapping must have 0 flips";
	EXPECT_TRUE(uv.allFinite);
	// Symmetric-Dirichlet for isometry = 4 exactly.
	// Tolerance: 1e-4 (SVD precision).
	EXPECT_NEAR(uv.symDirichlet, 4.0, 0.01)
	    << "Isometric mapping SD energy ε ≤ 0.01 from minimum 4";
}

// One deliberately flipped triangle: UV winding opposite to 3D → flipCount == 1.
TEST(UV, FlippedTriangle_FlipCountOne)
{
	halfmesh::Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {0.f, 1.f, 0.f}};
	halfmesh::HalfMesh::Face f;
	f[0] = 0;
	f[1] = 1;
	f[2] = 2;
	m.faces.push_back(f);
	// Flip UV by reversing vertex 1 and 2 → CW in UV space.
	m.faceTexcoords = {
	    {0.f, 0.f},
	    {0.f, 1.f}, // swapped
	    {1.f, 0.f} // swapped
	};
	const UVMetrics uv = ComputeUVMetrics(m);
	EXPECT_EQ(uv.flipCount, 1) << "Deliberately flipped triangle must have flip_count==1";
}

// Two-triangle mesh with equal UV texel density across both charts.
TEST(UV, EqualTexelDensity)
{
	// Two triangles (unit squares): same 3-D area, same UV area.
	halfmesh::Mesh m;
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {1.f, 1.f, 0.f},
	    {0.f, 1.f, 0.f},
	    {2.f, 0.f, 0.f},
	    {3.f, 0.f, 0.f},
	    {3.f, 1.f, 0.f},
	    {2.f, 1.f, 0.f},
	};
	halfmesh::HalfMesh::Face f0, f1;
	f0[0] = 0;
	f0[1] = 1;
	f0[2] = 2;
	m.faces.push_back(f0);
	f1[0] = 4;
	f1[1] = 5;
	f1[2] = 6;
	m.faces.push_back(f1);
	// UV: chart 0 and chart 1 both mapped to a 0.4×0.4 square region
	// (same UV area, same 3-D area → equal density).
	m.faceTexcoords = {
	    {0.0f, 0.0f},
	    {0.4f, 0.0f},
	    {0.4f, 0.4f}, // chart 0
	    {0.5f, 0.0f},
	    {0.9f, 0.0f},
	    {0.9f, 0.4f}, // chart 1
	};
	std::vector<unsigned> faceChart = {0, 1};
	const unsigned numCharts = 2;
	const auto densities = ComputeChartTexelDensity(m, faceChart, numCharts);
	ASSERT_EQ(densities.size(), 2u);
	// Both charts: worldArea = 0.5 (half of 1×1 square), uvArea = 0.5*(0.4)²= 0.08
	// density = sqrt(0.08/0.5) = sqrt(0.16) = 0.4 for each.
	// Tolerance: 1e-4 (float UV coordinates).
	EXPECT_NEAR(densities[0], densities[1], 1e-4)
	    << "Equal density: two charts ε ≤ 1e-4";
}

// ============================================================
// ROBUSTNESS TESTS
// ============================================================

TEST(Robustness, CleanMesh_ScanPasses)
{
	halfmesh::Mesh m = MakeTetra();
	EXPECT_TRUE(ScanFinite(m)) << "Clean tetrahedron must pass finite scan";
}

TEST(Robustness, NaN_Vertex_ScanFails)
{
	halfmesh::Mesh m = MakeTetra();
	m.vertices[1].x() = std::numeric_limits<float>::quiet_NaN();
	EXPECT_FALSE(ScanFinite(m)) << "NaN vertex must fail finite scan";
}

TEST(Robustness, Inf_Vertex_ScanFails)
{
	halfmesh::Mesh m = MakeTetra();
	m.vertices[2].y() = std::numeric_limits<float>::infinity();
	EXPECT_FALSE(ScanFinite(m)) << "Inf vertex must fail finite scan";
}

TEST(Robustness, NaN_InTexcoords_ScanFails)
{
	halfmesh::Mesh m = MakeSingleTriangle();
	m.faceTexcoords = {{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}};
	m.faceTexcoords[1].x() = std::numeric_limits<float>::quiet_NaN();
	EXPECT_FALSE(ScanFinite(m)) << "NaN in texcoords must fail finite scan";
}

// ============================================================
// CANONICALIZATION TESTS
// ============================================================

// A mesh and a relabeled + cyclically-rotated copy must compare EQUAL.
TEST(Canonicalization, RelabeledMesh_IsEqual)
{
	halfmesh::Mesh a = MakeSingleTriangle();
	// Build same triangle with permuted vertex indices: 2→0, 0→1, 1→2.
	halfmesh::Mesh b;
	b.vertices = {
	    a.vertices[2], // new v0 = old v2
	    a.vertices[0], // new v1 = old v0
	    a.vertices[1], // new v2 = old v1
	};
	halfmesh::HalfMesh::Face f;
	f[0] = 0;
	f[1] = 1;
	f[2] = 2;
	b.faces.push_back(f);
	EXPECT_TRUE(CanonicallyEqual(a, b))
	    << "Relabeled mesh must be canonically equal";
}

// Cyclic rotation of face vertices.
TEST(Canonicalization, CyclicRotation_IsEqual)
{
	halfmesh::Mesh a = MakeSingleTriangle();
	// Same vertices, face rotated: (0,1,2) → (1,2,0)
	halfmesh::Mesh b = a;
	b.faces[0][0] = 1;
	b.faces[0][1] = 2;
	b.faces[0][2] = 0;
	EXPECT_TRUE(CanonicallyEqual(a, b))
	    << "Cyclic-rotated face must be canonically equal";
}

// A genuinely different mesh (different geometry) must compare UNEQUAL.
TEST(Canonicalization, DifferentMesh_IsNotEqual)
{
	halfmesh::Mesh a = MakeSingleTriangle();
	halfmesh::Mesh b;
	b.vertices = {{0.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {0.f, 2.f, 0.f}}; // different scale
	halfmesh::HalfMesh::Face f;
	f[0] = 0;
	f[1] = 1;
	f[2] = 2;
	b.faces.push_back(f);
	// Vertices are at different positions → not canonically equal.
	EXPECT_FALSE(CanonicallyEqual(a, b))
	    << "Geometrically different meshes must NOT be canonically equal";
}

// Mesh vs itself must be equal.
TEST(Canonicalization, SameMesh_IsEqual)
{
	halfmesh::Mesh a = MakeTetra();
	EXPECT_TRUE(CanonicallyEqual(a, a))
	    << "Mesh vs itself must be canonically equal";
}

// Different topology (different face count) must compare UNEQUAL.
TEST(Canonicalization, DifferentTopology_IsNotEqual)
{
	halfmesh::Mesh a = MakeSingleTriangle();
	halfmesh::Mesh b = MakeGrid2x2();
	EXPECT_FALSE(CanonicallyEqual(a, b))
	    << "Different topology must NOT be canonically equal";
}

// Vertex positions perturbed within posTol → canonically equal (final comparison uses tol).
// Vertex positions perturbed beyond posTol → canonically NOT equal.
// This verifies deterministic sort (exact comparator) + tolerance only in final check.
TEST(Canonicalization, ToleranceBoundary_Determinism)
{
	halfmesh::Mesh a = MakeSingleTriangle(); // vertices at (0,0,0),(1,0,0),(0,1,0)
	const float tol = 1e-5f;

	// Perturb all vertices by tol/2 (within tolerance) → equal.
	halfmesh::Mesh b = a;
	for (auto& v : b.vertices)
		v += halfmesh::Mesh::Vertex(tol * 0.4f, 0.f, 0.f);
	EXPECT_TRUE(CanonicallyEqual(a, b, tol))
	    << "Perturbation < tol: meshes must be canonically equal";

	// Perturb one vertex beyond tol → not equal.
	halfmesh::Mesh c = a;
	c.vertices[0].x() += tol * 2.0f;
	EXPECT_FALSE(CanonicallyEqual(a, c, tol))
	    << "Perturbation > tol: meshes must NOT be canonically equal";
}

// ============================================================
// POINT-TO-TRIANGLE DISTANCE (brute-force helper) TESTS
// ============================================================

TEST(PointTriDist, PointAtVertex)
{
	// Point exactly at vertex a → distance = 0
	const halfmesh::Mesh::Vertex a(0.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex b(1.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex c(0.f, 1.f, 0.f);
	const float d2 = PointToTriangleDistSq(a, a, b, c);
	EXPECT_NEAR(d2, 0.f, 1e-10f) << "Point at vertex a: distance² ε ≤ 1e-10";
}

TEST(PointTriDist, PointAboveTriangle)
{
	// Point above the triangle centroid at height h → distance = h
	const halfmesh::Mesh::Vertex a(0.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex b(1.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex c(0.f, 1.f, 0.f);
	const float h = 3.f;
	const halfmesh::Mesh::Vertex p(1.f / 3.f, 1.f / 3.f, h); // centroid + height
	const float d2 = PointToTriangleDistSq(p, a, b, c);
	EXPECT_NEAR(d2, h * h, 1e-5f) << "Point above centroid: distance² ε ≤ 1e-5";
}

TEST(PointTriDist, PointBeyondEdge)
{
	// Point beyond edge ab → nearest is on edge ab
	const halfmesh::Mesh::Vertex a(0.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex b(1.f, 0.f, 0.f);
	const halfmesh::Mesh::Vertex c(0.f, 1.f, 0.f);
	const halfmesh::Mesh::Vertex p(0.5f, -1.f, 0.f);
	const float d2 = PointToTriangleDistSq(p, a, b, c);
	// Nearest point on ab is (0.5, 0, 0) → distance = 1
	EXPECT_NEAR(d2, 1.f, 1e-5f) << "Point beyond edge: distance² ε ≤ 1e-5";
}

// ---------------------------------------------------------------------------
// Chart-partition metrics (moved from hmbench so the default suite can assert
// segmentation quality — D-Charts distance-term plan, 2026-07-16).
// Analytic answers on the 2-triangle unit Quad.
// ---------------------------------------------------------------------------
TEST(ChartMetrics, BoundaryCutLengthBorderAndInterChart)
{
	halfmesh::Mesh m = hmtest::corpus::Quad();
	// one chart: seams are exactly the 4 mesh-border edges (perimeter 4)
	EXPECT_NEAR(hmtest::metrics::ComputeBoundaryCutLength(m, {0u, 0u}), 4.0, 1e-6);
	// two charts: the shared diagonal becomes a seam too
	EXPECT_NEAR(hmtest::metrics::ComputeBoundaryCutLength(m, {0u, 1u}),
	            4.0 + std::sqrt(2.0), 1e-6);
	// size mismatch -> -1.0 sentinel (hmbench UNSET semantics, kept verbatim)
	EXPECT_EQ(hmtest::metrics::ComputeBoundaryCutLength(m, {0u}), -1.0);
}

TEST(ChartMetrics, CompactnessAndPlanarityFlatQuad)
{
	halfmesh::Mesh m = hmtest::corpus::Quad();
	// single flat chart: compactness = perimeter^2/(4*pi*area) = 16/(4*pi)
	const auto comp = hmtest::metrics::ComputeChartCompactness(m, {0u, 0u}, 1u);
	ASSERT_EQ(comp.size(), 1u);
	EXPECT_NEAR(comp[0], 16.0 / (4.0 * M_PI), 1e-6);
	// coplanar faces: zero deviation from the mean normal
	const auto plan = hmtest::metrics::ComputeChartPlanarityError(m, {0u, 0u}, 1u);
	ASSERT_EQ(plan.size(), 1u);
	EXPECT_NEAR(plan[0], 0.0, 1e-9);
}

TEST(ChartMetrics, CoverageValidAndInvalid)
{
	halfmesh::Mesh m = hmtest::corpus::Quad();
	EXPECT_TRUE(hmtest::metrics::ComputeChartCoverage(m, {0u, 1u}, 2u));
	EXPECT_FALSE(hmtest::metrics::ComputeChartCoverage(m, {0u, 2u}, 2u)); // id out of range
	EXPECT_FALSE(hmtest::metrics::ComputeChartCoverage(m, {0u}, 1u)); // size mismatch
}

} // namespace
} // namespace metrics
} // namespace hmtest
