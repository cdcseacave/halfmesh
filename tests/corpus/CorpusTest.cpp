/*
* CorpusTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/corpus/CorpusTest.cpp — validate every generator via the metrics toolkit.
//
// Clean generators: assert known V/E/F/χ/genus/boundary-loop count/watertight,
//   plus geometry (area, angle-defect≈4π for sphere).
// Dirty generators: assert the metrics detect the exact injected defect.
// LargeMesh: builds, is watertight/valid, face count ≥ target/4.

#include "Corpus.h"
#include "Metrics.h"

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace hmtest {
namespace corpus {
namespace {

using metrics::ComputeTopology;
using metrics::TopologyCounts;
using metrics::ComputeSurfaceArea;
using metrics::ComputeSignedVolume;
using metrics::ScanFinite;
using metrics::ComputeAngleDefect;

// ============================================================
// Helper: assert topology matches KnownTopology
// ============================================================

static void AssertTopology(const halfmesh::Mesh& m,
                           const KnownTopology& kt,
                           const std::string& label)
{
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_EQ(tc.numVertices, kt.numVertices) << label << " num_vertices";
	EXPECT_EQ(tc.numFaces, kt.numFaces) << label << " num_faces";
	EXPECT_EQ(tc.numEdges, kt.numEdges) << label << " num_edges";
	EXPECT_EQ(tc.euler, kt.euler) << label << " euler";
	EXPECT_EQ(tc.genus, kt.genus) << label << " genus";
	EXPECT_EQ(tc.numBoundaryLoops, kt.numBoundaryLoops) << label << " boundary_loops";
	EXPECT_EQ(tc.isWatertight, kt.isWatertight) << label << " is_watertight";
	EXPECT_TRUE(tc.isEdgeManifold) << label << " is_edge_manifold";
	EXPECT_TRUE(ScanFinite(m)) << label << " all_finite";
}

// ============================================================
// CLEAN GENERATORS — topology
// ============================================================

TEST(CorpusClean, Triangle)
{
	AssertTopology(Triangle(), Triangle_Known(), "Triangle");
}

TEST(CorpusClean, Quad)
{
	AssertTopology(Quad(), Quad_Known(), "Quad");
}

TEST(CorpusClean, Tetrahedron)
{
	const halfmesh::Mesh m = TetrahedronMesh();
	AssertTopology(m, TetrahedronMesh_Known(), "Tetrahedron");
	EXPECT_TRUE(ComputeTopology(m).isVertexManifold) << "Tetrahedron vertex_manifold";
	// Surface area of regular tetrahedron edge=1: √3
	const double area = ComputeSurfaceArea(m);
	const double expected = std::sqrt(3.0);
	EXPECT_NEAR(area, expected, expected * 2e-4) << "Tetrahedron area";
	// Volume of regular tetrahedron edge=1: 1/(6√2)
	const double vol = std::abs(ComputeSignedVolume(m));
	const double expVol = 1.0 / (6.0 * std::sqrt(2.0));
	EXPECT_NEAR(vol, expVol, expVol * 2e-4) << "Tetrahedron volume";
}

TEST(CorpusClean, Cube)
{
	const halfmesh::Mesh m = CubeMesh(2.0f); // side=2
	// Known topology doesn't change with side (same connectivity).
	AssertTopology(m, CubeMesh_Known(), "Cube(2)");
	EXPECT_TRUE(ComputeTopology(m).isVertexManifold) << "Cube vertex_manifold";
	// Area = 6·s² = 24, Volume = s³ = 8
	const double area = ComputeSurfaceArea(m);
	EXPECT_NEAR(area, 24.0, 24.0 * 1e-4) << "Cube area";
	const double vol = std::abs(ComputeSignedVolume(m));
	EXPECT_NEAR(vol, 8.0, 8.0 * 1e-4) << "Cube volume";
}

TEST(CorpusClean, Icosahedron)
{
	const halfmesh::Mesh m = IcosahedronMesh();
	AssertTopology(m, IcosahedronMesh_Known(), "Icosahedron");
	EXPECT_TRUE(ComputeTopology(m).isVertexManifold) << "Icosahedron vertex_manifold";
	EXPECT_TRUE(ScanFinite(m)) << "Icosahedron all_finite";
}

TEST(CorpusClean, GridPlane_4)
{
	const halfmesh::Mesh m = GridPlane(4);
	AssertTopology(m, GridPlane_Known(4), "GridPlane(4)");
	EXPECT_FALSE(ComputeTopology(m).isWatertight) << "GridPlane open";
	// All z=0: surface area = 4×4 = 16 (flat grid, unit quads).
	const double area = ComputeSurfaceArea(m);
	EXPECT_NEAR(area, 16.0, 16.0 * 1e-4) << "GridPlane(4) area";
}

TEST(CorpusClean, GridPlane_1)
{
	// n=1: 4 verts, 2 tris, 1 boundary loop.
	const halfmesh::Mesh m = GridPlane(1);
	AssertTopology(m, GridPlane_Known(1), "GridPlane(1)");
}

TEST(CorpusClean, OpenCylinder_8_4)
{
	const halfmesh::Mesh m = OpenCylinder(8, 4);
	AssertTopology(m, OpenCylinder_Known(8, 4), "OpenCylinder(8,4)");
	EXPECT_FALSE(ComputeTopology(m).isWatertight) << "Cylinder open";
}

TEST(CorpusClean, Cone_8)
{
	const halfmesh::Mesh m = Cone(8);
	AssertTopology(m, Cone_Known(8), "Cone(8)");
	EXPECT_FALSE(ComputeTopology(m).isWatertight) << "Cone open";
}

TEST(CorpusClean, UVSphere_8_12)
{
	const halfmesh::Mesh m = UVSphere(8, 12);
	AssertTopology(m, UVSphere_Known(8, 12), "UVSphere(8,12)");
	EXPECT_TRUE(ComputeTopology(m).isVertexManifold) << "UVSphere vertex_manifold";
	// Gauss–Bonnet: integrated angle defect ≈ 4π for a sphere (genus 0).
	const double defect = ComputeAngleDefect(m);
	const double fourPi = 4.0 * std::numbers::pi;
	// Coarse discretization → allow 15% tolerance.
	EXPECT_NEAR(defect, fourPi, fourPi * 0.15)
	    << "UVSphere angle-defect ≈ 4π (Gauss–Bonnet)";
}

TEST(CorpusClean, UVSphere_Closed_ChiGenus)
{
	// Extra check: closed, χ=2, g=0 for a finer sphere.
	const halfmesh::Mesh m = UVSphere(16, 20);
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_EQ(tc.euler, 2) << "UVSphere(16,20) euler=2";
	EXPECT_EQ(tc.genus, 0) << "UVSphere(16,20) genus=0";
	EXPECT_TRUE(tc.isWatertight) << "UVSphere(16,20) watertight";
	// Finer sphere: angle defect should be closer to 4π.
	const double defect = ComputeAngleDefect(m);
	const double fourPi = 4.0 * std::numbers::pi;
	EXPECT_NEAR(defect, fourPi, fourPi * 0.05) << "UVSphere(16,20) angle-defect";
}

TEST(CorpusClean, TorusMesh_12_8)
{
	const halfmesh::Mesh m = TorusMesh(12, 8);
	AssertTopology(m, TorusMesh_Known(12, 8), "TorusMesh(12,8)");
	EXPECT_TRUE(ComputeTopology(m).isWatertight) << "Torus watertight";
	EXPECT_EQ(ComputeTopology(m).euler, 0) << "Torus χ=0";
	EXPECT_EQ(ComputeTopology(m).genus, 1) << "Torus genus=1";
	EXPECT_TRUE(ComputeTopology(m).isVertexManifold) << "Torus vertex_manifold";
}

// ============================================================
// LargeMesh — builds, is watertight/valid
// ============================================================

TEST(CorpusClean, LargeMesh_1000)
{
	unsigned actual = 0;
	const halfmesh::Mesh m = LargeMesh(1000, &actual);
	EXPECT_GE(actual, 1u) << "LargeMesh has at least 1 face";
	// actual ≥ target/4 (coarse estimate: slices/stacks might undershoot slightly)
	EXPECT_GE(actual, 1000u / 4u) << "LargeMesh face count ≥ target/4";
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_TRUE(tc.isWatertight) << "LargeMesh watertight";
	EXPECT_TRUE(tc.isEdgeManifold) << "LargeMesh edge_manifold";
	EXPECT_EQ(tc.euler, 2) << "LargeMesh euler=2 (sphere topology)";
	EXPECT_TRUE(ScanFinite(m)) << "LargeMesh all_finite";
}

TEST(CorpusClean, LargeMesh_10000)
{
	unsigned actual = 0;
	const halfmesh::Mesh m = LargeMesh(10000, &actual);
	EXPECT_GE(actual, 10000u / 4u) << "LargeMesh(10000) face count ≥ target/4";
	EXPECT_TRUE(ComputeTopology(m).isWatertight) << "LargeMesh(10000) watertight";
	EXPECT_TRUE(ScanFinite(m)) << "LargeMesh(10000) all_finite";
}

// ============================================================
// DIRTY GENERATORS — defects detected by metrics
// ============================================================

TEST(CorpusDirty, BowTie_VertexNonManifold)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyBowTie(&dd);
	ASSERT_TRUE(dd.hasBowTieVertex);
	const TopologyCounts tc = ComputeTopology(m);
	// Each edge has at most 2 incident faces → edge-manifold.
	EXPECT_TRUE(tc.isEdgeManifold) << "BowTie: still edge-manifold";
	// Vertex 0 has two disconnected fans → vertex-non-manifold.
	EXPECT_FALSE(tc.isVertexManifold) << "BowTie: vertex-non-manifold detected";
	EXPECT_TRUE(ScanFinite(m)) << "BowTie all_finite";
}

TEST(CorpusDirty, ThreeOnEdge_EdgeNonManifold)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyThreeOnEdge(&dd);
	ASSERT_TRUE(dd.hasThreeOnEdge);
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_FALSE(tc.isEdgeManifold) << "ThreeOnEdge: edge-non-manifold detected";
	EXPECT_TRUE(ScanFinite(m)) << "ThreeOnEdge all_finite";
}

TEST(CorpusDirty, DuplicateFaces_Count)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyDuplicateFaces(3, &dd);
	EXPECT_EQ(dd.duplicateFaces, 6u) // 3 pairs × 2
	    << "DuplicateFaces: 6 duplicate face count (both copies)";
	// Total faces: 12 (cube) + 3 duplicates = 15.
	EXPECT_EQ(m.faces.size(), 15u) << "DuplicateFaces: 15 total faces before repair";
	// After repair (remove_both=true): 15 - 6 = 9 faces remain.
	halfmesh::Mesh m2 = m;
	const halfmesh::Mesh::FIndex removed = m2.RemoveDuplicateFaces(/*removeBothFaces=*/true);
	EXPECT_EQ(removed, 6u) << "DuplicateFaces: RemoveDuplicateFaces removes 6 (3 pairs)";
	EXPECT_TRUE(ScanFinite(m)) << "DuplicateFaces all_finite";
}

TEST(CorpusDirty, DegenerateFaces_Count)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyDegenerateFaces(3, &dd);
	EXPECT_EQ(dd.degenerateFaces, 3u) << "DegenerateFaces: 3 injected";
	// Each degenerate face uses its own isolated collinear vertex triple, so
	// RemoveDegenerateFaces removes exactly the 3 injected zero-area faces.
	halfmesh::Mesh m2 = m;
	const halfmesh::Mesh::FIndex removed = m2.RemoveDegenerateFaces(1e-5f);
	EXPECT_EQ(removed, dd.degenerateFaces) << "DegenerateFaces: exactly 3 removed";
	EXPECT_TRUE(ScanFinite(m)) << "DegenerateFaces all_finite";
}

TEST(CorpusDirty, UnreferencedVertices_Count)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyUnreferencedVertices(5, &dd);
	EXPECT_EQ(dd.unreferencedVertices, 5u) << "UnrefVerts: 5 injected";
	// Cube has 8 verts; with 5 unreferenced: total = 8+3(from degen verts)+5 = 8+5+3.
	// Actually: CubeMesh=8 verts; we add 3 collinear verts (in DirtyDegen — but here
	// it's DirtyUnref which uses CubeMesh directly) + 5 orphan verts.
	// DirtyUnreferencedVertices starts from CubeMesh(8 verts) and appends 5 = 13 total.
	EXPECT_EQ(m.vertices.size(), 13u) << "UnrefVerts: 13 total vertices";
	// RemoveUnreferencedVertices returns number removed.
	halfmesh::Mesh m2 = m;
	const halfmesh::Mesh::VIndex removed = m2.RemoveUnreferencedVertices();
	EXPECT_EQ(removed, 5u) << "UnrefVerts: 5 removed by RemoveUnreferencedVertices";
	EXPECT_EQ(m2.vertices.size(), 8u) << "UnrefVerts: 8 remain (cube)";
	EXPECT_TRUE(ScanFinite(m)) << "UnrefVerts all_finite";
}

TEST(CorpusDirty, Holes_BoundaryLoops)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyHoles(2, &dd);
	EXPECT_EQ(dd.deletedFaces, 2u) << "Holes: 2 faces deleted";
	EXPECT_EQ(dd.expectedBoundaryLoops, 2u) << "Holes: 2 expected boundary loops";
	// Total faces: 12 - 2 = 10.
	EXPECT_EQ(m.faces.size(), 10u) << "Holes: 10 faces remain";
	const TopologyCounts tc = ComputeTopology(m);
	EXPECT_EQ(tc.numBoundaryLoops, 2u)
	    << "Holes: metrics detect 2 boundary loops";
	EXPECT_FALSE(tc.isWatertight) << "Holes: not watertight";
	EXPECT_TRUE(ScanFinite(m)) << "Holes all_finite";
}

TEST(CorpusDirty, ManyComponents_Count)
{
	DirtyDefects dd;
	const halfmesh::Mesh m = DirtyManyComponents(2, &dd);
	// DirtyManyComponents(2) = 1 cube (12F, 8V) + 1 small triangle (1F, 3V).
	EXPECT_EQ(dd.numComponents, 2u) << "ManyComponents: 2 components total";
	EXPECT_EQ(dd.smallComponents, 1u) << "ManyComponents: 1 small component";
	// Total: 8 + 3 = 11 verts, 12 + 1 = 13 faces.
	EXPECT_EQ(m.vertices.size(), 11u) << "ManyComponents: 11 verts";
	EXPECT_EQ(m.faces.size(), 13u) << "ManyComponents: 13 faces";
	EXPECT_TRUE(ScanFinite(m)) << "ManyComponents all_finite";

	// RemoveSmallComponents(threshold) removes components with face count < threshold.
	// With threshold = dd.smallComponentThreshold (= 2): the triangle (1 face) is
	// removed, the cube (12 faces) survives.  The return value is the number of
	// LARGE components (those that survived), which equals numComponents - smallComponents.
	halfmesh::Mesh m2 = m;
	const unsigned largeKept = m2.RemoveSmallComponents(dd.smallComponentThreshold);
	EXPECT_EQ(largeKept, dd.numComponents - dd.smallComponents)
	    << "ManyComponents: 1 large component survives RemoveSmallComponents";
	// After removal: only the cube remains.
	EXPECT_EQ(m2.faces.size(), dd.largeComponentFaces)
	    << "ManyComponents: 12 cube faces after repair";
	EXPECT_EQ(m2.vertices.size(), dd.largeComponentVerts)
	    << "ManyComponents: 8 cube verts after repair";
	// One component remains → metrics confirm.
	const TopologyCounts tc = ComputeTopology(m2);
	EXPECT_TRUE(tc.isWatertight) << "ManyComponents: surviving cube is watertight";
	EXPECT_EQ(tc.euler, 2) << "ManyComponents: surviving cube χ=2";
}

// ============================================================
// UV GROUND-TRUTH HELPERS — correctness checks
// ============================================================

// Helper: verify that a UV map has the expected count and all values in [0,1].
static void AssertUVBasics(const std::vector<UV2>& uvs,
                           unsigned expectedCount,
                           const std::string& label)
{
	EXPECT_EQ(uvs.size(), expectedCount) << label << " uv count";
	for (size_t i = 0; i < uvs.size(); ++i) {
		EXPECT_GE(uvs[i].u, 0.f) << label << " uv[" << i << "].u >= 0";
		EXPECT_LE(uvs[i].u, 1.f) << label << " uv[" << i << "].u <= 1";
		EXPECT_GE(uvs[i].v, 0.f) << label << " uv[" << i << "].v >= 0";
		EXPECT_LE(uvs[i].v, 1.f) << label << " uv[" << i << "].v <= 1";
	}
}

TEST(CorpusUV, GridPlane_ExpectedUV_Identity)
{
	// GridPlane(n=4): (n+1)²=25 vertices.
	// Corner vertices: (0,0)→(0,0), (4,0)→(1,0), (4,4)→(1,1), (0,4)→(0,1).
	constexpr unsigned n = 4;
	const auto uvs = GridPlane_ExpectedUV(n);
	AssertUVBasics(uvs, (n + 1) * (n + 1), "GridPlane(4)");
	// Spot-check corners (row-major: idx = row*(n+1) + col).
	EXPECT_NEAR(uvs[0].u, 0.f, 1e-5f) << "GridPlane corner (0,0).u";
	EXPECT_NEAR(uvs[0].v, 0.f, 1e-5f) << "GridPlane corner (0,0).v";
	const unsigned topRight = n * (n + 1) + n; // row=n, col=n
	EXPECT_NEAR(uvs[topRight].u, 1.f, 1e-5f) << "GridPlane corner (n,n).u";
	EXPECT_NEAR(uvs[topRight].v, 1.f, 1e-5f) << "GridPlane corner (n,n).v";
	// Centre vertex: row=2, col=2 → uv=(0.5, 0.5).
	const unsigned centre = 2 * (n + 1) + 2;
	EXPECT_NEAR(uvs[centre].u, 0.5f, 1e-5f) << "GridPlane centre.u";
	EXPECT_NEAR(uvs[centre].v, 0.5f, 1e-5f) << "GridPlane centre.v";
	// Count = (n+1)² = same as mesh vertex count.
	EXPECT_EQ(uvs.size(), GridPlane(n).vertices.size()) << "GridPlane UV count matches mesh";
}

TEST(CorpusUV, OpenCylinder_ExpectedUV)
{
	constexpr unsigned R = 8, H = 4;
	const auto uvs = OpenCylinder_ExpectedUV(R, H);
	AssertUVBasics(uvs, R * (H + 1), "OpenCylinder(8,4)");
	// Apex ring (h=0, r=0): u=0, v=0.
	EXPECT_NEAR(uvs[0].u, 0.f, 1e-5f) << "Cyl bottom-ring start u=0";
	EXPECT_NEAR(uvs[0].v, 0.f, 1e-5f) << "Cyl bottom-ring v=0";
	// Top ring (h=H, r=0): u=0, v=1.
	const unsigned top0 = H * R;
	EXPECT_NEAR(uvs[top0].u, 0.f, 1e-5f) << "Cyl top-ring u=0";
	EXPECT_NEAR(uvs[top0].v, 1.f, 1e-5f) << "Cyl top-ring v=1";
	// Middle ring (h=H/2, r=R/2): u=0.5, v=0.5.
	const unsigned mid = (H / 2) * R + (R / 2);
	EXPECT_NEAR(uvs[mid].u, 0.5f, 1e-5f) << "Cyl mid u=0.5";
	EXPECT_NEAR(uvs[mid].v, 0.5f, 1e-5f) << "Cyl mid v=0.5";
	// Count matches mesh.
	EXPECT_EQ(uvs.size(), OpenCylinder(R, H).vertices.size()) << "Cyl UV count matches mesh";
}

TEST(CorpusUV, Cone_ExpectedUV)
{
	constexpr unsigned R = 8;
	const auto uvs = Cone_ExpectedUV(R);
	AssertUVBasics(uvs, R + 1, "Cone(8)");
	// Apex (index R): uv = (0.5, 0.5).
	EXPECT_NEAR(uvs[R].u, 0.5f, 1e-5f) << "Cone apex u=0.5";
	EXPECT_NEAR(uvs[R].v, 0.5f, 1e-5f) << "Cone apex v=0.5";
	// Base vertex 0: phi=0, uv=(0.5+0.5*1, 0.5+0.5*0) = (1.0, 0.5).
	EXPECT_NEAR(uvs[0].u, 1.0f, 1e-5f) << "Cone base[0] u=1.0";
	EXPECT_NEAR(uvs[0].v, 0.5f, 1e-5f) << "Cone base[0] v=0.5";
	// All base vertices lie on a circle of radius 0.5 centred at (0.5, 0.5).
	for (unsigned r = 0; r < R; ++r) {
		const float du = uvs[r].u - 0.5f;
		const float dv = uvs[r].v - 0.5f;
		EXPECT_NEAR(std::sqrt(du * du + dv * dv), 0.5f, 1e-4f)
		    << "Cone base[" << r << "] on sector arc";
	}
	// Count matches mesh.
	EXPECT_EQ(uvs.size(), Cone(R).vertices.size()) << "Cone UV count matches mesh";
}

// ============================================================
// MAKE_CORPUS — overview check
// ============================================================

TEST(CorpusOverview, MakeCorpus_HasAllEntries)
{
	const auto entries = makeCorpus();
	EXPECT_GE(entries.size(), 17u)
	    << "make_corpus returns at least 17 entries (10 clean + 7 dirty)";
	int cleanCount = 0;
	int dirtyCount = 0;
	for (const auto& e : entries) {
		EXPECT_FALSE(e.name.empty()) << "Every entry has a name";
		if (e.isDirty)
			++dirtyCount;
		else
			++cleanCount;
	}
	EXPECT_GE(cleanCount, 10) << "At least 10 clean generators";
	EXPECT_GE(dirtyCount, 7) << "At least 7 dirty synthesizers";
}

} // namespace
} // namespace corpus
} // namespace hmtest
