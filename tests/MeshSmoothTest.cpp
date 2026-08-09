/*
* MeshSmoothTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for MeshSmooth.cpp:
//   Mesh::SmoothHCLaplacian — HC Laplacian smoothing (Vollmer/Mencl/Mueller
//   1999).
//   Mesh::SmoothTaubin — Taubin'95 lambda|mu band-pass smoothing; smooths
//   aggressively at ~zero shrinkage. Taubin* tests mirror the HC suite:
//   closed-form tetrahedron pin, locked vertices, determinism/composition/
//   no-ops, repair-on-build, attribute/normal-cache behavior,
//   denoise-without-shrink quality bounds, and a closed-form single-triangle
//   pin of the border-vertex boundary-curve rule (unlike SmoothHCLaplacian's
//   uniform ring).
//
// Tests:
//   1. Closed form: one iteration contracts a tetrahedron toward its centroid
//      by exactly 1/9 — pins the exact formula
//   2. Locked vertices stay bit-identical yet still feed neighbor averages
//   3. Determinism: identical runs are bit-identical
//   4. Composition: smooth(2) == smooth(1) applied twice, bit-identical
//   5. No-ops: empty mesh, zero/negative iterations
//   6. Unreferenced vertices are repaired away by the half-edge build (RemeshIsotropic semantics)
//   7. Faces/topology/attributes untouched; positions stay finite
//   8. Cached face normals are invalidated (cleared) by smoothing
//   9. Quality: noise on a sphere/grid strongly reduced, shrinkage bounded
//   10. Border rule: single-triangle closed form pins Taubin's boundary-curve
//       smoothing for border vertices (SmoothTaubin only; not shared with HC)
//   11. Unified Mesh::Smooth dispatcher: bit-identical to the individual
//       smoothers called with default params; Taubin is the default method

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include "metrics/Metrics.h"
#include "corpus/Corpus.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace halfmesh {
namespace {

// bit-exact equality of two vertex arrays (Mesh::Vertex is trivially copyable)
static bool BitEqual(const std::vector<Mesh::Vertex>& a, const std::vector<Mesh::Vertex>& b)
{
	return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Mesh::Vertex)) == 0);
}

static Mesh::Vertex Centroid(const std::vector<Mesh::Vertex>& verts)
{
	Mesh::Vertex c = Mesh::Vertex::Zero();
	for (const Mesh::Vertex& v : verts)
		c += v;
	return c / float(verts.size());
}

// Expected HC position on a tetrahedron: C + (1/9)(p - C).
// Derivation (any tetrahedron; every vertex is adjacent to the other three;
// S = sum of all vertices, C = S/4):
//   avg_i = (S - p_i)/3
//   b_i   = avg_i - p_i = (S - 4 p_i)/3
//   dif_i = mean of neighbor b_j = -b_i/3
//   p_i'  = avg_i - b_i/2 - dif_i/2 = C + (1/9)(p_i - C)
// The 1/9 factor pins the exact formula: adding the averaged neighbor
// correction instead of subtracting it would give 5/9.
static Mesh::Vertex TetraExpected(const Mesh::Vertex& p, const Mesh::Vertex& c)
{
	return c + (p - c) * (1.f / 9.f);
}

TEST(MeshSmooth, TetrahedronClosedForm)
{
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	mesh.SmoothHCLaplacian(1);
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = TetraExpected(orig[i], c);
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
}

TEST(MeshSmooth, LockedVertexHeldFixedButStillContributes)
{
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	std::vector<bool> locked(orig.size(), false);
	locked[0] = true;
	mesh.SmoothHCLaplacian(1, &locked);
	// the locked vertex is bit-identical
	EXPECT_TRUE(std::memcmp(&orig[0], &mesh.vertices[0], sizeof(Mesh::Vertex)) == 0);
	// pass 3 reads only pre-update positions, so the other vertices land
	// exactly on the unlocked closed-form result — proving the locked vertex
	// still contributed to their averages
	for (size_t i = 1; i < orig.size(); ++i) {
		const Mesh::Vertex expected = TetraExpected(orig[i], c);
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
}

TEST(MeshSmooth, Deterministic)
{
	Mesh a = hmtest::corpus::UVSphere(8, 12);
	Mesh b = hmtest::corpus::UVSphere(8, 12);
	a.SmoothHCLaplacian(3);
	b.SmoothHCLaplacian(3);
	EXPECT_TRUE(BitEqual(a.vertices, b.vertices));
}

TEST(MeshSmooth, IterationsCompose)
{
	// N iterations in one call == N single-iteration calls, bit-identical
	// (each iteration reads only positions produced by the previous one)
	Mesh a = hmtest::corpus::UVSphere(8, 12);
	Mesh b = hmtest::corpus::UVSphere(8, 12);
	a.SmoothHCLaplacian(2);
	b.SmoothHCLaplacian(1);
	b.SmoothHCLaplacian(1);
	EXPECT_TRUE(BitEqual(a.vertices, b.vertices));
}

TEST(MeshSmooth, NoOpCases)
{
	Mesh empty;
	empty.SmoothHCLaplacian(3); // must not crash
	EXPECT_TRUE(empty.Empty());

	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	mesh.SmoothHCLaplacian(0);
	EXPECT_TRUE(BitEqual(orig, mesh.vertices));
	mesh.SmoothHCLaplacian(-2);
	EXPECT_TRUE(BitEqual(orig, mesh.vertices));
}

TEST(MeshSmooth, UnreferencedVertexRepairedAway)
{
	// Unreferenced vertices violate HalfMesh::Build preconditions (debug ASSERT
	// src/HalfMesh.cpp:130; in release ConnectBorders rejects NO_ID anchors), so
	// ListHalfEdges falls back to ListHalfEdgesSafe, whose RemoveUnreferencedVertices
	// drops the loner before smoothing — same repair-on-build semantics as RemeshIsotropic.
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	const Mesh::Vertex loner(7.f, 8.f, 9.f);
	mesh.vertices.push_back(loner); // referenced by no face
	mesh.SmoothHCLaplacian(1);
	// the loner is removed by repair BEFORE smoothing
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	// the 4 surviving vertices match the closed-form result (repair runs before smoothing)
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = TetraExpected(orig[i], c);
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, TopologyAndAttributesUntouched)
{
	Mesh mesh = hmtest::corpus::UVSphere(8, 12);
	const std::vector<Mesh::Face> facesBefore = mesh.faces;
	const hmtest::metrics::TopologyCounts topoBefore = hmtest::metrics::ComputeTopology(mesh);
	mesh.SmoothHCLaplacian(3);
	ASSERT_EQ(facesBefore.size(), mesh.faces.size());
	EXPECT_TRUE(std::memcmp(facesBefore.data(), mesh.faces.data(), facesBefore.size() * sizeof(Mesh::Face)) == 0);
	const hmtest::metrics::TopologyCounts topoAfter = hmtest::metrics::ComputeTopology(mesh);
	EXPECT_EQ(topoBefore.numVertices, topoAfter.numVertices);
	EXPECT_EQ(topoBefore.numEdges, topoAfter.numEdges);
	EXPECT_EQ(topoBefore.numFaces, topoAfter.numFaces);
	EXPECT_EQ(topoBefore.euler, topoAfter.euler);
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, InvalidatesCachedFaceNormals)
{
	// SmoothHCLaplacian moves vertices without touching faces, so the repo's
	// size-based freshness idiom (faceNormals.size() == faces.size()) can't
	// detect staleness on its own — the cache must be explicitly cleared
	// (mirrors CloseHoles's faceNormals.clear() in src/MeshHoles.cpp).
	Mesh mesh = hmtest::corpus::UVSphere(8, 12);
	mesh.ComputeFaceNormals();
	ASSERT_EQ(mesh.faceNormals.size(), mesh.faces.size());
	mesh.SmoothHCLaplacian(1);
	EXPECT_TRUE(mesh.faceNormals.empty());
}

// deterministic hash-noise in [-1, 1] (integer mix; identical on every
// platform, unlike std::uniform_real_distribution which is
// implementation-defined)
static float HashNoise(uint32_t i)
{
	uint32_t h = i * 2654435761u;
	h ^= h >> 16;
	h *= 2246822519u;
	h ^= h >> 13;
	return float(h & 0xFFFFFFu) / float(0x800000u) - 1.f;
}

static double MeanRadius(const std::vector<Mesh::Vertex>& verts, const Mesh::Vertex& c)
{
	double mean = 0;
	for (const Mesh::Vertex& v : verts)
		mean += (v - c).norm();
	return mean / double(verts.size());
}

// radial standard deviation around the CURRENT mean radius: measures residual
// noise independently of any uniform shrinkage
static double RadialSpread(const std::vector<Mesh::Vertex>& verts, const Mesh::Vertex& c)
{
	const double mean = MeanRadius(verts, c);
	double s = 0;
	for (const Mesh::Vertex& v : verts) {
		const double d = (v - c).norm() - mean;
		s += d * d;
	}
	return std::sqrt(s / double(verts.size()));
}

TEST(MeshSmooth, DenoisesSphereWithBoundedShrinkage)
{
	Mesh mesh = hmtest::corpus::UVSphere(16, 24);
	const Mesh::Vertex c = Centroid(mesh.vertices);
	const double cleanR = MeanRadius(mesh.vertices, c);
	// radial noise, 3% of the radius
	for (size_t i = 0; i < mesh.vertices.size(); ++i) {
		Mesh::Vertex& v = mesh.vertices[i];
		v += (v - c).normalized() * (0.03f * float(cleanR) * HashNoise(uint32_t(i)));
	}
	const double spreadBefore = RadialSpread(mesh.vertices, c);
	mesh.SmoothHCLaplacian(2);
	const double spreadAfter = RadialSpread(mesh.vertices, c);
	// noise strongly reduced (measured ratio 0.393 after 2 iterations — threshold
	// has ~1.3x margin; per repo golden policy the fixture follows the algorithm —
	// recalibrate this pin if the default lambda/mu change)
	EXPECT_LT(spreadAfter, 0.5 * spreadBefore);
	// ...without shrinking: the HC correction barely contracts the sphere —
	// measured mean-radius ratio 0.9982 after 2 iterations
	EXPECT_GT(MeanRadius(mesh.vertices, c), 0.99 * cleanR);
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, FlattensNoisyGrid)
{
	Mesh mesh = hmtest::corpus::GridPlane(12);
	// the grid is planar: its plane normal is the degenerate bbox axis
	const Eigen::AlignedBox<Mesh::Type, 3> bbox = mesh.ComputeAABBox();
	const Eigen::Matrix<Mesh::Type, 3, 1> ext = bbox.sizes();
	int flatAxis = 0;
	if (ext[1] < ext[flatAxis])
		flatAxis = 1;
	if (ext[2] < ext[flatAxis])
		flatAxis = 2;
	const float diag = ext.norm();
	const float planeCoord = bbox.center()[flatAxis];
	for (size_t i = 0; i < mesh.vertices.size(); ++i)
		mesh.vertices[i][flatAxis] += 0.02f * diag * HashNoise(uint32_t(i));
	const auto offPlaneRms = [&](const std::vector<Mesh::Vertex>& verts) {
		double s = 0;
		for (const Mesh::Vertex& v : verts) {
			const double d = double(v[flatAxis]) - double(planeCoord);
			s += d * d;
		}
		return std::sqrt(s / double(verts.size()));
	};
	const double rmsBefore = offPlaneRms(mesh.vertices);
	mesh.SmoothHCLaplacian(2);
	const double rmsAfter = offPlaneRms(mesh.vertices);
	EXPECT_LT(rmsAfter, 0.5 * rmsBefore);
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

// Expected Taubin position on a tetrahedron after ONE lambda|mu iteration.
// Derivation (see TetraExpected above: avg_i - p_i = (4/3)(C - p_i)):
//   one pass with weight w: p' - C = (p - C) * (1 - 4w/3)
//   one iteration:          p' - C = (p - C) * (1 - 4*lambda/3)(1 - 4*mu/3)
// The defaults lambda = 0.65, mu = -0.69 give (0.4/3)*(5.76/3) = 0.256 (exactly
// 32/125) — pins both the update formula and the default parameter values.
static Mesh::Vertex TaubinTetraExpected(const Mesh::Vertex& p, const Mesh::Vertex& c, float lambda, float mu)
{
	const float factor = (1.f - 4.f * lambda / 3.f) * (1.f - 4.f * mu / 3.f);
	return c + (p - c) * factor;
}

TEST(MeshSmooth, TaubinTetrahedronClosedForm)
{
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	mesh.SmoothTaubin(1);
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = TaubinTetraExpected(orig[i], c, 0.65f, -0.69f);
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
}

TEST(MeshSmooth, TaubinTetrahedronClosedFormExplicitParams)
{
	// explicit lambda/mu (the classic 0.5/-0.53 pair) — pins parameter plumbing
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	mesh.SmoothTaubin(1, 0.5f, -0.53f);
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = TaubinTetraExpected(orig[i], c, 0.5f, -0.53f);
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
}

TEST(MeshSmooth, TaubinLockedVertexHeldFixedButStillContributes)
{
	// unlike the HC test, Taubin's second pass reads positions the first pass
	// already moved, so there is no simple closed form with a lock; instead
	// prove contribution differentially: shifting the LOCKED vertex must change
	// where the unlocked ones land
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	Mesh moved = hmtest::corpus::TetrahedronMesh();
	moved.vertices[0] += Mesh::Vertex(0.25f, 0.f, 0.f);
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const std::vector<Mesh::Vertex> movedOrig = moved.vertices;
	std::vector<bool> locked(orig.size(), false);
	locked[0] = true;
	mesh.SmoothTaubin(1, 0.65f, -0.69f, &locked);
	moved.SmoothTaubin(1, 0.65f, -0.69f, &locked);
	// the locked vertex is bit-identical in both runs
	EXPECT_TRUE(std::memcmp(&orig[0], &mesh.vertices[0], sizeof(Mesh::Vertex)) == 0);
	EXPECT_TRUE(std::memcmp(&movedOrig[0], &moved.vertices[0], sizeof(Mesh::Vertex)) == 0);
	for (size_t i = 1; i < orig.size(); ++i) {
		// unlocked vertices moved...
		EXPECT_FALSE(std::memcmp(&orig[i], &mesh.vertices[i], sizeof(Mesh::Vertex)) == 0) << "vertex " << i;
		// ...and land differently when the locked vertex was shifted — it
		// still contributed to their averages
		EXPECT_FALSE(std::memcmp(&mesh.vertices[i], &moved.vertices[i], sizeof(Mesh::Vertex)) == 0) << "vertex " << i;
	}
}

TEST(MeshSmooth, TaubinDeterministic)
{
	Mesh a = hmtest::corpus::UVSphere(8, 12);
	Mesh b = hmtest::corpus::UVSphere(8, 12);
	a.SmoothTaubin(3);
	b.SmoothTaubin(3);
	EXPECT_TRUE(BitEqual(a.vertices, b.vertices));
}

TEST(MeshSmooth, TaubinIterationsCompose)
{
	// N iterations in one call == N single-iteration calls, bit-identical
	// (each lambda|mu pair reads only positions produced by the previous one)
	Mesh a = hmtest::corpus::UVSphere(8, 12);
	Mesh b = hmtest::corpus::UVSphere(8, 12);
	a.SmoothTaubin(2);
	b.SmoothTaubin(1);
	b.SmoothTaubin(1);
	EXPECT_TRUE(BitEqual(a.vertices, b.vertices));
}

TEST(MeshSmooth, TaubinNoOpCases)
{
	Mesh empty;
	empty.SmoothTaubin(3); // must not crash
	EXPECT_TRUE(empty.Empty());

	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	mesh.SmoothTaubin(0);
	EXPECT_TRUE(BitEqual(orig, mesh.vertices));
	mesh.SmoothTaubin(-2);
	EXPECT_TRUE(BitEqual(orig, mesh.vertices));
}

TEST(MeshSmooth, TaubinUnreferencedVertexRepairedAway)
{
	// same repair-on-build semantics as SmoothHCLaplacian (see
	// UnreferencedVertexRepairedAway above for the mechanism)
	Mesh mesh = hmtest::corpus::TetrahedronMesh();
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	const Mesh::Vertex loner(7.f, 8.f, 9.f);
	mesh.vertices.push_back(loner); // referenced by no face
	mesh.SmoothTaubin(1);
	// the loner is removed by repair BEFORE smoothing
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = TaubinTetraExpected(orig[i], c, 0.65f, -0.69f);
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, TaubinTopologyAndAttributesUntouched)
{
	Mesh mesh = hmtest::corpus::UVSphere(8, 12);
	const std::vector<Mesh::Face> facesBefore = mesh.faces;
	const hmtest::metrics::TopologyCounts topoBefore = hmtest::metrics::ComputeTopology(mesh);
	mesh.SmoothTaubin(3);
	ASSERT_EQ(facesBefore.size(), mesh.faces.size());
	EXPECT_TRUE(std::memcmp(facesBefore.data(), mesh.faces.data(), facesBefore.size() * sizeof(Mesh::Face)) == 0);
	const hmtest::metrics::TopologyCounts topoAfter = hmtest::metrics::ComputeTopology(mesh);
	EXPECT_EQ(topoBefore.numVertices, topoAfter.numVertices);
	EXPECT_EQ(topoBefore.numEdges, topoAfter.numEdges);
	EXPECT_EQ(topoBefore.numFaces, topoAfter.numFaces);
	EXPECT_EQ(topoBefore.euler, topoAfter.euler);
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, TaubinInvalidatesCachedFaceNormals)
{
	// same rationale as InvalidatesCachedFaceNormals above
	Mesh mesh = hmtest::corpus::UVSphere(8, 12);
	mesh.ComputeFaceNormals();
	ASSERT_EQ(mesh.faceNormals.size(), mesh.faces.size());
	mesh.SmoothTaubin(1);
	EXPECT_TRUE(mesh.faceNormals.empty());
}

TEST(MeshSmooth, TaubinDenoisesSphereWithoutShrinking)
{
	Mesh mesh = hmtest::corpus::UVSphere(16, 24);
	const Mesh::Vertex c = Centroid(mesh.vertices);
	const double cleanR = MeanRadius(mesh.vertices, c);
	// radial noise, 3% of the radius (same fixture as DenoisesSphereWithBoundedShrinkage)
	for (size_t i = 0; i < mesh.vertices.size(); ++i) {
		Mesh::Vertex& v = mesh.vertices[i];
		v += (v - c).normalized() * (0.03f * float(cleanR) * HashNoise(uint32_t(i)));
	}
	const double spreadBefore = RadialSpread(mesh.vertices, c);
	mesh.SmoothTaubin(5);
	const double spreadAfter = RadialSpread(mesh.vertices, c);
	// noise strongly reduced (lab-measured ratio ~0.33 at 5 iterations — ~1.5x
	// margin; per repo golden policy the fixture follows the algorithm)
	EXPECT_LT(spreadAfter, 0.5 * spreadBefore);
	// ...at essentially ZERO shrinkage — the band-pass keeps the mean radius
	// within a fraction of a percent (measured ~1.002)
	const double r = MeanRadius(mesh.vertices, c);
	EXPECT_GT(r, 0.99 * cleanR);
	EXPECT_LT(r, 1.02 * cleanR);
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, TaubinFlattensNoisyGrid)
{
	Mesh mesh = hmtest::corpus::GridPlane(12);
	const Eigen::AlignedBox<Mesh::Type, 3> bbox = mesh.ComputeAABBox();
	const Eigen::Matrix<Mesh::Type, 3, 1> ext = bbox.sizes();
	int flatAxis = 0;
	if (ext[1] < ext[flatAxis])
		flatAxis = 1;
	if (ext[2] < ext[flatAxis])
		flatAxis = 2;
	const float diag = ext.norm();
	const float planeCoord = bbox.center()[flatAxis];
	for (size_t i = 0; i < mesh.vertices.size(); ++i)
		mesh.vertices[i][flatAxis] += 0.02f * diag * HashNoise(uint32_t(i));
	const auto offPlaneRms = [&](const std::vector<Mesh::Vertex>& verts) {
		double s = 0;
		for (const Mesh::Vertex& v : verts) {
			const double d = double(v[flatAxis]) - double(planeCoord);
			s += d * d;
		}
		return std::sqrt(s / double(verts.size()));
	};
	const double rmsBefore = offPlaneRms(mesh.vertices);
	// 5 iterations: Taubin's slowest-damped frequency is the k = 2 checkerboard
	// component, per-iteration factor |(1-2*lambda)(1-2*mu)| = 0.714 at the
	// defaults, so worst-case residual ~0.714^5 = 0.19 — comfortable margin
	mesh.SmoothTaubin(5);
	const double rmsAfter = offPlaneRms(mesh.vertices);
	EXPECT_LT(rmsAfter, 0.5 * rmsBefore);
	EXPECT_TRUE(hmtest::metrics::ScanFinite(mesh));
}

TEST(MeshSmooth, TaubinBorderTriangleClosedForm)
{
	// a single triangle: every vertex is a border vertex with exactly two
	// border neighbors, so the border rule gives avg_i = (p0+p1+p2)/3 = C for
	// every vertex and each pass contracts toward the centroid:
	//   p' - C = (p - C) * (1 - lambda)(1 - mu)  per iteration
	// (a plain uniform ring would give (1 - 1.5*lambda)(1 - 1.5*mu) instead —
	// 0.051 vs 0.5915 at the defaults — so this pins the border-curve rule)
	Mesh mesh;
	mesh.vertices = {Mesh::Vertex(0.f, 0.f, 0.f), Mesh::Vertex(1.f, 0.f, 0.f), Mesh::Vertex(0.3f, 0.9f, 0.2f)};
	mesh.faces = {Mesh::Face(0, 1, 2)};
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	mesh.SmoothTaubin(1);
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	const float factor = (1.f - 0.65f) * (1.f + 0.69f);
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = c + (orig[i] - c) * factor;
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
}

TEST(MeshSmooth, TaubinBorderQuadClosedForm)
{
	// two triangles forming a unit square: all four corners are border
	// vertices, but corners 0 and 2 also share the INTERIOR diagonal edge.
	// The border rule excludes interior edges from a border vertex's
	// average, so every corner gets avg_i = (self + 2 border neighbors)/3,
	// which for a square is avg_i - C = (p_i - C)/3 and each pass contracts
	// uniformly about the center:
	//   p' - C = (p - C) * (1 - 2w/3)  =>  per iteration (1 - 2*lambda/3)(1 - 2*mu/3)
	// A twin-check regression that let the diagonal leak into the ring would
	// change the factor for corners 0/2 only (breaking both the value and the
	// square's symmetry) — this pins the mixed border/interior case.
	Mesh mesh;
	mesh.vertices = {Mesh::Vertex(0.f, 0.f, 0.f), Mesh::Vertex(1.f, 0.f, 0.f), Mesh::Vertex(1.f, 1.f, 0.f), Mesh::Vertex(0.f, 1.f, 0.f)};
	mesh.faces = {Mesh::Face(0, 1, 2), Mesh::Face(0, 2, 3)};
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	const Mesh::Vertex c = Centroid(orig);
	mesh.SmoothTaubin(1);
	ASSERT_EQ(orig.size(), mesh.vertices.size());
	const float factor = (1.f - 2.f * 0.65f / 3.f) * (1.f - 2.f * -0.69f / 3.f);
	for (size_t i = 0; i < orig.size(); ++i) {
		const Mesh::Vertex expected = c + (orig[i] - c) * factor;
		EXPECT_NEAR(expected.x(), mesh.vertices[i].x(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.y(), mesh.vertices[i].y(), 1e-5f) << "vertex " << i;
		EXPECT_NEAR(expected.z(), mesh.vertices[i].z(), 1e-5f) << "vertex " << i;
	}
}

TEST(MeshSmooth, TaubinLockedBorderVertexHeldFixedButStillContributes)
{
	// same quad, with border vertex 0 locked; compare against the unlocked
	// run.  Within one iteration the lock only shows up in the mu pass (the
	// lambda pass reads pre-pass positions for every average), so vertex 0's
	// border neighbors 1 and 3 must land differently while the opposite
	// corner 2 (whose averages never read vertex 0) stays bit-identical —
	// pinning both the locked-border path and the accumulate-before-move
	// pass semantics.
	Mesh unlocked;
	unlocked.vertices = {Mesh::Vertex(0.f, 0.f, 0.f), Mesh::Vertex(1.f, 0.f, 0.f), Mesh::Vertex(1.f, 1.f, 0.f), Mesh::Vertex(0.f, 1.f, 0.f)};
	unlocked.faces = {Mesh::Face(0, 1, 2), Mesh::Face(0, 2, 3)};
	Mesh mesh = unlocked;
	const std::vector<Mesh::Vertex> orig = mesh.vertices;
	std::vector<bool> locked(orig.size(), false);
	locked[0] = true;
	unlocked.SmoothTaubin(1);
	mesh.SmoothTaubin(1, 0.65f, -0.69f, &locked);
	// the locked border vertex is bit-identical...
	EXPECT_TRUE(std::memcmp(&orig[0], &mesh.vertices[0], sizeof(Mesh::Vertex)) == 0);
	// ...its border neighbors moved, and moved DIFFERENTLY than unlocked
	for (const size_t i : {size_t(1), size_t(3)}) {
		EXPECT_FALSE(std::memcmp(&orig[i], &mesh.vertices[i], sizeof(Mesh::Vertex)) == 0) << "vertex " << i;
		EXPECT_FALSE(std::memcmp(&unlocked.vertices[i], &mesh.vertices[i], sizeof(Mesh::Vertex)) == 0) << "vertex " << i;
	}
	// ...and the opposite corner is untouched by the lock
	EXPECT_TRUE(std::memcmp(&unlocked.vertices[2], &mesh.vertices[2], sizeof(Mesh::Vertex)) == 0);
}

// --- Mesh::Smooth: unified dispatcher over the individual smoothers ---------
// Each test proves Smooth(iterations, method) is bit-identical to calling the
// matching smoother directly with its default parameters.  UVSphere at 3
// iterations makes the two methods produce distinct results, so a mis-wired
// switch case would break the bit-equality below.

TEST(MeshSmooth, UnifiedSmoothDefaultsToTaubin)
{
	Mesh unified = hmtest::corpus::UVSphere(8, 12);
	Mesh direct = hmtest::corpus::UVSphere(8, 12);
	unified.Smooth(3); // method defaults to Taubin
	direct.SmoothTaubin(3);
	EXPECT_TRUE(BitEqual(unified.vertices, direct.vertices));
}

TEST(MeshSmooth, UnifiedSmoothDispatchesHCLaplacian)
{
	Mesh unified = hmtest::corpus::UVSphere(8, 12);
	Mesh direct = hmtest::corpus::UVSphere(8, 12);
	unified.Smooth(3, Mesh::SmoothMethod::HCLaplacian);
	direct.SmoothHCLaplacian(3);
	EXPECT_TRUE(BitEqual(unified.vertices, direct.vertices));
}

} // namespace
} // namespace halfmesh
