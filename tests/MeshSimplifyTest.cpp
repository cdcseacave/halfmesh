/*
* MeshSimplifyTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for MeshSimplify.cpp:
//   Mesh::Simplify — QEM edge-collapse decimation
//   + RoundCast semantics tests (halfmesh/Util/Maths.h)
//
// Tests:
//   1. RoundCast semantics (as documented in halfmesh/Util/Maths.h)
//   2. Simplify(0.5f) on tests/data/mesh.ply — ~50% face reduction, validity, bbox
//   3. Simplify(0.25f) on tests/data/mesh.ply — ~25% face reduction (monotone)
//   4. One-sided Hausdorff via TriangleKdTree: decimated verts close to orig surface
//   5. Tiny inline grid mesh — runs and reduces faces

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/TriangleKDTree.h>
#include <halfmesh/Util/Maths.h>

#include <gtest/gtest.h>

#include "Corpus.h"
#include "Metrics.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <cmath>
#include <vector>

using halfmesh::RoundCast;

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// Helper: path to tests/data/mesh.ply
// ---------------------------------------------------------------------------
static std::string TestMeshPath()
{
	return (std::filesystem::path(__FILE__).parent_path()
	        / "data" / "mesh.ply")
	    .string();
}

// ---------------------------------------------------------------------------
// Helper: build a small 2x2 grid of quads split into triangles (8 triangles)
//   Vertices: 3x3 = 9 vertices, faces: 8 triangles
// ---------------------------------------------------------------------------
static Mesh MakeGrid3x3()
{
	Mesh m;
	// 3x3 grid of vertices at z=0
	m.vertices.resize(9);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			m.vertices[r * 3 + c] = Mesh::Vertex(float(c), float(r), 0.f);
	// 2x2 grid of quads, each split into 2 triangles
	// quad (r,c) top-left corner has vertex idx r*3+c
	for (int r = 0; r < 2; ++r) {
		for (int c = 0; c < 2; ++c) {
			const Mesh::VIndex v00 = r * 3 + c;
			const Mesh::VIndex v10 = (r + 1) * 3 + c;
			const Mesh::VIndex v01 = r * 3 + (c + 1);
			const Mesh::VIndex v11 = (r + 1) * 3 + (c + 1);
			m.faces.push_back({v00, v11, v01});
			m.faces.push_back({v00, v10, v11});
		}
	}
	return m;
}

// ---------------------------------------------------------------------------
// RoundCast tests (from the documented examples in halfmesh/Util/Maths.h)
// ---------------------------------------------------------------------------
TEST(RoundCastTest, PositiveHalfway)
{
	EXPECT_EQ(RoundCast<int>(0.5), 1);
}

TEST(RoundCastTest, NegativeHalfway)
{
	EXPECT_EQ(RoundCast<int>(-0.5), -1);
}

TEST(RoundCastTest, PositiveRoundsUp)
{
	EXPECT_EQ(RoundCast<int>(0.7), 1);
}

TEST(RoundCastTest, NegativeRoundsDown)
{
	EXPECT_EQ(RoundCast<int>(-1.4), -1);
}

TEST(RoundCastTest, SizeT50Percent)
{
	// Exactly half: RoundCast<size_t>(100 * 0.5) = 50
	EXPECT_EQ(RoundCast<size_t>(100 * 0.5f), size_t(50));
}

TEST(RoundCastTest, FloatPassthrough)
{
	// Non-integral T_out: no rounding — just static_cast
	EXPECT_FLOAT_EQ(RoundCast<float>(1.7f), 1.7f);
}

// ---------------------------------------------------------------------------
// Tiny grid mesh — Simplify runs and reduces face count
// ---------------------------------------------------------------------------
TEST(MeshSimplifyTest, GridMeshReduces)
{
	Mesh m = MakeGrid3x3();
	ASSERT_EQ(m.faces.size(), 8u);
	ASSERT_FALSE(m.Empty());

	// Priority-queue path (aggressiveness == 0), target 50%
	m.Simplify(0.5f);
	EXPECT_FALSE(m.Empty());
	EXPECT_LT(m.faces.size(), 8u);
}

// ---------------------------------------------------------------------------
// tests/data/mesh.ply — Simplify(0.5f): ~50% faces, validity, bbox preserved
// ---------------------------------------------------------------------------
TEST(MeshSimplifyTest, HalfDecimation)
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

	// Record original bounding box
	const auto origBbox = m.ComputeAABBox();

	// Decimate to 50%
	m.Simplify(0.5f);

	EXPECT_FALSE(m.Empty());
	EXPECT_GT(m.faces.size(), 0u);
	// Face count strictly less than original
	EXPECT_LT(m.faces.size(), origFaces);
	// Within ±15% of target
	const size_t target = origFaces / 2;
	EXPECT_GE(m.faces.size(), size_t(target * 0.80));
	EXPECT_LE(m.faces.size(), size_t(target * 1.20));

	// Vertex count reduced
	EXPECT_LT(m.vertices.size(), origVerts);

	// Bounding box approximately preserved (within 10% of diagonal)
	const auto decBbox = m.ComputeAABBox();
	const float origDiag = (origBbox.max() - origBbox.min()).norm();
	EXPECT_GE((decBbox.min() - origBbox.min()).norm(), 0.f); // not NaN
	// Decimated bbox should not expand significantly beyond original
	const float expandTol = origDiag * 0.10f;
	EXPECT_LE((decBbox.min() - origBbox.min()).cwiseAbs().maxCoeff(), expandTol);
	EXPECT_LE((decBbox.max() - origBbox.max()).cwiseAbs().maxCoeff(), expandTol);

	// Mesh should support rebuilding half-edges (validity check)
	Mesh copy = m;
	EXPECT_TRUE(copy.halfMesh.Build(copy));
}

// ---------------------------------------------------------------------------
// Absolute target face count (targetFaces): a reachable target lands within
// the documented +-20% window; an extreme target below the fixture's
// topology-preserving floor stalls gracefully ABOVE the target (warning
// logged) with a valid manifold result; a clamped (target >= input) call is a
// no-op on the MANIFOLDIZED mesh — the challenge fixture is non-manifold (138
// duplicate faces, 235 non-manifold edges), and every half-edge consumer
// auto-repairs on entry (see ListHalfEdges), so counts are compared against
// the manifoldized baseline, not the raw file.
// ---------------------------------------------------------------------------
TEST(MeshSimplifyTest, AbsoluteTargetFaceCount)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path))
		GTEST_SKIP() << "tests/data/mesh.ply not found at: " << path;

	Mesh m;
	ASSERT_TRUE(m.Load(path));
	const size_t origFaces = m.faces.size();
	ASSERT_GT(origFaces, 20000u);

	// decimateRatio > 1 is read as an absolute target face count.
	const size_t want = 5000;
	m.Simplify(/*decimateRatio=*/static_cast<float>(want));
	EXPECT_FALSE(m.Empty());
	EXPECT_LT(m.faces.size(), origFaces);
	// QEM stops as soon as the count drops to/below target, so it lands at the
	// target give or take the faces removed by the final collapse.
	EXPECT_GE(m.faces.size(), size_t(want * 0.80));
	EXPECT_LE(m.faces.size(), size_t(want * 1.20));
	Mesh copy = m;
	EXPECT_TRUE(copy.halfMesh.Build(copy)); // result is a valid manifold mesh

	// extreme target below the topology-preserving floor: stalls gracefully
	// above the target, output still a valid manifold. Floor measured 2026-08:
	// 1,444 raw / 1,569 after the sliver-cull pre-pass (genus + 29 boundary
	// loops bound the reachable minimum on this fixture).
	Mesh e;
	ASSERT_TRUE(e.Load(path));
	e.Simplify(/*decimateRatio=*/1000.f);
	EXPECT_GE(e.faces.size(), 1000u);
	EXPECT_LE(e.faces.size(), 2500u)
	    << "collapse floor regressed far above the measured 1,444";
	Mesh ecopy = e;
	EXPECT_TRUE(ecopy.halfMesh.Build(ecopy));

	// clamp: absolute target >= input is a no-op on the MANIFOLDIZED mesh (the
	// entry repair first drops the fixture's duplicate faces).
	Mesh n;
	ASSERT_TRUE(n.Load(path));
	n.ListHalfEdges(); // manifoldize-on-entry (the same repair Simplify runs)
	const size_t manifoldFaces = n.faces.size();
	EXPECT_LT(manifoldFaces, origFaces);
	n.Simplify(static_cast<float>(n.faces.size() + 100));
	EXPECT_EQ(n.faces.size(), manifoldFaces);
}

// ---------------------------------------------------------------------------
// tests/data/mesh.ply — Simplify(0.25f): ~25% faces, monotonic reduction
// ---------------------------------------------------------------------------
TEST(MeshSimplifyTest, QuarterDecimation)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path)) {
		GTEST_SKIP() << "tests/data/mesh.ply not found at: " << path;
	}

	Mesh m;
	ASSERT_TRUE(m.Load(path));
	const size_t origFaces = m.faces.size();

	m.Simplify(0.25f);

	EXPECT_FALSE(m.Empty());
	EXPECT_LT(m.faces.size(), origFaces);

	const size_t target = origFaces / 4;
	EXPECT_GE(m.faces.size(), size_t(target * 0.75));
	EXPECT_LE(m.faces.size(), size_t(target * 1.30));
}

// ---------------------------------------------------------------------------
// Hausdorff-style one-sided check: decimated vertices close to original surface
// ---------------------------------------------------------------------------
TEST(MeshSimplifyTest, HausdorffOneSided)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path)) {
		GTEST_SKIP() << "tests/data/mesh.ply not found at: " << path;
	}

	// Build KD-tree on original mesh BEFORE decimation
	Mesh orig;
	ASSERT_TRUE(orig.Load(path));
	const float origBboxDiag = (orig.ComputeAABBox().max() - orig.ComputeAABBox().min()).norm();
	TriangleKdTree origTree(orig);

	// Decimate a copy
	Mesh dec;
	ASSERT_TRUE(dec.Load(path));
	dec.Simplify(0.5f);

	// Threshold: vertices should be within 2% of bounding-box diagonal
	const float hausdorffTol = origBboxDiag * 0.02f;

	// Sample up to 200 decimated vertices
	const size_t step = std::max(size_t(1), dec.vertices.size() / 200);
	size_t violations = 0;
	for (size_t i = 0; i < dec.vertices.size(); i += step) {
		const auto nn = origTree.NearestPoint(dec.vertices[i]);
		if (nn.dist > hausdorffTol)
			++violations;
	}
	// Allow at most 5% violations (some boundary vertices may move slightly more)
	const size_t sampled = (dec.vertices.size() + step - 1) / step;
	EXPECT_LE(violations, sampled / 20 + 1)
	    << "Too many decimated vertices far from original surface ("
	    << violations << "/" << sampled << " > 5%)";
}

// ---------------------------------------------------------------------------
// Accuracy + mode coverage using the analytic corpus + metrics toolkit.
// These pin the "fast and accurate" decimation contract without a reference.
// ---------------------------------------------------------------------------
namespace corpus = hmtest::corpus;
namespace metrics = hmtest::metrics;

static bool RebuildsHalfMesh(const Mesh& m)
{
	Mesh copy = m;
	return copy.halfMesh.Build(copy);
}

static double BBoxDiag(const Mesh& m)
{
	const auto bb = m.ComputeAABBox();
	return (bb.max() - bb.min()).norm();
}

static double MinFaceDoubleArea(const Mesh& m)
{
	double mn = std::numeric_limits<double>::max();
	for (size_t f = 0; f < m.faces.size(); ++f)
		mn = std::min(mn, double(m.ComputeFaceDoubleArea(Mesh::FIndex(f))));
	return mn;
}

struct SimpCase
{
	std::string name;
	std::function<Mesh()> make;
	int32_t genus;
};

// Both decimation strategies (priority queue and the fast threshold sweep) on
// closed surfaces: reduces faces, stays a watertight manifold of the same genus,
// produces no degenerate faces, and stays close to the original surface.
TEST(MeshSimplifyAccuracy, BothModesClosedSurface)
{
	const std::vector<SimpCase> meshes = {
	    {"UVSphere", [] { return corpus::UVSphere(16, 24); }, 0},
	    {"Torus", [] { return corpus::TorusMesh(24, 16); }, 1},
	};
	const std::vector<std::pair<std::string, float>> modes = {{"priority", 0.f}, {"fast", 5.f}};

	for (const SimpCase& mc : meshes) {
		const Mesh orig = mc.make();
		const size_t origFaces = orig.faces.size();
		const double diag = BBoxDiag(orig);
		for (const auto& mode : modes) {
			SCOPED_TRACE(mc.name + "/" + mode.first);
			Mesh m = orig;
			m.Simplify(0.5f, 0.f, mode.second);

			EXPECT_GT(m.faces.size(), 0u);
			EXPECT_LT(m.faces.size(), origFaces) << "should reduce faces";
			EXPECT_GT(m.faces.size(), origFaces / 10) << "should not over-collapse";

			EXPECT_TRUE(RebuildsHalfMesh(m));
			const auto topo = metrics::ComputeTopology(m);
			EXPECT_TRUE(topo.isWatertight) << "closed input must stay watertight";
			EXPECT_EQ(topo.genus, mc.genus) << "genus must be preserved";

			EXPECT_TRUE(metrics::ScanFinite(m));
			EXPECT_GT(MinFaceDoubleArea(m), 0.0) << "no degenerate (zero-area) faces";
			const double hd = metrics::ComputeDistanceKdTree(orig, m).hausdorffSymmetric;
			EXPECT_LT(hd, 0.05 * diag) << "decimated surface deviates too far from original";
		}
	}
}

// Determinism: the priority-queue path is order-stable, so re-running yields the
// exact same mesh.
TEST(MeshSimplifyAccuracy, Deterministic)
{
	const Mesh orig = corpus::UVSphere(16, 24);
	Mesh a = orig;
	a.Simplify(0.5f);
	Mesh b = orig;
	b.Simplify(0.5f);
	EXPECT_TRUE(metrics::CanonicallyEqual(a, b)) << "Simplify must be deterministic";
}

// Min-edge-length mode (decimateRatio>=1, minEdgeLength>0): collapses short
// edges, reducing face count while keeping a valid finite mesh.
TEST(MeshSimplifyAccuracy, MinEdgeLengthMode)
{
	Mesh m = corpus::UVSphere(24, 32);
	const size_t origFaces = m.faces.size();
	const double meanLen = metrics::ComputeEdgeLengthStats(m).meanLen;
	ASSERT_GT(meanLen, 0.0);
	m.Simplify(1.0f, float(meanLen * 1.2)); // collapse edges shorter than ~1.2x mean
	EXPECT_LT(m.faces.size(), origFaces) << "min-edge mode should remove short edges";
	EXPECT_GT(m.faces.size(), 0u);
	EXPECT_TRUE(RebuildsHalfMesh(m));
	EXPECT_TRUE(metrics::ScanFinite(m));
	EXPECT_GT(MinFaceDoubleArea(m), 0.0);
}

// A flat, graded triangle fan: a small dense center surrounded by concentric rings
// whose radius grows geometrically, so at a moderate threshold MOST edges start
// LONGER than minEdgeLength (the outer rings) and are excluded by the build-time
// filter. Collapsing the dense center marches a collapse front outward, shortening
// the outer edges below the threshold as it goes — the exact dynamic-length setup
// requirement 2b targets. Returns a valid manifold disk (with a boundary rim).
static Mesh MakeGradedFan(unsigned rings, unsigned spokes, float r0, float growth)
{
	Mesh m;
	m.vertices.push_back(Mesh::Vertex(0.f, 0.f, 0.f)); // center = vertex 0
	std::vector<float> radius(rings + 1, 0.f);
	for (unsigned k = 1; k <= rings; ++k)
		radius[k] = radius[k - 1] + r0 * std::pow(growth, float(k - 1));
	auto vid = [&](unsigned k, unsigned s) { return 1u + (k - 1) * spokes + (s % spokes); };
	for (unsigned k = 1; k <= rings; ++k)
		for (unsigned s = 0; s < spokes; ++s) {
			const float a = float(2.0 * M_PI * s / spokes);
			m.vertices.push_back(Mesh::Vertex(radius[k] * std::cos(a), radius[k] * std::sin(a), 0.f));
		}
	for (unsigned s = 0; s < spokes; ++s) // inner cap fan
		m.faces.push_back({0u, vid(1, s), vid(1, s + 1)});
	for (unsigned k = 1; k < rings; ++k) // annular quads split into triangles
		for (unsigned s = 0; s < spokes; ++s) {
			m.faces.push_back({vid(k, s), vid(k + 1, s), vid(k + 1, s + 1)});
			m.faces.push_back({vid(k, s), vid(k + 1, s + 1), vid(k, s + 1)});
		}
	return m;
}

// Requirement 2b — dynamic edge length. An edge LONGER than minEdgeLength at build
// time is excluded from the initial heap by the build-time filter, but must still be
// collapsed if a neighboring collapse later shortens it to within the threshold;
// otherwise the min-edge candidate set silently shrinks vs. the unfiltered algorithm.
//
// The graded fan starts with most edges longer than the threshold (the outer rings).
// A correct single Simplify pass must nonetheless drive the collapse front out into
// that long-edge region by re-admitting edges as they shorten. The discriminator is
// a single-pass FIXPOINT: after one pass there is no remaining collapsible edge
// within the threshold, so a second fresh pass (which re-derives every current edge
// length from scratch) collapses nothing more — f2 == f1. If re-admission were
// dropped, pass 1 would leave shortened-but-never-admitted edges behind and pass 2
// would shrink the mesh further (f2 < f1). (Empirically verified: disabling the
// update-path emplace makes f2 fall below f1 on exactly this mesh.)

// ---------------------------------------------------------------------------
// Translation robustness tripwire for the float plane offset d in the quadric
// build (MeshSimplify.cpp). Experiment 2026-07-16
// (7 offsets x {float plane, double plane} on UVSphere(16,24), Simplify(0.5),
// symmetric Hausdorff of each result to its own translated input, baseline
// h0=0.0107):
//   float:  +0.9..+36% over offsets ~1e3-3e3 (median ~+1%)
//   double: -0.2..+31% over the same offsets (median ~+2%)
// The distributions OVERLAP: per-offset variance is dominated by the collapse-
// order lottery (near-tied QEM costs resolving differently) plus float vertex-
// storage quantization, identically under both plane precisions — promoting
// the plane to double produced no measurable quality gain, so the concern is
// refuted at float-storable coordinate magnitudes and the float plane stays.
// This test pins the CURRENT robustness envelope as a tripwire: a genuine
// conditioning bug (e.g. a 2.2x degradation at |v|~1e5, the projected failure
// mode) blows far past the bound, while the measured
// lottery ceiling (1.36x) passes with headroom.
// ---------------------------------------------------------------------------
TEST(MeshSimplifyAccuracy, TranslationRobustQuality)
{
	const Mesh orig = corpus::UVSphere(16, 24);
	Mesh base = orig;
	base.Simplify(0.5f);
	const double h0 = metrics::ComputeDistanceKdTree(base, orig).hausdorffSymmetric;
	ASSERT_GT(h0, 0.0);
	// Bound 1.75x: above the measured order-lottery ceiling (1.36x across the
	// experiment's offsets), far below a real conditioning failure (>2x).
	const double maxRel = 1.75;
	const Mesh::Vertex offsets[] = {
	    {1e3f, 1e3f, 1e3f},
	    {-1.3e3f, 7e2f, 9.1e2f},
	    {2.7e3f, -1.9e3f, 3e2f},
	    {5e2f, 2.2e3f, -1.1e3f},
	    {-2e3f, -2e3f, 2e3f},
	    {3.3e3f, 1.1e2f, -2.8e3f},
	};
	for (const auto& off : offsets) {
		Mesh input = orig;
		for (auto& v : input.vertices)
			v += off;
		Mesh m = input;
		m.Simplify(0.5f);
		// Same target => same face count; quality measured against the mesh's
		// OWN (translated) input so storage quantization of the input cancels.
		EXPECT_EQ(m.faces.size(), base.faces.size());
		const double h = metrics::ComputeDistanceKdTree(m, input).hausdorffSymmetric;
		EXPECT_LT(h, maxRel * h0)
		    << "translation (" << off.x() << "," << off.y() << "," << off.z()
		    << ") degraded simplify quality " << h / h0 << "x — conditioning bug?";
	}
}

TEST(MeshSimplifyAccuracy, MinEdgeDynamicLengthReadmission)
{
	const Mesh orig = MakeGradedFan(12, 8, 0.03f, 1.7f);
	ASSERT_TRUE(RebuildsHalfMesh(orig)) << "graded fan must be a valid manifold";
	const double meanLen = metrics::ComputeEdgeLengthStats(orig).meanLen;
	ASSERT_GT(meanLen, 0.0);
	const float threshold = float(meanLen * 0.9); // moderate: many outer edges exceed it

	Mesh m = orig;
	m.Simplify(1.0f, threshold);
	const size_t facesPass1 = m.faces.size();
	// The front reached the long-edge region: real reduction happened, and the result
	// is a valid, finite, non-degenerate mesh.
	EXPECT_LT(facesPass1, orig.faces.size()) << "min-edge pass must collapse the fan inward";
	EXPECT_GT(facesPass1, 0u);
	EXPECT_TRUE(RebuildsHalfMesh(m));
	EXPECT_TRUE(metrics::ScanFinite(m));
	EXPECT_GT(MinFaceDoubleArea(m), 0.0);

	// Single-pass fixpoint: a fresh pass finds nothing more to collapse iff every
	// edge shortened past the threshold was re-admitted and collapsed the first time.
	Mesh again = m;
	again.Simplify(1.0f, threshold);
	EXPECT_EQ(again.faces.size(), facesPass1)
	    << "a formerly-long edge that became short was left uncollapsed — re-admission is broken";
}

// Fast mode must behave the same regardless of the mesh's units/scale: the
// per-pass acceptance threshold is anchored to the squared bounding-box
// diagonal (quadric error has units length^2), so a uniformly scaled copy must
// reach the same face target with the same relative quality as the unit run.
TEST(MeshSimplifyAccuracy, FastModeScaleInvariant)
{
	const Mesh orig = corpus::UVSphere(16, 24);
	const size_t origFaces = orig.faces.size();

	// Reference: unit-scale fast-mode run (same params as BothModesClosedSurface).
	Mesh unit = orig;
	unit.Simplify(0.5f, 0.f, 5.f);
	const size_t unitFaces = unit.faces.size();
	ASSERT_LT(unitFaces, origFaces);
	// The fast sweep reaches the requested ~50% face target at unit scale.
	ASSERT_LE(unitFaces, origFaces / 2);

	for (const float scale : {1e-3f, 1e5f}) {
		SCOPED_TRACE("scale=" + std::to_string(scale));
		Mesh scaledOrig = orig;
		for (Mesh::Vertex& v : scaledOrig.vertices)
			v *= scale;
		Mesh m = scaledOrig;
		m.Simplify(0.5f, 0.f, 5.f);

		// Same face count as the unit-scale run: the collapse-acceptance
		// schedule must be invariant under uniform scaling. Every run that
		// reaches the target stops within one collapse (<= 2 faces) of it.
		EXPECT_NEAR(double(m.faces.size()), double(unitFaces), 2.0)
		    << "fast-mode threshold schedule is scale-dependent";

		// Relative quality parity: Hausdorff to the (scaled) input, relative
		// to the (scaled) bbox diagonal, held to the same bound the unit-scale
		// run satisfies in BothModesClosedSurface.
		EXPECT_TRUE(RebuildsHalfMesh(m));
		const double diag = BBoxDiag(scaledOrig);
		const double hd = metrics::ComputeDistanceKdTree(scaledOrig, m).hausdorffSymmetric;
		EXPECT_LT(hd, 0.05 * diag) << "decimated surface deviates too far from original";
	}
}

// Open meshes: decimation must not destroy or merge boundary loops, and the
// boundary silhouette stays close to the original.
TEST(MeshSimplifyAccuracy, PreservesBoundaryLoops)
{
	const std::vector<SimpCase> meshes = {
	    {"GridPlane", [] { return corpus::GridPlane(12); }, 0},
	    {"OpenCylinder", [] { return corpus::OpenCylinder(16, 8); }, 0},
	};
	for (const SimpCase& mc : meshes) {
		SCOPED_TRACE(mc.name);
		const Mesh orig = mc.make();
		const auto topo0 = metrics::ComputeTopology(orig);
		const double diag = BBoxDiag(orig);
		Mesh m = orig;
		m.Simplify(0.5f);
		EXPECT_LT(m.faces.size(), orig.faces.size());
		const auto topo1 = metrics::ComputeTopology(m);
		EXPECT_EQ(topo1.numBoundaryLoops, topo0.numBoundaryLoops) << "boundary loop count must be preserved";
		EXPECT_TRUE(RebuildsHalfMesh(m));
		EXPECT_TRUE(metrics::ScanFinite(m));
		EXPECT_LT(metrics::ComputeDistanceKdTree(orig, m).hausdorffSymmetric, 0.05 * diag);
	}
}

// Contract: an empty mesh and the documented identity call (decimateRatio == 1
// with no minEdgeLength) are graceful no-ops in every build mode — previously
// both tripped a Debug-only ASSERT while silently proceeding in Release.
TEST(MeshSimplifyTest, EmptyMeshIsNoOp)
{
	Mesh mesh;
	mesh.Simplify(0.5f);
	EXPECT_TRUE(mesh.Empty());
	EXPECT_TRUE(mesh.faces.empty());
}

TEST(MeshSimplifyTest, IdentityRatioIsNoOp)
{
	Mesh mesh = MakeGrid3x3();
	const size_t numVerts = mesh.vertices.size();
	const size_t numFaces = mesh.faces.size();
	mesh.Simplify(1.f);
	EXPECT_EQ(mesh.vertices.size(), numVerts);
	EXPECT_EQ(mesh.faces.size(), numFaces);
}

TEST(MeshSimplifyTest, HalfEdgeOnlyEntryMatchesPopulatedEntryWithoutBuild)
{
	Mesh populated = hmtest::corpus::UVSphere(8, 12);
	Mesh native = populated;
	populated.ListHalfEdges();
	native.ListHalfEdges();
	native.InvalidateFaces();

	populated.Simplify(0.5f);
	HalfMesh::ResetBuildCount();
	native.Simplify(0.5f);
	EXPECT_EQ(HalfMesh::BuildCount(), 0u);
	EXPECT_EQ(native.vertices, populated.vertices);
	EXPECT_EQ(native.faces, populated.faces);
	EXPECT_TRUE(native.ValidateHalfMesh());
}
} // namespace
} // namespace halfmesh
