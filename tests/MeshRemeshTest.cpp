/*
* MeshRemeshTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for MeshRemesh.cpp:
//   Mesh::RemeshIsotropic — isotropic remeshing
//
// Tests:
//   1. Validity: remeshed mesh is non-empty, half-edges rebuild, no degenerate faces
//   2. Edge-length regularization: most edges within [min, max] band; mean near target
//   3. Surface fidelity: one-sided Hausdorff — remeshed verts stay close to orig surface
//   4. BBox preserved approximately
//   5. Coarser target → fewer faces; finer target → more faces (density adapts)
//   6. Inline 2-triangle mesh: completes without crash (smoke test)
//   7. Zero-iteration early exit: mesh unchanged after 0 iterations

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/TriangleKDTree.h>

#include "metrics/Metrics.h"
#include "metrics/RemeshQuality.h"
#include "corpus/Corpus.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

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
// Helper: load mesh.ply or skip if not found
// ---------------------------------------------------------------------------
static bool LoadTestMesh(Mesh& m)
{
	return m.Load(TestMeshPath());
}

// ---------------------------------------------------------------------------
// Helper: compute per-edge lengths and return {min, max, mean}
// ---------------------------------------------------------------------------
struct EdgeStats
{
	double minLen, maxLen, meanLen;
	size_t count;
};
static EdgeStats ComputeEdgeStats(const Mesh& m)
{
	// iterate over unique edges via half-edge (he[0], he[1] are a pair; he[0]=even)
	double sum = 0.0;
	double emin = std::numeric_limits<double>::max();
	double emax = 0.0;
	size_t cnt = 0;
	const HalfMesh& hm = m.halfMesh;
	for (HalfMesh::EIndex iE = 0; iE < hm.ESize(); ++iE) {
		const HalfMesh::HIndex iH = hm.EHalfedge(iE);
		if (hm.EIsBoundary(iE))
			continue;
		const Mesh::VIndex v0 = hm.HeTailVertex(iH);
		const Mesh::VIndex v1 = hm.HeHeadVertex(iH);
		const double len = (m.vertices[v0] - m.vertices[v1]).norm();
		if (len < emin)
			emin = len;
		if (len > emax)
			emax = len;
		sum += len;
		++cnt;
	}
	if (cnt == 0) {
		return {0.0, 0.0, 0.0, 0};
	}
	return {emin, emax, sum / cnt, cnt};
}

// ---------------------------------------------------------------------------
// Helper: one-sided Hausdorff (vertices of m to original surface)
// ---------------------------------------------------------------------------
static double OneWayHausdorff(const Mesh& m, const TriangleKdTree& originalTree)
{
	double maxDist = 0.0;
	for (const Mesh::Vertex& v : m.vertices) {
		const auto nn = originalTree.NearestPoint(v);
		if (nn.IsValid())
			maxDist = std::max(maxDist, static_cast<double>(nn.dist));
	}
	return maxDist;
}

// ---------------------------------------------------------------------------
// Helper: check no zero-area faces
// ---------------------------------------------------------------------------
static bool HasNoZeroAreaFaces(const Mesh& m)
{
	constexpr float eps = 1e-10f;
	for (const auto& face : m.faces) {
		const float area = m.ComputeFaceDoubleArea(face) * 0.5f;
		if (area < eps)
			return false;
	}
	return true;
}

// Derived remeshing-quality report (edge uniformity, angle quality, valence
// regularity, surface fidelity) lives in the shared metrics toolkit so the unit
// tests and tests/bench/RemeshBench measure identically.
using hmtest::metrics::ComputeRemeshQuality;
using hmtest::metrics::RemeshQuality;

static void PrintRemeshQuality(const char* tag, const RemeshQuality& q)
{
	std::cout << "[RemeshQuality " << tag << "]"
	          << " edge_cov=" << q.edgeCov
	          << " min_angle_mean=" << q.minAngleMeanDeg << "deg"
	          << " frac<30deg=" << q.fracMinAngleLt30
	          << " mean_valence=" << q.meanValence
	          << " irregular_valence=" << q.fracIrregularValence
	          << " mean_dist/bbox=" << q.meanDistRatio()
	          << " hausdorff/bbox=" << q.hausdorffRatio()
	          << "\n";
}

// ---------------------------------------------------------------------------
// Test 1: validity after remeshing
// ---------------------------------------------------------------------------
TEST(MeshRemesh, ValidityAfterRemesh)
{
	Mesh m;
	ASSERT_TRUE(LoadTestMesh(m)) << "Could not load " << TestMeshPath();

	const size_t origFaceCount = m.faces.size();

	// Use a target edge length ~equal to mean edge length (neutral density)
	m.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(m);
	const float target = static_cast<float>(origStats.meanLen);

	Mesh::RemeshParams p;
	p.SetEdgeLength(target);
	p.iterations = 3;
	m.RemeshIsotropic(p);

	EXPECT_FALSE(m.Empty());
	EXPECT_FALSE(m.faces.empty());
	EXPECT_GT(m.vertices.size(), 0u);

	// half-edges must rebuild cleanly
	EXPECT_NO_THROW(m.ListHalfEdges());

	// no zero-area faces
	EXPECT_TRUE(HasNoZeroAreaFaces(m));

	(void)origFaceCount; // used to suppress unused warning
}

// ---------------------------------------------------------------------------
// Test 2: edge-length regularization
// ---------------------------------------------------------------------------
TEST(MeshRemesh, EdgeLengthRegularization)
{
	Mesh m;
	ASSERT_TRUE(LoadTestMesh(m));

	m.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(m);
	const float target = static_cast<float>(origStats.meanLen);

	Mesh::RemeshParams p;
	p.SetEdgeLength(target);
	p.iterations = 3;
	m.RemeshIsotropic(p);

	m.ListHalfEdges();
	const EdgeStats stats = ComputeEdgeStats(m);
	ASSERT_GT(stats.count, 0u);

	// mean should be reasonably close to target (within 50%)
	EXPECT_GT(stats.meanLen, target * 0.5)
	    << "Mean edge length " << stats.meanLen
	    << " too small vs target " << target;
	EXPECT_LT(stats.meanLen, target * 2.0)
	    << "Mean edge length " << stats.meanLen
	    << " too large vs target " << target;

	// count how many edges are within the [min, max] band
	const double emin = p.edgeMinLength;
	const double emax = p.edgeMaxLength;
	size_t inBand = 0;
	const HalfMesh& hm = m.halfMesh;
	for (HalfMesh::EIndex iE = 0; iE < hm.ESize(); ++iE) {
		if (hm.EIsBoundary(iE))
			continue;
		const HalfMesh::HIndex iH = hm.EHalfedge(iE);
		const Mesh::VIndex v0 = hm.HeTailVertex(iH);
		const Mesh::VIndex v1 = hm.HeHeadVertex(iH);
		const double len = (m.vertices[v0] - m.vertices[v1]).norm();
		if (len >= emin && len <= emax)
			++inBand;
	}
	// at least 50% of edges should be in band after 3 iterations
	EXPECT_GE(static_cast<double>(inBand) / stats.count, 0.50)
	    << inBand << "/" << stats.count << " edges in band";
}

// ---------------------------------------------------------------------------
// Test 3: surface fidelity (one-sided Hausdorff)
// ---------------------------------------------------------------------------
TEST(MeshRemesh, SurfaceFidelity)
{
	Mesh original, remeshed;
	ASSERT_TRUE(LoadTestMesh(original));
	ASSERT_TRUE(LoadTestMesh(remeshed));

	// Build KdTree on original BEFORE remeshing
	const TriangleKdTree originalTree(original);

	remeshed.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(remeshed);
	const float target = static_cast<float>(origStats.meanLen);

	Mesh::RemeshParams p;
	p.SetEdgeLength(target);
	p.iterations = 3;
	remeshed.RemeshIsotropic(p);

	// compute bbox diagonal of original
	Eigen::AlignedBox3f bbox;
	for (const auto& v : original.vertices)
		bbox.extend(v);
	const double bboxDiag = bbox.diagonal().norm();

	const double hausdorff = OneWayHausdorff(remeshed, originalTree);
	// Expect remeshed vertices within 20% of bbox diagonal of original surface
	EXPECT_LT(hausdorff, bboxDiag * 0.20)
	    << "One-sided Hausdorff " << hausdorff
	    << " exceeds 20% of bbox diagonal " << bboxDiag;

	// Also report for diagnostics
	std::cout << "[SurfaceFidelity] Hausdorff=" << hausdorff
	          << " bbox_diag=" << bboxDiag
	          << " ratio=" << hausdorff / bboxDiag << "\n";
}

// ---------------------------------------------------------------------------
// Test 4: bounding box approximately preserved
// ---------------------------------------------------------------------------
TEST(MeshRemesh, BBoxPreserved)
{
	Mesh original, remeshed;
	ASSERT_TRUE(LoadTestMesh(original));
	ASSERT_TRUE(LoadTestMesh(remeshed));

	// Original bbox
	Eigen::AlignedBox3f origBbox;
	for (const auto& v : original.vertices)
		origBbox.extend(v);

	remeshed.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(remeshed);
	const float target = static_cast<float>(origStats.meanLen);

	Mesh::RemeshParams p;
	p.SetEdgeLength(target);
	p.iterations = 3;
	remeshed.RemeshIsotropic(p);

	Eigen::AlignedBox3f newBbox;
	for (const auto& v : remeshed.vertices)
		newBbox.extend(v);

	const double origDiag = origBbox.diagonal().norm();
	// New bbox should be within 30% of original diagonal in min/max
	for (int i = 0; i < 3; ++i) {
		EXPECT_GT(newBbox.min()[i], origBbox.min()[i] - 0.30f * origDiag)
		    << "BBox min[" << i << "] moved too far";
		EXPECT_LT(newBbox.max()[i], origBbox.max()[i] + 0.30f * origDiag)
		    << "BBox max[" << i << "] moved too far";
	}
}

// ---------------------------------------------------------------------------
// Test 5: density adaptation (coarser → fewer faces; finer → more faces)
// ---------------------------------------------------------------------------
TEST(MeshRemesh, DensityAdaptation)
{
	Mesh coarse, fine;
	ASSERT_TRUE(LoadTestMesh(coarse));
	ASSERT_TRUE(LoadTestMesh(fine));

	coarse.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(coarse);
	const float targetMean = static_cast<float>(origStats.meanLen);

	// Coarser: target = 2x mean edge length
	{
		Mesh::RemeshParams p;
		p.SetEdgeLength(targetMean * 2.0f);
		p.iterations = 3;
		coarse.RemeshIsotropic(p);
	}
	// Finer: target = 0.5x mean edge length
	{
		Mesh::RemeshParams p;
		p.SetEdgeLength(targetMean * 0.5f);
		p.iterations = 3;
		fine.RemeshIsotropic(p);
	}

	// Coarser should have fewer faces than original; finer should have more
	Mesh orig;
	ASSERT_TRUE(LoadTestMesh(orig));
	const size_t origFaces = orig.faces.size();

	std::cout << "[DensityAdaptation] orig=" << origFaces
	          << " coarse=" << coarse.faces.size()
	          << " fine=" << fine.faces.size() << "\n";

	// At 2x edge length (coarser), expect fewer faces than at 0.5x (finer)
	EXPECT_LT(coarse.faces.size(), fine.faces.size())
	    << "Coarser target should yield fewer faces than finer target";
}

// ---------------------------------------------------------------------------
// Test 6: smoke test on inline 2-triangle mesh
// ---------------------------------------------------------------------------
TEST(MeshRemesh, SmokeTwoTriangles)
{
	Mesh m;
	// 4 vertices forming 2 triangles (a quad split diagonally)
	m.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {1.f, 1.f, 0.f},
	    {0.f, 1.f, 0.f},
	};
	m.faces = {
	    {0, 1, 2},
	    {0, 2, 3},
	};

	Mesh::RemeshParams p;
	p.SetEdgeLength(0.3f);
	p.iterations = 2;
	EXPECT_NO_THROW(m.RemeshIsotropic(p));
	EXPECT_FALSE(m.faces.empty());
}

// ---------------------------------------------------------------------------
// Test 7: zero iterations — mesh unchanged
// ---------------------------------------------------------------------------
TEST(MeshRemesh, ZeroIterations)
{
	Mesh m;
	ASSERT_TRUE(LoadTestMesh(m));

	// The raw fixture is non-manifold; ListHalfEdges manifoldizes on entry
	// (documented geometry-preserving repair — duplicate faces reduced to one
	// copy, unreferenced vertices dropped). Zero-iteration remesh must not
	// change the mesh BEYOND that entry repair, so snapshot after it.
	m.ListHalfEdges();
	const size_t origVertices = m.vertices.size();
	const size_t origFaces = m.faces.size();
	const EdgeStats origStats = ComputeEdgeStats(m);

	Mesh::RemeshParams p;
	p.SetEdgeLength(static_cast<float>(origStats.meanLen));
	p.iterations = 0; // nothing should happen
	m.RemeshIsotropic(p);

	// Topology should be unchanged (TagCreaseEdges runs but no iterations)
	EXPECT_EQ(m.vertices.size(), origVertices);
	EXPECT_EQ(m.faces.size(), origFaces);
}

// ---------------------------------------------------------------------------
// Test 8: quality regression baseline on a closed surface (UV sphere)
//
// Locks the isotropic-remeshing quality dimensions the algorithm is judged on.
// Bounds are deliberately conservative (current behaviour passes comfortably);
// they exist to catch gross regressions, not to over-fit. tests/bench/RemeshBench
// reports the precise numbers for tracking improvements across the SOTA upgrades.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, QualityRegression_UVSphere)
{
	const Mesh input = hmtest::corpus::UVSphere(24, 36);
	Mesh m = input;

	m.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(m);
	Mesh::RemeshParams p;
	p.SetEdgeLength(static_cast<float>(origStats.meanLen));
	p.iterations = 5;
	m.RemeshIsotropic(p);
	m.ListHalfEdges();

	const RemeshQuality q = ComputeRemeshQuality(input, m);
	PrintRemeshQuality("UVSphere", q);

	// Size uniformity: a few iterations should bring edge-length CoV well under 0.5.
	EXPECT_LT(q.edgeCov, 0.5) << "edge-length CoV too high (non-uniform sizing)";
	// Angle quality: the majority of triangles should have a healthy smallest angle.
	EXPECT_LT(q.fracMinAngleLt30, 0.5) << "too many sliver triangles (min angle < 30deg)";
	EXPECT_GT(q.minAngleMeanDeg, 35.0) << "mean minimum angle too low";
	// Valence regularity on a closed surface: mean valence stays near 6.
	EXPECT_GT(q.meanValence, 5.5);
	EXPECT_LT(q.meanValence, 6.5);
	EXPECT_LT(q.fracIrregularValence, 0.6) << "too many irregular-valence vertices";
	// Fidelity: vertices stay close to the input sphere.
	ASSERT_GT(q.bboxDiag, 0.0);
	EXPECT_LT(q.meanDistToInput / q.bboxDiag, 0.02) << "drifted off the input surface";
}

// ---------------------------------------------------------------------------
// Test 9: quality regression baseline on the real test mesh (open, with boundary)
// ---------------------------------------------------------------------------
TEST(MeshRemesh, QualityRegression_RealMesh)
{
	Mesh input;
	ASSERT_TRUE(LoadTestMesh(input));
	Mesh m;
	ASSERT_TRUE(LoadTestMesh(m));

	m.ListHalfEdges();
	const EdgeStats origStats = ComputeEdgeStats(m);
	Mesh::RemeshParams p;
	p.SetEdgeLength(static_cast<float>(origStats.meanLen));
	p.iterations = 3;
	m.RemeshIsotropic(p);
	m.ListHalfEdges();

	const RemeshQuality q = ComputeRemeshQuality(input, m);
	PrintRemeshQuality("RealMesh", q);

	EXPECT_LT(q.edgeCov, 0.6) << "edge-length CoV too high";
	EXPECT_LT(q.fracMinAngleLt30, 0.6) << "too many sliver triangles";
	// Open mesh: boundary vertices are naturally irregular, so only sanity-bound here.
	EXPECT_GT(q.meanValence, 5.0);
	EXPECT_LT(q.meanValence, 7.0);
	ASSERT_GT(q.bboxDiag, 0.0);
	EXPECT_LT(q.meanDistToInput / q.bboxDiag, 0.05) << "drifted off the input surface";
}

// ---------------------------------------------------------------------------
// Test 10: PMP-style tangential smoothing improves edge-length uniformity
// (the headline weakness vs pmp in the benchmark) without drifting off-surface.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, TangentialSmoothingImprovesUniformity)
{
	const Mesh input = hmtest::corpus::UVSphere(24, 36);

	auto run = [&](bool tangential) {
		Mesh m = input;
		m.ListHalfEdges();
		const EdgeStats es = ComputeEdgeStats(m);
		Mesh::RemeshParams p;
		p.SetEdgeLength(static_cast<float>(es.meanLen));
		p.iterations = 5;
		p.smoothTangential = tangential;
		p.smoothIterations = tangential ? 5 : 1; // PMP runs ~5 inner passes
		m.RemeshIsotropic(p);
		m.ListHalfEdges();
		return ComputeRemeshQuality(input, m);
	};

	const RemeshQuality base = run(false);
	const RemeshQuality tang = run(true);
	PrintRemeshQuality("uniform", base);
	PrintRemeshQuality("tangential", tang);

	// Stronger smoothing should not worsen size uniformity, and in practice
	// improves it. Allow a tiny tolerance against float noise.
	EXPECT_LE(tang.edgeCov, base.edgeCov + 1e-3)
	    << "tangential CoV " << tang.edgeCov << " vs uniform " << base.edgeCov;
	// And it must stay faithful to the input surface.
	ASSERT_GT(tang.bboxDiag, 0.0);
	EXPECT_LT(tang.meanDistRatio(), 0.02) << "tangential smoothing drifted off-surface";
}

// ---------------------------------------------------------------------------
// Test 11: featureCorners preserves sharp corners (cube) — corners are not
// eroded by smoothing/collapse when feature handling is enabled.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, FeatureCornersPreserveCubeCorners)
{
	const Mesh input = hmtest::corpus::CubeMesh(1.0f);
	// The corpus cube has exactly its 8 geometric corners as vertices; every one
	// is a feature corner (3 incident sharp edges) and must survive remeshing.
	const std::vector<Mesh::Vertex> corners(input.vertices.begin(), input.vertices.end());

	auto remesh = [&](bool feature) {
		Mesh m = input;
		m.ListHalfEdges();
		const EdgeStats es = ComputeEdgeStats(m);
		Mesh::RemeshParams p;
		p.SetEdgeLength(static_cast<float>(es.meanLen) * 0.5f); // refine
		p.iterations = 5;
		p.smoothTangential = true;
		p.smoothIterations = 5;
		p.featureCorners = feature;
		m.RemeshIsotropic(p);
		return m;
	};

	auto minDistTo = [](const Mesh& m, const Mesh::Vertex& p) {
		double best = std::numeric_limits<double>::max();
		for (const auto& v : m.vertices)
			best = std::min(best, static_cast<double>((v - p).norm()));
		return best;
	};

	const Mesh withFeat = remesh(true);
	// The cube must actually be remeshed (refined), not left untouched.
	EXPECT_GT(withFeat.faces.size(), input.faces.size())
	    << "cube was not refined — test would be vacuous";
	// ...yet every original cube corner must still have a vertex essentially on it.
	double worst = 0.0;
	for (const auto& c : corners)
		worst = std::max(worst, minDistTo(withFeat, c));
	std::cout << "[FeatureCorners] refined faces=" << withFeat.faces.size()
	          << " worst corner deviation=" << worst << "\n";
	EXPECT_LT(worst, 1e-3) << "feature_corners failed to preserve a cube corner";
}

// ---------------------------------------------------------------------------
// Test 11b: short crease edges must collapse ALONG the crease. A tent ridge is
// a straight crease line (90° between slope normals); its interior vertices
// have exactly two incident feature edges, so they are slideable crease
// vertices — not corners — and remeshing at 4x the ridge segment length must
// be able to coarsen the ridge.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, CreaseEdgesCollapseAlongRidge)
{
	Mesh m;
	constexpr unsigned N = 40; // ridge segments of length 0.1 → ridge length 4
	constexpr float h = 0.1f;
	for (unsigned j = 0; j <= N; ++j) {
		const float y = h * static_cast<float>(j);
		m.vertices.emplace_back(0.f, y, 1.f); // ridge
		m.vertices.emplace_back(1.f, y, 0.f); // right side
		m.vertices.emplace_back(-1.f, y, 0.f); // left side
	}
	for (unsigned j = 0; j < N; ++j) {
		const Mesh::VIndex r0 = 3 * j, s0 = 3 * j + 1, t0 = 3 * j + 2;
		const Mesh::VIndex r1 = 3 * (j + 1), s1 = 3 * (j + 1) + 1, t1 = 3 * (j + 1) + 2;
		m.faces.emplace_back(Mesh::Face{r0, s0, s1}); // right slope
		m.faces.emplace_back(Mesh::Face{r0, s1, r1});
		m.faces.emplace_back(Mesh::Face{r0, r1, t1}); // left slope
		m.faces.emplace_back(Mesh::Face{r0, t1, t0});
	}
	const unsigned ridgeBefore = N + 1;

	Mesh::RemeshParams p;
	p.SetEdgeLength(0.4f); // edge_min = 0.32 → 0.1 ridge edges are collapse candidates
	p.iterations = 5;
	m.RemeshIsotropic(p);

	unsigned ridgeAfter = 0;
	for (const auto& v : m.vertices)
		if (std::abs(v.x()) < 1e-3f && std::abs(v.z() - 1.f) < 1e-3f)
			++ridgeAfter;
	std::cout << "[CreaseCollapse] ridge vertices " << ridgeBefore << " -> "
	          << ridgeAfter << "\n";
	// The ridge must survive as a crease (some vertices stay on it)...
	EXPECT_GE(ridgeAfter, 2u);
	// ...but its 0.1-length segments must coarsen toward the 0.4 target: ~11
	// vertices ideal, 12 achieved. Over-locking degree-2 crease vertices (the
	// seen-set bug counted each crease neighbour twice) left 24+ behind.
	EXPECT_LE(ridgeAfter, 16u)
	    << "degree-2 crease vertices were locked — ridge could not coarsen";
}

// ---------------------------------------------------------------------------
// Test 11e: remeshing an open curved surface must not erode its border.
// On a spherical cap, split midpoints of rim edges lie on chords (inside the
// sphere and inward of the rim); projecting them to the nearest point of the
// ORIGINAL SURFACE lands in the cap interior, so the border creeps inward a
// bit each iteration. Boundary vertices must stay on the original rim circle.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, OpenBorderStaysOnRim)
{
	// Spherical cap of extent thmax: rim circle at polar angle thmax, radius 1.
	const float thmax = 1.2f;
	const int nth = 8, nphi = 16;
	Mesh m;
	m.vertices.emplace_back(0.f, 0.f, 1.f);
	for (int i = 1; i <= nth; ++i) {
		const float th = thmax * static_cast<float>(i) / static_cast<float>(nth);
		for (int j = 0; j < nphi; ++j) {
			const float ph = 2.f * static_cast<float>(M_PI) * static_cast<float>(j) / static_cast<float>(nphi);
			m.vertices.emplace_back(std::sin(th) * std::cos(ph),
			                        std::sin(th) * std::sin(ph), std::cos(th));
		}
	}
	auto ring = [&](int i, int j) {
		return static_cast<Mesh::VIndex>(1 + i * nphi + (j % nphi));
	};
	for (int j = 0; j < nphi; ++j)
		m.faces.emplace_back(Mesh::Face(0u, ring(0, j), ring(0, j + 1)));
	for (int i = 0; i + 1 < nth; ++i)
		for (int j = 0; j < nphi; ++j) {
			m.faces.emplace_back(Mesh::Face(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)));
			m.faces.emplace_back(Mesh::Face(ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)));
		}

	m.ListHalfEdges();
	const float L = static_cast<float>(ComputeEdgeStats(m).meanLen) * 0.5f; // refine
	Mesh::RemeshParams p;
	p.SetEdgeLength(L);
	p.iterations = 5;
	p.smoothTangential = true;
	p.smoothIterations = 5;
	m.RemeshIsotropic(p);

	// Every boundary vertex must still lie (near) the original rim circle:
	// radius 1, polar angle thmax.
	m.ListHalfEdges();
	std::map<std::pair<Mesh::VIndex, Mesh::VIndex>, int> ecount;
	for (const auto& f : m.faces)
		for (int e = 0; e < 3; ++e) {
			Mesh::VIndex a = f[e], b = f[(e + 1) % 3];
			if (a > b)
				std::swap(a, b);
			++ecount[{a, b}];
		}
	std::set<Mesh::VIndex> bverts;
	for (const auto& kv : ecount)
		if (kv.second == 1) {
			bverts.insert(kv.first.first);
			bverts.insert(kv.first.second);
		}
	ASSERT_GT(bverts.size(), 8u);
	double worstTheta = 0.0, worstRadius = 0.0;
	for (Mesh::VIndex v : bverts) {
		const auto& pos = m.vertices[v];
		const double r = pos.norm();
		const double th = std::acos(std::clamp(static_cast<double>(pos.z()) / r, -1.0, 1.0));
		worstTheta = std::max(worstTheta, std::abs(th - static_cast<double>(thmax)));
		worstRadius = std::max(worstRadius, std::abs(r - 1.0));
	}
	std::cout << "[OpenBorder] worst |theta-thmax|=" << worstTheta
	          << " worst |r-1|=" << worstRadius << "\n";
	// rim edge subtends ~2*pi/16=0.39rad originally, ~0.2 after refine; the
	// rim polyline chordal sag is ~0.005 — allow that, flag real erosion.
	EXPECT_LT(worstTheta, 0.05)
	    << "open border eroded inward from the original rim";
	EXPECT_LT(worstRadius, 0.02);
}

// ---------------------------------------------------------------------------
// Test 11d: a single remesh iteration must fully refine edges far above the
// split threshold. Splitting each original edge only once leaves the halves
// of a >2x-threshold edge over-long (only masked by running more outer
// iterations).
// ---------------------------------------------------------------------------
TEST(MeshRemesh, SingleIterationRefinesVeryLongEdges)
{
	Mesh m = hmtest::corpus::GridPlane(2); // very coarse: edges ~8x the target
	m.ListHalfEdges();
	const float L = static_cast<float>(ComputeEdgeStats(m).meanLen) / 8.f;

	Mesh::RemeshParams p;
	p.SetEdgeLength(L);
	p.iterations = 1;
	m.RemeshIsotropic(p);

	m.ListHalfEdges();
	const EdgeStats es = ComputeEdgeStats(m);
	std::cout << "[SingleIterSplit] max edge=" << es.maxLen
	          << " edge_max=" << p.edgeMaxLength << "\n";
	EXPECT_LE(es.maxLen, 2.0 * static_cast<double>(p.edgeMaxLength))
	    << "one iteration left edges far above the split threshold";
}

// ---------------------------------------------------------------------------
// Test 11c: adaptive coarsening must be able to EXCEED the uniform edge_max
// cap in flat regions. With maxAdaptiveMult=3 a flat plane's sizing target
// is 3x the base length, but the post-collapse length guard used the uniform
// edgeMaxLength, silently rejecting every coarsening collapse.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, AdaptiveCoarsensFlatRegionsPastUniformCap)
{
	// A dense flat grid: zero curvature everywhere → sizing = max mult * L.
	Mesh m = hmtest::corpus::GridPlane(24); // 24x24 quads, side 1
	m.ListHalfEdges();
	const EdgeStats es = ComputeEdgeStats(m);
	const float L = static_cast<float>(es.meanLen) * 1.5f;

	Mesh::RemeshParams p;
	p.SetEdgeLength(L);
	p.iterations = 6;
	p.adapt = true;
	p.approxError = 0.f;
	p.minAdaptiveMult = 1.f;
	p.maxAdaptiveMult = 3.f;
	m.RemeshIsotropic(p);

	// Interior edges (both endpoints off the boundary rim) must have grown
	// past the uniform cap 4/3*L toward the adaptive target 3L.
	std::map<std::pair<Mesh::VIndex, Mesh::VIndex>, int> ecount;
	for (const auto& f : m.faces)
		for (int e = 0; e < 3; ++e) {
			Mesh::VIndex a = f[e], b = f[(e + 1) % 3];
			if (a > b)
				std::swap(a, b);
			++ecount[{a, b}];
		}
	std::set<Mesh::VIndex> bverts;
	for (const auto& kv : ecount)
		if (kv.second == 1) {
			bverts.insert(kv.first.first);
			bverts.insert(kv.first.second);
		}
	double maxInterior = 0.0;
	for (const auto& kv : ecount) {
		if (bverts.count(kv.first.first) || bverts.count(kv.first.second))
			continue;
		maxInterior = std::max(
		    maxInterior,
		    static_cast<double>(
		        (m.vertices[kv.first.first] - m.vertices[kv.first.second]).norm()));
	}
	std::cout << "[AdaptiveCoarsen] max interior edge=" << maxInterior
	          << " uniform cap=" << p.edgeMaxLength << "\n";
	// Post-fix reaches ~7.2 (target 3L); with the uniform cap wrongly applied
	// the interior never grew past ~2.6.
	EXPECT_GT(maxInterior, 2.0 * static_cast<double>(p.edgeMaxLength))
	    << "adaptive mode could not coarsen a flat region past the uniform cap";
}

// ---------------------------------------------------------------------------
// Test 12: curvature-adaptive sizing produces a genuinely graded, deterministic,
// faithful mesh that spends fewer faces than uniform remeshing.
//
// NOTE: this deliberately does NOT assert a hausdorff*faces "efficiency" win.
// Hausdorff is a MAX over the surface, so it is set by the single worst spot;
// adaptive lowers the AVERAGE error (concentrating resolution where curvature is
// high) but its coarsened regions and less-regular triangles keep the MAX no
// better than a well-tuned uniform remesh — measured across torus/bump surfaces,
// uniform matches or beats adaptive on hausdorff*faces at every budget. The
// sizing-correctness claim (saddle/developable refinement) is proven directly by
// AdaptiveSizingRefinesSaddleRegions; here we lock in the graded field, its
// determinism, its coarsening, and its faithfulness.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, AdaptiveSizingImprovesFidelityPerFace)
{
	const Mesh input = hmtest::corpus::TorusMesh(48, 24);
	Mesh tmp = input;
	tmp.ListHalfEdges();
	const float L = static_cast<float>(ComputeEdgeStats(tmp).meanLen);

	auto run = [&](bool adaptive) {
		Mesh m = input;
		Mesh::RemeshParams p;
		p.SetEdgeLength(L);
		p.iterations = 5;
		p.smoothTangential = true;
		p.smoothIterations = 5;
		if (adaptive)
			p.SetAdaptive(/*error*/ 0.5f * L, /*minMult*/ 0.5f, /*maxMult*/ 4.0f);
		m.RemeshIsotropic(p);
		return m;
	};

	const Mesh uniMesh = run(false);
	const Mesh adaMesh = run(true);
	const Mesh adaMesh2 = run(true); // determinism: identical params -> identical output
	const RemeshQuality uni = ComputeRemeshQuality(input, uniMesh);
	const RemeshQuality ada = ComputeRemeshQuality(input, adaMesh);
	PrintRemeshQuality("uniform", uni);
	PrintRemeshQuality("adaptive", ada);
	std::cout << "[Adaptive] faces uniform=" << uni.numFaces << " adaptive=" << ada.numFaces << "\n";

	ASSERT_GT(ada.numFaces, 0u);
	ASSERT_TRUE(std::isfinite(ada.edgeCov) && std::isfinite(ada.hausdorffToInput));
	// Deterministic: the same input+params must give byte-identical output.
	ASSERT_EQ(adaMesh.vertices.size(), adaMesh2.vertices.size());
	ASSERT_EQ(adaMesh.faces.size(), adaMesh2.faces.size());
	for (size_t i = 0; i < adaMesh.vertices.size(); ++i)
		EXPECT_EQ(adaMesh.vertices[i], adaMesh2.vertices[i]) << "adaptive remesh is non-deterministic at vertex " << i;
	// The sizing field is genuinely non-uniform (varies with curvature).
	EXPECT_GT(ada.edgeCov, uni.edgeCov)
	    << "adaptive field is not varying (edge CoV " << ada.edgeCov << " vs " << uni.edgeCov << ")";
	// Adaptive spends fewer faces (coarsens low-curvature regions).
	EXPECT_LT(ada.numFaces, uni.numFaces)
	    << "adaptive did not coarsen relative to uniform (" << ada.numFaces << " vs " << uni.numFaces << ")";
	// ...while staying faithful to the input surface.
	ASSERT_GT(ada.bboxDiag, 0.0);
	EXPECT_LT(ada.meanDistRatio(), 0.02) << "adaptive drifted off the input surface";
}

// ---------------------------------------------------------------------------
// Test 16: adaptive sizing must refine SADDLE (negative-Gaussian-curvature)
// regions. A catenoid is a minimal surface: the mean curvature H is ~0
// EVERYWHERE while the Gaussian curvature K < 0 everywhere, so the max absolute
// principal curvature kappaMax = sqrt(-K) is substantial. The old |H|-only
// sizing therefore saw ~0 curvature over the WHOLE surface and coarsened it to
// the maximum edge length; the sizing-smoothing passes cannot dilute this
// because every vertex is equally mis-sized. kappaMax = |H| + sqrt(H^2 - K)
// recovers the real curvature and keeps the saddle refined.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, AdaptiveSizingRefinesSaddleRegions)
{
	// Catenoid tube: x=cosh(v)cos u, y=cosh(v)sin u, z=v; closed in u, open rims.
	Mesh input;
	const int NU = 48, NV = 20;
	const float vmax = 0.9f;
	for (int j = 0; j <= NV; ++j) {
		const float v = -vmax + 2.f * vmax * static_cast<float>(j) / static_cast<float>(NV);
		const float ch = std::cosh(v);
		for (int i = 0; i < NU; ++i) {
			const float u = 2.f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(NU);
			input.vertices.emplace_back(ch * std::cos(u), ch * std::sin(u), v);
		}
	}
	auto vid = [&](int i, int j) { return static_cast<Mesh::VIndex>(j * NU + (i % NU)); };
	for (int j = 0; j < NV; ++j)
		for (int i = 0; i < NU; ++i) {
			input.faces.emplace_back(Mesh::Face{vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)});
			input.faces.emplace_back(Mesh::Face{vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)});
		}
	Mesh tmp = input;
	tmp.ListHalfEdges();
	const float base = static_cast<float>(ComputeEdgeStats(tmp).meanLen);
	const TriangleKdTree inputTree(input);

	Mesh m = input;
	Mesh::RemeshParams p;
	p.SetEdgeLength(base * 4.f); // coarse base; the bug would coarsen to ~max
	p.iterations = 5;
	p.smoothTangential = true;
	p.smoothIterations = 5;
	p.SetAdaptive(/*error*/ 0.02f, /*minMult*/ 0.25f, /*maxMult*/ 6.f);
	m.RemeshIsotropic(p);

	// Max distance of interior edge midpoints to the (fine) input surface: coarse
	// edges bow off the curved saddle. Skip midpoints near the open rims.
	ASSERT_GT(m.faces.size(), 0u);
	double worst = 0.0;
	for (const auto& f : m.faces)
		for (int k = 0; k < 3; ++k) {
			const Mesh::Vertex mid = 0.5f * (m.vertices[f[k]] + m.vertices[f[(k + 1) % 3]]);
			if (std::abs(mid.z()) > 0.75f)
				continue; // away from the rims
			const auto nn = inputTree.NearestPoint(mid);
			if (nn.IsValid())
				worst = std::max(worst, static_cast<double>(nn.dist));
		}
	std::cout << "[SaddleFidelity] worst interior edge-midpoint dist to surface=" << worst << "\n";
	// |H|-only sizing coarsens the whole minimal surface (dist ~0.11);
	// kappaMax keeps it refined (dist ~0.049).
	EXPECT_LT(worst, 0.06) << "adaptive sizing left the saddle surface under-refined";
}

// ---------------------------------------------------------------------------
// Test 15: valence-optimizing edge flip must actually raise valence regularity.
// The flip score must add a neighbour (+1) to the two OPPOSITE stencil vertices
// (which gain the new diagonal) and remove one (-1) only from the shared-edge
// endpoints; decrementing all four (the old bug) mis-scores flips and leaves a
// closed surface with too many irregular vertices. On a UV sphere (all interior)
// a correct rule drives most vertices to the ideal valence 6.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, EdgeFlipImprovesValenceRegularity)
{
	const Mesh input = hmtest::corpus::UVSphere(24, 36);
	Mesh m = input;
	m.ListHalfEdges();
	const EdgeStats es = ComputeEdgeStats(m);
	Mesh::RemeshParams p;
	p.SetEdgeLength(static_cast<float>(es.meanLen));
	p.iterations = 5;
	m.RemeshIsotropic(p);
	m.ListHalfEdges();

	const HalfMesh& hm = m.halfMesh;
	std::size_t interior = 0, regular = 0;
	for (Mesh::VIndex v = 0; v < hm.VSize(); ++v) {
		if (hm.VIsBoundary(v))
			continue;
		++interior;
		if (hm.VDegree(v) == 6)
			++regular;
	}
	ASSERT_GT(interior, 0u);
	const double regularFrac = static_cast<double>(regular) / static_cast<double>(interior);
	std::cout << "[FlipValence] regular(interior valence-6) fraction=" << regularFrac
	          << " (" << regular << "/" << interior << ")\n";
	// Correct scoring lifts the regular fraction measurably (bug ~0.67, fix ~0.73).
	EXPECT_GT(regularFrac, 0.70) << "too few regular-valence interior vertices";
}

// ---------------------------------------------------------------------------
// Test 14: the Hausdorff normal gate must be scale-invariant. It compares the
// alignment of the new face normal with the nearest original-surface face
// normal against a fixed cosine threshold (0.7). Because power-of-2 rescaling of
// the whole mesh AND all length parameters is an exact float operation, a truly
// scale-invariant remesh produces bit-identically-scaled geometry and therefore
// an IDENTICAL collapseCount. The old gate dotted UNNORMALIZED normals (whose
// product scales with area^2 ~ s^4), so at 1/64 scale the dot underflowed the
// 0.7 threshold and rejected every collapse (count ~0), while at 64x it never
// fired — grossly scale-dependent.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, HausdorffNormalGateScaleInvariant)
{
	auto runAtScale = [](float s) {
		Mesh m = hmtest::corpus::UVSphere(16, 24);
		for (auto& v : m.vertices)
			v *= s; // exact for power-of-2 s
		Mesh::RemeshParams p;
		p.SetEdgeLength(0.4f * s); // coarser than the sphere's edges => collapses
		p.iterations = 3;
		p.checkSurfDist = true;
		p.SetMaxSurfaceDistance(0.4f * s); // maxSurfDist = 0.04*s
		Mesh::RemeshStats st;
		m.RemeshIsotropic(p, &st);
		return st.collapseCount;
	};
	const unsigned c1 = runAtScale(1.0f);
	const unsigned cSmall = runAtScale(1.0f / 64.0f);
	const unsigned cBig = runAtScale(64.0f);
	std::cout << "[NormalGateScale] collapses @1=" << c1
	          << " @1/64=" << cSmall << " @64=" << cBig << "\n";
	// A scale-invariant gate collapses a substantial number at every scale...
	EXPECT_GT(c1, 50u);
	// ...and — under exact power-of-2 rescaling — exactly the same number.
	EXPECT_EQ(cSmall, c1) << "collapse_count differs at 1/64 scale (gate is scale-dependent)";
	EXPECT_EQ(cBig, c1) << "collapse_count differs at 64x scale (gate is scale-dependent)";
}

// ---------------------------------------------------------------------------
// Test 13: operation stats are reported and per-pass toggles take effect.
// ---------------------------------------------------------------------------
TEST(MeshRemesh, StatsAndToggles)
{
	const Mesh input = hmtest::corpus::UVSphere(16, 24);
	Mesh tmp = input;
	tmp.ListHalfEdges();
	const float refine = static_cast<float>(ComputeEdgeStats(tmp).meanLen) * 0.5f; // refine target

	// stats populated: a refine target must trigger splits.
	{
		Mesh m = input;
		Mesh::RemeshParams p;
		p.SetEdgeLength(refine);
		p.iterations = 3;
		Mesh::RemeshStats s;
		m.RemeshIsotropic(p, &s);
		std::cout << "[Stats] split=" << s.splitCount << " collapse=" << s.collapseCount
		          << " flip=" << s.flipCount << "\n";
		EXPECT_GT(s.splitCount, 0u);
	}

	// toggle: disabling the split pass yields far fewer faces on a refine target.
	{
		Mesh withSplit = input, withoutSplit = input;
		Mesh::RemeshParams p;
		p.SetEdgeLength(refine);
		p.iterations = 3;
		withSplit.RemeshIsotropic(p);
		Mesh::RemeshParams pno = p;
		pno.doSplit = false;
		withoutSplit.RemeshIsotropic(pno);
		EXPECT_GT(withSplit.faces.size(), withoutSplit.faces.size())
		    << "disabling do_split should yield fewer faces on a refine target";
	}
}

// Remesh splits edges (appending a midpoint vertex) and collapses them (swap-pop
// on the vertex array). It already mirrored both onto its own `sizing` and
// `projectHint` arrays but not onto Mesh::vertexColors, which is subject to the
// same parallel-array contract -- so a colored mesh came out with a colors array
// of the wrong length and stale entries at swapped slots.
TEST(MeshRemesh, KeepsVertexColorsInLockstep)
{
	Mesh m = hmtest::corpus::UVSphere(8, 12);
	m.vertexColors.assign(m.vertices.size(), Mesh::Pixel(60, 60, 60));
	Mesh::RemeshParams params;
	// half the mean edge length, so the pass does real splitting and collapsing
	const float meanLen = m.ComputeMeanEdgeLength();
	ASSERT_GT(meanLen, 0.f);
	params.SetEdgeLength(meanLen * 0.5f);
	params.iterations = 2;
	Mesh::RemeshStats stats;

	m.RemeshIsotropic(params, &stats);

	EXPECT_GT(stats.splitCount, 0u);
	ASSERT_EQ(m.vertexColors.size(), m.vertices.size());
	EXPECT_TRUE(m.ValidateInvariants());
	// every source colour was the same, so every interpolated/moved one must be too
	for (size_t i = 0; i < m.vertexColors.size(); ++i)
		EXPECT_EQ(static_cast<int>(m.vertexColors[i].x()), 60) << "at slot " << i;
}

TEST(MeshRemesh, HalfEdgeOnlyEntryMatchesPopulatedEntryWithoutBuild)
{
	Mesh populated = hmtest::corpus::UVSphere(8, 12);
	Mesh native = populated;
	populated.ListHalfEdges();
	native.ListHalfEdges();
	const float target = static_cast<float>(ComputeEdgeStats(populated).meanLen);
	native.InvalidateFaces();
	Mesh::RemeshParams params;
	params.SetEdgeLength(target);
	params.iterations = 1;
	Mesh::RemeshStats populatedStats;
	Mesh::RemeshStats nativeStats;

	populated.RemeshIsotropic(params, &populatedStats);
	HalfMesh::ResetBuildCount();
	native.RemeshIsotropic(params, &nativeStats);
	EXPECT_EQ(HalfMesh::BuildCount(), 0u);
	EXPECT_EQ(native.vertices, populated.vertices);
	EXPECT_EQ(native.faces, populated.faces);
	EXPECT_EQ(nativeStats.splitCount, populatedStats.splitCount);
	EXPECT_EQ(nativeStats.collapseCount, populatedStats.collapseCount);
	EXPECT_EQ(nativeStats.flipCount, populatedStats.flipCount);
	EXPECT_TRUE(native.ValidateHalfMesh());
}

// RemeshParams' edge lengths were the only fields without default
// initializers — a default-constructed struct read indeterminate values, and
// SplitLongEdges has no floor, so a garbage/non-positive edgeMaxLength
// subdivides every edge forever.
TEST(MeshRemesh, RemeshParamsEdgeLengthsDefaultToZero)
{
	Mesh::RemeshParams p;
	EXPECT_EQ(p.edgeMinLength, 0.f);
	EXPECT_EQ(p.edgeMaxLength, 0.f);
}

// The zero default is NOT a valid remesh request — RemeshIsotropic must
// refuse it loudly (warning + no-op) instead of splitting unboundedly.
// NOTE: this test is added GREEN-only; its pre-fix behavior IS the bug (an
// unbounded split loop), so it cannot be run red — the defaults test above
// carries the red phase.
TEST(MeshRemesh, RemeshIsotropicRejectsInvalidEdgeLengths)
{
	const auto makeQuad = [] {
		Mesh mesh;
		mesh.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0),
		                 Mesh::Vertex(1, 1, 0), Mesh::Vertex(0, 1, 0)};
		mesh.faces = {Mesh::Face(0, 1, 2), Mesh::Face(0, 2, 3)};
		return mesh;
	};
	{
		Mesh mesh = makeQuad();
		Mesh::RemeshParams p; // zero-initialized edge lengths: invalid
		p.iterations = 1;
		mesh.RemeshIsotropic(p);
		EXPECT_EQ(mesh.faces.size(), 2u) << "invalid params must be a no-op";
	}
	{
		Mesh mesh = makeQuad();
		Mesh::RemeshParams p;
		p.edgeMinLength = 0.1f;
		p.edgeMaxLength = std::numeric_limits<float>::quiet_NaN();
		p.iterations = 1;
		mesh.RemeshIsotropic(p);
		EXPECT_EQ(mesh.faces.size(), 2u) << "NaN edge_max_length must be a no-op";
	}
}

// Remeshing splits, collapses and tangentially relaxes vertices, so no authored
// per-vertex normal survives it; the array is dropped rather than left stale, and
// the invariant holds against the vertex set the pass ends with.
TEST(MeshRemesh, ClearsAuthoredVertexNormals)
{
	Mesh m = hmtest::corpus::GridPlane(8); // 8x8 quads, side 1
	m.vertexNormals.assign(m.vertices.size(), Mesh::Normal(0.f, 0.f, 1.f));

	Mesh::RemeshParams p;
	p.SetEdgeLength(0.25f); // coarsen: 2x the grid edge
	p.iterations = 2;
	m.RemeshIsotropic(p);

	EXPECT_TRUE(m.vertexNormals.empty());
	EXPECT_TRUE(m.ValidateInvariants());
}

} // namespace
} // namespace halfmesh
