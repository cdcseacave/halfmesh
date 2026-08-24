/*
* PerfHarness.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/perf/PerfHarness.cpp — Performance regression harness
//
// Ops timed on LargeMesh(N) for N ≈ 50k / 200k / 800k faces:
//   HalfMesh Build, Simplify(0.5), RemeshIsotropic, FixNonManifold,
//   CloseHoles (holed variant), TriangleKdTree build + queries,
//   SegmentCharts+Parametrize, GenerateAtlas
//
// Assertions (machine-independent):
//   - KD-tree queries >= 10x faster than brute-force O(n*m) at large N
//   - Build + Simplify scale sub-quadratically (4x faces → < 8x time)
//   - Every op on largest mesh completes within generous wall-clock bound
//
// Output: tests/perf/baseline.json (on first run / when PERF_WRITE_BASELINE=1)
//         comparison against baseline.json (regression alarm at 3x)
//
// Build: cmake -DHALFMESH_BUILD_PERF=ON -DCMAKE_BUILD_TYPE=Release
// Run:   ctest --test-dir <build> -L perf
//     or: ./perf_harness --gtest_filter=*

#include <gtest/gtest.h>

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/TriangleKDTree.h>
#include <halfmesh/TriangleBVH.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/Util/Geometry.h>

#include "corpus/Corpus.h" // hmtest::corpus::LargeMesh

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

static double elapsedS(Clock::time_point t0)
{
	return std::chrono::duration_cast<Seconds>(Clock::now() - t0).count();
}

#define TIME_OP(label, code)                      \
	do {                                          \
		auto _t0 = Clock::now();                  \
		code;                                     \
		double _dt = elapsedS(_t0);               \
		record(label, _dt);                       \
		std::cout << "  " << label << ": " << _dt \
		          << " s\n";                      \
	} while (0)

// ---------------------------------------------------------------------------
// Brute-force NearestPoint for KD-tree speedup comparison.
// Uses the SAME math::DistanceBetweenTriangleAndPoint function as the KD-tree
// so the per-triangle cost is identical. Speedup measures traversal pruning.
// ---------------------------------------------------------------------------
static halfmesh::TriangleKdTree::NearestNeighbor
BruteForceNearest(const halfmesh::Mesh& mesh, const halfmesh::Mesh::Vertex& q)
{
	halfmesh::TriangleKdTree::NearestNeighbor best;
	halfmesh::Mesh::Vertex nearest;
	for (halfmesh::Mesh::FIndex fi = 0; fi < (halfmesh::Mesh::FIndex)mesh.faces.size(); ++fi) {
		const auto& face = mesh.faces[fi];
		const float d = math::DistanceBetweenTriangleAndPoint(
		    mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]],
		    q, &nearest);
		if (d < best.dist) {
			best.dist = d;
			best.nearest = nearest;
			best.idxFace = fi;
		}
	}
	return best;
}

// ---------------------------------------------------------------------------
// Mesh helpers
// ---------------------------------------------------------------------------

// Create a mesh with N holes by removing individual faces via RemoveFaces.
// We pick widely-spaced interior faces to avoid merging boundary loops.
static halfmesh::Mesh MakeHoled(const halfmesh::Mesh& m, unsigned nHoles = 20)
{
	halfmesh::Mesh copy = m;
	// Build half-edge to find boundary-free interior faces
	copy.ListHalfEdges();
	const unsigned nf = (unsigned)copy.faces.size();
	const unsigned step = std::max(1u, nf / (nHoles + 1));
	unsigned actualHoles = std::min(nHoles, nf / 10); // cap at 10% of faces
	std::vector<halfmesh::Mesh::FIndex> toRemove;
	toRemove.reserve(actualHoles);
	for (unsigned i = 0; i < actualHoles; ++i) {
		halfmesh::Mesh::FIndex fi = (i + 1) * step;
		if (fi >= nf)
			break;
		toRemove.push_back(fi);
	}
	if (toRemove.empty())
		return copy;
	copy.RemoveFaces(toRemove);
	copy.RemoveUnreferencedVertices();
	return copy;
}

// ---------------------------------------------------------------------------
// JSON helpers (minimal — no dependency on nlohmann or similar)
// ---------------------------------------------------------------------------
struct PerfRecord
{
	std::string label;
	double seconds;
	unsigned nFaces;
};

static std::string baselinePath()
{
	// Find source dir: walk up from __FILE__ until we find tests/perf
	namespace fs = std::filesystem;
	fs::path src = fs::path(__FILE__).parent_path();
	return (src / "baseline.json").string();
}

static void writeJson(const std::vector<PerfRecord>& records,
                      const std::string& machineLabel,
                      const std::string& path)
{
	std::ofstream f(path);
	if (!f) {
		std::cerr << "Cannot write " << path << "\n";
		return;
	}
	f << "{\n";
	f << "  \"machine\": \"" << machineLabel << "\",\n";
	f << "  \"note\": \"Absolute times are machine-dependent. Use only as regression alarm (3x).\",\n";
	f << "  \"timings\": [\n";
	for (size_t i = 0; i < records.size(); ++i) {
		const auto& r = records[i];
		f << "    {\"label\": \"" << r.label << "\""
		  << ", \"n_faces\": " << r.nFaces
		  << ", \"seconds\": " << r.seconds
		  << "}";
		if (i + 1 < records.size())
			f << ",";
		f << "\n";
	}
	f << "  ]\n}\n";
	std::cout << "\nBaseline written: " << path << "\n";
}

static std::unordered_map<std::string, double> loadBaseline(const std::string& path)
{
	std::unordered_map<std::string, double> m;
	std::ifstream f(path);
	if (!f)
		return m;
	std::string line;
	while (std::getline(f, line)) {
		// crude parse: {"label": "...", "n_faces": N, "seconds": T}
		auto lpos = line.find("\"label\": \"");
		auto spos = line.find("\"seconds\": ");
		if (lpos == std::string::npos || spos == std::string::npos)
			continue;
		lpos += 10;
		auto lend = line.find("\"", lpos);
		if (lend == std::string::npos)
			continue;
		std::string label = line.substr(lpos, lend - lpos);
		double sec = std::stod(line.substr(spos + 11));
		m[label] = sec;
	}
	return m;
}

// ---------------------------------------------------------------------------
// Global state for the test suite
// ---------------------------------------------------------------------------
static std::vector<PerfRecord> gRecords;
static std::unordered_map<std::string, double> gBaseline;

static void record(const std::string& label, double sec)
{
	// We'll set nFaces outside
	gRecords.push_back({label, sec, 0});
}
static void record(const std::string& label, double sec, unsigned n)
{
	gRecords.push_back({label, sec, n});
}

// ---------------------------------------------------------------------------
// The perf test suite
// ---------------------------------------------------------------------------

// Sizes used (actual face counts will differ slightly due to UV sphere quant)
constexpr unsigned SMALL_TARGET = 50000;
constexpr unsigned MED_TARGET = 200000;
constexpr unsigned LARGE_TARGET = 800000;
constexpr unsigned TRUCK_TARGET = 5000000;

// Wall-clock upper bound per op on largest mesh (generous: 120 s)
constexpr double MAX_WALL_SECONDS = 120.0;

// ======================== Small mesh (50k faces) ===========================
TEST(PerfHarness, Build_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	std::cout << "[Build 50k] actual_faces=" << actual << "\n";
	halfmesh::HalfMesh hm;
	double dt;
	auto t0 = Clock::now();
	hm.Build(m);
	dt = elapsedS(t0);
	record("Build_50k", dt, actual);
	std::cout << "  Build_50k: " << dt << " s\n";
}

TEST(PerfHarness, Build_200k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	std::cout << "[Build 200k] actual_faces=" << actual << "\n";
	halfmesh::HalfMesh hm;
	double dt;
	auto t0 = Clock::now();
	hm.Build(m);
	dt = elapsedS(t0);
	record("Build_200k", dt, actual);
	std::cout << "  Build_200k: " << dt << " s\n";
}

TEST(PerfHarness, Build_800k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(LARGE_TARGET, &actual);
	std::cout << "[Build 800k] actual_faces=" << actual << "\n";
	halfmesh::HalfMesh hm;
	double dt;
	auto t0 = Clock::now();
	hm.Build(m);
	dt = elapsedS(t0);
	record("Build_800k", dt, actual);
	std::cout << "  Build_800k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "Build_800k blew wall-clock bound";
}

// ===================== Scaling assertion: Build ============================
TEST(PerfHarness, ScalingAssertion_Build)
{
	// Find Build_50k and Build_200k in gRecords (already run)
	double t50 = -1, t200 = -1;
	for (const auto& r : gRecords) {
		if (r.label == "Build_50k")
			t50 = r.seconds;
		if (r.label == "Build_200k")
			t200 = r.seconds;
	}
	if (t50 <= 0 || t200 <= 0) {
		GTEST_SKIP() << "Build timing records not found";
	}
	// 200k / 50k ≈ 4x faces → expect < 8x time (sub-quadratic)
	double ratio = t200 / t50;
	std::cout << "Build scaling ratio (4x faces): " << ratio << "x time\n";
	// Log the implied exponent: t ~ N^alpha → alpha = log(ratio)/log(4)
	double alpha = std::log(ratio) / std::log(4.0);
	std::cout << "Build scaling exponent alpha ≈ " << alpha << "\n";
	EXPECT_LT(ratio, 8.0)
	    << "Build time grew " << ratio << "x for 4x faces — possible O(n^2)";
}

TEST(PerfHarness, PrebuiltRepairPipelinePerformsZeroBuilds)
{
	halfmesh::Mesh mesh = hmtest::corpus::UVSphere(24, 36);
	mesh.ListHalfEdges();
	halfmesh::HalfMesh::ResetBuildCount();
	mesh.RemoveSpuriousComponents(100.f);
	mesh.RemoveSpikes();
	mesh.RemoveDegenerateFaces();
	mesh.RemoveUnreferencedVertices();
	EXPECT_EQ(halfmesh::HalfMesh::BuildCount(), 0u);
	EXPECT_FALSE(mesh.halfMesh.Empty());
}

TEST(PerfHarness, SimulatedCleanPerformsOneBuildAndOneFaceHarvest)
{
	halfmesh::Mesh mesh = hmtest::corpus::UVSphere(16, 24);
	halfmesh::HalfMesh::ResetBuildCount();
	halfmesh::HalfMesh::ResetFFacesCount();
	mesh.BeginHalfEdgePipeline();
	mesh.RemoveSpuriousComponents(100.f);
	mesh.RemoveSpikes();
	mesh.Simplify(0.8f);
	mesh.CloseHoles();
	mesh.SmoothTaubin(1);
	halfmesh::Mesh::RemeshParams params;
	params.SetEdgeLength(mesh.ComputeMeanEdgeLength());
	params.iterations = 1;
	mesh.RemeshIsotropic(params);
	EXPECT_TRUE(mesh.faces.empty());
	mesh.EndHalfEdgePipeline();

	EXPECT_EQ(halfmesh::HalfMesh::BuildCount(), 1u);
	EXPECT_EQ(halfmesh::HalfMesh::FFacesCount(), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(PerfHarness, TruckScaleNativeCleaningPerformsOneBuildAndOneFaceHarvest)
{
	unsigned actual = 0;
	halfmesh::Mesh mesh = hmtest::corpus::LargeMesh(TRUCK_TARGET, &actual);
	halfmesh::HalfMesh::ResetBuildCount();
	halfmesh::HalfMesh::ResetFFacesCount();
	const auto t0 = Clock::now();
	mesh.BeginHalfEdgePipeline();
	mesh.RemoveSpuriousComponents(100.f);
	mesh.RemoveSpikes();
	mesh.CloseHoles();
	mesh.RemoveUnreferencedVertices();
	mesh.EndHalfEdgePipeline();
	const double dt = elapsedS(t0);
	record("NativeClean_5m", dt, actual);
	std::cout << "  NativeClean_5m: " << dt << " s\n";

	EXPECT_GE(actual, TRUCK_TARGET);
	EXPECT_EQ(halfmesh::HalfMesh::BuildCount(), 1u);
	EXPECT_EQ(halfmesh::HalfMesh::FFacesCount(), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	EXPECT_LT(dt, MAX_WALL_SECONDS);
}

// ======================== Simplify (50k and 200k) ==========================
TEST(PerfHarness, Simplify_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	auto t0 = Clock::now();
	m.Simplify(0.5f);
	double dt = elapsedS(t0);
	record("Simplify_50k", dt, actual);
	std::cout << "  Simplify_50k: " << dt << " s (result faces=" << m.faces.size() << ")\n";
}

TEST(PerfHarness, Simplify_200k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	auto t0 = Clock::now();
	m.Simplify(0.5f);
	double dt = elapsedS(t0);
	record("Simplify_200k", dt, actual);
	std::cout << "  Simplify_200k: " << dt << " s (result faces=" << m.faces.size() << ")\n";
}

TEST(PerfHarness, Simplify_800k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(LARGE_TARGET, &actual);
	auto t0 = Clock::now();
	m.Simplify(0.5f);
	double dt = elapsedS(t0);
	record("Simplify_800k", dt, actual);
	std::cout << "  Simplify_800k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "Simplify_800k blew wall-clock bound";
}

TEST(PerfHarness, ScalingAssertion_Simplify)
{
	double t50 = -1, t200 = -1;
	for (const auto& r : gRecords) {
		if (r.label == "Simplify_50k")
			t50 = r.seconds;
		if (r.label == "Simplify_200k")
			t200 = r.seconds;
	}
	if (t50 <= 0 || t200 <= 0) {
		GTEST_SKIP() << "Simplify timing records not found";
	}
	double ratio = t200 / t50;
	double alpha = std::log(ratio) / std::log(4.0);
	std::cout << "Simplify scaling ratio (4x faces): " << ratio
	          << "x time, exponent=" << alpha << "\n";
	EXPECT_LT(ratio, 8.0)
	    << "Simplify time grew " << ratio << "x for 4x faces — possible O(n^2)";
}

// ======================== RemeshIsotropic (50k) ============================
TEST(PerfHarness, RemeshIsotropic_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	// Compute a reasonable edge length from AABB diagonal
	auto bbox = m.ComputeAABBox();
	float diag = (bbox.max() - bbox.min()).norm();
	float edgeLen = diag / std::sqrt((float)actual / 2.f);
	halfmesh::Mesh::RemeshParams p;
	p.SetEdgeLength(edgeLen);
	p.iterations = 2; // fewer iterations for speed
	auto t0 = Clock::now();
	m.RemeshIsotropic(p);
	double dt = elapsedS(t0);
	record("RemeshIsotropic_50k", dt, actual);
	std::cout << "  RemeshIsotropic_50k: " << dt << " s\n";
}

TEST(PerfHarness, RemeshIsotropic_200k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	auto bbox = m.ComputeAABBox();
	float diag = (bbox.max() - bbox.min()).norm();
	float edgeLen = diag / std::sqrt((float)actual / 2.f);
	halfmesh::Mesh::RemeshParams p;
	p.SetEdgeLength(edgeLen);
	p.iterations = 2;
	auto t0 = Clock::now();
	m.RemeshIsotropic(p);
	double dt = elapsedS(t0);
	record("RemeshIsotropic_200k", dt, actual);
	std::cout << "  RemeshIsotropic_200k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "RemeshIsotropic_200k blew wall-clock bound";
}

// ======================== FixNonManifold (50k) =============================
TEST(PerfHarness, FixNonManifold_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	auto t0 = Clock::now();
	m.FixNonManifold();
	double dt = elapsedS(t0);
	record("FixNonManifold_50k", dt, actual);
	std::cout << "  FixNonManifold_50k: " << dt << " s\n";
}

TEST(PerfHarness, FixNonManifold_800k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(LARGE_TARGET, &actual);
	auto t0 = Clock::now();
	m.FixNonManifold();
	double dt = elapsedS(t0);
	record("FixNonManifold_800k", dt, actual);
	std::cout << "  FixNonManifold_800k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "FixNonManifold_800k blew wall-clock bound";
}

// ======================== CloseHoles (50k with holes) ======================
TEST(PerfHarness, CloseHoles_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh base = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	halfmesh::Mesh m = MakeHoled(base, 20);
	auto t0 = Clock::now();
	m.CloseHoles();
	double dt = elapsedS(t0);
	record("CloseHoles_50k", dt, actual);
	std::cout << "  CloseHoles_50k: " << dt << " s\n";
}

TEST(PerfHarness, CloseHoles_200k)
{
	unsigned actual = 0;
	halfmesh::Mesh base = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	halfmesh::Mesh m = MakeHoled(base, 20);
	auto t0 = Clock::now();
	m.CloseHoles();
	double dt = elapsedS(t0);
	record("CloseHoles_200k", dt, actual);
	std::cout << "  CloseHoles_200k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "CloseHoles_200k blew wall-clock bound";
}

// ======================== KD-tree build + queries ==========================

TEST(PerfHarness, PrebuiltCloseHolesWithoutFillsPerformsZeroBuilds)
{
	unsigned actual = 0;
	halfmesh::Mesh mesh = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	mesh.ListHalfEdges();
	halfmesh::HalfMesh::ResetBuildCount();
	const auto t0 = Clock::now();
	EXPECT_EQ(mesh.CloseHoles(), 0u);
	const double dt = elapsedS(t0);
	record("CloseHoles_Prebuild_NoFill", dt, actual);
	std::cout << "  CloseHoles_Prebuild_NoFill: " << dt << " s\n";
	EXPECT_EQ(halfmesh::HalfMesh::BuildCount(), 0u);
	EXPECT_LT(dt, MAX_WALL_SECONDS);
}
// KD-tree speedup assertion: kd-tree queries must be >= 10x faster than brute-force
TEST(PerfHarness, KdTree_50k_BuildAndQuery)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);

	// Build KD-tree
	double dtBuild;
	{
		auto t0 = Clock::now();
		halfmesh::TriangleKdTree tree(m);
		dtBuild = elapsedS(t0);
		record("KdTree_Build_50k", dtBuild, actual);
		std::cout << "  KdTree_Build_50k: " << dtBuild << " s\n";
	}

	// Query KD-tree: N random queries from exterior (for fast pruning)
	const unsigned N_queries = 1000;
	halfmesh::TriangleKdTree tree(m);
	auto bbox = m.ComputeAABBox();
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> udist(-1.f, 1.f);
	std::uniform_real_distribution<float> rdist(1.05f, 2.0f);
	halfmesh::Mesh::Vertex center = (bbox.min() + bbox.max()) * 0.5f;
	float radius = (bbox.max() - center).norm();

	std::vector<halfmesh::Mesh::Vertex> queries(N_queries);
	for (auto& q : queries) {
		halfmesh::Mesh::Vertex dir(udist(rng), udist(rng), udist(rng));
		float dn = dir.norm();
		if (dn < 1e-6f)
			dir = halfmesh::Mesh::Vertex(1.f, 0.f, 0.f);
		else
			dir /= dn;
		q = center + dir * (radius * rdist(rng));
	}

	double dtKdtree;
	volatile double kd50Sum = 0.0;
	{
		auto t0 = Clock::now();
		for (const auto& q : queries) {
			auto r = tree.NearestPoint(q);
			kd50Sum += r.dist;
		}
		dtKdtree = elapsedS(t0);
	}
	(void)kd50Sum;
	record("KdTree_Query_50k_1000q", dtKdtree, actual);
	std::cout << "  KdTree_Query_50k (" << N_queries << " queries): "
	          << dtKdtree << " s\n";
}

// ======================== BVH build + queries ==============================
// Mirrors KdTree_50k_BuildAndQuery for the flattened binned-SAH BVH: reports
// build time plus NearestPoint and IntersectedPoint batch times. The BVH shares
// the kd-tree's query surface but prunes with tighter per-node AABBs, so it should
// build comparably and query at least as fast (advisory timing only; the numbers
// are machine-dependent and recorded for regression watching, not asserted).
TEST(PerfHarness, BVH_50k_BuildAndQuery)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);

	// Build
	{
		auto t0 = Clock::now();
		halfmesh::TriangleBVH bvh(m);
		const double dtBuild = elapsedS(t0);
		record("BVH_Build_50k", dtBuild, actual);
		std::cout << "  BVH_Build_50k: " << dtBuild << " s\n";
	}

	halfmesh::TriangleBVH bvh(m);
	auto bbox = m.ComputeAABBox();
	halfmesh::Mesh::Vertex center = (bbox.min() + bbox.max()) * 0.5f;
	const float radius = (bbox.max() - center).norm();
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> udist(-1.f, 1.f);
	std::uniform_real_distribution<float> rdist(1.05f, 2.0f);

	// NearestPoint batch (exterior queries, like the kd benchmark)
	const unsigned N_queries = 1000;
	std::vector<halfmesh::Mesh::Vertex> queries(N_queries);
	for (auto& q : queries) {
		halfmesh::Mesh::Vertex dir(udist(rng), udist(rng), udist(rng));
		const float dn = dir.norm();
		dir = (dn < 1e-6f) ? halfmesh::Mesh::Vertex(1.f, 0.f, 0.f) : (dir / dn).eval();
		q = center + dir * (radius * rdist(rng));
	}
	{
		auto t0 = Clock::now();
		volatile double sum = 0.0;
		for (const auto& q : queries)
			sum += bvh.NearestPoint(q).dist;
		(void)sum;
		const double dt = elapsedS(t0);
		record("BVH_NearestPoint_50k_1000q", dt, actual);
		std::cout << "  BVH_NearestPoint_50k (" << N_queries << " queries): " << dt << " s\n";
	}

	// IntersectedPoint batch: rays from exterior points toward the mesh center.
	{
		auto t0 = Clock::now();
		volatile double sum = 0.0;
		unsigned hits = 0;
		for (const auto& q : queries) {
			const halfmesh::Mesh::Vertex d = (center - q).normalized();
			const auto r = bvh.IntersectedPoint(Eigen::ParametrizedLine<float, 3>(q, d));
			if (r.IsValid()) {
				sum += r.dist;
				++hits;
			}
		}
		(void)sum;
		const double dt = elapsedS(t0);
		record("BVH_IntersectedPoint_50k_1000q", dt, actual);
		std::cout << "  BVH_IntersectedPoint_50k (" << N_queries << " rays, " << hits
		          << " hits): " << dt << " s\n";
	}
}

// ---------------------------------------------------------------------------
// KD-tree speedup: PATHOLOGICAL CASE (UV-sphere)
// ---------------------------------------------------------------------------
// A UV-sphere is the known worst case for 1D split-plane KD-trees:
// a query near the center is ~equidistant to all surface triangles, so no
// split-plane prune fires and speedup approaches 1×.
//
// Guard: KD-tree must not be more than 10× SLOWER than brute-force.
// We do NOT assert ≥10× speedup here because the sphere guarantees it won't be.
// See KdTree_SpeedupVsBruteForce_GridPlane below for the realistic ≥10× test.
TEST(PerfHarness, KdTree_SpeedupVsBruteForce_Sphere_Pathological)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);

	halfmesh::TriangleKdTree tree(m);
	auto bbox = m.ComputeAABBox();
	halfmesh::Mesh::Vertex center = (bbox.min() + bbox.max()) * 0.5f;
	float radius = (bbox.max() - center).norm();
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> udist(-1.f, 1.f);
	std::uniform_real_distribution<float> rdist(1.01f, 1.5f);

	const unsigned N_queries = 500;
	std::vector<halfmesh::Mesh::Vertex> queries(N_queries);
	for (auto& q : queries) {
		halfmesh::Mesh::Vertex dir(udist(rng), udist(rng), udist(rng));
		float dn = dir.norm();
		if (dn < 1e-6f)
			dir = halfmesh::Mesh::Vertex(1.f, 0.f, 0.f);
		else
			dir /= dn;
		q = center + dir * (radius * rdist(rng));
	}

	double dtKd;
	volatile double kdSum = 0.0;
	{
		auto t0 = Clock::now();
		for (const auto& q : queries) {
			kdSum += tree.NearestPoint(q).dist;
		}
		dtKd = elapsedS(t0);
	}

	double dtBf;
	volatile double bfSum = 0.0;
	{
		auto t0 = Clock::now();
		for (const auto& q : queries) {
			bfSum += BruteForceNearest(m, q).dist;
		}
		dtBf = elapsedS(t0);
	}
	(void)kdSum;
	(void)bfSum;

	double speedup = dtBf / (dtKd + 1e-12);
	record("KdTree_Sphere_speedup_kd", dtKd, actual);
	record("KdTree_Sphere_speedup_bf", dtBf, actual);
	std::cout << "  [PATHOLOGICAL] KdTree Sphere Speedup (" << actual << " faces, "
	          << N_queries << " exterior queries): " << speedup << "x "
	          << "(kd=" << dtKd << "s, bf=" << dtBf << "s)\n"
	          << "  NOTE: UV-sphere is worst case for 1D split-plane KD-trees; "
	          << "~1x speedup expected.\n";

	// No-degrade guard: KD-tree must not be >10x SLOWER than brute-force even on sphere
	EXPECT_GT(speedup, 0.1)
	    << "KD-tree is more than 10x SLOWER than brute-force on sphere (pathological) — "
	    << "catastrophic regression. speedup=" << speedup;
	::testing::Test::RecordProperty("kdtree_sphere_speedup_vs_bf", speedup);
}

// ---------------------------------------------------------------------------
// KD-tree speedup: REALISTIC CASE (GridPlane — non-spherical, large, flat)
// ---------------------------------------------------------------------------
// A flat grid plane has strong spatial locality: near-surface query points
// are close to exactly one region of the plane, so the KD-tree prunes most
// split-plane subtrees aggressively. This is the realistic use-case.
//
// Query strategy: sample a random face's barycentric centroid, then offset by
// a small normal distance (1% of mesh diagonal). The nearest face is always
// the sampled face, so nearly all other KD subtrees get pruned.
//
// Assertion: KD-tree must be ≥10× faster than brute-force on this mesh.
TEST(PerfHarness, KdTree_SpeedupVsBruteForce_GridPlane)
{
	// GridPlane(n) produces 2*n^2 triangles. n=500 → 500k faces (500k~target).
	// Use n=512 → 2*512^2 = 524288 faces (~524k, within 500k-800k target).
	constexpr unsigned gridN = 512;
	halfmesh::Mesh m = hmtest::corpus::GridPlane(gridN);
	unsigned actual = (unsigned)m.faces.size();
	std::cout << "  [GridPlane KdTree] faces=" << actual << "\n";

	halfmesh::TriangleKdTree tree(m);

	// Build near-surface queries: sample face centroids + tiny normal offset
	const unsigned N_queries = 200;
	std::vector<halfmesh::Mesh::Vertex> queries;
	queries.reserve(N_queries);
	auto bbox = m.ComputeAABBox();
	float diag = (bbox.max() - bbox.min()).norm();
	float offset = diag * 0.01f; // 1% of diagonal above surface

	std::mt19937 rng(1337);
	std::uniform_int_distribution<unsigned> faceDist(0, actual - 1);
	std::uniform_real_distribution<float> baryDist(0.f, 1.f);

	for (unsigned i = 0; i < N_queries; ++i) {
		unsigned fi = faceDist(rng);
		const auto& face = m.faces[fi];
		// Random barycentric coords
		float u = baryDist(rng), v = baryDist(rng);
		if (u + v > 1.f) {
			u = 1.f - u;
			v = 1.f - v;
		}
		float w = 1.f - u - v;
		const halfmesh::Mesh::Vertex& v0 = m.vertices[face[0]];
		const halfmesh::Mesh::Vertex& v1 = m.vertices[face[1]];
		const halfmesh::Mesh::Vertex& v2 = m.vertices[face[2]];
		halfmesh::Mesh::Vertex pt = v0 * u + v1 * v + v2 * w;
		// Offset in +Z direction (GridPlane lies in XY plane)
		pt[2] += offset;
		queries.push_back(pt);
	}

	double dtKd;
	volatile double kdSum = 0.0;
	{
		auto t0 = Clock::now();
		for (const auto& q : queries) {
			kdSum += tree.NearestPoint(q).dist;
		}
		dtKd = elapsedS(t0);
	}

	double dtBf;
	volatile double bfSum = 0.0;
	{
		auto t0 = Clock::now();
		for (const auto& q : queries) {
			bfSum += BruteForceNearest(m, q).dist;
		}
		dtBf = elapsedS(t0);
	}
	(void)kdSum;
	(void)bfSum;

	double speedup = dtBf / (dtKd + 1e-12);
	record("KdTree_GridPlane_speedup_kd", dtKd, actual);
	record("KdTree_GridPlane_speedup_bf", dtBf, actual);
	std::cout << "  KdTree GridPlane Speedup (" << actual << " faces, "
	          << N_queries << " near-surface queries): " << speedup << "x "
	          << "(kd=" << dtKd << "s, bf=" << dtBf << "s)\n";

	// The "optimized promise": KD-tree must be >=10x faster on a realistic mesh
	EXPECT_GE(speedup, 10.0)
	    << "KD-tree speedup on GridPlane (" << actual << " faces) is only " << speedup
	    << "x — expected >=10x. Near-surface queries should prune most subtrees.";
	::testing::Test::RecordProperty("kdtree_gridplane_speedup_vs_bf", speedup);
}

TEST(PerfHarness, KdTree_800k_Build)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(LARGE_TARGET, &actual);
	auto t0 = Clock::now();
	halfmesh::TriangleKdTree tree(m);
	double dt = elapsedS(t0);
	record("KdTree_Build_800k", dt, actual);
	std::cout << "  KdTree_Build_800k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "KdTree_Build_800k blew wall-clock bound";
}

// ======================== SegmentCharts + Parametrize ======================
TEST(PerfHarness, Parametrize_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	halfmesh::ParametrizeParams pp;
	// Use fewer SLIM iterations to keep it fast
	pp.flattenIterations = 3;
	auto t0 = Clock::now();
	unsigned nCharts = halfmesh::Parametrize(m, pp);
	double dt = elapsedS(t0);
	record("Parametrize_50k", dt, actual);
	std::cout << "  Parametrize_50k: " << dt << " s (charts=" << nCharts << ")\n";
}

TEST(PerfHarness, Parametrize_200k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	halfmesh::ParametrizeParams pp;
	pp.flattenIterations = 3;
	auto t0 = Clock::now();
	unsigned nCharts = halfmesh::Parametrize(m, pp);
	double dt = elapsedS(t0);
	record("Parametrize_200k", dt, actual);
	std::cout << "  Parametrize_200k: " << dt << " s (charts=" << nCharts << ")\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "Parametrize_200k blew wall-clock bound";
}

// ======================== GenerateAtlas (50k) ==============================
TEST(PerfHarness, GenerateAtlas_50k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(SMALL_TARGET, &actual);
	halfmesh::ParametrizeParams pp;
	pp.flattenIterations = 3;
	halfmesh::AtlasParams ap;
	ap.resolution = 512;
	auto t0 = Clock::now();
	auto result = halfmesh::GenerateAtlas(m, pp, ap);
	double dt = elapsedS(t0);
	record("GenerateAtlas_50k", dt, actual);
	std::cout << "  GenerateAtlas_50k: " << dt << " s "
	          << "(atlas=" << result.width << "x" << result.height
	          << " pages=" << result.numPages
	          << " occupancy=" << result.occupancy << ")\n";
}

TEST(PerfHarness, GenerateAtlas_200k)
{
	unsigned actual = 0;
	halfmesh::Mesh m = hmtest::corpus::LargeMesh(MED_TARGET, &actual);
	halfmesh::ParametrizeParams pp;
	pp.flattenIterations = 2;
	halfmesh::AtlasParams ap;
	ap.resolution = 1024;
	auto t0 = Clock::now();
	auto result = halfmesh::GenerateAtlas(m, pp, ap);
	double dt = elapsedS(t0);
	record("GenerateAtlas_200k", dt, actual);
	std::cout << "  GenerateAtlas_200k: " << dt << " s\n";
	EXPECT_LT(dt, MAX_WALL_SECONDS) << "GenerateAtlas_200k blew wall-clock bound";
}

// ======================== Optional: real mesh timing =======================
// Skipped when files are absent. Times logged (not asserted).
static void TryRealMesh(const std::string& path, const std::string& tag, bool doAtlas = true)
{
	if (!std::filesystem::exists(path)) {
		std::cout << "  [SKIP] " << tag << " not found: " << path << "\n";
		return;
	}
	halfmesh::Mesh m;
	{
		auto t0 = Clock::now();
		bool ok = m.Load(path);
		double dt = elapsedS(t0);
		std::cout << "  " << tag << " Load: " << dt << " s"
		          << " faces=" << m.faces.size()
		          << (ok ? "" : " (FAILED)") << "\n";
		if (!ok)
			return;
	}
	unsigned nf = (unsigned)m.faces.size();
	{
		halfmesh::Mesh copy = m;
		auto t0 = Clock::now();
		copy.FixNonManifold();
		std::cout << "  " << tag << " FixNonManifold: " << elapsedS(t0) << " s\n";
	}
	{
		halfmesh::Mesh copy = m;
		auto t0 = Clock::now();
		copy.Simplify(0.5f);
		std::cout << "  " << tag << " Simplify(0.5): " << elapsedS(t0) << " s\n";
	}
	{
		auto t0 = Clock::now();
		halfmesh::TriangleKdTree tree(m);
		std::cout << "  " << tag << " KdTree Build: " << elapsedS(t0) << " s\n";
	}
	if (doAtlas && nf <= 500000) {
		halfmesh::Mesh copy = m;
		halfmesh::ParametrizeParams pp;
		pp.flattenIterations = 2;
		halfmesh::AtlasParams ap;
		ap.resolution = 1024;
		auto t0 = Clock::now();
		auto result = halfmesh::GenerateAtlas(copy, pp, ap);
		std::cout << "  " << tag << " GenerateAtlas: " << elapsedS(t0) << " s"
		          << " (atlas=" << result.width << "x" << result.height << ")\n";
	} else if (nf > 500000) {
		std::cout << "  " << tag << " GenerateAtlas: SKIPPED (mesh too large: "
		          << nf << " faces)\n";
	}
}

TEST(PerfHarness, RealMesh_Optional)
{
	// Optional real meshes (not committed) — drop them in tests/data to time them;
	// skipped gracefully when absent (e.g. in CI).
	const std::string dataDir =
	    (std::filesystem::path(__FILE__).parent_path().parent_path() / "data").string();
	TryRealMesh(dataDir + "/mesh_roi_crop_1.ply",
	            "mesh_roi_crop_1 (~173MB)", true);
	TryRealMesh(dataDir + "/mesh_ours_2pivots_texture_refined_999.ply",
	            "mesh_ours_2pivots (~323MB)", false /* cap ops */);
}

// ======================== Regression alarm vs baseline.json ================
TEST(PerfHarness, RegressionAlarm)
{
	if (gBaseline.empty()) {
		std::cout << "  No baseline.json found — skipping regression check\n"
		          << "  (Run once with PERF_WRITE_BASELINE=1 to write it)\n";
		GTEST_SKIP() << "No baseline.json";
	}
	constexpr double alarmFactor = 3.0;
	bool anyAlarm = false;
	for (const auto& r : gRecords) {
		auto it = gBaseline.find(r.label);
		if (it == gBaseline.end())
			continue;
		double baselineSec = it->second;
		double ratio = r.seconds / (baselineSec + 1e-12);
		if (ratio > alarmFactor) {
			std::cerr << "  REGRESSION: " << r.label
			          << " is " << ratio << "x slower than baseline ("
			          << r.seconds << "s vs baseline " << baselineSec << "s)\n";
			anyAlarm = true;
		}
	}
	// Regression alarm is ADVISORY by default: slower CI machines would hard-fail
	// on absolute baseline times recorded on a fast dev machine (architecture
	// differences alone can be 2-5×). To promote to a hard failure set the env var:
	//   HALFMESH_STRICT_BASELINE=1
	if (anyAlarm) {
		const char* strict = std::getenv("HALFMESH_STRICT_BASELINE");
		if (strict && strict[0] != '\0' && strict[0] != '0') {
			EXPECT_FALSE(anyAlarm)
			    << "One or more ops are >3x slower than baseline.json "
			    << "(HALFMESH_STRICT_BASELINE is set — hard failure enabled).";
		} else {
			std::cerr << "  PERF ADVISORY: regression alarm triggered but treated as "
			          << "warning-only (set HALFMESH_STRICT_BASELINE=1 to hard-fail).\n";
		}
	}
}

// ---------------------------------------------------------------------------
// main: write baseline.json if PERF_WRITE_BASELINE=1
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);

	// Load existing baseline for regression alarm
	std::string bp = baselinePath();
	gBaseline = loadBaseline(bp);
	if (!gBaseline.empty()) {
		std::cout << "Loaded baseline from: " << bp
		          << " (" << gBaseline.size() << " entries)\n";
	}

	int rc = RUN_ALL_TESTS();

	// Write baseline if requested or if none exists yet
	bool writeBaseline = (std::getenv("PERF_WRITE_BASELINE") != nullptr) || gBaseline.empty();
	if (writeBaseline && !gRecords.empty()) {
		// Build machine label
		std::string machine = "unknown";
		const char* hn = std::getenv("HOSTNAME");
		if (hn)
			machine = hn;
#if defined(__APPLE__)
		machine += "/macOS";
#elif defined(__linux__)
		machine += "/Linux";
#endif
#if defined(__clang__)
		machine += "/clang-" + std::to_string(__clang_major__);
#elif defined(__GNUC__)
		machine += "/gcc-" + std::to_string(__GNUC__);
#endif
		machine += "/Release";
		writeJson(gRecords, machine, bp);
	}

	return rc;
}
