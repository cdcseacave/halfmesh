/*
* GoldenDiffTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/golden/GoldenDiffTest.cpp — current-output vs frozen golden regression (§4b).
//
// Runs in the DEFAULT build (HALFMESH_BUILD_TESTS=ON).  For every spec in
// GoldenSpecs.h it recomputes the op on the deterministic input, loads the
// committed frozen snapshot (tests/data/golden/<mesh>__<op>.ply + .json), and
// asserts they agree within the §5 tolerance for that op's CompareMode.
//
// The golden fixtures are committed known-good outputs; regenerate them only when
// behaviour legitimately changes (reviewed, intentional update).

#include <gtest/gtest.h>

#include <halfmesh/Mesh.h>

#include "GoldenIO.h"
#include "GoldenSpecs.h"
#include "Metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <string>

namespace {

using hmtest::golden::CompareMode;
using hmtest::golden::GoldenMetrics;
using hmtest::golden::GoldenSpec;
using hmtest::metrics::CanonicallyEqual;

// Load a golden fixture pair; returns false (with a gtest failure) if missing —
// fixtures are committed, so a missing one is a real error, not a skip.
bool LoadGolden(const GoldenSpec& spec, halfmesh::Mesh& mesh, GoldenMetrics& m)
{
	const std::string dir = hmtest::golden::GoldenDir();
	const std::string stem = hmtest::golden::FixtureStem(spec.meshName, spec.opName);
	const std::string ply = dir + "/" + stem + ".ply";
	const std::string json = dir + "/" + stem + ".json";
	if (!mesh.Load(ply)) {
		ADD_FAILURE() << "missing golden mesh fixture: " << ply
		              << "  (regenerate: see tests/data/golden/README.md)";
		return false;
	}
	if (!hmtest::golden::ReadMetricsJson(json, m)) {
		ADD_FAILURE() << "missing/invalid golden metrics fixture: " << json;
		return false;
	}
	return true;
}

constexpr float POS_TOL = 1e-5f; // canonical vertex position tolerance
constexpr double AREA_REL = 1e-4; // accumulated-quantity relative tolerance (§5)

void ExpectMetricsClose(const halfmesh::Mesh& result,
                        const halfmesh::Mesh& input,
                        const GoldenMetrics& golden,
                        uint32_t resultCount,
                        bool exactCounts)
{
	const GoldenMetrics pm =
	    hmtest::golden::ComputeGoldenMetrics(result, input, resultCount);

	if (exactCounts) {
		EXPECT_EQ(pm.numVertices, golden.numVertices) << "vertex count";
		EXPECT_EQ(pm.numFaces, golden.numFaces) << "face count";
		EXPECT_EQ(pm.opCount, golden.opCount) << "op count";
	}
	// Surface area: relative ε (accumulated quantity). Float/build-flag
	// sensitive ops (tolerant mode) legitimately differ by a few 1e-4 — a
	// config-drifted cascade keeps different vertices — so they get a wider
	// bound that still catches real damage (a lost region is >> 0.1%).
	const double areaRel = exactCounts ? AREA_REL : 10.0 * AREA_REL;
	const double areaDen = std::max(1e-12, std::abs(golden.surfaceArea));
	EXPECT_LT(std::abs(pm.surfaceArea - golden.surfaceArea) / areaDen, areaRel)
	    << "surface area (" << pm.surfaceArea << " vs " << golden.surfaceArea << ")";
	// Bounding box: absolute ε scaled to mesh size.
	double diag = 0.0;
	for (int k = 0; k < 3; ++k)
		diag += (golden.bboxMax[k] - golden.bboxMin[k]) * (golden.bboxMax[k] - golden.bboxMin[k]);
	diag = std::sqrt(std::max(1e-12, diag));
	for (int k = 0; k < 3; ++k) {
		EXPECT_LT(std::abs(pm.bboxMin[k] - golden.bboxMin[k]) / diag, 1e-4) << "bbox_min[" << k << "]";
		EXPECT_LT(std::abs(pm.bboxMax[k] - golden.bboxMax[k]) / diag, 1e-4) << "bbox_max[" << k << "]";
	}
}

// Signed relative change (new vs old) as a percentage string, guarded against a
// zero baseline (relative % is undefined there — report "n/a" instead).
std::string RelChangePct(double oldv, double newv)
{
	if (std::abs(oldv) < 1e-12)
		return std::abs(newv) < 1e-12 ? "+0.000%" : "n/a (old=0)";
	char buf[32];
	std::snprintf(buf, sizeof(buf), "%+.3f%%", (newv - oldv) / std::abs(oldv) * 100.0);
	return buf;
}

// Regen delta report: one grep-able "[golden-regen]" line per metric (old -> new
// + relative %), so a regeneration can be quantified and pasted into its commit
// message (owner policy — every regen is justified with evidence). A Hausdorff
// increase of >10% raises a clearly-marked QUALITY REGRESSION warning; the regen
// still proceeds (a human judges whether the change is intended).
void PrintRegenDelta(const std::string& stem, const GoldenMetrics& o, const GoldenMetrics& n)
{
	auto row = [&](const char* name, double ov, double nv) {
		std::cout << "[golden-regen] " << stem << "  " << name << ": " << ov << " -> "
		          << nv << "  (" << RelChangePct(ov, nv) << ")\n";
	};
	row("num_vertices", o.numVertices, n.numVertices);
	row("num_faces", o.numFaces, n.numFaces);
	row("op_count", o.opCount, n.opCount);
	row("surface_area", o.surfaceArea, n.surfaceArea);
	row("hausdorff_to_input", o.hausdorffToInput, n.hausdorffToInput);
	row("edge_len_min", o.edgeLenMin, n.edgeLenMin);
	row("edge_len_max", o.edgeLenMax, n.edgeLenMax);
	row("edge_len_mean", o.edgeLenMean, n.edgeLenMean);

	// Bounding box: summarize as the largest absolute per-component shift over
	// both corners (an absolute delta — a relative % on a coordinate is not
	// meaningful, e.g. a component that crosses zero).
	double bboxDelta = 0.0;
	for (int k = 0; k < 3; ++k) {
		bboxDelta = std::max(bboxDelta, std::abs(n.bboxMin[k] - o.bboxMin[k]));
		bboxDelta = std::max(bboxDelta, std::abs(n.bboxMax[k] - o.bboxMax[k]));
	}
	std::cout << "[golden-regen] " << stem << "  bbox: max abs component delta = "
	          << bboxDelta << "\n";

	if (n.hausdorffToInput > o.hausdorffToInput * 1.10)
		std::cout << "[golden-regen] " << stem
		          << "  *** QUALITY REGRESSION: hausdorff_to_input rose >10% ("
		          << o.hausdorffToInput << " -> " << n.hausdorffToInput
		          << "); review before committing ***\n";
}

} // namespace

// Pretty-print a spec (keeps gtest / ctest labels readable instead of dumping
// the raw GoldenSpec bytes, which contain std::function objects).
namespace hmtest {
namespace golden {
void PrintTo(const GoldenSpec& s, std::ostream* os)
{
	*os << s.meshName << "__" << s.opName;
}
} // namespace golden
} // namespace hmtest

// One parameterized test per (mesh, op) spec.
class GoldenDiff : public ::testing::TestWithParam<GoldenSpec>
{
};

TEST_P(GoldenDiff, MatchesFrozenGolden)
{
	const GoldenSpec& spec = GetParam();
	SCOPED_TRACE(spec.meshName + "__" + spec.opName);

	halfmesh::Mesh goldenMesh;
	GoldenMetrics goldenMetrics;
	ASSERT_TRUE(LoadGolden(spec, goldenMesh, goldenMetrics));

	const halfmesh::Mesh input = spec.makeInput();
	halfmesh::Mesh result = input;
	const uint32_t resultCount = spec.runOp(result);

	if (spec.mode == CompareMode::CANONICAL_EXACT) {
		// Deterministic, build-flag-stable op: exact canonical mesh equality.
		EXPECT_EQ(resultCount, goldenMetrics.opCount) << "op count";
		EXPECT_TRUE(CanonicallyEqual(result, goldenMesh, POS_TOL))
		    << "result != frozen golden (canonical)";
		ExpectMetricsClose(result, input, goldenMetrics, resultCount, /*exactCounts=*/true);
	} else {
		// Tolerant (float/build-flag sensitive, e.g. Remesh): element counts
		// within a per-op-contract bound + Hausdorff(result, golden) ≈ 0 relative
		// to mesh scale.
		// Count tolerance follows what the OP guarantees. Ratio-mode Simplify and
		// Remesh drive toward an explicit size target, so counts are pinned tightly
		// (5%; in practice they land identical across build flags). Min-edge mode's
		// stopping criterion is a LENGTH threshold — "collapse every edge shorter
		// than L" — so its final count is emergent from tie ordering and legitimately
		// spreads wider across ISAs (measured 5.4% default-vs-x86-64-v3 on the
		// re-baselined fixture); 10% bounds that spread while still catching a
		// runaway (e.g. re-admission loss would leave >>10% extra faces).
		const double countRelTarget = 0.05; // count-targeted ops (ratio Simplify, Remesh)
		const double countRelEmergent = 0.10; // length-criterion ops (SimplifyMinEdge)
		const double countRel =
		    spec.opName.find("MinEdge") != std::string::npos ? countRelEmergent : countRelTarget;
		const double fRel =
		    std::abs(double(result.faces.size()) - double(goldenMesh.faces.size())) / std::max<double>(1.0, double(goldenMesh.faces.size()));
		const double vRel =
		    std::abs(double(result.vertices.size()) - double(goldenMesh.vertices.size())) / std::max<double>(1.0, double(goldenMesh.vertices.size()));
		EXPECT_LT(fRel, countRel) << "face count rel-diff";
		EXPECT_LT(vRel, countRel) << "vertex count rel-diff";

		double diag = 0.0;
		for (int k = 0; k < 3; ++k)
			diag += (goldenMetrics.bboxMax[k] - goldenMetrics.bboxMin[k]) * (goldenMetrics.bboxMax[k] - goldenMetrics.bboxMin[k]);
		diag = std::sqrt(std::max(1e-12, diag));
		// Gross-divergence bound only: decimation keeps DIFFERENT original
		// vertices per collapse cascade, so a config-drifted (equally valid)
		// result sits a few 1e-3 of the diagonal from the frozen one (Remesh,
		// which converges to the same limit surface, stays far below this).
		const double h =
		    hmtest::metrics::ComputeDistanceKdTree(result, goldenMesh).hausdorffSymmetric;
		EXPECT_LT(h / diag, 1e-2) << "Hausdorff(result, golden) relative to bbox diagonal";
		// The real regression net: quality parity vs the INPUT. Whatever
		// collapse/relocation cascade this build picked, it may not approximate
		// the input meaningfully worse than the frozen reference did.
		const double hIn =
		    hmtest::metrics::ComputeDistanceKdTree(result, input).hausdorffSymmetric;
		EXPECT_LT(hIn, std::max(goldenMetrics.hausdorffToInput * 1.5, 1e-3 * diag))
		    << "result approximates the input worse than the frozen reference ("
		    << hIn << " vs golden " << goldenMetrics.hausdorffToInput << ")";

		// Isotropy ratchet (Remesh only). Uniform edge length IS
		// RemeshIsotropic's contract, and the Hausdorff parity net above cannot
		// see isotropy loss: a mesh can still hug the input surface while its
		// edge-length distribution drifts non-uniform. So pin the result's edge
		// stats to the frozen baseline's. Bounds are baseline-relative with
		// build-flag headroom analogous to the 1.5x Hausdorff net — the result
		// recomputes ~identically today (edge stats ≈ baseline), so they pass
		// now and trip only on a genuine isotropy regression. (TESTING.md §5:
		// every tolerance named and justified, no silent magic numbers.)
		if (spec.opName.find("Remesh") != std::string::npos) {
			// Reuse the shared metrics path (edge stats computed exactly as the
			// frozen sidecar was) for an apples-to-apples comparison.
			const GoldenMetrics pm =
			    hmtest::golden::ComputeGoldenMetrics(result, input, resultCount);
			// Mean edge length is the target the op converges to → tight bound.
			constexpr double edgeMeanRel = 0.10;
			// Min/max are distribution tails (single-edge outliers) → looser bound.
			constexpr double edgeExtremeRel = 0.25;
			const double meanDen = std::max(1e-12, goldenMetrics.edgeLenMean);
			EXPECT_LT(std::abs(pm.edgeLenMean - goldenMetrics.edgeLenMean) / meanDen, edgeMeanRel)
			    << "remesh edge_len_mean drifted from baseline (" << pm.edgeLenMean
			    << " vs " << goldenMetrics.edgeLenMean << ")";
			const double minDen = std::max(1e-12, goldenMetrics.edgeLenMin);
			EXPECT_LT(std::abs(pm.edgeLenMin - goldenMetrics.edgeLenMin) / minDen, edgeExtremeRel)
			    << "remesh edge_len_min drifted from baseline (" << pm.edgeLenMin
			    << " vs " << goldenMetrics.edgeLenMin << ")";
			const double maxDen = std::max(1e-12, goldenMetrics.edgeLenMax);
			EXPECT_LT(std::abs(pm.edgeLenMax - goldenMetrics.edgeLenMax) / maxDen, edgeExtremeRel)
			    << "remesh edge_len_max drifted from baseline (" << pm.edgeLenMax
			    << " vs " << goldenMetrics.edgeLenMax << ")";
		}

		ExpectMetricsClose(result, input, goldenMetrics, resultCount, /*exactCounts=*/false);
	}
}

// Fixture (re)generation — opt-in via the GOLDEN_REGEN env var (the regen helper
// referenced in tests/data/golden/README.md). Regenerate ONLY when behaviour
// legitimately changes; review the committed diff. GOLDEN_REGEN may be "1"/"all"
// to regenerate every spec, or a substring to regenerate matching specs only
// (e.g. GOLDEN_REGEN=RemeshIsotropic). Skipped otherwise.
TEST_P(GoldenDiff, RegenerateFixtureIfRequested)
{
	const char* regen = std::getenv("GOLDEN_REGEN");
	if (regen == nullptr)
		GTEST_SKIP() << "set GOLDEN_REGEN=1|all|<substring> to regenerate fixtures";
	const GoldenSpec& spec = GetParam();
	const std::string stem = hmtest::golden::FixtureStem(spec.meshName, spec.opName);
	const std::string filter = regen;
	if (filter != "1" && filter != "all" && stem.find(filter) == std::string::npos)
		GTEST_SKIP() << "GOLDEN_REGEN='" << filter << "' does not match " << stem;

	const halfmesh::Mesh input = spec.makeInput();
	halfmesh::Mesh result = input;
	const uint32_t count = spec.runOp(result);

	const std::string dir = hmtest::golden::GoldenDir();
	const std::string plyPath = dir + "/" + stem + ".ply";
	const std::string jsonPath = dir + "/" + stem + ".json";

	// Compute the new metrics up front so we can diff against the previous
	// fixture BEFORE overwriting it. Owner policy: every regeneration must be
	// justified with quantified evidence — this delta report is pasted into the
	// regen commit message (see tests/data/golden/README.md, docs/TESTING.md §4).
	const GoldenMetrics m = hmtest::golden::ComputeGoldenMetrics(result, input, count);
	GoldenMetrics prev;
	if (hmtest::golden::ReadMetricsJson(jsonPath, prev))
		PrintRegenDelta(stem, prev, m);
	else
		std::cout << "[golden-regen] " << stem << ": new fixture (no previous .json)\n";

	ASSERT_TRUE(result.Save(plyPath)) << "write " << stem << ".ply";
	ASSERT_TRUE(hmtest::golden::WriteMetricsJson(jsonPath, m))
	    << "write " << stem << ".json";
	std::cout << "[golden-regen] wrote " << stem << " (V=" << result.vertices.size()
	          << " F=" << result.faces.size() << ")\n";
}

// Determinism backstop in the default build too (run-twice identical output).
TEST_P(GoldenDiff, OpDeterministic)
{
	const GoldenSpec& spec = GetParam();
	SCOPED_TRACE(spec.meshName + "__" + spec.opName);

	const halfmesh::Mesh input = spec.makeInput();
	halfmesh::Mesh a = input, b = input;
	const uint32_t ca = spec.runOp(a);
	const uint32_t cb = spec.runOp(b);
	EXPECT_EQ(ca, cb) << "run-twice op count";
	EXPECT_TRUE(CanonicallyEqual(a, b, 0.0f)) << "run-twice output determinism";
}

INSTANTIATE_TEST_SUITE_P(
    GoldenCorpus, GoldenDiff,
    ::testing::ValuesIn(hmtest::golden::GoldenCorpus()),
    [](const ::testing::TestParamInfo<GoldenSpec>& info) {
	    return info.param.meshName + "_" + info.param.opName;
    });
