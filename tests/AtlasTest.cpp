/*
* AtlasTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/AtlasTest.cpp — property-based tests for NormalizeChartDensity.
//
// Tests:
//   1. UniformDensityInvariant: two planar patches of different 3-D area each
//      assigned its own chart → after normalisation both have the same
//      sqrt(uvArea / worldArea).
//   2. ExplicitTexelsPerUnit: density matches the requested value per chart.
//   3. AutoResolutionArea: total UV area ≈ resolution² (within 1 %).
//   4. MeshPlyEndToEnd: segment+flatten+normalize on mesh.ply; density
//      variance across charts is small and all UVs are finite.

#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>
#include <halfmesh/Mesh.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/RectPacking.h>

// Internal Module A<->B bridge header (src/ on this target's include path — see
// tests/CMakeLists.txt): brings in detail::AtlasSegmentStats + the cache-aware
// detail::SegmentCharts overload the stats test below calls directly.
#include "ChartFlattenCache.h"

// Test mesh corpus (hmtest::corpus): GridPlane feeds the carve-seam test below
// (halfmesh_corpus is linked for this target — see tests/CMakeLists.txt).
#include "Corpus.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace halfmesh {
namespace detail {
// Test seam: 3-arg fold verdict (defined in src/Parametrize.cpp).
bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params);
// Test seam (§6.1, defined in src/AtlasCharting.cpp): delegate to
// CarveFailureRegion with a default-params SegmentState built from `mesh`.
// Mirrors the ComputeSegmentationSeeds seam pattern below.
bool CarveFailureRegionForTest(Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                               const std::vector<Mesh::FIndex>& badFaces, unsigned rings,
                               std::vector<Mesh::FIndex>& A, std::vector<Mesh::FIndex>& B);
} // namespace detail
} // namespace halfmesh

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string TestMeshPath()
{
	return (std::filesystem::path(__FILE__).parent_path()
	        / "data" / "mesh.ply")
	    .string();
}

// Signed 2-D area of a triangle (helper — same formula as in src/AtlasCharting.cpp).
static float SignedDoubleArea2D(const Mesh::TexCoord& a,
                                const Mesh::TexCoord& b,
                                const Mesh::TexCoord& c)
{
	return (b.x() - a.x()) * (c.y() - a.y())
	       - (c.x() - a.x()) * (b.y() - a.y());
}

// Per-chart UV area and world area.
struct ChartMeasure
{
	double worldArea = 0.0;
	double uvArea = 0.0;
};

std::vector<ChartMeasure> MeasureCharts(const Mesh& mesh,
                                        const std::vector<unsigned>& faceChart,
                                        unsigned numCharts)
{
	std::vector<ChartMeasure> m(numCharts);
	for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;
		const Mesh::Face& f = mesh.faces[fi];
		const double da3d = (mesh.vertices[f[1]] - mesh.vertices[f[0]])
		                        .cross(mesh.vertices[f[2]] - mesh.vertices[f[0]])
		                        .norm();
		m[cid].worldArea += 0.5 * da3d;

		const Mesh::TexCoord& t0 = mesh.faceTexcoords[fi * 3 + 0];
		const Mesh::TexCoord& t1 = mesh.faceTexcoords[fi * 3 + 1];
		const Mesh::TexCoord& t2 = mesh.faceTexcoords[fi * 3 + 2];
		const float da2d = std::abs(SignedDoubleArea2D(t0, t1, t2));
		m[cid].uvArea += 0.5 * static_cast<double>(da2d);
	}
	return m;
}

// Build two axis-aligned planar patches sharing a contiguous vertex / face list.
// Patch 0: unit square (1×1), 2 triangles.
// Patch 1: 3×3 square, 2 triangles.
// Each patch is given flat local UVs that mirror the 3-D layout but at a
// DIFFERENT scale so density is not initially equal — this is the pre-condition
// for the normalisation test.
static void BuildTwoPlanarPatches(Mesh& mesh,
                                  std::vector<unsigned>& faceChart,
                                  unsigned& numCharts)
{
	// Patch 0 vertices (z=0, unit square).
	mesh.vertices = {
	    {0.f, 0.f, 0.f},
	    {1.f, 0.f, 0.f},
	    {1.f, 1.f, 0.f},
	    {0.f, 1.f, 0.f},
	    // Patch 1 (3×3, offset so not overlapping).
	    {5.f, 0.f, 0.f},
	    {8.f, 0.f, 0.f},
	    {8.f, 3.f, 0.f},
	    {5.f, 3.f, 0.f},
	};
	mesh.faces = {
	    // Patch 0 — 2 triangles
	    {0, 1, 2},
	    {0, 2, 3},
	    // Patch 1 — 2 triangles
	    {4, 5, 6},
	    {4, 6, 7},
	};
	numCharts = 2;
	faceChart = {0, 0, 1, 1};

	// Set up local UVs: patch 0 at scale 2 (UV side = 2 while world side = 1),
	// patch 1 at scale 0.5 (UV side = 1.5 while world side = 3).
	// After normalisation both must reach the same density.
	mesh.faceTexcoords.resize(mesh.faces.size() * 3);

	// Patch 0 face 0: vertices 0,1,2 → UV (0,0),(2,0),(2,2)
	mesh.faceTexcoords[0] = {0.f, 0.f};
	mesh.faceTexcoords[1] = {2.f, 0.f};
	mesh.faceTexcoords[2] = {2.f, 2.f};
	// Patch 0 face 1: vertices 0,2,3 → UV (0,0),(2,2),(0,2)
	mesh.faceTexcoords[3] = {0.f, 0.f};
	mesh.faceTexcoords[4] = {2.f, 2.f};
	mesh.faceTexcoords[5] = {0.f, 2.f};

	// Patch 1 face 0: vertices 4,5,6 → UV (0,0),(1.5,0),(1.5,1.5)
	mesh.faceTexcoords[6] = {0.f, 0.f};
	mesh.faceTexcoords[7] = {1.5f, 0.f};
	mesh.faceTexcoords[8] = {1.5f, 1.5f};
	// Patch 1 face 1: vertices 4,6,7 → UV (0,0),(1.5,1.5),(0,1.5)
	mesh.faceTexcoords[9] = {0.f, 0.f};
	mesh.faceTexcoords[10] = {1.5f, 1.5f};
	mesh.faceTexcoords[11] = {0.f, 1.5f};
}

// ---------------------------------------------------------------------------
// Test 1 — Uniform-density invariant.
// ---------------------------------------------------------------------------
TEST(NormalizeChartDensity, UniformDensityInvariant)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildTwoPlanarPatches(mesh, faceChart, numCharts);

	AtlasParams params;
	params.texelsPerUnit = 10.f; // explicit density so we can predict results

	const float returnedDensity =
	    NormalizeChartDensity(mesh, faceChart, numCharts, params);
	ASSERT_NEAR(returnedDensity, 10.f, 1e-4f);

	auto m = MeasureCharts(mesh, faceChart, numCharts);

	// The uniform-density invariant: sqrt(uvArea / worldArea) == D for each chart.
	for (unsigned c = 0; c < numCharts; ++c) {
		ASSERT_GT(m[c].worldArea, 0.0) << "chart " << c << " degenerate";
		ASSERT_GT(m[c].uvArea, 0.0) << "chart " << c << " zero UV area";

		const double measuredDensity =
		    std::sqrt(m[c].uvArea / m[c].worldArea);
		EXPECT_NEAR(measuredDensity,
		            static_cast<double>(returnedDensity),
		            1e-4)
		    << "chart " << c << " density mismatch";
	}

	// Also verify the two charts have equal density relative to each other.
	const double d0 = std::sqrt(m[0].uvArea / m[0].worldArea);
	const double d1 = std::sqrt(m[1].uvArea / m[1].worldArea);
	EXPECT_NEAR(d0, d1, 1e-4) << "Density is NOT uniform across charts";
}

// ---------------------------------------------------------------------------
// Test 2 — Explicit texelsPerUnit: each chart's measured density ≈ d.
// ---------------------------------------------------------------------------
TEST(NormalizeChartDensity, ExplicitTexelsPerUnit)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildTwoPlanarPatches(mesh, faceChart, numCharts);

	const float targetDensity = 5.f;
	AtlasParams params;
	params.texelsPerUnit = targetDensity;

	const float d = NormalizeChartDensity(mesh, faceChart, numCharts, params);
	ASSERT_NEAR(d, targetDensity, 1e-4f);

	auto m = MeasureCharts(mesh, faceChart, numCharts);
	for (unsigned c = 0; c < numCharts; ++c) {
		if (m[c].worldArea <= 0.0 || m[c].uvArea <= 0.0)
			continue;
		const double measured = std::sqrt(m[c].uvArea / m[c].worldArea);
		EXPECT_NEAR(measured, static_cast<double>(targetDensity), 5e-4)
		    << "chart " << c;
	}
}

// ---------------------------------------------------------------------------
// Test 3 — Auto resolution: total UV area ≈ resolution².
// ---------------------------------------------------------------------------
TEST(NormalizeChartDensity, AutoResolutionArea)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildTwoPlanarPatches(mesh, faceChart, numCharts);

	const unsigned resolution = 512;
	AtlasParams params;
	params.texelsPerUnit = 0.f; // auto
	params.resolution = resolution;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);

	auto m = MeasureCharts(mesh, faceChart, numCharts);

	double totalUvArea = 0.0;
	for (unsigned c = 0; c < numCharts; ++c)
		totalUvArea += m[c].uvArea;

	// Total UV area should be resolution² (within 1 %).
	const double expected = static_cast<double>(resolution) * static_cast<double>(resolution);
	EXPECT_NEAR(totalUvArea, expected, expected * 0.01)
	    << "Total UV area should be ~resolution² with auto-density";
}

// ---------------------------------------------------------------------------
// Test 4 — mesh.ply end-to-end: segment → flatten → normalize.
// ---------------------------------------------------------------------------
TEST(NormalizeChartDensity, MeshPlyEndToEnd)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path)) {
		GTEST_SKIP() << "mesh.ply not found at " << path;
	}

	Mesh mesh;
	ASSERT_TRUE(mesh.Load(path)) << "Failed to load " << path;
	ASSERT_FALSE(mesh.faces.empty());

	ParametrizeParams pparams;
	pparams.flattenIterations = 3; // fast for tests
	std::vector<unsigned> faceChart;
	const unsigned numCharts = SegmentCharts(mesh, pparams, faceChart);
	ASSERT_GE(numCharts, 1u);

	// Flatten.
	ParametrizeCharts(mesh, faceChart, numCharts, pparams);
	ASSERT_EQ(mesh.faceTexcoords.size(), mesh.faces.size() * 3);

	// Normalize density.
	AtlasParams aparams;
	aparams.texelsPerUnit = 0.f;
	aparams.resolution = 1024;
	const float density = NormalizeChartDensity(mesh, faceChart, numCharts, aparams);
	ASSERT_GT(density, 0.f) << "Returned density must be positive";

	// All UVs must be finite.
	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x())) << "Non-finite UV.x";
		ASSERT_TRUE(std::isfinite(uv.y())) << "Non-finite UV.y";
	}

	// Measure per-chart densities.
	auto m = MeasureCharts(mesh, faceChart, numCharts);

	std::vector<double> densities;
	densities.reserve(numCharts);
	for (unsigned c = 0; c < numCharts; ++c) {
		if (m[c].worldArea <= 0.0 || m[c].uvArea <= 0.0)
			continue;
		densities.push_back(std::sqrt(m[c].uvArea / m[c].worldArea));
	}
	ASSERT_FALSE(densities.empty());

	const double meanD =
	    std::accumulate(densities.begin(), densities.end(), 0.0) / static_cast<double>(densities.size());

	double variance = 0.0;
	for (double d : densities)
		variance += (d - meanD) * (d - meanD);
	variance /= static_cast<double>(densities.size());
	const double stddev = std::sqrt(variance);
	const double relStddev = (meanD > 0.0) ? (stddev / meanD) : 0.0;

	std::printf("[atlas_test] num_charts=%u  density=%.4f  "
	            "stddev=%.6f  rel_stddev=%.4f%%\n",
	            numCharts, density, stddev, relStddev * 100.0);

	// Density variance across charts should be very small (< 1 % relative).
	EXPECT_LT(relStddev, 0.01)
	    << "Density relative std-dev too large: " << relStddev * 100.0 << "%";
}

// ---------------------------------------------------------------------------
// Helpers for PackAtlas / GenerateAtlas tests
// ---------------------------------------------------------------------------

// Compute per-chart bounding rect (texel space) from faceTexcoords.
struct BRect
{
	float x0, y0, x1, y1;
	unsigned page;
};

static std::vector<BRect> ChartBBoxes(
    const Mesh& mesh,
    const std::vector<unsigned>& faceChart,
    unsigned numCharts,
    const std::vector<unsigned>& chartPage,
    unsigned pageW, unsigned pageH)
{
	std::vector<BRect> rects(numCharts,
	                         {std::numeric_limits<float>::max(),
	                          std::numeric_limits<float>::max(),
	                          std::numeric_limits<float>::lowest(),
	                          std::numeric_limits<float>::lowest(), 0u});
	for (unsigned c = 0; c < numCharts; ++c)
		rects[c].page = chartPage[c];

	for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;
		BRect& r = rects[cid];
		for (int k = 0; k < 3; ++k) {
			const Mesh::TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
			// Convert from normalized [0,1] back to texel coords for easier math.
			const float tx = uv.x() * static_cast<float>(pageW);
			const float ty = uv.y() * static_cast<float>(pageH);
			r.x0 = std::min(r.x0, tx);
			r.y0 = std::min(r.y0, ty);
			r.x1 = std::max(r.x1, tx);
			r.y1 = std::max(r.y1, ty);
		}
	}
	return rects;
}

// Check that no two charts on the same page have overlapping bounding rects
// (with a small epsilon to allow touching edges).
static bool BoundingRectsDisjoint(const std::vector<BRect>& rects, unsigned numCharts)
{
	constexpr float eps = 1e-3f;
	for (unsigned i = 0; i < numCharts; ++i) {
		const BRect& a = rects[i];
		if (a.x1 <= a.x0 || a.y1 <= a.y0)
			continue; // skip degenerate
		for (unsigned j = i + 1; j < numCharts; ++j) {
			const BRect& b = rects[j];
			if (a.page != b.page)
				continue; // different pages — ok
			if (b.x1 <= b.x0 || b.y1 <= b.y0)
				continue;

			// AABB overlap test.
			const bool overlap =
			    a.x0 < b.x1 - eps && a.x1 > b.x0 + eps && a.y0 < b.y1 - eps && a.y1 > b.y0 + eps;
			if (overlap)
				return false;
		}
	}
	return true;
}

// Build a mesh with `n` synthetic axis-aligned square charts of varying sizes.
// Each chart = 2 triangles; chart c has world side 1+c so areas differ.
// faceTexcoords are set to the unit square (NormalizeChartDensity will rescale).
static void BuildSyntheticCharts(Mesh& mesh,
                                 std::vector<unsigned>& faceChart,
                                 unsigned& numCharts,
                                 unsigned n)
{
	numCharts = n;
	faceChart.clear();
	mesh.vertices.clear();
	mesh.faces.clear();
	mesh.faceTexcoords.clear();

	float offset = 0.f;
	for (unsigned c = 0; c < n; ++c) {
		const float side = static_cast<float>(c + 1); // world side
		const unsigned base = static_cast<unsigned>(mesh.vertices.size());

		mesh.vertices.push_back({offset, 0.f, 0.f});
		mesh.vertices.push_back({offset + side, 0.f, 0.f});
		mesh.vertices.push_back({offset + side, side, 0.f});
		mesh.vertices.push_back({offset, side, 0.f});

		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});

		faceChart.push_back(c);
		faceChart.push_back(c);

		// Local UVs: unit square (NormalizeChartDensity will fix the scale).
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});

		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 1.f});

		offset += side + 2.f; // avoid overlapping world positions
	}
}

// Build `n` UNIFORM tiny square charts (side `side` world units) plus `nBig`
// large ones (side 40·side). Mimics the production regime: a huge tail of
// near-identical tiny charts and a small head of large ones.
static void BuildMixedCharts(Mesh& mesh,
                             std::vector<unsigned>& faceChart,
                             unsigned& numCharts,
                             unsigned n, unsigned nBig, float side)
{
	numCharts = n + nBig;
	faceChart.clear();
	mesh.vertices.clear();
	mesh.faces.clear();
	mesh.faceTexcoords.clear();
	float offset = 0.f;
	for (unsigned c = 0; c < numCharts; ++c) {
		const float s = (c < nBig) ? 40.f * side : side;
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({offset, 0.f, 0.f});
		mesh.vertices.push_back({offset + s, 0.f, 0.f});
		mesh.vertices.push_back({offset + s, s, 0.f});
		mesh.vertices.push_back({offset, s, 0.f});
		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(c);
		faceChart.push_back(c);
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 1.f});
		offset += s + 2.f;
	}
}

// ---------------------------------------------------------------------------
// Two-tier packing: a production-shaped input (2000 tiny + 4 large charts)
// must place every chart overlap-free at sane occupancy through the shelf
// tier. This is correctness/regression coverage of the two-tier path
// (disjointness, in-bounds UVs, occupancy floor) — the quadratic blowup the
// two-tier split targets only manifests at production scale (100k+ charts,
// measured: 78% of a 3h45m unwrap) and is checked by the release perf runs,
// not by this unit-scale fixture.
// ---------------------------------------------------------------------------
TEST(PackAtlas, TwoTierManyTinyChartsDisjointAndDense)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	const unsigned nTiny = 2000u, nBig = 4u;
	BuildMixedCharts(mesh, faceChart, numCharts, nTiny, nBig, 1.f);

	AtlasParams params;
	params.resolution = 1024;
	params.padding = 2;
	params.allowRotation = true;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	ASSERT_EQ(res.chartPage.size(), numCharts);
	for (unsigned c = 0; c < numCharts; ++c)
		EXPECT_LT(res.chartPage[c], res.numPages);
	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x()));
		ASSERT_TRUE(std::isfinite(uv.y()));
		EXPECT_GE(uv.x(), 0.f - 1e-4f);
		EXPECT_LE(uv.x(), 1.f + 1e-4f);
		EXPECT_GE(uv.y(), 0.f - 1e-4f);
		EXPECT_LE(uv.y(), 1.f + 1e-4f);
	}
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts)) << "two charts overlap in the atlas";
	EXPECT_GT(res.occupancy, 0.4f) << "shelf tier wastes too much: " << res.occupancy;

	// Fixture-validity tripwire: BRect's size survives packing unchanged (PackRects
	// places each chart at its own unpadded {cr.w, cr.h} — see AtlasPacking.cpp step 5 —
	// rotation only swaps which axis holds w vs h), so max(rect width, rect height) + 2*pad
	// reproduces exactly the padded long side AtlasPacking.cpp's `tierThreshold` tests.
	// Nothing else here proves the 2000 tiny charts actually took the shelf path instead of
	// the skyline (head) tier; density-normalization drift could silently push them all
	// above threshold and this test would keep passing without ever exercising the shelf
	// tier it exists to cover.
	const auto paddedLongSide = [&](const BRect& r) {
		return std::max(r.x1 - r.x0, r.y1 - r.y0) + 2.f * static_cast<float>(params.padding);
	};
	const float tierThreshold = static_cast<float>(res.width) / 32.f;
	unsigned belowThreshold = 0;
	for (unsigned c = nBig; c < numCharts; ++c)
		if (paddedLongSide(rects[c]) < tierThreshold)
			++belowThreshold;
	EXPECT_GT(belowThreshold, static_cast<unsigned>(0.9f * static_cast<float>(nTiny)))
	    << "fixture did not exercise the shelf tier: only " << belowThreshold << "/" << nTiny
	    << " tiny charts fell below the pageW/32 tier threshold (" << tierThreshold
	    << " texels, pageW=" << res.width << ")";

	std::printf("[PackAtlas] TwoTier: pages=%u occupancy=%.3f dims=%ux%u\n",
	            res.numPages, res.occupancy, res.width, res.height);
}

// ---------------------------------------------------------------------------
// Shelf-tier rotation: tall skinny tiny charts must be laid down (rotated) in
// shelves without breaking the winding-preserving 90° UV-rewrite convention —
// disjointness + in-bounds UVs + positive UV area per face prove it.
// ---------------------------------------------------------------------------
TEST(PackAtlas, TwoTierShelfRotationKeepsWindingAndBounds)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	mesh.vertices.clear();
	// 4 big charts absorb the auto density so the 500 skinny ones land UNDER
	// the pageW/32 tier threshold (in the shelf tier) — without them the
	// skinny charts normalize to ~60 texels tall and take the skyline path.
	const unsigned nBig = 4u, n = nBig + 500u;
	float off = 0.f;
	for (unsigned c = 0; c < n; ++c) {
		const float w = (c < nBig) ? 200.f : 1.f;
		const float h = (c < nBig) ? 200.f : 6.f; // tall & skinny → shelf tier wants them rotated
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({off, 0.f, 0.f});
		mesh.vertices.push_back({off + w, 0.f, 0.f});
		mesh.vertices.push_back({off + w, h, 0.f});
		mesh.vertices.push_back({off, h, 0.f});
		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(c);
		faceChart.push_back(c);
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, h});
		off += 210.f;
	}
	AtlasParams params;
	params.resolution = 512;
	params.padding = 2;
	params.allowRotation = true;

	NormalizeChartDensity(mesh, faceChart, n, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, n, params);

	const auto rects = ChartBBoxes(mesh, faceChart, n, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, n));
	// Winding preserved: every face keeps POSITIVE signed UV area (a mirrored
	// placement would flip the sign).
	for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
		const float a2 = SignedDoubleArea2D(mesh.faceTexcoords[f * 3 + 0],
		                                    mesh.faceTexcoords[f * 3 + 1],
		                                    mesh.faceTexcoords[f * 3 + 2]);
		EXPECT_GT(a2, 0.f) << "face " << f << " mirrored by shelf rotation";
	}
}

// ---------------------------------------------------------------------------
// Test 5 — BoundingRectsDisjoint + AllPlaced: basic packing of synthetic charts.
// ---------------------------------------------------------------------------
TEST(PackAtlas, BoundingRectsDisjointAndAllPlaced)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildSyntheticCharts(mesh, faceChart, numCharts, 8u);

	AtlasParams params;
	params.resolution = 512;
	params.padding = 2;
	params.allowRotation = true;
	params.powerOfTwo = false;
	params.square = false;

	// Normalize density first (required before PackAtlas).
	NormalizeChartDensity(mesh, faceChart, numCharts, params);

	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	// All charts placed.
	ASSERT_EQ(res.chartPage.size(), numCharts);
	for (unsigned c = 0; c < numCharts; ++c) {
		EXPECT_LT(res.chartPage[c], res.numPages)
		    << "chart " << c << " has invalid page";
	}

	// All UVs finite and in [0,1].
	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x())) << "Non-finite UV.x after packing";
		ASSERT_TRUE(std::isfinite(uv.y())) << "Non-finite UV.y after packing";
		EXPECT_GE(uv.x(), 0.f - 1e-4f);
		EXPECT_LE(uv.x(), 1.f + 1e-4f);
		EXPECT_GE(uv.y(), 0.f - 1e-4f);
		EXPECT_LE(uv.y(), 1.f + 1e-4f);
	}

	// No overlap.
	const auto rects = ChartBBoxes(
	    mesh, faceChart, numCharts,
	    res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts))
	    << "Two charts overlap in the atlas!";

	// Reasonable occupancy (synthetic charts should fill >30% of the page area).
	EXPECT_GT(res.occupancy, 0.3f)
	    << "Occupancy too low: " << res.occupancy;

	std::printf("[PackAtlas] BoundingRectsDisjointAndAllPlaced: num_pages=%u  "
	            "occupancy=%.3f  dims=%ux%u\n",
	            res.numPages, res.occupancy, res.width, res.height);
}

// ---------------------------------------------------------------------------
// Test 6 — PowerOfTwo: result dims are powers of two when requested.
// ---------------------------------------------------------------------------
TEST(PackAtlas, PowerOfTwo)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildSyntheticCharts(mesh, faceChart, numCharts, 4u);

	AtlasParams params;
	params.resolution = 256;
	params.powerOfTwo = true;
	params.square = false;
	params.padding = 1;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	// Both dims must be powers of two.
	auto isPow2 = [](unsigned v) -> bool {
		return v > 0 && (v & (v - 1)) == 0;
	};
	EXPECT_TRUE(isPow2(res.width)) << "width not power of two: " << res.width;
	EXPECT_TRUE(isPow2(res.height)) << "height not power of two: " << res.height;
}

// ---------------------------------------------------------------------------
// Test 7 — Square: w == h when square=true.
// ---------------------------------------------------------------------------
TEST(PackAtlas, Square)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildSyntheticCharts(mesh, faceChart, numCharts, 4u);

	AtlasParams params;
	params.resolution = 256;
	params.square = true;
	params.padding = 1;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	EXPECT_EQ(res.width, res.height) << "Atlas is not square";
}

// ---------------------------------------------------------------------------
// Test 7b — Rotation preserves UV winding: a 90°-rotated chart must come out
// rotated (Jacobian det = +1), not transposed (det = -1). A transposed chart
// has mirror-wound UV triangles, which inverts tangent frames and breaks
// normal-map baking for every rotated chart.
// ---------------------------------------------------------------------------
TEST(PackAtlas, RotationPreservesWinding)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	// UVs in texel space with bbox-min at origin (PackAtlas precondition),
	// CCW winding. World positions are irrelevant to the packer.
	auto addChart = [&](float w, float h, unsigned cid) {
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({0.f, 0.f, static_cast<float>(cid)});
		mesh.vertices.push_back({w, 0.f, static_cast<float>(cid)});
		mesh.vertices.push_back({w, h, static_cast<float>(cid)});
		mesh.vertices.push_back({0.f, h, static_cast<float>(cid)});
		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(cid);
		faceChart.push_back(cid);
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, h});
	};
	// Chart 0: 40x40 square packs first at (0,0) of a 64x64 page. Chart 1:
	// 30x10 rect fits unrotated only above the square (top edge 50) but
	// rotated in the right-hand valley at (40,0) (top edge 30) — the skyline
	// bottom-left rule deterministically rotates it.
	addChart(40.f, 40.f, 0u);
	addChart(30.f, 10.f, 1u);
	const unsigned numCharts = 2;

	AtlasParams params;
	params.resolution = 64;
	params.padding = 0;
	params.allowRotation = true;
	params.orientCharts = false; // isolate the packer's own 90° rotation
	params.powerOfTwo = false;
	params.square = false;

	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);
	ASSERT_EQ(res.numPages, 1u);

	// Chart 1 must actually have been rotated (bbox now taller than wide),
	// otherwise the winding assertion below would be vacuous.
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts,
	                               res.chartPage, res.width, res.height);
	EXPECT_GT((rects[1].y1 - rects[1].y0) - (rects[1].x1 - rects[1].x0), 15.f)
	    << "chart 1 was not rotated — test setup no longer exercises rotation";

	// Every UV triangle must keep its input CCW winding.
	for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
		const Mesh::TexCoord& a = mesh.faceTexcoords[fi * 3 + 0];
		const Mesh::TexCoord& b = mesh.faceTexcoords[fi * 3 + 1];
		const Mesh::TexCoord& c = mesh.faceTexcoords[fi * 3 + 2];
		const float signedArea2 = (b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y());
		EXPECT_GT(signedArea2, 0.f)
		    << "face " << fi << " (chart " << faceChart[fi]
		    << ") came out mirror-wound after packing";
	}
}

// ---------------------------------------------------------------------------
// Test 8 — Rotation: allowRotation packs correctly (all placed, no overlap).
// ---------------------------------------------------------------------------
TEST(PackAtlas, RotationAllPlaced)
{
	// Use some tall and some wide rects to exercise rotation.
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 4;
	faceChart.clear();
	mesh.vertices.clear();
	mesh.faces.clear();
	mesh.faceTexcoords.clear();

	// Chart 0: tall 1×3
	// Chart 1: wide 3×1
	// Chart 2: tall 1×4
	// Chart 3: wide 4×1
	const float widths[] = {1.f, 3.f, 1.f, 4.f};
	const float heights[] = {3.f, 1.f, 4.f, 1.f};

	float offset = 0.f;
	for (unsigned c = 0; c < numCharts; ++c) {
		const unsigned base = static_cast<unsigned>(mesh.vertices.size());
		const float w = widths[c], h = heights[c];

		mesh.vertices.push_back({offset, 0.f, 0.f});
		mesh.vertices.push_back({offset + w, 0.f, 0.f});
		mesh.vertices.push_back({offset + w, h, 0.f});
		mesh.vertices.push_back({offset, h, 0.f});

		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(c);
		faceChart.push_back(c);

		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 1.f});
		mesh.faceTexcoords.push_back({0.f, 1.f});

		offset += w + 2.f;
	}

	AtlasParams params;
	params.resolution = 256;
	params.padding = 1;
	params.allowRotation = true;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	ASSERT_EQ(res.chartPage.size(), numCharts);
	for (unsigned c = 0; c < numCharts; ++c) {
		EXPECT_LT(res.chartPage[c], res.numPages);
	}

	const auto rects = ChartBBoxes(
	    mesh, faceChart, numCharts,
	    res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts))
	    << "Overlap detected after rotation packing";
}

// ---------------------------------------------------------------------------
// Test 9 — MultiAtlasOverflow: tiny resolution forces numPages > 1.
// ---------------------------------------------------------------------------
TEST(PackAtlas, MultiAtlasOverflow)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	// 6 charts each with side ~1; resolution=16 should overflow to multiple pages.
	BuildSyntheticCharts(mesh, faceChart, numCharts, 6u);

	AtlasParams params;
	params.resolution = 16; // very small → forces overflow
	params.padding = 1;
	params.allowRotation = true;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	// All charts placed.
	ASSERT_EQ(res.chartPage.size(), numCharts);
	for (unsigned c = 0; c < numCharts; ++c) {
		EXPECT_LT(res.chartPage[c], res.numPages)
		    << "chart " << c << " not placed";
	}

	// No overlap within each page.
	const auto rects = ChartBBoxes(
	    mesh, faceChart, numCharts,
	    res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts))
	    << "Overlap in multi-page atlas";

	std::printf("[PackAtlas] MultiAtlasOverflow: num_pages=%u\n", res.numPages);
	// With tiny resolution we expect more than 1 page.
	EXPECT_GT(res.numPages, 1u) << "Expected overflow to multiple pages";
}

// ---------------------------------------------------------------------------
// Test 9b — Degenerate chart: a chart whose UVs are COLLINEAR (zero UV area) with a
// large raw extent, packed alongside normal charts. NormalizeChartDensity skips it
// (scale 0), so before the fix PackAtlas stacked it at the page-0 origin with its raw
// extent — pushing UVs far outside [0,1] and over the chart packed at the origin.
// After the fix it collapses to its own >=1-texel slot centre.
// ---------------------------------------------------------------------------
TEST(PackAtlas, DegenerateChartStaysInBounds)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildSyntheticCharts(mesh, faceChart, numCharts, 3u); // charts 0..2 are normal

	// Append a 4th chart: one triangle with COLLINEAR UVs spanning 0..6000 (zero area).
	const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back({100.f, 0.f, 0.f});
	mesh.vertices.push_back({101.f, 0.f, 0.f});
	mesh.vertices.push_back({102.f, 1.f, 0.f});
	mesh.faces.push_back({base, base + 1, base + 2});
	faceChart.push_back(3u);
	mesh.faceTexcoords.push_back({0.f, 0.f});
	mesh.faceTexcoords.push_back({3000.f, 0.f});
	mesh.faceTexcoords.push_back({6000.f, 0.f});
	numCharts = 4u;

	AtlasParams params;
	params.resolution = 256;
	params.padding = 2;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x())) << "non-finite UV.x";
		ASSERT_TRUE(std::isfinite(uv.y())) << "non-finite UV.y";
		EXPECT_GE(uv.x(), 0.f - 1e-4f) << "UV.x below 0 (degenerate chart bled out)";
		EXPECT_LE(uv.x(), 1.f + 1e-4f) << "UV.x above 1 (degenerate chart bled out)";
		EXPECT_GE(uv.y(), 0.f - 1e-4f) << "UV.y below 0";
		EXPECT_LE(uv.y(), 1.f + 1e-4f) << "UV.y above 1";
	}
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts)) << "degenerate chart overlaps a real chart";
}

// ---------------------------------------------------------------------------
// A flip-free but severely area-compressed sliver chart (uvArea tiny yet
// POSITIVE -- it passes the uvArea<=0 guard) must not collapse its siblings:
// pre-fix its unbounded NormalizeChartDensity scale blew up its bbox, and
// fitToResolution's global solve then shrank every OTHER chart to sub-texel
// size with no error. Occupancy can still look healthy afterwards
// (the blown-up sliver fills the page), so assert per-sibling texel extents,
// not occupancy.
// ---------------------------------------------------------------------------
TEST(PackAtlas, SliverChartDoesNotCollapseSiblingCharts)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildSyntheticCharts(mesh, faceChart, numCharts, 8u); // charts 0..7 are normal

	// Append chart 8: normal world area (2.0) but a near-collinear UV triangle
	// with uvArea = 5e-10 -- flip-free, positive, magnification ~6.3e4.
	const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
	mesh.vertices.push_back({100.f, 0.f, 0.f});
	mesh.vertices.push_back({102.f, 0.f, 0.f});
	mesh.vertices.push_back({102.f, 2.f, 0.f});
	mesh.faces.push_back({base, base + 1, base + 2});
	faceChart.push_back(8u);
	mesh.faceTexcoords.push_back({0.f, 0.f});
	mesh.faceTexcoords.push_back({1.f, 0.f});
	mesh.faceTexcoords.push_back({1.f, 1e-9f});
	numCharts = 9u;

	AtlasParams params;
	params.resolution = 256;
	params.padding = 2;
	params.fitToResolution = true; // the vulnerable path (GenerateAtlas default)

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x()));
		ASSERT_TRUE(std::isfinite(uv.y()));
	}
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts));
	// every NORMAL chart keeps the packer's own >=1-texel invariant
	for (unsigned c = 0; c < 8u; ++c) {
		EXPECT_GE(rects[c].x1 - rects[c].x0, 1.f) << "chart " << c << " collapsed (width)";
		EXPECT_GE(rects[c].y1 - rects[c].y0, 1.f) << "chart " << c << " collapsed (height)";
	}
}

// ---------------------------------------------------------------------------
// The sibling-collapse guard above only bites when the compressed chart's RAW
// extent is small. Slitting a tube open (cutToDisk) produces the dangerous
// shape instead: a ribbon whose uvArea is tiny yet positive AND whose raw UV
// extent is large. NormalizeChartDensity used to leave such a chart entirely
// unnormalized (scale 0 -> raw UVs passed through), and PackAtlas's degenerate
// rescue does not catch it either, because that only tests for a ZERO-width or
// ZERO-height rect (AtlasPacking.cpp: `cr.degenerate = (cr.w <= 0 || cr.h <= 0)`)
// while a ribbon has a large w and a small-but-positive h. It therefore entered
// fitToResolution's global solve, whose binding term is MAX DIMENSION -- every
// chart shrinks until the widest one fits -- and so it set the page scale for
// every sibling. (Not the sum-of-w*h term: a ribbon's bbox area is near zero.)
//
// Measured on a real mesh (Ignatius, 536k faces, cutToDisk on): one triangle
// spanning 4092 of 4096 texels held triangle coverage at 0.0189, against 0.2017
// once bounded and 0.2226 on 0.3.0, with occupancy still reporting a plausible
// 0.196 and no error anywhere. Assert the
// invariant that actually broke -- adding one degenerate chart must not destroy
// the atlas -- rather than the >=1-texel rect floor, which the packer clamps and
// which therefore holds even under total collapse.
// ---------------------------------------------------------------------------
TEST(PackAtlas, LargeExtentSliverDoesNotSetPageScale)
{
	// `uvHeight` sets the sliver's magnification: 1e-3 keeps it under the
	// maxScaleMagnitude area guard (~816) so only the page clamp can catch it,
	// 1e-9 puts it far over (~2.6e7) so it takes the degenerate-flatten branch —
	// which is the path the real Ignatius ribbon takes, and the one the original
	// code skipped into raw UVs. Both must hold.
	auto coverageOf = [](float uvHeight) {
		Mesh mesh;
		std::vector<unsigned> faceChart;
		unsigned numCharts = 0;
		BuildSyntheticCharts(mesh, faceChart, numCharts, 8u);

		if (uvHeight > 0.f) {
			// World area 2.0, a near-collinear UV triangle, raw UV extent 6000 --
			// the slit-ribbon shape: negligible area over an enormous span.
			const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
			mesh.vertices.push_back({100.f, 0.f, 0.f});
			mesh.vertices.push_back({102.f, 0.f, 0.f});
			mesh.vertices.push_back({102.f, 2.f, 0.f});
			mesh.faces.push_back({base, base + 1, base + 2});
			faceChart.push_back(8u);
			mesh.faceTexcoords.push_back({0.f, 0.f});
			mesh.faceTexcoords.push_back({6000.f, 0.f});
			mesh.faceTexcoords.push_back({6000.f, uvHeight});
			numCharts = 9u;
		}

		AtlasParams params;
		params.resolution = 256;
		params.padding = 2;
		params.fitToResolution = true; // the vulnerable path (GenerateAtlas default)

		NormalizeChartDensity(mesh, faceChart, numCharts, params);
		const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);
		for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
			EXPECT_TRUE(std::isfinite(uv.x()));
			EXPECT_TRUE(std::isfinite(uv.y()));
		}
		return res.coverage;
	};

	const float without = coverageOf(0.f);
	ASSERT_GT(without, 0.05f) << "control atlas is itself empty — test is vacuous";

	for (const float uvHeight : {1e-3f, 1e-9f}) {
		const float with = coverageOf(uvHeight);
		EXPECT_GT(with, without * 0.5f)
		    << "a large-extent sliver chart (uv height " << uvHeight
		    << ") collapsed the atlas: coverage " << without << " -> " << with;
	}
}

// ---------------------------------------------------------------------------
// Test 9c — fit-to-resolution must ITERATE (not a single open-loop 0.82 solve): a
// corpus of high-aspect charts wastes >18% of the skyline, so the single-scale solve
// overflows to a nearly-empty second page. The iterate-and-shrink loop must bring it
// back onto ONE page at the requested resolution.
// ---------------------------------------------------------------------------
TEST(PackAtlas, FitToResolutionIteratesToOnePage)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	const unsigned numCharts = 40u;

	// Alternating extreme-aspect rects (80×2 wide, 2×80 tall). With rotation disabled
	// the packer cannot normalise their orientation, so the skyline buries >18% of the
	// page — the single open-loop 0.82 solve then overflows to a second, near-empty page.
	float off = 0.f;
	for (unsigned c = 0; c < numCharts; ++c) {
		const float w = (c % 2) ? 2.f : 80.f;
		const float h = (c % 2) ? 80.f : 2.f;
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({off, 0.f, 0.f});
		mesh.vertices.push_back({off + w, 0.f, 0.f});
		mesh.vertices.push_back({off + w, h, 0.f});
		mesh.vertices.push_back({off, h, 0.f});
		mesh.faces.push_back({base + 0, base + 1, base + 2});
		mesh.faces.push_back({base + 0, base + 2, base + 3});
		faceChart.push_back(c);
		faceChart.push_back(c);
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({w, h});
		mesh.faceTexcoords.push_back({0.f, h});
		off += 200.f;
	}

	AtlasParams params;
	params.resolution = 256;
	params.padding = 2;
	params.allowRotation = false; // keep the extreme aspects → force high skyline waste
	params.fitToResolution = true; // the behaviour under test

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	std::printf("[PackAtlas] FitToResolutionIteratesToOnePage: num_pages=%u occupancy=%.3f\n",
	            res.numPages, res.occupancy);
	EXPECT_EQ(res.numPages, 1u)
	    << "fit-to-resolution must iterate the scale until the atlas fits ONE page";

	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts)) << "overlap after fit-to-resolution";
}

// ---------------------------------------------------------------------------
// Analytic fit shrink: the fit loop must converge in [2,3] probe packs on an
// input that GENUINELY overflows the one-page area budget (not merely a
// waste-driven page-count overflow).
//
// Why "genuinely": fitToResolution's quadratic pre-solves k so that, AT ITS
// UNCLAMPED ROOT, the padded rect-area sum is EXACTLY targetFill*resolution^2
// (see PackAtlas §1.5) -- so on ordinary (non-degenerate) charts, probeArea
// can never exceed that budget. The analytic shrink's raw factor
// sqrt(budget/probeArea) is then always >= 1 and clamps to the SAME 0.95
// ceiling the old blind ladder used, so a purely waste-driven overflow (e.g.
// FitToResolutionIteratesToOnePage's extreme-aspect rects) cannot
// discriminate the two algorithms -- both take an identical attempt count.
// Real area overflow needs area the quadratic solve cannot see: DEGENERATE
// (zero-UV-area) charts are excluded from its a/b/c terms entirely
// (NormalizeChartDensity / PackAtlas skip them) yet still occupy a real
// (1+2*padding)^2 padded slot in the actual probe pack. 50 such charts on top
// of 50 ordinary tiny squares (resolution=128, padding=4) inject ~30% real
// overflow (measured: probeArea=17485 against a 13435 budget on the first
// probe) -- confirmed by temporarily reverting to the old `kf *= 0.95` ladder,
// which needs 5 probes to claw back under budget; the analytic
// sqrt(budget/probeArea) shrink (clamped [0.80,0.95]) gets there in 3.
// ---------------------------------------------------------------------------
TEST(PackAtlas, FitToResolutionConvergesInFewAttempts)
{
	Mesh mesh;
	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;
	BuildMixedCharts(mesh, faceChart, numCharts, 50u, 0u, 1.f);

	// Append 50 degenerate (collinear-UV) charts: excluded from the
	// fitToResolution quadratic's a/b/c terms, but each still occupies a real
	// padded texel slot in the probe pack -- see the comment above for why
	// this (not a bigger tiny-chart count or a smaller resolution alone) is
	// the mechanism that forces genuine, not just waste-driven, overflow.
	const unsigned numDegenerate = 50u;
	for (unsigned i = 0; i < numDegenerate; ++i) {
		const auto base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.push_back({1000.f + static_cast<float>(i), 0.f, 0.f});
		mesh.vertices.push_back({1001.f + static_cast<float>(i), 0.f, 0.f});
		mesh.vertices.push_back({1002.f + static_cast<float>(i), 1.f, 0.f});
		mesh.faces.push_back({base, base + 1, base + 2});
		faceChart.push_back(numCharts);
		// Collinear (zero-area) UVs confined to a small [0,1] extent, so the
		// degenerate chart's own footprint stays bounded to ~1 texel (unlike
		// the wide 0..6000-texel span DegenerateChartStaysInBounds uses to
		// exercise the "raw extent bleeds outside [0,1]" regression).
		mesh.faceTexcoords.push_back({0.f, 0.f});
		mesh.faceTexcoords.push_back({0.5f, 0.f});
		mesh.faceTexcoords.push_back({1.f, 0.f});
		++numCharts;
	}

	AtlasParams params;
	params.resolution = 128; // small: 50 fixed padded slots are a meaningful fraction of the budget
	params.padding = 4; // padding-dominated: the production pathology
	params.allowRotation = true;
	params.fitToResolution = true;

	NormalizeChartDensity(mesh, faceChart, numCharts, params);
	const AtlasResult res = PackAtlas(mesh, faceChart, numCharts, params);

	std::printf("[PackAtlas] FitConverges: attempts=%u pages=%u occupancy=%.3f\n",
	            res.fitAttempts, res.numPages, res.occupancy);
	EXPECT_EQ(res.numPages, 1u);
	EXPECT_GE(res.fitAttempts, 2u) << "converged trivially -- the analytic shrink math never ran";
	EXPECT_LE(res.fitAttempts, 3u) << "fit loop is still ladder-stepping";
	const auto rects = ChartBBoxes(mesh, faceChart, numCharts, res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts));
}

// ---------------------------------------------------------------------------
// Test 10 — GenerateAtlas end-to-end on mesh.ply.
// ---------------------------------------------------------------------------
TEST(GenerateAtlas, MeshPlyEndToEnd)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path)) {
		GTEST_SKIP() << "mesh.ply not found at " << path;
	}

	Mesh mesh;
	ASSERT_TRUE(mesh.Load(path)) << "Failed to load " << path;
	ASSERT_FALSE(mesh.faces.empty());

	ParametrizeParams pparams;
	pparams.flattenIterations = 3; // fast for tests

	AtlasParams aparams;
	aparams.resolution = 1024;
	aparams.padding = 2;
	aparams.allowRotation = true;
	aparams.powerOfTwo = false;
	aparams.square = false;

	const AtlasResult res = GenerateAtlas(mesh, pparams, aparams);

	ASSERT_GT(res.numPages, 0u);
	const unsigned numCharts =
	    static_cast<unsigned>(res.chartPage.size());
	ASSERT_GT(numCharts, 0u);

	// faceChart must now be populated by GenerateAtlas via PackAtlas.
	ASSERT_EQ(res.faceChart.size(), mesh.faces.size())
	    << "result.face_chart must have one entry per face";

	std::printf("[GenerateAtlas] mesh.ply: charts=%u  pages=%u  "
	            "dims=%ux%u  occupancy=%.3f\n",
	            numCharts, res.numPages,
	            res.width, res.height, res.occupancy);

	// All UVs must be finite and within [0,1].
	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		ASSERT_TRUE(std::isfinite(uv.x())) << "Non-finite UV.x";
		ASSERT_TRUE(std::isfinite(uv.y())) << "Non-finite UV.y";
		EXPECT_GE(uv.x(), 0.f - 1e-3f) << "UV.x below 0";
		EXPECT_LE(uv.x(), 1.f + 1e-3f) << "UV.x above 1";
		EXPECT_GE(uv.y(), 0.f - 1e-3f) << "UV.y below 0";
		EXPECT_LE(uv.y(), 1.f + 1e-3f) << "UV.y above 1";
	}

	// Occupancy must be a meaningful fill fraction: measured 0.820 on the
	// reference box (matches the packer's 0.82 open-loop target). 0.2 catches
	// the "packing regression leaves the page mostly empty" class while
	// staying safe against build-flag FP drift.
	EXPECT_GT(res.occupancy, 0.2f) << "Occupancy too low (packing regression?): " << res.occupancy;
	EXPECT_LE(res.occupancy, 1.f) << "Occupancy must not exceed 1.0";

	// Key invariant: no two charts on the same page may overlap.
	// Use result.faceChart (now exposed by GenerateAtlas) + the ChartBBoxes /
	// BoundingRectsDisjoint helpers so the real mesh.ply data exercises the same path as
	// the synthetic-chart tests.
	const auto rects = ChartBBoxes(
	    mesh, res.faceChart, numCharts,
	    res.chartPage, res.width, res.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, numCharts))
	    << "Chart bounding rects overlap on mesh.ply — packing invariant violated";
}

// ---------------------------------------------------------------------------
// Test 11 — GenerateAtlas honors an explicit density: with texelsPerUnit > 0
// and fitToResolution = false it preserves density and overflows into multiple
// pages (instead of force-fitting everything onto one). texelsPerUnit == 0
// keeps the single-page fit (covered by Test 10).
// ---------------------------------------------------------------------------
TEST(GenerateAtlas, ExplicitDensityOverflowsToMultiPage)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path))
		GTEST_SKIP() << "mesh.ply not found at " << path;

	Mesh mesh;
	ASSERT_TRUE(mesh.Load(path));
	ASSERT_FALSE(mesh.faces.empty());

	ParametrizeParams pparams;
	pparams.flattenIterations = 3;

	// Choose a density so the total packed UV area ≈ (4·resolution)² ⇒ needs ~16
	// pages. Scale-independent: derive from the mesh's own world area.
	const float worldSide = std::sqrt(static_cast<float>(mesh.ComputeArea()));
	ASSERT_GT(worldSide, 0.f);

	AtlasParams aparams;
	aparams.resolution = 512;
	aparams.padding = 2;
	aparams.texelsPerUnit = 4.f * 512.f / worldSide; // explicit density
	aparams.fitToResolution = false; // must be honored now

	const AtlasResult res = GenerateAtlas(mesh, pparams, aparams);
	EXPECT_GT(res.numPages, 1u)
	    << "GenerateAtlas must honor fit_to_resolution=false with an explicit "
	       "density and overflow into multiple pages";
}

// ---------------------------------------------------------------------------
// Test 11b — GenerateAtlas reports true triangle coverage (§6.4): rect
// occupancy (~0.82 by construction of the fit solve) is blind to per-chart
// bbox waste and the padding tax, so consumers sizing an atlas resolution
// from occupancy alone overestimate texel density. Cross-checked against a
// direct triangle-area integral of the final, normalized output UVs (no
// rasterization needed).
// ---------------------------------------------------------------------------
TEST(GenerateAtlas, ReportsTriangleCoverage)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path))
		GTEST_SKIP() << "mesh.ply not found at " << path;

	Mesh mesh;
	ASSERT_TRUE(mesh.Load(path)) << "Failed to load " << path;
	ASSERT_FALSE(mesh.faces.empty());

	ParametrizeParams pparams;
	pparams.flattenIterations = 3; // fast for tests

	AtlasParams aparams;
	aparams.resolution = 256;

	const AtlasResult result = GenerateAtlas(mesh, pparams, aparams);

	// Triangle coverage: real texels under UV triangles. Always positive for a
	// non-degenerate atlas, never above the padded-rect occupancy (rects contain
	// their triangles plus padding), never above 1.
	EXPECT_GT(result.coverage, 0.f);
	EXPECT_LE(result.coverage, result.occupancy + 1e-3f);
	EXPECT_LE(result.coverage, 1.f);

	// Cross-check against a direct rasterization-free integral of the output UVs.
	double tri = 0.0;
	for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
		const auto& t0 = mesh.faceTexcoords[fi * 3 + 0];
		const auto& t1 = mesh.faceTexcoords[fi * 3 + 1];
		const auto& t2 = mesh.faceTexcoords[fi * 3 + 2];
		tri += 0.5 * std::abs(double(t1.x() - t0.x()) * (t2.y() - t0.y()) - double(t2.x() - t0.x()) * (t1.y() - t0.y()));
	}
	EXPECT_NEAR(result.coverage, static_cast<float>(tri / result.numPages), 1e-4f);
}

// ---------------------------------------------------------------------------
// Test 12 — cache parity: Parametrize()'s shared flip-repair -> flatten cache
// (src/ChartFlattenCache.h, the GenerateAtlas call site ~src/AtlasCharting.cpp:1417
// and the lookup/consume site ~src/Parametrize.cpp:2042-2095) must be perfectly
// transparent to callers. Pin BOTH cache branches:
//   (a) cutToDisk=true — the flip-repair's accepting verdict runs CutChartToDisk
//       and the cache carries that CUT ChartMesh forward for ParametrizeCharts to
//       write back (UVs alone can't be written back through a fresh, uncut
//       re-extraction because CutChartToDisk duplicates vertices).
//   (b) developableMaxUvDistortion > 0 — ChartFolds' distortion-budget path
//       always ships the FULL init+SLIM map on acceptance (finalUv=true), so the
//       cache's "reuse the shipped map directly" branch is what fires.
// For both, the public cache-backed Parametrize() must produce mesh.faceTexcoords
// EXACTLY (bitwise) equal to the public two-call SegmentCharts + ParametrizeCharts
// pipeline, which always receives a nullptr cache (see the public overloads
// declared in halfmesh/AtlasCharting.h / halfmesh/Parametrize.h — the cache-aware
// overloads live only in namespace detail). A cache bug can only ever surface as a
// DIFFERENCE here: a miss silently recomputes and is indistinguishable from "no
// cache", so any observed hit this test exercises is what actually pins the
// branch.
//
// Fixture: an open tube (cylinder wall — two boundary loops, top and bottom
// rings, no end caps). It is perfectly developable, so cone-Lloyd + merge
// collapse the whole tube to ONE chart before flip-repair ever runs, and that
// chart's 2 boundary loops force ChartFolds' cutToDisk block for variant (a).
//
// Cache-hit coverage was confirmed manually rather than via a build-time tally
// gated into this committed test (HM_ATLAS_DEBUG stays OFF in the default/CI
// build, per house convention — see CMakeLists.txt's HALFMESH_ATLAS_DEBUG
// option). A -DHALFMESH_ATLAS_DEBUG=ON scratch build of this EXACT fixture/params
// printed, for the cached (Parametrize) run:
//   variant (a): "[flatten] charts=1 lscm=0 tutte=0 cut2disk=0 pca_fallback=0 cached=1"
//   variant (b): "[flatten] charts=4 lscm=0 tutte=0 cut2disk=0 pca_fallback=0 cached=4"
// i.e. ParametrizeCharts performed ZERO fresh flattens in either run — every
// shipped chart was served from the cache (cached == charts). The matching
// UNCACHED (public two-call) run on the same fixture/params printed
// "cut2disk=1 lscm=1" for (a) and "lscm=4" for (b), confirming which code path
// those cached entries were captured from (ChartFolds' cutToDisk block is
// byte-identical to FlattenChart's; src/Parametrize.cpp's ChartFacesFold moves
// that post-cut `cm` straight into the cache entry — see the comment at its
// `slot.entry->cm = std::move(cm);`). Both variants' cached vs. uncached
// faceTexcoords were verified bitwise identical in that same scratch harness
// before this test was written.
// ---------------------------------------------------------------------------
static Mesh MakeOpenTubeForCacheParity(int nth, int nz, float r, float h)
{
	Mesh m;
	auto idx = [&](int iz, int it) { return static_cast<uint32_t>(iz * nth + (it % nth)); };
	for (int iz = 0; iz <= nz; ++iz)
		for (int it = 0; it < nth; ++it) {
			const float a = 2.f * static_cast<float>(M_PI) * static_cast<float>(it) / static_cast<float>(nth);
			m.vertices.emplace_back(r * std::cos(a), r * std::sin(a), h * static_cast<float>(iz));
		}
	for (int iz = 0; iz < nz; ++iz)
		for (int it = 0; it < nth; ++it) {
			m.faces.emplace_back(idx(iz, it), idx(iz, it + 1), idx(iz + 1, it + 1));
			m.faces.emplace_back(idx(iz, it), idx(iz + 1, it + 1), idx(iz + 1, it));
		}
	return m;
}

TEST(Parametrize, CachedPipelineMatchesUncachedTwoCallPipeline)
{
	auto check = [](const char* label, const Mesh& base, const ParametrizeParams& pp) {
		SCOPED_TRACE(label);

		Mesh cached = base;
		cached.ListHalfEdges();
		cached.ComputeFaceNormals();
		const unsigned n1 = Parametrize(cached, pp); // public, cache-backed entry point

		Mesh uncached = base;
		uncached.ListHalfEdges();
		uncached.ComputeFaceNormals();
		std::vector<unsigned> faceChart;
		const unsigned n2 = SegmentCharts(uncached, pp, faceChart); // public, nullptr cache
		ParametrizeCharts(uncached, faceChart, n2, pp); // public, nullptr cache

		// Guard against vacuity: both pipelines must have actually produced a
		// non-trivial result, not silently no-op'd on an empty/degenerate mesh.
		ASSERT_GT(n1, 0u) << "expected at least one chart";
		ASSERT_FALSE(cached.faceTexcoords.empty());
		// Parametrize()'s public signature does not expose its internal
		// faceChart, so numCharts is the coarsest cross-check available on the
		// segmentation side; SegmentCharts's cache-aware overload is documented
		// (and structurally guaranteed — the cache pointer is only ever WRITTEN
		// during flip-repair's accept path, never READ by its bisection
		// decisions) to return output identical to the public nullptr-cache
		// overload, so this is not a vacuous check.
		ASSERT_EQ(n1, n2) << "chart count differs cached vs uncached";
		ASSERT_EQ(cached.faceTexcoords.size(), base.faces.size() * 3);
		ASSERT_EQ(cached.faceTexcoords.size(), uncached.faceTexcoords.size());

		// The actual pin: EXACT (bitwise) equality, corner-for-corner. No
		// tolerance — a cache bug must show up as a hard mismatch, not a "close
		// enough" pass.
		for (size_t i = 0; i < cached.faceTexcoords.size(); ++i) {
			EXPECT_EQ(cached.faceTexcoords[i].x(), uncached.faceTexcoords[i].x())
			    << "UV.x differs at corner " << i;
			EXPECT_EQ(cached.faceTexcoords[i].y(), uncached.faceTexcoords[i].y())
			    << "UV.y differs at corner " << i;
		}
	};

	const Mesh tube = MakeOpenTubeForCacheParity(24, 6, 1.0f, 0.5f);

	// (a) cutToDisk=true: exercises the cut-ChartMesh writeback branch.
	{
		ParametrizeParams pp;
		pp.method = ParametrizeParams::FlattenMethod::SLIM;
		pp.cutToDisk = true;
		check("cut_to_disk=true", tube, pp);
	}

	// (b) developableMaxUvDistortion > 0: exercises the full-map-reuse
	// (finalUv=true) branch.
	{
		ParametrizeParams pp;
		pp.method = ParametrizeParams::FlattenMethod::SLIM;
		pp.developableMaxUvDistortion = 4.4f;
		check("developable_max_uv_distortion=4.4", tube, pp);
	}
}

// A single extreme-aspect chart: PackRects grows the page to swallow any
// oversized rect, and the fit-to-resolution shrink loop used to test only the
// page COUNT — so the "one page" came back ~10x the requested resolution,
// silently violating the documented one resolution² page contract (2026-08
// review). The max-chart-side clamp must keep the page within resolution.
TEST(PackAtlas, FitToResolutionHonorsPageDimensions)
{
	Mesh mesh;
	mesh.vertices = {{0.f, 0.f, 0.f}, {100.f, 0.f, 0.f}, {100.f, 1.f, 0.f}, {0.f, 1.f, 0.f}};
	mesh.faces = {{0, 1, 2}, {0, 2, 3}};
	mesh.faceTexcoords = {{0.f, 0.f}, {10240.f, 0.f}, {10240.f, 102.f}, {0.f, 0.f}, {10240.f, 102.f}, {0.f, 102.f}};
	const std::vector<unsigned> faceChart = {0, 0};

	AtlasParams params;
	params.resolution = 1024;
	params.padding = 2;
	params.fitToResolution = true;

	const AtlasResult res = PackAtlas(mesh, faceChart, 1u, params);
	EXPECT_EQ(res.numPages, 1u);
	EXPECT_LE(res.width, params.resolution);
	EXPECT_LE(res.height, params.resolution);
}

TEST(PackAtlas, NonFitModeGrowsPageForOversizedChart)
{
	Mesh mesh;
	mesh.vertices = {{0.f, 0.f, 0.f}, {100.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
	mesh.faces = {{0, 1, 2}};
	mesh.faceTexcoords = {{0.f, 0.f}, {100.f, 0.f}, {0.f, 1.f}};
	const std::vector<unsigned> faceChart = {0};

	AtlasParams params;
	params.resolution = 16;
	params.padding = 2;
	params.fitToResolution = false;

	const AtlasResult res = PackAtlas(mesh, faceChart, 1u, params);
	EXPECT_EQ(res.numPages, 1u);
	EXPECT_GE(res.width, 104u);
	EXPECT_GE(res.height, 5u);
	for (const Mesh::TexCoord& uv : mesh.faceTexcoords) {
		EXPECT_GE(uv.x(), 0.f);
		EXPECT_LE(uv.x(), 1.f);
		EXPECT_GE(uv.y(), 0.f);
		EXPECT_LE(uv.y(), 1.f);
	}
}

// K disjoint unit quads (2 triangles each), spaced 3 world units apart so each
// is trivially its own connected component -- used as a crafted multi-chart
// fixture (one chart per quad, assigned by hand) for the per-size padding test.
static Mesh DisjointQuads(int k)
{
	Mesh m;
	for (int i = 0; i < k; ++i) {
		const float x = static_cast<float>(i) * 3.f;
		const unsigned b = static_cast<unsigned>(m.vertices.size());
		m.vertices.emplace_back(x, 0.f, 0.f);
		m.vertices.emplace_back(x + 1.f, 0.f, 0.f);
		m.vertices.emplace_back(x + 1.f, 1.f, 0.f);
		m.vertices.emplace_back(x, 1.f, 0.f);
		m.faces.emplace_back(b, b + 1, b + 2);
		m.faces.emplace_back(b, b + 2, b + 3);
	}
	return m;
}

// K DisjointQuads charts (2 faces each), one chart per quad, all sharing the
// same 4-texel-square local UVs -- the shared fixture for both §6.5 per-size
// padding trigger tests below (tinyChartSide via chart side, debrisChartFaces
// via chart face count).
static void BuildTinyChartFixture(int k, Mesh& mesh, std::vector<unsigned>& faceChart)
{
	mesh = DisjointQuads(k);
	faceChart.resize(mesh.faces.size());
	for (size_t f = 0; f < faceChart.size(); ++f)
		faceChart[f] = static_cast<unsigned>(f / 2);
	mesh.faceTexcoords.assign(mesh.faces.size() * 3, Mesh::TexCoord(0.f, 0.f));
	for (size_t f = 0; f < mesh.faces.size(); f += 2) {
		const Mesh::TexCoord q[4] = {{0.f, 0.f}, {4.f, 0.f}, {4.f, 4.f}, {0.f, 4.f}};
		mesh.faceTexcoords[f * 3 + 0] = q[0];
		mesh.faceTexcoords[f * 3 + 1] = q[1];
		mesh.faceTexcoords[f * 3 + 2] = q[2];
		mesh.faceTexcoords[(f + 1) * 3 + 0] = q[0];
		mesh.faceTexcoords[(f + 1) * 3 + 1] = q[2];
		mesh.faceTexcoords[(f + 1) * 3 + 2] = q[3];
	}
}

// §6.5: tiny charts (max unpadded side <= tinyChartSide) get a 1-texel gutter
// instead of the uniform `padding`. With K=64 identical 4-texel charts packed
// (no fitToResolution) at a page side of 64, the padded rect area at a 4-texel
// gutter (64 * 12*12 = 9216) does not fit one 64*64=4096 page (needs several),
// while the same charts at a 1-texel gutter (64 * 6*6 = 2304) fit one page --
// LESS wasted page area for the SAME geometry, so triangle coverage (Σ triangle
// area / numPages, see AtlasResult::coverage) must go up. (`resolution` is
// chosen small enough to force this multi-page split for the uniform case;
// PackRects only grows the page beyond `resolution` for a single oversized
// chart, never for aggregate content, so at a resolution that dwarfs every
// individual padded chart -- e.g. 128 -- both paddings fit one page and
// coverage cannot differ; that would falsely pass a no-op implementation.)
// Also verifies the per-chart pad vector cannot make charts bleed into each
// other: every padded chart bbox must stay disjoint from every other chart's
// on the same page (ChartBBoxes / BoundingRectsDisjoint, defined above).
TEST(AtlasTest, TinyChartPaddingRaisesCoverage)
{
	constexpr int K = 64;
	Mesh mesh;
	std::vector<unsigned> faceChart;
	BuildTinyChartFixture(K, mesh, faceChart);
	halfmesh::AtlasParams uniform;
	uniform.resolution = 64; // small enough that padding=4 needs >1 page; see comment above
	uniform.padding = 4;
	Mesh meshU = mesh;
	const auto rU = halfmesh::PackAtlas(meshU, faceChart, K, uniform);
	halfmesh::AtlasParams tiny = uniform;
	tiny.tinyChartSide = 8.f; // every 4-texel chart qualifies → pad 1
	Mesh meshT = mesh;
	const auto rT = halfmesh::PackAtlas(meshT, faceChart, K, tiny);
	EXPECT_GT(rT.coverage, rU.coverage); // less gutter, same triangles

	const auto rects = ChartBBoxes(meshT, faceChart, K, rT.chartPage, rT.width, rT.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, K)) << "tinyChartSide padding let charts overlap";
}

// Same mechanism as TinyChartPaddingRaisesCoverage above (same fixture, same
// resolution=64 rationale), through the OTHER §6.5 trigger: every
// BuildTinyChartFixture chart has exactly 2 faces, so debrisChartFaces=2
// qualifies all of them (tinyChartSide stays 0, so only debris fires).
TEST(AtlasTest, DebrisChartPaddingRaisesCoverage)
{
	constexpr int K = 64;
	Mesh mesh;
	std::vector<unsigned> faceChart;
	BuildTinyChartFixture(K, mesh, faceChart);
	halfmesh::AtlasParams uniform;
	uniform.resolution = 64;
	uniform.padding = 4;
	Mesh meshU = mesh;
	const auto rU = halfmesh::PackAtlas(meshU, faceChart, K, uniform);
	halfmesh::AtlasParams debris = uniform;
	debris.debrisChartFaces = 2; // every 2-face chart qualifies → pad 1
	Mesh meshD = mesh;
	const auto rD = halfmesh::PackAtlas(meshD, faceChart, K, debris);
	EXPECT_GT(rD.coverage, rU.coverage); // less gutter, same triangles

	const auto rects = ChartBBoxes(meshD, faceChart, K, rD.chartPage, rD.width, rD.height);
	EXPECT_TRUE(BoundingRectsDisjoint(rects, K)) << "debrisChartFaces padding let charts overlap";
}

// Deterministic "staircase terrain": an n×n grid whose vertex heights are
// quantized random levels — many high-angle-defect vertices, like a
// tetra-extracted MVS surface. Cone-Lloyd fragments it, flip repair splits
// further; the post-repair merge must claw a meaningful share back.
static void BuildStaircaseTerrain(Mesh& mesh, unsigned n, float step)
{
	std::mt19937 rng(42u);
	std::uniform_int_distribution<int> lvl(0, 4);
	std::vector<float> h((n + 1) * (n + 1));
	for (float& z : h)
		z = step * static_cast<float>(lvl(rng));
	for (unsigned y = 0; y <= n; ++y)
		for (unsigned x = 0; x <= n; ++x)
			mesh.vertices.push_back({static_cast<float>(x), static_cast<float>(y), h[y * (n + 1) + x]});
	for (unsigned y = 0; y < n; ++y)
		for (unsigned x = 0; x < n; ++x) {
			const unsigned a = y * (n + 1) + x, b = a + 1, c = a + (n + 1), d = c + 1;
			mesh.faces.push_back({a, b, d});
			mesh.faces.push_back({a, d, c});
		}
}

TEST(SegmentCharts, PostRepairMergeReducesChartsFoldFree)
{
	Mesh base;
	BuildStaircaseTerrain(base, 48u, 0.75f);

	ParametrizeParams p0;
	p0.postRepairMergeRounds = 0;
	Mesh m0 = base;
	std::vector<unsigned> chart0;
	const unsigned n0 = SegmentCharts(m0, p0, chart0);

	ParametrizeParams p2;
	p2.postRepairMergeRounds = 2;
	Mesh m2 = base;
	std::vector<unsigned> chart2;
	const unsigned n2 = SegmentCharts(m2, p2, chart2);

	std::printf("[SegmentCharts] PostRepairMerge: %u -> %u charts\n", n0, n2);
	// Precondition: the fixture actually fragments (otherwise the test is vacuous).
	ASSERT_GT(n0, 50u) << "fixture did not fragment — increase `step` or grid size";
	EXPECT_LT(n2, n0) << "re-merge recombined nothing";

	// Every face charted, ids compact.
	ASSERT_EQ(chart2.size(), m2.faces.size());
	std::vector<char> seen(n2, 0);
	for (unsigned c : chart2) {
		ASSERT_LT(c, n2);
		seen[c] = 1;
	}
	for (unsigned c = 0; c < n2; ++c)
		EXPECT_TRUE(seen[c]) << "chart id " << c << " is empty";

	// Fold-free guarantee survives the merge: no >2-face chart folds.
	std::vector<std::vector<Mesh::FIndex>> fl(n2);
	for (Mesh::FIndex f = 0; f < static_cast<Mesh::FIndex>(m2.faces.size()); ++f)
		fl[chart2[f]].push_back(f);
	for (unsigned c = 0; c < n2; ++c) {
		if (fl[c].size() <= 2)
			continue;
		EXPECT_FALSE(detail::ChartFacesFold(m2, fl[c], p2)) << "chart " << c << " folds after re-merge";
	}
}

// ---------------------------------------------------------------------------
// Test — detail::SegmentCharts's opt-in AtlasSegmentStats out-param (§6.3):
// the post-repair-merge loop's per-round counters (budget vs. wouldEnclose
// rejects, accepted merges, dirty vs. resplit charts) are what Task 7 uses to
// diagnose why postRepairMergeRounds recovers so few charts in practice.
// Exercises the cache-aware detail:: overload directly on the challenge
// fixture (tests/data/mesh.ply) — no public API exposes stats, so this test
// is only reachable via ChartFlattenCache.h (this target's src/ include dir,
// see tests/CMakeLists.txt).
// ---------------------------------------------------------------------------
TEST(SegmentCharts, SegmentationStatsPopulatedOnChallengeMesh)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path))
		GTEST_SKIP() << "mesh.ply not found at " << path;

	Mesh mesh;
	ASSERT_TRUE(mesh.Load(path)) << "Failed to load " << path;
	ASSERT_FALSE(mesh.faces.empty());

	ParametrizeParams params;
	std::vector<unsigned> faceChart;
	detail::AtlasSegmentStats stats;
	const unsigned n = detail::SegmentCharts(mesh, params, faceChart, nullptr, &stats);

	std::printf("[SegmentCharts] stats: lloyd=%u merged=%u repaired=%u final=%u repairSplits=%u rounds=%zu\n",
	            stats.lloydCharts, stats.mergedCharts, stats.repairedCharts, stats.finalCharts,
	            stats.repairSplits, stats.rounds.size());
	for (size_t i = 0; i < stats.rounds.size(); ++i) {
		const auto& r = stats.rounds[i];
		std::printf("[SegmentCharts]   round %zu: pushed=%u budgetRej=%u encloseRej=%u merges=%u dirty=%u resplit=%u after=%u\n",
		            i, r.pairsPushed, r.pairsBudgetRejected, r.pairsEncloseRejected,
		            r.merges, r.dirtyCharts, r.resplitCharts, r.chartsAfter);
	}

	EXPECT_EQ(stats.finalCharts, n);
	EXPECT_GE(stats.lloydCharts, stats.mergedCharts); // merge only reduces
	EXPECT_GE(stats.repairedCharts, stats.mergedCharts); // repair only splits
	ASSERT_GE(stats.rounds.size(), 1u); // postRepairMergeRounds=2 default
	for (const auto& r : stats.rounds) {
		EXPECT_GE(r.merges, r.dirtyCharts ? 1u : 0u);
		EXPECT_LE(r.resplitCharts, r.dirtyCharts);
	}
}

// ---------------------------------------------------------------------------
// Test — §6.3 post-repair fold-blacklist: a merged pair whose resulting
// chart the repair wave split right back (a "fold") is recorded per-round in
// MergeRound::refoldedPairs, keyed by the two sides' smallest global face ids
// (invariant under the Compact() id relabelling that happens between rounds).
// The blacklist must make that memoized pair unavailable to every later
// round, so no key may ever appear twice across stats.rounds.
// ---------------------------------------------------------------------------
TEST(AtlasTest, PostRepairMergeNeverRetriesAFoldedUnion)
{
	const std::string path = TestMeshPath();
	if (!std::filesystem::exists(path))
		GTEST_SKIP() << "mesh.ply not found at " << path;

	Mesh mesh;
	ASSERT_TRUE(mesh.Load(path)) << "Failed to load " << path;

	ParametrizeParams params;
	std::vector<unsigned> faceChart;
	detail::AtlasSegmentStats stats;
	detail::SegmentCharts(mesh, params, faceChart, nullptr, &stats);

	std::set<std::pair<Mesh::FIndex, Mesh::FIndex>> seen;
	for (const auto& r : stats.rounds)
		for (const auto& p : r.refoldedPairs) {
			// Contract (minFidA < minFidB, per the brief): asserted rather than
			// silently canonicalized here, so a producer-side regression that
			// starts emitting raw root order shows up as a failure at THIS line
			// instead of being masked by the test re-sorting around it.
			EXPECT_LT(p.first, p.second) << "refoldedPairs entry not canonically ordered";
			EXPECT_TRUE(seen.insert(p).second) << "pair re-merged after folding";
		}
}

// ---------------------------------------------------------------------------
// Test — §6.1 failure-localized carve seam (detail::CarveFailureRegionForTest,
// defined in src/AtlasCharting.cpp): on a 20x20 grid, carving 2 TopoNeighbor
// rings around one corner face must produce {small local region, the rest},
// covering every face exactly once. A non-localized failure (every face is
// "bad") must fall back (return false) — the repair then uses BisectFaces.
// ---------------------------------------------------------------------------
TEST(AtlasTest, CarveFailureRegionSplitsAroundBadFaces)
{
	Mesh mesh = hmtest::corpus::GridPlane(20);
	mesh.ListHalfEdges();
	std::vector<Mesh::FIndex> faces(mesh.faces.size());
	std::iota(faces.begin(), faces.end(), 0u);
	const std::vector<Mesh::FIndex> bad = {0u}; // one corner face
	std::vector<Mesh::FIndex> A, B;
	ASSERT_TRUE(detail::CarveFailureRegionForTest(mesh, faces, bad, 2u, A, B));
	EXPECT_FALSE(A.empty());
	EXPECT_FALSE(B.empty());
	EXPECT_EQ(A.size() + B.size(), faces.size());
	// A contains the bad face and stays local: within 2 rings of one corner
	// face of a large grid, far fewer than half the faces.
	EXPECT_NE(std::find(A.begin(), A.end(), 0u), A.end());
	EXPECT_LT(A.size(), faces.size() / 4);
	// Not-localized failure falls back (returns false): bad = every face.
	std::vector<Mesh::FIndex> A2, B2;
	EXPECT_FALSE(detail::CarveFailureRegionForTest(mesh, faces, faces, 2u, A2, B2));
}

// ---------------------------------------------------------------------------
// Test — the connectivity guard's DECLINE path specifically. The size guard
// (badFaces >= half the chart) above never exercises it: a scattered/snaking
// failure region can be well under half the chart by AREA yet still sever
// the chart into disconnected pieces. On a 20x20 grid, a single full-width
// quad row (40 of 800 faces) floods (rings=2) to 5 rows — 200 faces, 25% of
// the chart, comfortably under the size guard — but the band spans the
// ENTIRE grid width, so removing it severs the remainder into two pieces
// (the rows above vs. below). The connectivity guard must decline (return
// false), matching the method's own name: carving off THE (singular)
// topo-connected neighborhood, not a cut that fragments the rest.
// A same-size (40-face) but CORNER-localized cluster is the positive
// control: same badFaces count, same rings, no severing — must still carve.
// ---------------------------------------------------------------------------
TEST(AtlasTest, CarveFailureRegionDeclinesWhenRemainderWouldSplit)
{
	Mesh mesh = hmtest::corpus::GridPlane(20);
	mesh.ListHalfEdges();
	std::vector<Mesh::FIndex> faces(mesh.faces.size());
	std::iota(faces.begin(), faces.end(), 0u);
	constexpr unsigned n = 20;

	std::vector<Mesh::FIndex> band; // full-width row j=9 (40 faces)
	for (unsigned i = 0; i < n; ++i) {
		band.push_back(2u * (9u * n + i));
		band.push_back(2u * (9u * n + i) + 1u);
	}
	std::vector<Mesh::FIndex> A, B;
	EXPECT_FALSE(detail::CarveFailureRegionForTest(mesh, faces, band, 2u, A, B))
	    << "a full-width band severs the remainder — the connectivity guard must decline";

	std::vector<Mesh::FIndex> corner; // 2 rows x 10 cols in one corner (40 faces)
	for (unsigned j = 0; j < 2; ++j)
		for (unsigned i = 0; i < 10; ++i) {
			corner.push_back(2u * (j * n + i));
			corner.push_back(2u * (j * n + i) + 1u);
		}
	std::vector<Mesh::FIndex> A2, B2;
	ASSERT_TRUE(detail::CarveFailureRegionForTest(mesh, faces, corner, 2u, A2, B2))
	    << "a same-size corner-localized cluster must still carve";
	EXPECT_FALSE(A2.empty());
	EXPECT_FALSE(B2.empty());
	EXPECT_EQ(A2.size() + B2.size(), faces.size());
}

TEST(RectPacking, UsesCvRectsAndPreservesInputOrder)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(17, 23, 4, 8),
	    cv::Rect(0, 0, 8, 4)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 0;
	params.mode = RectPackMode::FixedMultiPage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	ASSERT_EQ(placements.size(), rects.size());
	EXPECT_EQ(result.numPacked, 2u);
	EXPECT_TRUE(placements[0].packed);
	EXPECT_TRUE(placements[1].packed);
	EXPECT_EQ(placements[0].rect.area(), rects[0].area());
	EXPECT_EQ(placements[1].rect.area(), rects[1].area());
}

TEST(RectPacking, FixedSinglePageReportsUnpackedInputs)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(0, 0, 8, 8),
	    cv::Rect(0, 0, 8, 8)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 0;
	params.mode = RectPackMode::FixedSinglePage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_EQ(result.numPages, 1u);
	EXPECT_EQ(result.numPacked, 1u);
	EXPECT_NE(placements[0].packed, placements[1].packed);
}

TEST(RectPacking, FixedSinglePageReportsPageWhenAllInputsAreOversized)
{
	const std::vector<cv::Rect> rects{cv::Rect(0, 0, 9, 9)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 0;
	params.mode = RectPackMode::FixedSinglePage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_EQ(result.numPages, 1u);
	EXPECT_EQ(result.numPacked, 0u);
}

TEST(RectPacking, UnlimitedModeOpensAdditionalPages)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(0, 0, 8, 8),
	    cv::Rect(0, 0, 8, 8)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 0;
	params.mode = RectPackMode::FixedMultiPage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_EQ(result.numPages, 2u);
	EXPECT_EQ(result.numPacked, 2u);
}

TEST(RectPacking, GrowToFitExpandsPageForOversizedInput)
{
	const std::vector<cv::Rect> rects{cv::Rect(0, 0, 12, 7)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 2;
	params.mode = RectPackMode::GrowSinglePage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_GE(result.pageSize.width, 16);
	EXPECT_GE(result.pageSize.height, 11);
	ASSERT_EQ(placements.size(), 1u);
	EXPECT_TRUE(placements[0].packed);
}

TEST(RectPacking, GrowSinglePageRepacksUntilAllRectsFit)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(0, 0, 8, 8),
	    cv::Rect(0, 0, 8, 8)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 0;
	params.mode = RectPackMode::GrowSinglePage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_EQ(result.numPages, 1u);
	EXPECT_EQ(result.numPacked, 2u);
	EXPECT_GE(result.pageSize.width, 16);
	EXPECT_GE(result.pageSize.height, 16);
}

TEST(RectPacking, GrowSinglePageHonorsMaximumSize)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(0, 0, 8, 8),
	    cv::Rect(0, 0, 8, 8)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.maxPageSize = cv::Size(8, 8);
	params.padding = 0;
	params.mode = RectPackMode::GrowSinglePage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_EQ(result.pageSize, cv::Size(8, 8));
	EXPECT_EQ(result.numPacked, 1u);
}

// GrowSinglePage sized its page with `rect.width + 2*padding` in int, which is
// signed overflow -- UB -- for an extent near INT_MAX, and it ran before the
// int64_t `impossible` test that rejects such a rect. Sizing now widens to
// int64_t and saturates, so this is a well-defined "does not fit" instead.
// Under UBSan a regression here trips the signed-overflow check.
TEST(RectPacking, GrowSinglePageDoesNotOverflowOnHugeRects)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(0, 0, std::numeric_limits<int>::max(), 4),
	    cv::Rect(0, 0, 8, 8)};
	RectPackParams params;
	params.pageSize = cv::Size(16, 16);
	params.maxPageSize = cv::Size(64, 64);
	params.padding = 4;
	params.allowRotation = false;
	params.mode = RectPackMode::GrowSinglePage;
	std::vector<RectPlacement> placements;

	const RectPackResult result = PackRectangles(rects, params, placements);

	ASSERT_EQ(placements.size(), rects.size());
	EXPECT_FALSE(placements[0].packed) << "an INT_MAX-wide rect cannot fit a 64px cap";
	EXPECT_LE(result.pageSize.width, 64);
	EXPECT_GT(result.pageSize.width, 0) << "page width must not wrap negative";
	EXPECT_GT(result.pageSize.height, 0);
}

// Same arithmetic, but uncapped: growth must saturate at INT_MAX rather than
// wrap, and the call must terminate.
TEST(RectPacking, GrowSinglePageSaturatesUncappedGrowth)
{
	const std::vector<cv::Rect> rects{cv::Rect(0, 0, std::numeric_limits<int>::max(), 2)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.padding = 2;
	params.mode = RectPackMode::GrowSinglePage;
	std::vector<RectPlacement> placements;

	const RectPackResult result = PackRectangles(rects, params, placements);

	ASSERT_EQ(placements.size(), 1u);
	EXPECT_GT(result.pageSize.width, 0) << "page width must not wrap negative";
	EXPECT_GT(result.pageSize.height, 0);
}

TEST(RectPacking, GrowSinglePageStopsWhenAxisCapMakesInputImpossible)
{
	const std::vector<cv::Rect> rects{cv::Rect(0, 0, 12, 7)};
	RectPackParams params;
	params.pageSize = cv::Size(8, 8);
	params.maxPageSize = cv::Size(8, 0);
	params.padding = 2;
	params.mode = RectPackMode::GrowSinglePage;
	std::vector<RectPlacement> placements;
	const RectPackResult result = PackRectangles(rects, params, placements);
	EXPECT_EQ(result.pageSize, cv::Size(8, 11));
	EXPECT_EQ(result.numPages, 1u);
	EXPECT_EQ(result.numPacked, 0u);
}

TEST(RectPacking, EstimatesRoundedSquareTextureSize)
{
	const std::vector<cv::Rect> rects{
	    cv::Rect(0, 0, 10, 10),
	    cv::Rect(0, 0, 10, 10)};
	EXPECT_EQ(EstimateSquareTextureSize(rects, 8, 1.f), 16);
	EXPECT_EQ(EstimateSquareTextureSize(rects, 0, 1.f), 16);
}

} // namespace
} // namespace halfmesh
