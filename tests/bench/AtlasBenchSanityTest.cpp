/*
* AtlasBenchSanityTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/AtlasBenchSanityTest.cpp — self-verification of the harness.
//
// Proves the (new) bench metrics are correctly normalized and that the engines
// are deterministic, so the comparison numbers can be trusted.  Labelled
// "bench" for ctest.
#include <gtest/gtest.h>

#include "BenchMetrics.h"
#include "BenchTypes.h"
#include "engines/EngineHalfmesh.h"

#include <halfmesh/Mesh.h>

#include <cmath>
#include <vector>

namespace {

// Build a flat n×n quad grid in the z=0 plane whose per-corner UVs equal the XY
// positions — i.e. an EXACTLY isometric parametrization (one chart).  Used as
// the analytic ground truth: every distortion metric must report its optimum.
halfmesh::Mesh MakeFlatGrid(int n)
{
	halfmesh::Mesh m;
	const int side = n + 1;
	for (int j = 0; j <= n; ++j)
		for (int i = 0; i <= n; ++i)
			m.vertices.emplace_back(static_cast<float>(i), static_cast<float>(j), 0.f);
	auto vid = [&](int i, int j) { return static_cast<uint32_t>(j * side + i); };
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			// CCW winding so signed UV area is positive (no flips).
			m.faces.push_back({vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)});
			m.faces.push_back({vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)});
		}
	}
	m.faceTexcoords.resize(m.faces.size() * 3);
	for (size_t f = 0; f < m.faces.size(); ++f)
		for (int c = 0; c < 3; ++c) {
			const auto& v = m.vertices[m.faces[f][c]];
			m.faceTexcoords[f * 3 + c] = halfmesh::Mesh::TexCoord(v.x(), v.y());
		}
	return m;
}

} // namespace

// ---------------------------------------------------------------------------
// Identity: an exactly-isometric flat map must hit every distortion optimum.
// ---------------------------------------------------------------------------
TEST(AtlasBenchSanity, IdentityFlatGridOptimal)
{
	const halfmesh::Mesh grid = MakeFlatGrid(5);
	const std::vector<unsigned> faceChart(grid.faces.size(), 0u);

	hmbench::AtlasMetrics m;
	hmbench::FillParametrization(m, grid, faceChart, 1);

	EXPECT_EQ(m.flipCount, 0) << "isometric map must have no flips";
	EXPECT_NEAR(m.symDirichlet, 4.0, 1e-3) << "symmetric-Dirichlet floor is 4";
	EXPECT_NEAR(m.stretchL2, 1.0, 1e-3) << "Sander L2 stretch floor is 1";
	EXPECT_NEAR(m.quasiconformal, 1.0, 1e-3) << "conformal => sigma_max/sigma_min = 1";
	EXPECT_NEAR(m.areaDistortionRatio, 1.0, 1e-3) << "equiareal => ratio 1";
	EXPECT_TRUE(m.allFinite);
}

// ---------------------------------------------------------------------------
// Segmentation metrics on the analytic grid (single chart).
// ---------------------------------------------------------------------------
TEST(AtlasBenchSanity, SegmentationMetricsOnFlatGrid)
{
	const halfmesh::Mesh grid = MakeFlatGrid(5); // 5x5 = unit cells, 6x6 verts
	const std::vector<unsigned> faceChart(grid.faces.size(), 0u);

	hmbench::AtlasMetrics m;
	hmbench::FillSegmentation(m, grid, faceChart, 1);

	EXPECT_TRUE(m.fullCoverage);
	EXPECT_EQ(m.chartCount, 1u);
	// A flat chart has ~zero planarity error.
	EXPECT_LT(m.meanPlanarityError, 1e-4);
	// Single chart: boundary-cut length == outer perimeter == 4*5 = 20.
	EXPECT_NEAR(m.boundaryCutLength, 20.0, 1e-3);
}

// ---------------------------------------------------------------------------
// Determinism: the same input yields identical metrics across runs.
// ---------------------------------------------------------------------------
TEST(AtlasBenchSanity, HalfmeshDeterministic)
{
	const halfmesh::Mesh grid = MakeFlatGrid(8);
	hmbench::BenchConfig cfg;
	cfg.resolution = 256;

	const hmbench::EngineResult a = hmbench::RunHalfmesh(grid, cfg);
	const hmbench::EngineResult b = hmbench::RunHalfmesh(grid, cfg);

	EXPECT_EQ(a.metrics.chartCount, b.metrics.chartCount);
	EXPECT_EQ(a.metrics.flipCount, b.metrics.flipCount);
	EXPECT_NEAR(a.metrics.symDirichlet, b.metrics.symDirichlet, 1e-9);
	EXPECT_NEAR(a.metrics.occupancyTri, b.metrics.occupancyTri, 1e-9);
	EXPECT_TRUE(a.validOutput);
}

// ---------------------------------------------------------------------------
// A flat grid flattens with near-zero distortion through the full pipeline.
// ---------------------------------------------------------------------------
TEST(AtlasBenchSanity, HalfmeshFlatGridLowDistortion)
{
	const halfmesh::Mesh grid = MakeFlatGrid(6);
	hmbench::BenchConfig cfg;
	cfg.resolution = 256;
	const hmbench::EngineResult r = hmbench::RunHalfmesh(grid, cfg);

	EXPECT_TRUE(r.validOutput);
	EXPECT_EQ(r.metrics.flipCount, 0);
	// A developable plane flattens near-isometrically with the LSCM init.
	// (Regression guard: the old circle-Tutte init left ~6% distortion here.)
	EXPECT_LT(r.metrics.symDirichlet, 4.2) << "flat grid should be near-isometric";
	EXPECT_LT(r.metrics.quasiconformal, 1.02) << "flat grid should be near-conformal";
	EXPECT_GE(r.metrics.quasiconformal, 1.0);
}
