/*
* EngineHalfmesh.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include "engines/EngineHalfmesh.h"
#include "BenchMetrics.h"

#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>

#include "ChartFlattenCache.h" // detail:: cache-aware overloads, as GenerateAtlas uses

#include <chrono>

namespace hmbench {

namespace {
inline double Now()
{
	using clk = std::chrono::steady_clock;
	return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Apply the bench config's halfmesh-specific overrides (init method, iters).
inline void ApplyParamOverrides(halfmesh::ParametrizeParams& pp, const BenchConfig& cfg)
{
	if (cfg.flattenIterations >= 0)
		pp.flattenIterations = static_cast<unsigned>(cfg.flattenIterations);
	if (cfg.initMethod == "lscm")
		pp.initMethod = halfmesh::ParametrizeParams::InitMethod::LSCM;
	else if (cfg.initMethod == "tutte")
		pp.initMethod = halfmesh::ParametrizeParams::InitMethod::Tutte;
	if (cfg.seedMult >= 0.f)
		pp.seedExtraMult = cfg.seedMult;
	if (cfg.devSmooth >= 0)
		pp.developableSmoothIters = static_cast<unsigned>(cfg.devSmooth);
	if (cfg.flipRounds >= 0)
		pp.developableFlipRepairRounds = static_cast<unsigned>(cfg.flipRounds);
	if (cfg.coneErr >= 0.f)
		pp.developableMaxConeError = cfg.coneErr;
	if (cfg.vdefect >= 0.f)
		pp.developableMaxVertexDefect = cfg.vdefect;
	if (cfg.maxDistortion >= 0.f)
		pp.developableMaxUvDistortion = cfg.maxDistortion;
	if (cfg.cutToDisk)
		pp.cutToDisk = true;
	if (cfg.repairCarveRings >= 0)
		pp.repairCarveRings = static_cast<unsigned>(cfg.repairCarveRings);
	if (cfg.foldRescueSlits >= 0)
		pp.foldRescueSlits = static_cast<unsigned>(cfg.foldRescueSlits);
}

// Apply the bench config's per-size padding overrides (AtlasParams).
inline void ApplyAtlasOverrides(halfmesh::AtlasParams& ap, const BenchConfig& cfg)
{
	if (cfg.tinyChartSide >= 0.f)
		ap.tinyChartSide = cfg.tinyChartSide;
	if (cfg.debrisChartFaces >= 0)
		ap.debrisChartFaces = static_cast<unsigned>(cfg.debrisChartFaces);
}
} // namespace

EngineResult RunHalfmesh(halfmesh::Mesh mesh, const BenchConfig& cfg)
{
	EngineResult r;
	r.engine = "halfmesh";

	halfmesh::ParametrizeParams pp; // defaults: developable D-Charts segment + SLIM flatten
	ApplyParamOverrides(pp, cfg);
	halfmesh::AtlasParams ap;
	ap.resolution = cfg.resolution;
	ap.fitToResolution = true; // fit one page of `resolution`, like xatlas
	if (cfg.orient == "on")
		ap.orientCharts = true;
	else if (cfg.orient == "off")
		ap.orientCharts = false;
	ApplyAtlasOverrides(ap, cfg);

	std::vector<unsigned> faceChart;
	unsigned numCharts = 0;

	// Segmentation and flattening share one cache, exactly as GenerateAtlas
	// does: the flip-repair's accepting verdict already flattened every
	// shipping chart, so Module B resumes from those artifacts instead of
	// redoing the work. Without this the harness flattens every chart twice
	// and reports a cost no caller of GenerateAtlas ever pays — measured ~14x
	// slower end-to-end on a 100k-chart mesh. Output is unaffected: the
	// cache-aware overloads are byte-identical for any cache state.
	halfmesh::detail::ChartFlattenCache flattenCache;

	// --- Module A: segmentation ---------------------------------------------
	{
		const double t0 = Now();
		numCharts = halfmesh::detail::SegmentCharts(mesh, pp, faceChart, &flattenCache);
		r.segmentation.wallSeconds = Now() - t0;
		r.segmentation.completed = true;
	}
	FillSegmentation(r.metrics, mesh, faceChart, numCharts);

	// --- Module B: per-chart flattening -------------------------------------
	{
		const double t0 = Now();
		halfmesh::detail::ParametrizeCharts(mesh, faceChart, numCharts, pp, &flattenCache);
		r.parametrization.wallSeconds = Now() - t0;
		r.parametrization.completed = true;
	}

	// --- Measure parametrization quality (unit-scale, handled internally) ---
	FillParametrization(r.metrics, mesh, faceChart, numCharts);

	// --- Module C: density normalization (uniform texel density) ------------
	double packSeconds = 0.0;
	{
		const double t0 = Now();
		halfmesh::NormalizeChartDensity(mesh, faceChart, numCharts, ap);
		packSeconds += Now() - t0;
	}

	// --- Module D: packing --------------------------------------------------
	halfmesh::AtlasResult atlas;
	{
		const double t0 = Now();
		atlas = halfmesh::PackAtlas(mesh, faceChart, numCharts, ap);
		packSeconds += Now() - t0;
	}
	r.packing.wallSeconds = packSeconds;
	r.packing.completed = true;
	FillPacking(r.metrics, mesh);
	r.metrics.occupancyRect = atlas.occupancy; // engine-reported rectangle fill
	r.metrics.numPages = atlas.numPages;

	r.totalWallSeconds = r.segmentation.wallSeconds + r.parametrization.wallSeconds + r.packing.wallSeconds;

	r.validOutput = r.metrics.allFinite && r.metrics.chartCount > 0;
	r.peakRssBytes = PeakRssBytes();
	return r;
}

EngineResult RunHalfmeshParam(const halfmesh::Mesh& mesh,
                              const std::vector<unsigned>& faceChart,
                              unsigned numCharts, const BenchConfig& cfg)
{
	EngineResult r;
	r.engine = "halfmesh";
	FillSegmentation(r.metrics, mesh, faceChart, numCharts);

	halfmesh::Mesh m = mesh;
	halfmesh::ParametrizeParams pp;
	ApplyParamOverrides(pp, cfg);
	{
		const double t0 = Now();
		halfmesh::ParametrizeCharts(m, faceChart, numCharts, pp);
		r.parametrization.wallSeconds = Now() - t0;
		r.parametrization.completed = true;
	}
	r.totalWallSeconds = r.parametrization.wallSeconds;
	FillParametrization(r.metrics, m, faceChart, numCharts);
	r.validOutput = r.metrics.allFinite && numCharts > 0;
	r.peakRssBytes = PeakRssBytes();
	return r;
}

} // namespace hmbench
