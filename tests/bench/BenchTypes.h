/*
* BenchTypes.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/BenchTypes.h — shared data structures for the atlasbench harness.
//
// Every benchmark engine (halfmesh, xatlas, libigl, pmp, BFF, CGAL) produces a
// halfmesh::Mesh whose faceTexcoords hold the packed atlas UVs plus a parallel
// per-face chart-id vector.  The SAME hmtest::metrics functions then measure all
// engines, so the comparison is apples-to-apples.  These structs collect the
// measured numbers + per-stage timing into one record per engine.
#pragma once

#include <string>
#include <vector>

namespace hmbench {

// ---------------------------------------------------------------------------
// Per-stage timing + completion status (robustness signal).
// ---------------------------------------------------------------------------
struct StageTiming
{
	double wallSeconds = 0.0;
	bool completed = false; // did the stage finish without throwing?
	bool timedOut = false; // did the soft watchdog fire?
};

// ---------------------------------------------------------------------------
// All metrics measurable on a finished (segmented + flattened + packed) atlas.
// Not every engine fills every field (e.g. param-only baselines leave packing
// fields at their defaults); unset numeric fields stay at the sentinel below.
// ---------------------------------------------------------------------------
constexpr double UNSET = -1.0;

struct AtlasMetrics
{
	// ---- segmentation -------------------------------------------------------
	unsigned chartCount = 0;
	double boundaryCutLength = UNSET; // total 3-D seam length between charts
	double meanPlanarityError = UNSET; // area-weighted mean normal deviation (rad)
	double meanChartCompactness = UNSET; // mean perimeter^2 / (4*pi*area)
	bool fullCoverage = false; // every face labeled exactly once

	// ---- parametrization (from ComputeUVMetrics + BenchMetrics) -------------
	int flipCount = -1; // triangles flipped in UV (-1 = not measured)
	double symDirichlet = UNSET; // area-weighted symmetric-Dirichlet (min 4)
	double stretchL2 = UNSET; // Sander L2 stretch
	double quasiconformal = UNSET; // mean sigma_max/sigma_min of UV Jacobian (1 = conformal)
	double areaDistortionRatio = UNSET; // global scaleMax/scaleMin
	bool allFinite = true;

	// ---- packing ------------------------------------------------------------
	double occupancyRect = UNSET; // engine-reported rectangle fill (halfmesh AtlasResult)
	double occupancyTri = UNSET; // true triangle fill (UVMetrics.atlasOccupancy)
	unsigned numPages = 0;
	bool hasBboxOverlaps = false;
};

// ---------------------------------------------------------------------------
// One engine's full result on one input mesh.
// ---------------------------------------------------------------------------
struct EngineResult
{
	std::string engine; // "halfmesh" | "xatlas" | "libigl-lscm" | ...
	bool validOutput = false; // produced a finite, measurable atlas
	std::size_t peakRssBytes = 0;

	// Per-stage timing.  For end-to-end engines (halfmesh, xatlas) the three
	// stage timers may be summed into totalWallSeconds; param-only baselines
	// fill only `parametrization`.
	StageTiming segmentation;
	StageTiming parametrization;
	StageTiming packing;
	double totalWallSeconds = 0.0;

	AtlasMetrics metrics;
	std::string note; // free-form (e.g. "timed out in flatten", "no segmentation")
};

// ---------------------------------------------------------------------------
// Benchmark configuration (parsed from the CLI).
// ---------------------------------------------------------------------------
struct BenchConfig
{
	std::string meshPath;
	std::string tier = "small"; // small | medium | large
	unsigned resolution = 1024;
	unsigned decimateTargetFaces = 0; // 0 = no decimation
	double perStageTimeoutS = 120.0;
	std::vector<std::string> engines; // selected engine names
	std::string stage = "all"; // all | segment | param | pack
	std::string outDir = "."; // where report.{json,md,csv} are written
	unsigned seed = 42;
	int flattenIterations = -1; // override halfmesh flattenIterations (<0 = default, 0 = init-only)
	std::string initMethod = "default"; // halfmesh UV init: default | lscm | tutte
	std::string orient = "default"; // halfmesh chart orientation: default | on | off
	std::string saveMesh; // if set: save the (decimated) mesh here and exit
	bool repair = false; // clean the mesh (dedup/degenerate/non-manifold) after load
	bool fixManifold = false; // minimal: only FixNonManifold (keeps all faces)
	bool diagnose = false; // if set: print mesh-health stats (manifoldness, etc.) and exit
	// developable (D-Charts) segmentation tuning; <0 = library default
	int devSmooth = -1; // developableSmoothIters (Taubin geometry denoise)
	int flipRounds = -1; // developableFlipRepairRounds (0 = off)
	float seedMult = -1.f; // initial farthest-point extra-seed multiple
	float coneErr = -1.f; // developableMaxConeError
	float vdefect = -1.f; // developableMaxVertexDefect (radians)
	float maxDistortion = -1.f; // developableMaxUvDistortion (sym-Dir cap; 0=off; <0 = default)
	bool cutToDisk = false; // Seamster cut-to-disk instead of disk-split (fewer MVS charts)
};

} // namespace hmbench
