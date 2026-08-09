/*
* BenchMetrics.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/BenchMetrics.h — atlas-quality metrics for atlasbench.
//
// Wraps the shared hmtest::metrics toolkit (flips, sym-Dirichlet, Sander
// stretch, triangle occupancy) and adds the segmentation / distortion metrics
// the literature uses but that Metrics.h does not yet provide.  Everything here
// operates on the public halfmesh::Mesh + a parallel per-face chart-id vector,
// so every engine's converted output is measured identically.
//
// Style mirrors hmtest::metrics; the chart-partition set has been promoted
// (see below) and the distortion metrics may follow later.
#pragma once

#include "BenchTypes.h"
#include "Metrics.h" // hmtest::metrics (shared toolkit)

#include <halfmesh/Mesh.h>

#include <cstddef>
#include <vector>

namespace hmbench {

// ---------------------------------------------------------------------------
// Stage-targeted measurement.  Parametrization quality MUST be measured on the
// per-chart flatten output (after density normalization, BEFORE packing):
// packing applies per-chart rotations/reflections + the global flip heuristic
// and scale-sensitive energies are only meaningful pre-pack.  Occupancy MUST be
// measured AFTER packing (UVs in [0,1] atlas space).
// ---------------------------------------------------------------------------

// Fill segmentation fields (independent of UVs — needs only the chart labels).
void FillSegmentation(AtlasMetrics& m, const halfmesh::Mesh& mesh,
                      const std::vector<unsigned>& faceChart, unsigned numCharts);

// Fill parametrization fields (flips, sym-Dirichlet, stretch, quasiconformal,
// area distortion, finiteness).  The mesh's per-corner UVs may be at any scale
// (raw per-chart flatten OR packed atlas); this internally normalizes a copy to
// unit world scale per chart (uvArea == worldArea) so scale-sensitive
// energies are comparable across engines.  Distortion metrics are rotation- and
// reflection-invariant, so packed (rotated) UVs measure the same as pre-pack.
void FillParametrization(AtlasMetrics& m, const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart, unsigned numCharts);

// Fill packing fields (triangle occupancy, bbox overlaps) from the mesh's
// current (packed, [0,1]) UVs.  occupancyRect / numPages come from the engine.
void FillPacking(AtlasMetrics& m, const halfmesh::Mesh& mesh);

// Convenience: measure everything from ONE mesh state (used by param-isolation
// baselines that do not pack — occupancy is then meaningless and left unset).
AtlasMetrics MeasureAtlas(const halfmesh::Mesh& mesh,
                          const std::vector<unsigned>& faceChart,
                          unsigned numCharts);

// ---------------------------------------------------------------------------
// Individual segmentation metrics — defined in the shared toolkit
// (tests/metrics/Metrics.h, namespace hmtest::metrics) so the DEFAULT test
// suite can assert on segmentation quality (segment_quality_test); re-exported
// here for the hmbench call sites.
// ---------------------------------------------------------------------------
using hmtest::metrics::ComputeBoundaryCutLength;
using hmtest::metrics::ComputeChartPlanarityError;
using hmtest::metrics::ComputeChartCompactness;
using hmtest::metrics::ComputeChartCoverage;

// ---------------------------------------------------------------------------
// Individual parametrization-distortion metrics (operate on faceTexcoords).
// ---------------------------------------------------------------------------

// Area-weighted mean quasi-conformal distortion: sigma_max/sigma_min of the
// per-face 2x2 Jacobian of the 3D->UV map (1 = conformal).
double ComputeQuasiConformalDistortion(const halfmesh::Mesh& mesh);

struct AreaDistortion
{
	double scaleMin = 0.0;
	double scaleMax = 0.0;
	double ratio = 0.0; // scaleMax / scaleMin (1 = equiareal up to global scale)
};

AreaDistortion ComputeAreaDistortion(const halfmesh::Mesh& mesh);

// ---------------------------------------------------------------------------
// Process peak resident-set size in bytes (cross-cutting cost metric).
// ---------------------------------------------------------------------------
std::size_t PeakRssBytes();

} // namespace hmbench
