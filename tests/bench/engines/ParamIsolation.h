/*
* ParamIsolation.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/ParamIsolation.h — parametrization-only comparison.
//
// Holds the chart partition FIXED (halfmesh's SegmentCharts) and flattens the
// SAME charts with each baseline parametrizer, so distortion deltas are
// attributable to the flattener alone — not to a different upstream charting.
// A chart a baseline cannot handle (no boundary, non-disk) falls back to a PCA
// planar projection and is counted, so it never contaminates the metrics with
// NaNs.
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>

#include <Eigen/Core>

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace hmbench {

// A single chart extracted as a standalone (local) submesh.
struct ChartSubmesh
{
	std::vector<Eigen::Vector3d> V; // local vertex positions
	std::vector<std::array<int, 3>> F; // local faces (indices into V)
	std::vector<uint32_t> globalFaces; // parallel to F: original mesh face id
};

// Split the mesh into one ChartSubmesh per chart id.
std::vector<ChartSubmesh> ExtractCharts(const halfmesh::Mesh& mesh,
                                        const std::vector<unsigned>& faceChart,
                                        unsigned numCharts);

// Universal fallback: project a chart's vertices onto their best-fit plane
// (PCA top-2 axes).  Always finite; used when a baseline flattener fails.
bool PcaProject(const ChartSubmesh& chart, std::vector<Eigen::Vector2d>& uv);

// flatten(chart, uv): fill `uv` (one per chart.V) with per-vertex 2D UVs.
// Return false to trigger the PCA fallback for that chart.
using ChartFlattener = std::function<bool(const ChartSubmesh&, std::vector<Eigen::Vector2d>&)>;

// Flatten every chart with `flatten` (PCA fallback on failure), assemble the
// per-corner UVs, and measure parametrization quality on the FIXED partition.
EngineResult RunPerChartFlatten(const std::string& name,
                                const halfmesh::Mesh& mesh,
                                const std::vector<unsigned>& faceChart,
                                unsigned numCharts,
                                const ChartFlattener& flatten);

} // namespace hmbench
