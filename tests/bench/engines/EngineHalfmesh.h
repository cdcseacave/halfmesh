/*
* EngineHalfmesh.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/EngineHalfmesh.h — halfmesh atlas engine for atlasbench.
//
// Runs the native pipeline stage-by-stage (SegmentCharts → ParametrizeCharts →
// NormalizeChartDensity → PackAtlas) so each stage is timed independently, then
// measures the packed atlas with the shared hmtest::metrics toolkit.
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>
#include <halfmesh/AtlasPacking.h>

namespace hmbench {

// Run halfmesh end-to-end on a COPY of `mesh` (caller's mesh is not modified —
// the mesh is passed by value).  Honors cfg.resolution and cfg.stage.
EngineResult RunHalfmesh(halfmesh::Mesh mesh, const BenchConfig& cfg);

// Parametrization-isolation: flatten a FIXED chart partition with halfmesh's
// own SLIM/ARAP flattener (no segmentation of its own).  For --stage param.
EngineResult RunHalfmeshParam(const halfmesh::Mesh& mesh,
                              const std::vector<unsigned>& faceChart,
                              unsigned numCharts, const BenchConfig& cfg);

} // namespace hmbench
