/*
* EngineCgal.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/EngineCgal.h — CGAL baselines (compiled when
// HALFMESH_BENCH_WITH_CGAL=ON).
//   - RunCgalLscm / RunCgalArap: per-chart parametrization (Surface_mesh_
//     parameterization), plugged into the same fixed-chart isolation path as
//     the libigl/pmp baselines.
//   - RunCgalSegment: SDF-based mesh segmentation (segmentation metrics only),
//     a segmentation baseline alongside xatlas.
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>

#include <vector>

namespace hmbench {

EngineResult RunCgalLscm(const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart,
                         unsigned numCharts, const BenchConfig& cfg);

EngineResult RunCgalArap(const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart,
                         unsigned numCharts, const BenchConfig& cfg);

// CGAL SDF-graph-cut segmentation; fills only the segmentation metrics.
EngineResult RunCgalSegment(const halfmesh::Mesh& mesh, const BenchConfig& cfg);

} // namespace hmbench
