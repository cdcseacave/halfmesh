/*
* EngineBff.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/EngineBff.h — Boundary First Flattening baseline via
// geometry-central (compiled when HALFMESH_BENCH_WITH_BFF=ON).  Per-chart
// conformal flattening, plugged into the same fixed-chart isolation path as the
// libigl/pmp/CGAL baselines.
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>

#include <vector>

namespace hmbench {

EngineResult RunBffParam(const halfmesh::Mesh& mesh,
                         const std::vector<unsigned>& faceChart,
                         unsigned numCharts, const BenchConfig& cfg);

} // namespace hmbench
