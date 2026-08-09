/*
* EngineLibigl.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/EngineLibigl.h — libigl parametrization baseline.
// Per-chart LSCM (compiled when HALFMESH_BENCH_WITH_LIBIGL=ON).
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>

#include <vector>

namespace hmbench {

EngineResult RunLibiglLscm(const halfmesh::Mesh& mesh,
                           const std::vector<unsigned>& faceChart,
                           unsigned numCharts, const BenchConfig& cfg);

} // namespace hmbench
