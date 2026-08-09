/*
* EnginePmp.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/EnginePmp.h — pmp-library parametrization baselines.
// Per-chart LSCM and discrete-harmonic flattening (compiled when
// HALFMESH_BENCH_WITH_PMP=ON).
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>

#include <vector>

namespace hmbench {

EngineResult RunPmpLscm(const halfmesh::Mesh& mesh,
                        const std::vector<unsigned>& faceChart,
                        unsigned numCharts, const BenchConfig& cfg);

EngineResult RunPmpHarmonic(const halfmesh::Mesh& mesh,
                            const std::vector<unsigned>& faceChart,
                            unsigned numCharts, const BenchConfig& cfg);

} // namespace hmbench
