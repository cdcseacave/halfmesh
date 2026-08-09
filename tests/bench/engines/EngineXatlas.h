/*
* EngineXatlas.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/engines/EngineXatlas.h — xatlas baseline engine for atlasbench.
//
// Runs the full xatlas pipeline (ComputeCharts → PackCharts) and converts the
// output back into a halfmesh::Mesh (per-corner UVs + per-face chart labels via
// xref→original vertices) so the shared metrics measure it identically.
#pragma once

#include "BenchTypes.h"

#include <halfmesh/Mesh.h>

namespace hmbench {

EngineResult RunXatlas(const halfmesh::Mesh& mesh, const BenchConfig& cfg);

} // namespace hmbench
