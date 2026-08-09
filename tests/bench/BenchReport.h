/*
* BenchReport.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/bench/BenchReport.h — comparison-report writers for atlasbench.
//
// Emits the same EngineResult set three ways: hand-rolled JSON (machine), a
// Markdown table per stage with a winner column (human), and a flat CSV (one
// row per engine×stage, for plotting).  No external JSON dependency — mirrors
// the report style in tests/perf/PerfHarness.cpp.
#pragma once

#include "BenchTypes.h"

#include <string>
#include <vector>

namespace hmbench {

// Render the Markdown report to a string (also what gets printed to stdout).
std::string RenderMarkdown(const BenchConfig& cfg,
                           const std::vector<EngineResult>& results);

// Write report.{json,md,csv} into cfg.outDir.  Returns false on write error.
bool WriteReports(const BenchConfig& cfg,
                  const std::vector<EngineResult>& results);

} // namespace hmbench
