/*
* GoldenIO.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/golden/GoldenIO.h — shared helpers for the golden-fixture layer (§4b).
//
// Used by GoldenDiffTest.cpp (default build) to read committed frozen snapshots
// and compare current outputs within tolerance.
//
// Fixtures live in tests/data/golden/ and are committed for regression testing
// without any external dependency.  Two files per (mesh, op): a binary .ply
// result (lossless float positions + integer faces) and a small .json metrics
// sidecar.
//
// Namespace: hmtest::golden
#pragma once

#include <halfmesh/Mesh.h>

#include <map>
#include <string>
#include <vector>

namespace hmtest {
namespace golden {

// Absolute path to tests/data/golden/ (resolved from __FILE__ at compile time of
// the including TU via GoldenDir()).
std::string GoldenDir();

// File stem for a (mesh, op) fixture, e.g. "Cube__RemoveDuplicateFaces".
std::string FixtureStem(const std::string& meshName, const std::string& opName);

// Scalar metrics frozen alongside each result mesh.  All are recomputed from the
// result mesh with the shared metrics toolkit, so the .json is a redundant,
// human-readable cross-check on top of the lossless .ply.
struct GoldenMetrics
{
	uint32_t numVertices = 0;
	uint32_t numFaces = 0;
	uint32_t opCount = 0; // op-specific fix/removal count (0 if N/A)
	double surfaceArea = 0.0;
	double bboxMin[3] = {0, 0, 0};
	double bboxMax[3] = {0, 0, 0};
	double hausdorffToInput = 0.0; // symmetric Hausdorff result<->input
	double edgeLenMin = 0.0;
	double edgeLenMax = 0.0;
	double edgeLenMean = 0.0;
};

// Compute the golden metrics for a result mesh against its input.
GoldenMetrics ComputeGoldenMetrics(const halfmesh::Mesh& result,
                                   const halfmesh::Mesh& input,
                                   uint32_t opCount);

// JSON (de)serialization of GoldenMetrics — minimal hand-rolled writer/reader so
// the golden layer adds no new dependency to the default build.
bool WriteMetricsJson(const std::string& path, const GoldenMetrics& m);
bool ReadMetricsJson(const std::string& path, GoldenMetrics& out);

} // namespace golden
} // namespace hmtest
