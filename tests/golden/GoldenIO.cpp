/*
* GoldenIO.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/golden/GoldenIO.cpp — shared golden-fixture helpers (see GoldenIO.h).

#include "GoldenIO.h"

#include "Metrics.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace hmtest {
namespace golden {

std::string GoldenDir()
{
	// __FILE__ is tests/golden/GoldenIO.cpp → repo/tests/golden → repo/tests/data/golden.
	const std::filesystem::path here(__FILE__);
	return (here.parent_path().parent_path() / "data" / "golden").string();
}

std::string FixtureStem(const std::string& meshName, const std::string& opName)
{
	return meshName + "__" + opName;
}

GoldenMetrics ComputeGoldenMetrics(const halfmesh::Mesh& result,
                                   const halfmesh::Mesh& input,
                                   uint32_t opCount)
{
	GoldenMetrics m;
	m.numVertices = static_cast<uint32_t>(result.vertices.size());
	m.numFaces = static_cast<uint32_t>(result.faces.size());
	m.opCount = opCount;
	m.surfaceArea = metrics::ComputeSurfaceArea(result);

	const metrics::AABB box = metrics::ComputeAABB(result);
	for (int k = 0; k < 3; ++k) {
		m.bboxMin[k] = box.minPt[k];
		m.bboxMax[k] = box.maxPt[k];
	}

	if (!result.faces.empty() && !input.faces.empty()) {
		m.hausdorffToInput =
		    metrics::ComputeDistanceKdTree(result, input).hausdorffSymmetric;
	}

	const metrics::EdgeLengthStats els = metrics::ComputeEdgeLengthStats(result);
	m.edgeLenMin = els.minLen;
	m.edgeLenMax = els.maxLen;
	m.edgeLenMean = els.meanLen;
	return m;
}

bool WriteMetricsJson(const std::string& path, const GoldenMetrics& m)
{
	std::ofstream f(path);
	if (!f)
		return false;
	f.precision(17); // round-trip doubles exactly
	f << "{\n";
	f << "  \"num_vertices\": " << m.numVertices << ",\n";
	f << "  \"num_faces\": " << m.numFaces << ",\n";
	f << "  \"op_count\": " << m.opCount << ",\n";
	f << "  \"surface_area\": " << m.surfaceArea << ",\n";
	f << "  \"bbox_min\": [" << m.bboxMin[0] << ", " << m.bboxMin[1] << ", "
	  << m.bboxMin[2] << "],\n";
	f << "  \"bbox_max\": [" << m.bboxMax[0] << ", " << m.bboxMax[1] << ", "
	  << m.bboxMax[2] << "],\n";
	f << "  \"hausdorff_to_input\": " << m.hausdorffToInput << ",\n";
	f << "  \"edge_len_min\": " << m.edgeLenMin << ",\n";
	f << "  \"edge_len_max\": " << m.edgeLenMax << ",\n";
	f << "  \"edge_len_mean\": " << m.edgeLenMean << "\n";
	f << "}\n";
	return static_cast<bool>(f);
}

namespace {

// Minimal numeric extractor for our own fixed-shape JSON.  Finds `"key"` and
// parses the following number (handles ints/doubles, ignores arrays for scalars).
bool FindScalar(const std::string& s, const std::string& key, double& out)
{
	const std::string needle = "\"" + key + "\"";
	const size_t k = s.find(needle);
	if (k == std::string::npos)
		return false;
	size_t i = s.find(':', k);
	if (i == std::string::npos)
		return false;
	++i;
	while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n'))
		++i;
	if (i < s.size() && s[i] == '[')
		return false; // not a scalar
	char* end = nullptr;
	out = std::strtod(s.c_str() + i, &end);
	return end != s.c_str() + i;
}

bool FindVec3(const std::string& s, const std::string& key, double out[3])
{
	const std::string needle = "\"" + key + "\"";
	const size_t k = s.find(needle);
	if (k == std::string::npos)
		return false;
	size_t i = s.find('[', k);
	if (i == std::string::npos)
		return false;
	++i;
	for (int j = 0; j < 3; ++j) {
		char* end = nullptr;
		out[j] = std::strtod(s.c_str() + i, &end);
		if (end == s.c_str() + i)
			return false;
		i = static_cast<size_t>(end - s.c_str());
		const size_t comma = s.find(',', i);
		if (j < 2) {
			if (comma == std::string::npos)
				return false;
			i = comma + 1;
		}
	}
	return true;
}

} // namespace

bool ReadMetricsJson(const std::string& path, GoldenMetrics& out)
{
	std::ifstream f(path);
	if (!f)
		return false;
	std::stringstream ss;
	ss << f.rdbuf();
	const std::string s = ss.str();

	double tmp = 0.0;
	if (!FindScalar(s, "num_vertices", tmp))
		return false;
	out.numVertices = static_cast<uint32_t>(tmp);
	if (!FindScalar(s, "num_faces", tmp))
		return false;
	out.numFaces = static_cast<uint32_t>(tmp);
	if (!FindScalar(s, "op_count", tmp))
		return false;
	out.opCount = static_cast<uint32_t>(tmp);
	if (!FindScalar(s, "surface_area", out.surfaceArea))
		return false;
	if (!FindVec3(s, "bbox_min", out.bboxMin))
		return false;
	if (!FindVec3(s, "bbox_max", out.bboxMax))
		return false;
	if (!FindScalar(s, "hausdorff_to_input", out.hausdorffToInput))
		return false;
	if (!FindScalar(s, "edge_len_min", out.edgeLenMin))
		return false;
	if (!FindScalar(s, "edge_len_max", out.edgeLenMax))
		return false;
	if (!FindScalar(s, "edge_len_mean", out.edgeLenMean))
		return false;
	return true;
}

} // namespace golden
} // namespace hmtest
