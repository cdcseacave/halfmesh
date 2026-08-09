/*
* RemeshQuality.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/metrics/RemeshQuality.h — derived remeshing-quality scalars.
//
// Header-only helper (namespace hmtest::metrics) that distils the analytic
// metrics toolkit into the handful of scalar quantities an isotropic remesher is
// judged on: edge-length size uniformity, triangle-angle quality, vertex valence
// regularity, and fidelity to the input surface.  Used by both the remesh unit
// tests (MeshRemeshTest.cpp) and the benchmark (tests/bench/RemeshBench.cpp) so
// they measure quality in exactly the same way.
#pragma once

#include "Metrics.h"

#include <halfmesh/Mesh.h>

#include <cstdint>
#include <map>
#include <vector>

namespace hmtest {
namespace metrics {

struct RemeshQuality
{
	std::size_t numVertices = 0;
	std::size_t numFaces = 0;
	double edgeMean = 0; // mean edge length
	double edgeCov = 0; // coefficient of variation = stddev/mean (size uniformity; 0 = perfect)
	double minAngleMeanDeg = 0; // mean over faces of the per-face minimum angle
	double fracMinAngleLt30 = 0; // fraction of faces whose smallest angle < 30 deg
	double meanValence = 0; // mean vertex valence (incident-face count)
	double fracIrregularValence = 0; // fraction of vertices with valence != 6
	double meanDistToInput = 0; // mean per-vertex nearest-surface distance to the input
	double hausdorffToInput = 0; // symmetric Hausdorff vs the input mesh
	double bboxDiag = 0; // input bbox diagonal (for normalizing distances)

	double meanDistRatio() const { return bboxDiag > 0 ? meanDistToInput / bboxDiag : 0.0; }
	double hausdorffRatio() const { return bboxDiag > 0 ? hausdorffToInput / bboxDiag : 0.0; }
};

// Compute the derived quality report for `out`, measuring fidelity against `input`.
inline RemeshQuality ComputeRemeshQuality(const halfmesh::Mesh& input,
                                          const halfmesh::Mesh& out)
{
	RemeshQuality q;
	q.numVertices = out.vertices.size();
	q.numFaces = out.faces.size();

	const EdgeLengthStats els = ComputeEdgeLengthStats(out);
	q.edgeMean = els.meanLen;
	q.edgeCov = els.meanLen > 0.0 ? els.stddev / els.meanLen : 0.0;

	const std::vector<TriangleQuality> tq = ComputeAllTriangleQualities(out);
	double sumMin = 0.0;
	std::size_t below = 0;
	for (const auto& t : tq) {
		sumMin += t.minAngleDeg;
		if (t.minAngleDeg < 30.f)
			++below;
	}
	q.minAngleMeanDeg = tq.empty() ? 0.0 : sumMin / static_cast<double>(tq.size());
	q.fracMinAngleLt30 = tq.empty() ? 0.0 : static_cast<double>(below) / static_cast<double>(tq.size());

	const std::map<uint32_t, uint32_t> vh = ComputeValenceHistogram(out);
	std::uint64_t valenceSum = 0, vcount = 0, irregular = 0;
	for (const auto& kv : vh) {
		valenceSum += static_cast<std::uint64_t>(kv.first) * kv.second;
		vcount += kv.second;
		if (kv.first != 6u)
			irregular += kv.second;
	}
	q.meanValence = vcount ? static_cast<double>(valenceSum) / static_cast<double>(vcount) : 0.0;
	q.fracIrregularValence = vcount ? static_cast<double>(irregular) / static_cast<double>(vcount) : 0.0;

	const DistanceResult dr = ComputeDistanceKdTree(out, input);
	q.meanDistToInput = dr.meanSurfaceDist;
	q.hausdorffToInput = dr.hausdorffSymmetric;

	const AABB bb = ComputeAABB(input);
	q.bboxDiag = (bb.maxPt - bb.minPt).norm();
	return q;
}

} // namespace metrics
} // namespace hmtest
