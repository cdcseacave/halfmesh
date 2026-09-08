/*
* PartitionChecks.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// tests/corpus/PartitionChecks.h — chart-partition invariants shared by the
// segmentation suites (ParametrizeTest, SegmentQualityTest). Header-only; the
// including TU must link gtest.
// Namespace: hmtest::checks
#pragma once

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <queue>
#include <vector>

namespace hmtest {
namespace checks {

// Valid partition: every face has a chart id in [0, n), and no id is empty.
inline void ExpectValidPartition(const std::vector<unsigned>& fc, unsigned n, size_t numFaces)
{
	ASSERT_EQ(fc.size(), numFaces);
	ASSERT_GE(n, 1u);
	std::vector<char> seen(n, 0);
	for (unsigned c : fc) {
		ASSERT_LT(c, n) << "chart id out of range";
		seen[c] = 1;
	}
	for (unsigned c = 0; c < n; ++c)
		EXPECT_TRUE(seen[c]) << "chart " << c << " is empty (ids not compact)";
}

// Each chart is a single connected face set over TOPO edges — every edge EXCEPT
// the seams a chart may never span (mesh border, non-manifold, texblob border).
// Charts span creases freely when the surface is developable across them, so
// connectivity is checked across creases too. An empty chart fails.
inline bool AllChartsConnectedTopo(const halfmesh::Mesh& m, const std::vector<unsigned>& fc, unsigned n)
{
	using halfmesh::HalfMesh;
	const HalfMesh& hm = m.halfMesh;
	const size_t nf = m.faces.size();
	const bool hasTexblobs = m.faceTexblobs.size() == m.faces.size();
	auto topoNb = [&](HalfMesh::HIndex iHe) -> HalfMesh::FIndex {
		const HalfMesh::HIndex tw = hm.HeTwin(iHe);
		if (hm.HeIsBoundary(iHe) || hm.HeIsBoundary(tw))
			return math::NO_ID;
		if (hm.EDegree(hm.HeEdge(iHe)) != 2)
			return math::NO_ID;
		const HalfMesh::FIndex nb = hm.HeFace(tw);
		if (nb == math::NO_ID)
			return math::NO_ID;
		if (hasTexblobs && m.faceTexblobs[hm.HeFace(iHe)] != m.faceTexblobs[nb])
			return math::NO_ID;
		return nb;
	};
	std::vector<char> visited(nf, 0);
	for (unsigned c = 0; c < n; ++c) {
		size_t seed = nf, total = 0;
		for (size_t f = 0; f < nf; ++f)
			if (fc[f] == c) {
				if (seed == nf)
					seed = f;
				++total;
			}
		if (seed == nf)
			return false; // empty chart
		std::queue<size_t> q;
		q.push(seed);
		visited[seed] = 1;
		size_t count = 1;
		while (!q.empty()) {
			const size_t f = q.front();
			q.pop();
			for (HalfMesh::HIndex iHe : hm.FAdjacentHalfedges(static_cast<HalfMesh::FIndex>(f))) {
				const HalfMesh::FIndex nb = topoNb(iHe);
				if (nb == math::NO_ID || visited[nb] || fc[nb] != c)
					continue;
				visited[nb] = 1;
				++count;
				q.push(nb);
			}
		}
		if (count != total)
			return false; // chart c is disconnected via topo edges
	}
	return true;
}

} // namespace checks
} // namespace hmtest
