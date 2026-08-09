/*
* ChartFlattenCache.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// src/ChartFlattenCache.h — INTERNAL (not installed) bridge types between the
// segmentation flip-repair (src/AtlasCharting.cpp, Module A) and the per-chart
// flattener (src/Parametrize.cpp, Module B).
//
// RepairDevelopableFlips accepts a chart only after detail::ChartFacesFold
// reports it does not fold — a verdict computed by flattening the chart exactly
// the way ParametrizeCharts ships it. Without a cache those accepted-chart UVs
// are discarded and immediately recomputed. The types below let the accepting
// verdict hand its artifacts (cut ChartMesh + UVs) forward so ParametrizeCharts
// can resume instead of recomputing; the output is bitwise identical because the
// artifacts ARE the flatten state ParametrizeCharts would reproduce.
//
// The per-chart entry type is defined in src/Parametrize.cpp (it wraps the
// TU-local ChartMesh) and stays opaque here; slots and the cache are movable
// handles whose entry-touching members are also defined there.
//
// Cache key = the chart's smallest global face id (face lists are sorted
// ascending). Shipped charts own disjoint, never-mutated face sets, so the key
// is unique — and, unlike a chart id, INVARIANT under the id relabelling done
// by EnforceConnectivity/Compact, so no key remap is needed across the
// flip-repair's trailing Compact. Consumers must still verify the full face
// list matches before reuse (a miss falls back to recomputing).

#pragma once

#include <halfmesh/Parametrize.h>

#include <unordered_map>
#include <vector>

namespace halfmesh {
namespace detail {

// One accepted chart's flatten artifacts. Defined in src/Parametrize.cpp.
struct ChartFlattenEntry;

// Move-only owning handle to a ChartFlattenEntry. Entry-deleting members are
// defined in src/Parametrize.cpp (the only TU with the complete entry type).
class ChartFlattenSlot
{
	public:
	ChartFlattenSlot() noexcept = default;
	~ChartFlattenSlot();
	ChartFlattenSlot(ChartFlattenSlot&& o) noexcept :
	    entry(o.entry) { o.entry = nullptr; }
	ChartFlattenSlot& operator=(ChartFlattenSlot&& o) noexcept;
	ChartFlattenSlot(const ChartFlattenSlot&) = delete;
	ChartFlattenSlot& operator=(const ChartFlattenSlot&) = delete;
	explicit operator bool() const noexcept { return entry != nullptr; }

	ChartFlattenEntry* entry = nullptr;
};

// Accepted-chart artifacts keyed by chart identity (smallest global face id).
// Write discipline: wave workers fill DISJOINT ChartFlattenSlot elements in
// parallel; Store() is called SERIALLY from the wave harvest. After the
// segmentation returns, the cache is read-only (concurrent find() from the
// ParametrizeCharts workers is safe).
class ChartFlattenCache
{
	public:
	ChartFlattenCache() = default;
	ChartFlattenCache(const ChartFlattenCache&) = delete;
	ChartFlattenCache& operator=(const ChartFlattenCache&) = delete;

	// Take ownership of an accepted chart's artifacts (no-op on an empty slot).
	// Defined in src/Parametrize.cpp.
	void Store(ChartFlattenSlot&& slot);

	// Look up a chart's cached artifacts, enforcing the verify-before-reuse
	// contract: key on the smallest global face id, then confirm the full face
	// list matches. Returns nullptr on miss. Const + read-only: safe to call
	// concurrently from flatten workers while Store() runs serially.
	// Defined in src/Parametrize.cpp.
	ChartFlattenEntry* Find(const std::vector<Mesh::FIndex>& globalFid) const;

	private:
	std::unordered_map<Mesh::FIndex, ChartFlattenSlot> entries;
};

// Fold bridge (defined in src/Parametrize.cpp): true if the chart made of these
// global faces folds when flattened exactly the way ParametrizeCharts ships it.
bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params);

// Extended fold bridge: identical verdict; additionally, when `out` is non-null
// and the chart does NOT fold (it ships), deposits the flatten artifacts of the
// accepting verdict into *out for ChartFlattenCache::Store.
bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params, ChartFlattenSlot* out);

// Cache-aware pipeline internals. Byte-identical output to the public overloads
// for ANY cache state (a lookup miss recomputes); the cache only removes the
// duplicate flatten work. Defined in src/AtlasCharting.cpp / src/Parametrize.cpp.
unsigned SegmentCharts(Mesh& mesh, const ParametrizeParams& params,
                       std::vector<unsigned>& faceChart, ChartFlattenCache* cache);
void ParametrizeCharts(Mesh& mesh, const std::vector<unsigned>& faceChart,
                       unsigned numCharts, const ParametrizeParams& params,
                       ChartFlattenCache* cache);

} // namespace detail
} // namespace halfmesh
