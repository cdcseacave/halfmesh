/*
* FootprintPacking.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// src/FootprintPacking.cpp — mesh-independent footprint (mask) packing: bottom-left
// first fit of block-quantized masks over per-row free intervals; see
// halfmesh/RectPacking.h for the API and the method.

#include <halfmesh/RectPacking.h>
#include <halfmesh/Util/Assert.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace halfmesh {

namespace {

// a page dimension: up to the multiple, or to a power of two
int RoundPageSize(int size, int multiple)
{
	ASSERT(size > 0);
	if (multiple > 0)
		return (size + multiple - 1) / multiple * multiple;
	int pow2 = 1;
	while (pow2 < size)
		pow2 *= 2;
	return pow2;
}

// one orientation of a footprint on the block grid: per block row the block
// columns [start,end) it spans (start >= end: empty)
struct Profile
{
	int cols = 0;
	std::vector<int> start, end;
	std::vector<int> rowOrder; // the non-empty block rows, longest span first
	uint64_t area = 0; // blocks covered by the spans
};

struct Footprint
{
	cv::Size size; // of the mask, as given
	Profile profiles[2]; // as given, and rotated (cv::ROTATE_90_COUNTERCLOCKWISE)
};

void ComputeProfile(const cv::Mat& mask, int block, Profile& profile)
{
	const int rows = (mask.rows + block - 1) / block;
	profile.cols = (mask.cols + block - 1) / block;
	profile.start.assign(rows, profile.cols);
	profile.end.assign(rows, 0);
	for (int r = 0; r < mask.rows; ++r) {
		const uint8_t* const row = mask.ptr<uint8_t>(r);
		int c = 0;
		while (c < mask.cols && !row[c])
			++c;
		if (c == mask.cols)
			continue;
		int e = mask.cols;
		while (!row[e - 1])
			--e;
		const int br = r / block;
		profile.start[br] = std::min(profile.start[br], c / block);
		profile.end[br] = std::max(profile.end[br], (e + block - 1) / block);
	}
	profile.rowOrder.clear();
	profile.area = 0;
	for (int br = 0; br < rows; ++br) {
		if (profile.start[br] >= profile.end[br])
			continue;
		profile.rowOrder.emplace_back(br);
		profile.area += static_cast<uint64_t>(profile.end[br] - profile.start[br]);
	}
	std::sort(profile.rowOrder.begin(), profile.rowOrder.end(), [&profile](int a, int b) {
		return profile.end[a] - profile.start[a] > profile.end[b] - profile.start[b];
	});
}

struct Placed
{
	uint32_t idx;
	int x, y; // block coordinates
	int orientation;
};

// Bottom-left first fit of the footprints, in the given order, into one page of
// the given width (blocks): each page block row keeps its free block-column
// intervals; a footprint takes, in whichever allowed orientation goes lower, the
// lowest row, then the leftmost column, where each of its row spans lies in a free
// interval of the page row it falls on. Footprints that would pass maxHeight
// (blocks) are left over.
void PackPage(const std::vector<Footprint>& footprints, const std::vector<uint32_t>& order, int W, int H,
              bool allowRotation, std::vector<Placed>& page, std::vector<uint32_t>& leftover)
{
	ASSERT(W > 0 && H > 0);
	using Interval = std::pair<int, int>; // [first,second)
	using Intervals = std::vector<Interval>;
	std::vector<Intervals> freeRows(H, Intervals{Interval(0, W)});
	std::vector<int> maxFree(H, W); // longest free interval of each page row
	Intervals feasible, next;
	// per footprint size (block columns, rows): the row below which the last search
	// for that size found no place; space is only ever taken, so a footprint of the
	// same size starts there
	std::unordered_map<uint64_t, int> cursors;
	for (const uint32_t idx : order) {
		const Footprint& footprint = footprints[idx];
		int bestX = 0, bestY = std::numeric_limits<int>::max(), bestO = -1;
		for (int o = 0; o < (allowRotation ? 2 : 1); ++o) {
			const Profile& p = footprint.profiles[o];
			const int rows = static_cast<int>(p.start.size());
			if (p.cols > W)
				continue;
			const int r0 = p.rowOrder.front(), len0 = p.end[r0] - p.start[r0];
			int& cursor = cursors[(static_cast<uint64_t>(p.cols) << 32) | static_cast<uint32_t>(rows)];
			int y = cursor;
			for (; y + rows <= H && y < bestY; ++y) {
				if (maxFree[y + r0] < len0)
					continue;
				// cheap rejection first: a row whose longest free interval is shorter than its span
				bool fits = true;
				for (const int r : p.rowOrder)
					if (maxFree[y + r] < p.end[r] - p.start[r]) {
						fits = false;
						break;
					}
				if (!fits)
					continue;
				// the columns the footprint can start at: for each of its rows, those that put
				// the row's span inside a free interval of the page row, intersected over the rows
				feasible.assign(1, Interval(0, W - p.cols + 1));
				for (const int r : p.rowOrder) {
					const int start = p.start[r], end = p.end[r], len = end - start;
					const Intervals& freeRow = freeRows[y + r];
					next.clear();
					for (const Interval& g : feasible) {
						// the free intervals that can hold the span for some x in g
						auto it = std::lower_bound(freeRow.cbegin(), freeRow.cend(), g.first + end,
						                           [](const Interval& gap, int v) { return gap.second < v; });
						for (; it != freeRow.cend() && it->first <= g.second - 1 + start; ++it) {
							if (it->second - it->first < len)
								continue;
							const Interval common(std::max(g.first, it->first - start), std::min(g.second, it->second - end + 1));
							if (common.first < common.second)
								next.emplace_back(common);
						}
					}
					feasible.swap(next);
					if (feasible.empty())
						break;
				}
				if (!feasible.empty()) {
					bestX = feasible.front().first;
					bestY = y;
					bestO = o;
					break;
				}
			}
			cursor = y;
		}
		if (bestO < 0) {
			leftover.emplace_back(idx);
			continue;
		}
		// occupy the spans
		const Profile& p = footprint.profiles[bestO];
		for (const int r : p.rowOrder) {
			Intervals& freeRow = freeRows[bestY + r];
			const Interval span(bestX + p.start[r], bestX + p.end[r]);
			const auto it = std::lower_bound(freeRow.begin(), freeRow.end(), span.second,
			                                 [](const Interval& gap, int v) { return gap.second < v; });
			ASSERT(it != freeRow.end() && it->first <= span.first && span.second <= it->second);
			const Interval gap = *it;
			if (gap.first < span.first && span.second < gap.second) {
				it->second = span.first;
				freeRow.insert(it + 1, Interval(span.second, gap.second));
			} else if (gap.first < span.first) {
				it->second = span.first;
			} else if (span.second < gap.second) {
				it->first = span.second;
			} else {
				freeRow.erase(it);
			}
			if (gap.second - gap.first == maxFree[bestY + r]) {
				int longest = 0;
				for (const Interval& f : freeRow)
					longest = std::max(longest, f.second - f.first);
				maxFree[bestY + r] = longest;
			}
		}
		page.push_back(Placed{idx, bestX, bestY, bestO});
	}
}

} // namespace

FootprintPackResult PackFootprints(const std::vector<cv::Mat>& masks,
                                   const FootprintPackParams& params,
                                   std::vector<FootprintPlacement>& placements)
{
	ASSERT(params.blockSize > 0 && params.maxPageSize >= 0 && params.sizeMultiple >= 0);
	const int block = static_cast<int>(params.blockSize);
	const int multiple = params.sizeMultiple;
	FootprintPackResult result;
	placements.assign(masks.size(), FootprintPlacement{});

	// the footprints, both orientations; an empty mask or one larger than a bounded
	// page in both orientations is not packed
	std::vector<Footprint> footprints(masks.size());
	std::vector<uint32_t> remaining;
	remaining.reserve(masks.size());
	for (size_t i = 0; i < masks.size(); ++i) {
		const cv::Mat& mask = masks[i];
		ASSERT(mask.type() == CV_8UC1);
		if (mask.empty() || mask.type() != CV_8UC1 || cv::countNonZero(mask) == 0)
			continue;
		if (params.maxPageSize > 0 && (mask.cols > params.maxPageSize || mask.rows > params.maxPageSize) && (!params.allowRotation || mask.rows > params.maxPageSize || mask.cols > params.maxPageSize))
			continue;
		Footprint& footprint = footprints[i];
		footprint.size = mask.size();
		ComputeProfile(mask, block, footprint.profiles[0]);
		if (params.allowRotation) {
			cv::Mat rotated;
			cv::rotate(mask, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
			ComputeProfile(rotated, block, footprint.profiles[1]);
		}
		remaining.emplace_back(static_cast<uint32_t>(i));
	}
	std::stable_sort(remaining.begin(), remaining.end(), [&footprints](uint32_t a, uint32_t b) {
		return footprints[a].profiles[0].area > footprints[b].profiles[0].area;
	});

	while (!remaining.empty()) {
		// every footprint has to fit an empty page alone: the page is at least as wide
		// as the narrower side of each (the other side runs down the page)
		uint64_t area = 0;
		int minWidth = 0, minHeight = 0;
		for (const uint32_t idx : remaining) {
			const Footprint& footprint = footprints[idx];
			area += footprint.profiles[0].area * static_cast<uint64_t>(block * block);
			// in whole blocks: a page W blocks wide holds a footprint up to W block columns
			const cv::Size size((footprint.size.width + block - 1) / block * block, (footprint.size.height + block - 1) / block * block);
			const bool across = params.allowRotation && size.height < size.width; // placed rotated when alone
			minWidth = std::max(minWidth, across ? size.height : size.width);
			minHeight = std::max(minHeight, across ? size.width : size.height);
		}
		int side = RoundPageSize(std::max(static_cast<int>(std::ceil(std::sqrt(static_cast<double>(area)))), 1), multiple);
		if (params.maxPageSize > 0)
			side = std::min(side, params.maxPageSize);
		const int maxHeight = params.maxPageSize > 0 ? params.maxPageSize : std::max(side * 4, minHeight);
		std::vector<int> widths;
		for (const int w : {side / 2, side, side * 2}) {
			int width = std::max(RoundPageSize(std::max(w, 1), multiple), RoundPageSize(minWidth, multiple));
			if (params.maxPageSize > 0)
				width = std::min(width, params.maxPageSize);
			if (std::find(widths.begin(), widths.end(), width) == widths.end())
				widths.emplace_back(width);
		}
		std::vector<Placed> bestPage;
		std::vector<uint32_t> bestLeftover;
		cv::Size bestSize;
		for (const int width : widths) {
			if (!bestPage.empty() && bestLeftover.empty() && static_cast<int64_t>(bestSize.area()) <= static_cast<int64_t>(width) * RoundPageSize(std::max(static_cast<int>((area + width - 1) / width), 1), multiple))
				continue; // this width cannot make a smaller page
			std::vector<Placed> page;
			std::vector<uint32_t> leftover;
			PackPage(footprints, remaining, width / block, maxHeight / block, params.allowRotation, page, leftover);
			if (page.empty())
				continue;
			cv::Size extent(0, 0);
			for (const Placed& placed : page) {
				const cv::Size size = placed.orientation ? cv::Size(footprints[placed.idx].size.height, footprints[placed.idx].size.width) : footprints[placed.idx].size;
				extent.width = std::max(extent.width, placed.x * block + size.width);
				extent.height = std::max(extent.height, placed.y * block + size.height);
			}
			const cv::Size size(std::min(RoundPageSize(extent.width, multiple), width), std::min(RoundPageSize(extent.height, multiple), maxHeight));
			if (bestPage.empty() || leftover.size() < bestLeftover.size() || (leftover.size() == bestLeftover.size() && size.area() < bestSize.area())) {
				bestPage.swap(page);
				bestLeftover.swap(leftover);
				bestSize = size;
			}
		}
		if (bestPage.empty())
			break; // nothing left fits a page: reported as unpacked
		const unsigned pageIdx = static_cast<unsigned>(result.pageSizes.size());
		result.pageSizes.emplace_back(bestSize);
		for (const Placed& placed : bestPage) {
			const Footprint& footprint = footprints[placed.idx];
			FootprintPlacement& placement = placements[placed.idx];
			placement.rotated = placed.orientation == 1;
			placement.rect = cv::Rect(cv::Point(placed.x * block, placed.y * block),
			                          placement.rotated ? cv::Size(footprint.size.height, footprint.size.width) : footprint.size);
			placement.page = pageIdx;
			placement.packed = true;
			++result.numPacked;
			result.footprintArea += footprint.profiles[placed.orientation].area * static_cast<uint64_t>(block * block);
		}
		remaining.swap(bestLeftover);
	}
	return result;
}

} // namespace halfmesh
