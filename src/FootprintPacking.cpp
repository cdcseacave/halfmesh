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

#include "FootprintPacking.h"

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
	cv::Size size; // of the mask, in this orientation
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
	profile.size = mask.size();
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
// width x height texels: each page block row keeps its free block-column
// intervals; a footprint takes, in whichever allowed orientation goes lower, the
// lowest row, then the leftmost column, where each of its row spans lies in a free
// interval of the page row it falls on and its mask lies inside the page. The
// search starts on the row the last one for the same bounding size ended on (the
// rows below are mostly taken by then); a footprint that finds no place from there
// searches the rows it skipped, so it is left over only if the page has no place
// for it. With stopAtLeftover the packing ends at the first one left over.
void PackPage(const std::vector<Footprint>& footprints, const std::vector<uint32_t>& order,
              int width, int height, int block, bool allowRotation, bool stopAtLeftover,
              std::vector<Placed>& page, std::vector<uint32_t>& leftover)
{
	ASSERT(width > 0 && height > 0);
	// the grid covers the page; a partial last block column or row takes only the
	// footprints whose mask ends inside the page
	const int W = (width + block - 1) / block, H = (height + block - 1) / block;
	using Interval = std::pair<int, int>; // [first,second)
	using Intervals = std::vector<Interval>;
	std::vector<Intervals> freeRows(H, Intervals{Interval(0, W)});
	std::vector<int> maxFree(H, W); // longest free interval of each page row
	Intervals feasible, next;
	// the leftmost block column at which the profile fits with its first row on
	// page row y, at most maxX; -1 if none
	const auto fitAt = [&](const Profile& p, int y, int maxX) {
		// cheap rejection first: a row whose longest free interval is shorter than its span
		for (const int r : p.rowOrder)
			if (maxFree[y + r] < p.end[r] - p.start[r])
				return -1;
		// the columns the footprint can start at: for each of its rows, those that put
		// the row's span inside a free interval of the page row, intersected over the rows
		feasible.assign(1, Interval(0, maxX + 1));
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
				return -1;
		}
		return feasible.front().first;
	};
	// per footprint size (block columns, rows): the row the last search for that
	// size ended on
	std::unordered_map<uint64_t, int> cursors;
	const auto cursor = [&cursors](const Profile& p) -> int& {
		return cursors[(static_cast<uint64_t>(p.cols) << 32) | static_cast<uint32_t>(p.start.size())];
	};
	const int numOrientations = allowRotation ? 2 : 1;
	for (const uint32_t idx : order) {
		const Footprint& footprint = footprints[idx];
		int bestX = 0, bestY = std::numeric_limits<int>::max(), bestO = -1;
		// the page rows [from,to) for orientation o, lowest first, up to the best
		// place so far; returns the row the search ended on
		const auto search = [&](int o, int from, int to) {
			const Profile& p = footprint.profiles[o];
			const int maxX = (width - p.size.width) / block;
			ASSERT(maxX + p.cols <= W && (height - p.size.height) / block + static_cast<int>(p.start.size()) <= H);
			int y = from;
			for (; y < to && y < bestY; ++y) {
				const int x = fitAt(p, y, maxX);
				if (x >= 0) {
					bestX = x;
					bestY = y;
					bestO = o;
					break;
				}
			}
			return y;
		};
		int starts[2] = {0, 0}, ends[2] = {0, 0};
		for (int o = 0; o < numOrientations; ++o) {
			const Profile& p = footprint.profiles[o];
			if (p.size.width > width || p.size.height > height)
				continue;
			ends[o] = (height - p.size.height) / block + 1; // past the last row the mask can start on
			int& y = cursor(p);
			starts[o] = std::min(y, ends[o]);
			y = search(o, starts[o], ends[o]);
		}
		// nothing from the cursors: the rows they skipped
		if (bestO < 0)
			for (int o = 0; o < numOrientations; ++o)
				if (starts[o] > 0) {
					search(o, 0, starts[o]);
					if (bestO == o)
						cursor(footprint.profiles[o]) = bestY;
				}
		if (bestO < 0) {
			leftover.emplace_back(idx);
			if (stopAtLeftover)
				return;
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

// the footprints of the masks, both orientations (footprints[i] for masks[i]);
// returns the indices of those to pack, largest first: an empty mask or one larger
// than a bounded page in both orientations is not packed
std::vector<uint32_t> MakeFootprints(const std::vector<cv::Mat>& masks, const FootprintPackParams& params, std::vector<Footprint>& footprints)
{
	const int block = static_cast<int>(params.blockSize);
	footprints.assign(masks.size(), Footprint{});
	std::vector<uint32_t> order;
	order.reserve(masks.size());
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
		order.emplace_back(static_cast<uint32_t>(i));
	}
	std::stable_sort(order.begin(), order.end(), [&footprints](uint32_t a, uint32_t b) {
		return footprints[a].profiles[0].area > footprints[b].profiles[0].area;
	});
	return order;
}

// texels of the given footprints, block-quantized, as unrotated
uint64_t FootprintArea(const std::vector<Footprint>& footprints, const std::vector<uint32_t>& indices, int block)
{
	uint64_t area = 0;
	for (const uint32_t idx : indices)
		area += footprints[idx].profiles[0].area;
	return area * static_cast<uint64_t>(block) * static_cast<uint64_t>(block);
}

// the page widths to try for the given footprints, of `area` texels in all, and
// the page height: every footprint has to fit an empty page alone, so the page is
// at least as wide as the narrower side of each (the other side runs down the page)
void PageCandidates(const std::vector<Footprint>& footprints, const std::vector<uint32_t>& indices, uint64_t area,
                    const FootprintPackParams& params, std::vector<int>& widths, int& maxHeight)
{
	const int block = static_cast<int>(params.blockSize), multiple = params.sizeMultiple;
	int minWidth = 0, minHeight = 0;
	for (const uint32_t idx : indices) {
		const Footprint& footprint = footprints[idx];
		// in whole blocks: a page W blocks wide holds a footprint up to W block columns
		const cv::Size size((footprint.size.width + block - 1) / block * block, (footprint.size.height + block - 1) / block * block);
		const bool across = params.allowRotation && size.height < size.width; // placed rotated when alone
		minWidth = std::max(minWidth, across ? size.height : size.width);
		minHeight = std::max(minHeight, across ? size.width : size.height);
	}
	int side = RoundPageSize(std::max(static_cast<int>(std::ceil(std::sqrt(static_cast<double>(area)))), 1), multiple);
	if (params.maxPageSize > 0)
		side = std::min(side, params.maxPageSize);
	maxHeight = params.maxPageSize > 0 ? params.maxPageSize : std::max(side * 4, minHeight);
	widths.clear();
	for (const int w : {side / 2, side, side * 2}) {
		int width = std::max(RoundPageSize(std::max(w, 1), multiple), RoundPageSize(minWidth, multiple));
		if (params.maxPageSize > 0)
			width = std::min(width, params.maxPageSize);
		if (std::find(widths.begin(), widths.end(), width) == widths.end())
			widths.emplace_back(width);
	}
}

bool ValidParams(const FootprintPackParams& params)
{
	ASSERT(params.blockSize > 0 && params.blockSize <= FootprintPackParams::maxBlockSize && params.maxPageSize >= 0 && params.sizeMultiple >= 0);
	return params.blockSize > 0 && params.blockSize <= FootprintPackParams::maxBlockSize;
}

} // namespace

FootprintPackResult PackFootprints(const std::vector<cv::Mat>& masks,
                                   const FootprintPackParams& params,
                                   std::vector<FootprintPlacement>& placements)
{
	FootprintPackResult result;
	placements.assign(masks.size(), FootprintPlacement{});
	if (!ValidParams(params))
		return result;
	const int block = static_cast<int>(params.blockSize);
	const uint64_t blockArea = static_cast<uint64_t>(block) * static_cast<uint64_t>(block);
	const int multiple = params.sizeMultiple;
	std::vector<Footprint> footprints;
	std::vector<uint32_t> remaining = MakeFootprints(masks, params, footprints);
	std::vector<int> widths;
	while (!remaining.empty()) {
		const uint64_t area = FootprintArea(footprints, remaining, block);
		int maxHeight;
		PageCandidates(footprints, remaining, area, params, widths, maxHeight);
		std::vector<Placed> bestPage;
		std::vector<uint32_t> bestLeftover;
		cv::Size bestSize;
		for (const int width : widths) {
			if (!bestPage.empty() && bestLeftover.empty() && static_cast<int64_t>(bestSize.area()) <= static_cast<int64_t>(width) * RoundPageSize(std::max(static_cast<int>((area + width - 1) / width), 1), multiple))
				continue; // this width cannot make a smaller page
			std::vector<Placed> page;
			std::vector<uint32_t> leftover;
			PackPage(footprints, remaining, width, maxHeight, block, params.allowRotation, false, page, leftover);
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
			result.footprintArea += footprint.profiles[placed.orientation].area * blockArea;
		}
		remaining.swap(bestLeftover);
	}
	return result;
}

namespace detail {

bool FootprintsFitOnePage(const std::vector<cv::Mat>& masks, const FootprintPackParams& params, uint64_t& footprintArea)
{
	footprintArea = 0;
	if (!ValidParams(params))
		return false;
	const int block = static_cast<int>(params.blockSize);
	std::vector<Footprint> footprints;
	const std::vector<uint32_t> order = MakeFootprints(masks, params, footprints);
	footprintArea = FootprintArea(footprints, order, block);
	if (order.empty() || order.size() < masks.size())
		return false; // a mask PackFootprints leaves unpacked
	std::vector<int> widths;
	int maxHeight;
	PageCandidates(footprints, order, footprintArea, params, widths, maxHeight);
	std::vector<Placed> page;
	std::vector<uint32_t> leftover;
	for (const int width : widths) {
		page.clear();
		leftover.clear();
		PackPage(footprints, order, width, maxHeight, block, params.allowRotation, true, page, leftover);
		if (leftover.empty())
			return true;
	}
	return false;
}

} // namespace detail

} // namespace halfmesh
