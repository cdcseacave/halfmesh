/*
* RectPacking.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// halfmesh/RectPacking.h — mesh-independent rectangle bin packing.
//
// This mesh-independent counterpart to the atlas pipeline's float chart packer
// accepts integer pixel rectangles, so texture-atlas repacking, lightmap layout,
// sprite sheets, and similar callers can pack their existing rectangles without
// going through charting or touching a Mesh.
//
// The algorithm is a two-tier first-fit-decreasing over a set of skyline
// (min-waste) bins. Large rects go through the full min-waste skyline scan;
// rects whose padded long side falls under 1/32 of the page are placed on
// height-sorted shelves allocated through that same skyline, which keeps packing
// near-linear when the input runs to 100k+ tiny rects. Every page stays open, so
// a later small rect can fill space an earlier large one left behind — provably
// never more pages than closing the active bin on first overflow, usually fewer.
// Page dimensions and page count can be hard-bounded; unpacked inputs are
// reported explicitly instead of silently enlarging the atlas.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace halfmesh {

// Where an input cv::Rect landed. `rect` excludes the surrounding padding.
// Only the input width/height are consumed; x/y are ignored. A rotated rect is
// turned 90 degrees in the winding-preserving direction.
struct RectPlacement
{
	cv::Rect rect;
	bool rotated = false;
	unsigned page = 0;
	bool packed = false;
};

enum class RectPackMode
{
	// Repack from scratch while doubling page dimensions until every input fits
	// one page (or maxPageSize is reached).
	GrowSinglePage,
	// Never resize and use exactly one page; report inputs that do not fit.
	FixedSinglePage,
	// Never resize; open as many fixed-size pages as needed.
	FixedMultiPage
};

struct RectPackParams
{
	// Initial page dimensions in texels.
	cv::Size pageSize{1024, 1024};
	RectPackMode mode = RectPackMode::GrowSinglePage;
	// Growth ceiling for GrowSinglePage; an empty dimension means unbounded.
	cv::Size maxPageSize;
	// Gutter texels kept around every rect (applied on all four sides).
	unsigned padding = 2;
	// Permit 90-degree rotation while packing.
	bool allowRotation = true;
	// Round the page dimensions up to the next power of two.
	bool powerOfTwo = false;
	// Force square pages.
	bool square = false;
};

struct RectPackResult
{
	unsigned numPages = 0;
	unsigned numPacked = 0;
	cv::Size pageSize;
	// Total placed area INCLUDING padding, i.e. the same basis as
	// pageSize.area()*numPages, so occupancy is their ratio.
	uint64_t packedArea = 0;
};

// Pack `rects` according to `mode`:
//   GrowSinglePage  — retry with doubled dimensions until one page fits all;
//   FixedSinglePage — one bounded probe, exposing inputs that did not fit;
//   FixedMultiPage  — as many equal fixed-size pages as required.
// `placements` is indexed in lockstep with `rects`, so callers do not need an
// index wrapper even though packing reorders inputs internally. Degenerate,
// oversized, and growth-cap-limited inputs have packed=false.
RectPackResult PackRectangles(const std::vector<cv::Rect>& rects,
                              const RectPackParams& params,
                              std::vector<RectPlacement>& placements);

// Approximate the smallest square page for these rects at the requested target
// occupancy. If `multiple` is non-zero, round up to that multiple; otherwise
// round up to a power of two.
int EstimateSquareTextureSize(const std::vector<cv::Rect>& rects,
                              int multiple = 0,
                              float targetOccupancy = 0.9f);

} // namespace halfmesh
