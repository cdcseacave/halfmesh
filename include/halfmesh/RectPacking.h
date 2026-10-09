/*
* RectPacking.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// halfmesh/RectPacking.h — mesh-independent rectangle and footprint bin packing.
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

enum class RectPackMode {
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

// ---------------------------------------------------------------------------
// Footprint packing: items are binary masks, not rectangles. Two items may share
// texels of their rectangles wherever neither mask is set, so irregular shapes
// (texture patches, UV charts) nest into each other's empty corners instead of
// each claiming its whole bounding box.
//
// The masks are quantized to a grid of blockSize x blockSize texels (a block is
// taken if any of its texels is set), each block row of a footprint described by
// the one span of block columns it covers (holes inside a row are not reused).
// Bottom-left first fit: largest footprint first, each takes the lowest row, then
// the leftmost column, where every one of its row spans lies in a free interval
// of the page row it falls on, in whichever orientation sits lower. A page is
// tried at three widths around the square root of the footprint area (half,
// equal, double) and the one leaving the fewest footprints over, then the
// smallest, is kept; its dimensions are cropped to the extent the footprints use.
// Footprints a page cannot take open the next page.
// ---------------------------------------------------------------------------

// Where an input mask landed. `rect` is the mask's rectangle on its page
// (width/height swapped when rotated); a rotated mask is turned 90 degrees in
// the winding-preserving direction of RectPlacement: the texel (x,y) of a mask
// w texels wide lands at (rect.x + y, rect.y + w-1 - x), i.e. the mask is placed
// as cv::rotate(mask, cv::ROTATE_90_COUNTERCLOCKWISE).
struct FootprintPlacement
{
	cv::Rect rect;
	bool rotated = false;
	unsigned page = 0;
	bool packed = false;
};

struct FootprintPackParams
{
	// Placement grid in texels; any gutter must already be part of the masks.
	unsigned blockSize = 4;
	// Page side bound in texels (0: unbounded, the page height is then capped at
	// four times its width, so a long tail still opens a new page).
	int maxPageSize = 0;
	// Round each page dimension up to this multiple; 0 rounds to a power of two.
	int sizeMultiple = 0;
	// Permit the 90-degree rotation.
	bool allowRotation = true;
};

struct FootprintPackResult
{
	std::vector<cv::Size> pageSizes; // one per page, each cropped to its content
	unsigned numPacked = 0;
	// Texels of the block-quantized footprints, on the same basis as the page
	// areas, so occupancy is footprintArea / sum(pageSizes area).
	uint64_t footprintArea = 0;
};

// Pack the masks (CV_8UC1, non-zero = footprint, each with at least one texel
// set). `placements` is indexed in lockstep with `masks`. A mask wider and taller
// than maxPageSize in both orientations is left unpacked (packed=false).
FootprintPackResult PackFootprints(const std::vector<cv::Mat>& masks,
                                   const FootprintPackParams& params,
                                   std::vector<FootprintPlacement>& placements);

} // namespace halfmesh
