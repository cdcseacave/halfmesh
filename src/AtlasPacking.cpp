/*
* AtlasPacking.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// src/AtlasPacking.cpp — Module D of the atlas pipeline: pack the per-chart UV
// islands (already density-normalised by NormalizeChartDensity) into a single
// texture page. A skyline bottom-left bin packer with a min-area-rect
// pre-orientation per chart; see halfmesh/AtlasPacking.h for the API.

#include <halfmesh/AtlasPacking.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/RectPacking.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace halfmesh {

namespace {

using TexCoord = Mesh::TexCoord;
using FIndex = Mesh::FIndex;

struct Rect
{
	float x = 0.f, y = 0.f, w = 0.f, h = 0.f;
	bool rotated = false; // true if the chart was rotated 90° to fit here
};

// ============================================================================
// Skyline bin packer (Jylänki "Skyline" / stb_rect_pack style, bottom-left).
//
// MaxRects above is O(free_rects²) per insert (the Prune step), which explodes
// to minutes on atlases with many thousands of charts. The skyline packer keeps
// only the upper contour as a list of horizontal segments, so each insert is
// O(#segments) and the contour stays small — packing 10k+ charts in well under a
// second with occupancy on par with MaxRects. Same Insert() interface.
// ============================================================================
struct SkylineBin
{
	float binW, binH;
	struct Node
	{
		float x, y, width;
	}; // segment: left edge at x, top at y, spanning `width`
	std::vector<Node> nodes;

	SkylineBin(float w, float h) :
	    binW(w), binH(h)
	{
		nodes.push_back({0.f, 0.f, w});
	}

	// Does a (rw × rh) rect placed with its left edge at nodes[i].x fit? If so,
	// outY = the y it would rest at (max contour height over its width) and
	// outWaste = the area buried under the placement — Σ (outY − nodes[j].y)·overlap
	// over the segments it spans. That buried area is permanently lost, so preferring
	// the least-waste placement (Jyländki Skyline-MW) beats pure bottom-left.
	bool RectFits(int i, float rw, float rh, float& outY, float& outWaste) const
	{
		const float x = nodes[i].x;
		if (x + rw > binW + 1e-3f)
			return false;
		float y = nodes[i].y;
		float widthLeft = rw;
		int j = i;
		while (widthLeft > 0.f) {
			if (j >= static_cast<int>(nodes.size()))
				return false;
			y = std::max(y, nodes[j].y);
			if (y + rh > binH + 1e-3f)
				return false;
			widthLeft -= nodes[j].width;
			++j;
		}
		outY = y;
		// Second pass (now that the resting height y is known): sum the gap area between
		// the rect bottom (y) and each spanned segment's top.
		float waste = 0.f;
		float wl = rw;
		for (int k = i; wl > 0.f && k < static_cast<int>(nodes.size()); ++k) {
			const float overlap = std::min(wl, nodes[k].width);
			waste += (y - nodes[k].y) * overlap;
			wl -= nodes[k].width;
		}
		outWaste = waste;
		return true;
	}

	bool Insert(float rw, float rh, bool allowRotation, Rect& out)
	{
		float bestWaste = std::numeric_limits<float>::max();
		float bestY = std::numeric_limits<float>::max();
		float bestX = std::numeric_limits<float>::max();
		int bestI = -1;
		bool bestRot = false;
		float bestPlaceY = 0.f, bestPw = 0.f, bestPh = 0.f;

		auto consider = [&](float pw, float ph, bool rot) {
			for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
				float y, waste;
				if (!RectFits(i, pw, ph, y, waste))
					continue;
				const float top = y + ph;
				// Skyline min-waste: least buried area first, then bottom-left (min top
				// edge, then leftmost x). STRICT inequalities so the first-considered
				// placement wins exact ties — non-rotated is considered first, so an
				// equal-cost non-rotated placement is preferred over a rotation.
				if (waste < bestWaste
				    || (waste == bestWaste
				        && (top < bestY || (top == bestY && nodes[i].x < bestX)))) {
					bestWaste = waste;
					bestY = top;
					bestX = nodes[i].x;
					bestI = i;
					bestRot = rot;
					bestPlaceY = y;
					bestPw = pw;
					bestPh = ph;
				}
			}
		};
		consider(rw, rh, false);
		if (allowRotation && rw != rh)
			consider(rh, rw, true);

		if (bestI < 0)
			return false;

		out = {bestX, bestPlaceY, bestPw, bestPh, bestRot};
		AddLevel(bestI, bestX, bestPlaceY + bestPh, bestPw);
		return true;
	}

	private:
	// Raise the skyline: insert a segment at top `newY` over [x, x+w], clip the
	// segments it covers, then merge same-height neighbors to keep it compact.
	void AddLevel(int idx, float x, float newY, float w)
	{
		nodes.insert(nodes.begin() + idx, Node{x, newY, w});
		for (int i = idx + 1; i < static_cast<int>(nodes.size());) {
			const float prevRight = nodes[i - 1].x + nodes[i - 1].width;
			if (nodes[i].x < prevRight) {
				const float shrink = prevRight - nodes[i].x;
				nodes[i].x += shrink;
				nodes[i].width -= shrink;
				if (nodes[i].width <= 0.f) {
					nodes.erase(nodes.begin() + i);
					continue;
				}
			}
			break;
		}
		for (int i = 0; i + 1 < static_cast<int>(nodes.size());) {
			if (nodes[i].y == nodes[i + 1].y) {
				nodes[i].width += nodes[i + 1].width;
				nodes.erase(nodes.begin() + i + 1);
			} else {
				++i;
			}
		}
	}
};

// A rect's extent plus the gutter kept on both sides, widened to int64_t so the
// sum cannot overflow whatever int-sized extent the caller handed in. Every
// padded-size decision goes through this one place; computing it in int at any
// of them is undefined behaviour for extents near INT_MAX.
inline int64_t PaddedExtent(int extent, unsigned padding)
{
	return static_cast<int64_t>(extent) + 2 * static_cast<int64_t>(padding);
}

// Round up to next power-of-two >= v.
inline unsigned NextPow2(unsigned v)
{
	if (v == 0)
		return 1;
	--v;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

inline double Cross2(const Eigen::Vector2d& o, const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
	return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
}

// Convex hull (Andrew's monotone chain), CCW, no repeated endpoint.
std::vector<Eigen::Vector2d> ConvexHull2D(std::vector<Eigen::Vector2d> p)
{
	std::sort(p.begin(), p.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
		return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
	});
	p.erase(std::unique(p.begin(), p.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
		        return a.x() == b.x() && a.y() == b.y();
	        }),
	        p.end());
	const int n = static_cast<int>(p.size());
	if (n < 3)
		return p;
	std::vector<Eigen::Vector2d> h(2 * n);
	int k = 0;
	for (int i = 0; i < n; ++i) {
		while (k >= 2 && Cross2(h[k - 2], h[k - 1], p[i]) <= 0.0)
			--k;
		h[k++] = p[i];
	}
	for (int i = n - 2, t = k + 1; i >= 0; --i) {
		while (k >= t && Cross2(h[k - 2], h[k - 1], p[i]) <= 0.0)
			--k;
		h[k++] = p[i];
	}
	h.resize(k - 1);
	return h;
}

// Rotation angle (radians) that, applied as a rotation by -angle, aligns the
// chart's minimum-area bounding rectangle with the axes. The min-area enclosing
// rectangle always has one edge collinear with a convex-hull edge, so we test
// each hull-edge orientation and keep the smallest-area box. Returns 0 on a
// degenerate (sub-triangle) point set.
double MinAreaRectAngle(const std::vector<Eigen::Vector2d>& pts)
{
	const std::vector<Eigen::Vector2d> hull = ConvexHull2D(pts);
	const int m = static_cast<int>(hull.size());
	if (m < 3)
		return 0.0;
	double bestArea = std::numeric_limits<double>::max();
	double bestAng = 0.0;
	for (int i = 0; i < m; ++i) {
		const Eigen::Vector2d e = hull[(i + 1) % m] - hull[i];
		const double len = e.norm();
		if (len < 1e-20)
			continue;
		const Eigen::Vector2d dir = e / len;
		const Eigen::Vector2d perp(-dir.y(), dir.x());
		double mind = 1e300, maxd = -1e300, minp = 1e300, maxp = -1e300;
		for (const Eigen::Vector2d& q : hull) {
			const double a = q.dot(dir), b = q.dot(perp);
			mind = std::min(mind, a);
			maxd = std::max(maxd, a);
			minp = std::min(minp, b);
			maxp = std::max(maxp, b);
		}
		const double area = (maxd - mind) * (maxp - minp);
		if (area < bestArea) {
			bestArea = area;
			bestAng = std::atan2(dir.y(), dir.x());
		}
	}
	return bestAng;
}

// Per-chart axis-aligned rect (texels, pre-padding) plus its UV bbox. `degenerate`
// marks a zero-UV-area chart (collapsed to a point/segment by the flattener): it is
// clamped to a ≥1-texel slot and its UVs are collapsed to the slot centre so its raw,
// unnormalized extent cannot bleed over neighbours or exceed [0,1].
struct ChartRect
{
	// Keep pre-pack dimensions as floats because fit-to-resolution repeatedly
	// rescales UVs; rounding at every probe would accumulate density error.
	float w = 0.f, h = 0.f; // size in texels (before padding)
	float uvMinX = 0.f, uvMinY = 0.f; // bbox min (should be ~0)
	float uvMaxX = 0.f, uvMaxY = 0.f;
	bool degenerate = false;
};

// Placement of a chart in the packed atlas (texel space, before UV normalization).
struct Placement
{
	float x = 0.f, y = 0.f;
	bool rotated = false;
	unsigned page = 0;
};

// Placement produced by the shared two-tier packer, in texel space: (x, y) is the
// top-left corner of the rect ITSELF, with the padding already stepped over.
struct PackedRect
{
	float x = 0.f, y = 0.f;
	bool rotated = false;
	unsigned page = 0;
	bool packed = false;
};

// Two-tier first-fit-decreasing over a growing set of skyline bins, shared by the
// float chart packer and the public integer rectangle API:
//   - head (padded long side >= pageW/32): full min-waste skyline scan. Insert() is
//     O(#segments) per PROBE and every rect probes every segment, so pushing 100k+
//     tiny rects through it is quadratic in rect count (measured: 78% of a 3h45m
//     production unwrap).
//   - tail (everything smaller): height-descending shelves, so each shelf's FIRST
//     rect is its tallest and everything after it fits the shelf height. A shelf is
//     allocated THROUGH the same skyline as one wide pseudo-rect, so shelves nestle
//     into the contour the head left and the multi-page logic is untouched; inside a
//     shelf, placement is O(1). At most ~32x32 rects per page can exceed the
//     threshold, so the skyline tier stays small and keeps its quality where it
//     matters. Unfilled shelf remainder counts as waste (packedArea sums only real
//     padded rect areas) -- honest, and small under the height sort.
// Every page stays open, so a later small rect can fill the space an earlier large
// one left behind -- provably never more pages than closing the active bin on the
// first overflow, usually fewer.
//  - sizes: each input's UNPADDED (width, height); a non-positive entry is skipped
//  - singlePage: never open a second page; rects that do not fit stay unpacked
// return the number of pages opened, and accumulate the placed area INCLUDING
// padding (the same basis as pageW*pageH*pages, so occupancy is their ratio)
unsigned PackTwoTier(const std::vector<Eigen::Vector2f>& sizes,
                     float pageW, float pageH, float pad, bool allowRotation,
                     bool singlePage, std::vector<PackedRect>& placements,
                     double& packedArea)
{
	const unsigned numRects = static_cast<unsigned>(sizes.size());
	placements.assign(numRects, PackedRect{});
	std::vector<unsigned> order(numRects);
	std::iota(order.begin(), order.end(), 0u);
	std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
		return sizes[a].x() * sizes[a].y() > sizes[b].x() * sizes[b].y();
	});

	std::vector<SkylineBin> bins;
	bins.emplace_back(pageW, pageH);
	packedArea = 0;

	const float tierThreshold = pageW / 32.f;
	struct TailRect
	{
		unsigned ci;
		float rw, rh;
		bool rot;
	};
	std::vector<unsigned> head;
	std::vector<TailRect> tail;
	for (unsigned ci : order) {
		if (sizes[ci].x() <= 0.f || sizes[ci].y() <= 0.f)
			continue;
		float rw = sizes[ci].x() + 2.f * pad;
		float rh = sizes[ci].y() + 2.f * pad;
		if ((rw > pageW || rh > pageH) && !(allowRotation && rh <= pageW && rw <= pageH))
			continue; // does not fit a page in either orientation
		if (std::max(rw, rh) >= tierThreshold) {
			head.push_back(ci);
			continue;
		}
		// Shelf tier: pre-decide the rotation (lowest profile: height <= width),
		// same winding-preserving 90-degree convention as the skyline placements.
		bool rot = false;
		if (allowRotation && rh > rw) {
			std::swap(rw, rh);
			rot = true;
		}
		tail.push_back({ci, rw, rh, rot});
	}

	for (unsigned ci : head) {
		const float rw = sizes[ci].x() + 2.f * pad;
		const float rh = sizes[ci].y() + 2.f * pad;
		Rect placed;
		unsigned page = 0;
		bool ok = false;
		// First-fit across every open page (Insert is O(#segments), so a few probes
		// per rect are cheap); single-page results are unchanged.
		for (unsigned p = 0; p < static_cast<unsigned>(bins.size()); ++p) {
			if (bins[p].Insert(rw, rh, allowRotation, placed)) {
				page = p;
				ok = true;
				break;
			}
		}
		if (!ok) {
			if (singlePage)
				continue;
			page = static_cast<unsigned>(bins.size());
			bins.emplace_back(pageW, pageH);
			if (!bins[page].Insert(rw, rh, allowRotation, placed)) {
				bins.pop_back();
				continue;
			}
		}
		placements[ci] = {placed.x + pad, placed.y + pad, placed.rotated, page, true};
		packedArea += static_cast<double>(placed.w) * placed.h;
	}

	std::sort(tail.begin(), tail.end(), [](const TailRect& a, const TailRect& b) {
		return a.rh > b.rh;
	});
	struct Shelf
	{
		float x = 0.f, y = 0.f, w = 0.f, h = 0.f, cursor = 0.f;
		unsigned page = 0;
		bool open = false;
	};
	Shelf shelf;
	const auto OpenShelf = [&](float rw, float rh) {
		// Prefer wide shelves; fall back to narrower ones that still slot into
		// leftover contour gaps.
		for (const int denom : {1, 2, 4, 8}) {
			const float sw = std::max(rw, pageW / static_cast<float>(denom));
			for (unsigned p = 0; p < static_cast<unsigned>(bins.size()); ++p) {
				Rect rect;
				if (bins[p].Insert(sw, rh, false, rect)) {
					shelf = {rect.x, rect.y, sw, rh, rect.x, p, true};
					return true;
				}
			}
		}
		if (singlePage)
			return false;
		const unsigned page = static_cast<unsigned>(bins.size());
		bins.emplace_back(pageW, pageH);
		Rect rect;
		const float sw = std::max(rw, pageW);
		if (!bins[page].Insert(sw, rh, false, rect)) {
			bins.pop_back();
			return false;
		}
		shelf = {rect.x, rect.y, sw, rh, rect.x, page, true};
		return true;
	};
	for (const TailRect& t : tail) {
		if ((!shelf.open || shelf.cursor + t.rw > shelf.x + shelf.w + 1e-3f || t.rh > shelf.h + 1e-3f)
		    && !OpenShelf(t.rw, t.rh))
			continue;
		placements[t.ci] = {shelf.cursor + pad, shelf.y + pad, t.rot, shelf.page, true};
		shelf.cursor += t.rw;
		packedArea += static_cast<double>(t.rw) * t.rh;
	}
	return static_cast<unsigned>(bins.size());
}

// Pack the chart extents as floats: fit-to-resolution rescales the UVs and repacks
// on every probe, and rounding to whole texels at each one would accumulate density
// error. The public rectangle API below is integral, hence the two thin wrappers
// over the shared packer.
void PackRects(const std::vector<ChartRect>& crects, unsigned numCharts,
               const AtlasParams& params, unsigned pad,
               std::vector<Placement>& placements, unsigned& outPages,
               unsigned& outPw, unsigned& outPh, float& outPackedArea)
{
	// Page dims: requested resolution grown to fit the largest padded chart, so
	// every chart is placeable and nothing is dropped.
	std::vector<Eigen::Vector2f> sizes(numCharts);
	unsigned pageW = params.resolution;
	unsigned pageH = params.resolution;
	for (unsigned ci = 0; ci < numCharts; ++ci) {
		const ChartRect& cr = crects[ci];
		sizes[ci] = Eigen::Vector2f(cr.w, cr.h);
		if (cr.w <= 0.f || cr.h <= 0.f)
			continue;
		pageW = std::max(pageW, static_cast<unsigned>(std::ceil(cr.w + 2.f * pad)));
		pageH = std::max(pageH, static_cast<unsigned>(std::ceil(cr.h + 2.f * pad)));
	}
	if (params.powerOfTwo) {
		pageW = NextPow2(pageW);
		pageH = NextPow2(pageH);
	}
	if (params.square)
		pageW = pageH = std::max(pageW, pageH);

	std::vector<PackedRect> packed;
	double packedArea = 0;
	outPages = PackTwoTier(sizes, static_cast<float>(pageW), static_cast<float>(pageH),
	                       static_cast<float>(pad), params.allowRotation,
	                       /*singlePage*/ false, packed, packedArea);
	placements.assign(numCharts, Placement{});
	for (unsigned ci = 0; ci < numCharts; ++ci) {
		ASSERT(packed[ci].packed || crects[ci].w <= 0.f || crects[ci].h <= 0.f);
		placements[ci] = {packed[ci].x, packed[ci].y, packed[ci].rotated, packed[ci].page};
	}
	outPw = pageW;
	outPh = pageH;
	outPackedArea = static_cast<float>(packedArea);
}

} // namespace

// ---------------------------------------------------------------------------
// Generic rectangle bin packing — the API declared in halfmesh/RectPacking.h.
// Two-tier first-fit-decreasing over a bounded or unbounded set of skyline bins,
// opening a new page only when no existing one fits. Keeping every page open (instead of
// discarding the active bin on the first overflow) lets a later small rect fill
// the free space a large rect left on an earlier page — provably never more
// pages, usually fewer.
// ---------------------------------------------------------------------------
RectPackResult PackRectangles(const std::vector<cv::Rect>& rects,
                              const RectPackParams& params,
                              std::vector<RectPlacement>& placements)
{
	if (params.mode == RectPackMode::GrowSinglePage) {
		RectPackParams probe(params);
		probe.mode = RectPackMode::FixedSinglePage;
		probe.maxPageSize = cv::Size();
		probe.powerOfTwo = false;
		probe.square = false;

		int pageW = std::max(params.pageSize.width, 1);
		int pageH = std::max(params.pageSize.height, 1);
		unsigned targetCount = 0;
		// Grow in int64_t and saturate on the way back: a rect near INT_MAX plus
		// its gutter does not fit an int, and the `impossible` test that would
		// reject it runs further down.
		const auto GrowToFit = [](int& page, int64_t padded) {
			page = static_cast<int>(std::min<int64_t>(std::max<int64_t>(page, padded),
			                                          std::numeric_limits<int>::max()));
		};
		for (const cv::Rect& rect : rects) {
			if (rect.width <= 0 || rect.height <= 0)
				continue;
			++targetCount;
			GrowToFit(pageW, PaddedExtent(rect.width, params.padding));
			GrowToFit(pageH, PaddedExtent(rect.height, params.padding));
		}
		const auto NormalizePageSize = [&](int& width, int& height) {
			if (params.powerOfTwo) {
				width = static_cast<int>(NextPow2(static_cast<unsigned>(width)));
				height = static_cast<int>(NextPow2(static_cast<unsigned>(height)));
			}
			if (params.square)
				width = height = std::max(width, height);
			if (params.maxPageSize.width > 0)
				width = std::min(width, params.maxPageSize.width);
			if (params.maxPageSize.height > 0)
				height = std::min(height, params.maxPageSize.height);
		};
		NormalizePageSize(pageW, pageH);
		const int64_t maxPageW = params.maxPageSize.width > 0
		                             ? params.maxPageSize.width
		                             : std::numeric_limits<int>::max();
		const int64_t maxPageH = params.maxPageSize.height > 0
		                             ? params.maxPageSize.height
		                             : std::numeric_limits<int>::max();
		const bool impossible = std::any_of(rects.begin(), rects.end(), [&](const cv::Rect& rect) {
			if (rect.width <= 0 || rect.height <= 0)
				return false;
			const int64_t paddedW = PaddedExtent(rect.width, params.padding);
			const int64_t paddedH = PaddedExtent(rect.height, params.padding);
			return !((paddedW <= maxPageW && paddedH <= maxPageH)
			         || (params.allowRotation && paddedH <= maxPageW && paddedW <= maxPageH));
		});
		if (impossible) {
			probe.pageSize = cv::Size(pageW, pageH);
			return PackRectangles(rects, probe, placements);
		}
		for (;;) {
			probe.pageSize = cv::Size(pageW, pageH);
			RectPackResult result = PackRectangles(rects, probe, placements);
			if (result.numPacked == targetCount)
				return result;
			int nextW = pageW <= std::numeric_limits<int>::max() / 2 ? pageW * 2 : pageW;
			int nextH = pageH <= std::numeric_limits<int>::max() / 2 ? pageH * 2 : pageH;
			NormalizePageSize(nextW, nextH);
			if (nextW == pageW && nextH == pageH)
				return result;
			pageW = nextW;
			pageH = nextH;
		}
	}

	// Fixed-size core, also used by every GrowSinglePage probe above.
	unsigned pageW = static_cast<unsigned>(std::max(params.pageSize.width, 1));
	unsigned pageH = static_cast<unsigned>(std::max(params.pageSize.height, 1));
	if (params.powerOfTwo) {
		pageW = NextPow2(pageW);
		pageH = NextPow2(pageH);
	}
	if (params.square)
		pageW = pageH = std::max(pageW, pageH);

	std::vector<Eigen::Vector2f> sizes(rects.size());
	for (size_t i = 0; i < rects.size(); ++i)
		sizes[i] = Eigen::Vector2f(static_cast<float>(rects[i].width), static_cast<float>(rects[i].height));

	std::vector<PackedRect> packed;
	double packedArea = 0;
	const unsigned numPages = PackTwoTier(
	    sizes, static_cast<float>(pageW), static_cast<float>(pageH),
	    static_cast<float>(params.padding), params.allowRotation,
	    params.mode == RectPackMode::FixedSinglePage, packed, packedArea);

	RectPackResult result;
	result.pageSize = cv::Size(static_cast<int>(pageW), static_cast<int>(pageH));
	result.packedArea = static_cast<uint64_t>(packedArea);
	placements.assign(rects.size(), RectPlacement{});
	for (size_t i = 0; i < rects.size(); ++i) {
		if (!packed[i].packed)
			continue;
		// A rotated rect keeps its area but swaps its extents.
		placements[i] = {cv::Rect(static_cast<int>(packed[i].x), static_cast<int>(packed[i].y),
		                          packed[i].rotated ? rects[i].height : rects[i].width,
		                          packed[i].rotated ? rects[i].width : rects[i].height),
		                 packed[i].rotated, packed[i].page, true};
		++result.numPacked;
	}
	// With nothing packed the multi-page modes opened no page at all; the
	// single-page mode still reports its one (empty) page, so the caller always has
	// a page size to work from.
	result.numPages = result.numPacked > 0 || params.mode == RectPackMode::FixedSinglePage
	                      ? numPages
	                      : 0u;
	return result;
}

int EstimateSquareTextureSize(const std::vector<cv::Rect>& rects,
                              int multiple,
                              float targetOccupancy)
{
	ASSERT(targetOccupancy > 0.f && targetOccupancy <= 1.f);
	uint64_t area = 0;
	int maxSide = 0;
	for (const cv::Rect& rect : rects) {
		if (rect.width <= 0 || rect.height <= 0)
			continue;
		area += static_cast<uint64_t>(rect.width) * rect.height;
		maxSide = std::max(maxSide, std::max(rect.width, rect.height));
	}
	const int side = std::max(
	    static_cast<int>(std::ceil(std::sqrt(static_cast<double>(area) / targetOccupancy))),
	    maxSide);
	if (multiple > 0)
		return ((side + multiple - 1) / multiple) * multiple;
	return static_cast<int>(NextPow2(static_cast<unsigned>(side)));
}

// ---------------------------------------------------------------------------
// Module D: skyline (min-waste) first-fit-decreasing multi-page packing + UV
// rewrite to atlas space.
// ---------------------------------------------------------------------------
AtlasResult PackAtlas(Mesh& mesh,
                      const std::vector<unsigned>& faceChart,
                      unsigned numCharts,
                      const AtlasParams& params)
{
	mesh.SyncFaces();
	const size_t nf = mesh.faces.size();
	AtlasResult result;
	if (nf == 0 || numCharts == 0)
		return result;

	// Fail fast in every build mode: these were Debug-only asserts, so a mesh
	// without per-corner UVs was an out-of-bounds read in Release.
	if (faceChart.size() != nf || mesh.faceTexcoords.size() != nf * 3) {
		REPORT_WARNING("PackAtlas: per-face chart ids ({} for {} faces) and per-corner UVs "
		               "({} for {} needed) are required; nothing packed",
		               faceChart.size(), nf, mesh.faceTexcoords.size(), nf * 3);
		return result;
	}

	// ------------------------------------------------------------------
	// 0. Pre-orient each chart to its MINIMUM-AREA bounding rectangle.
	//    Rotates each chart's UVs (rigidly, about its centroid) so its tightest
	//    oriented box aligns with the axes — minimizing the axis-aligned bbox
	//    the packer actually uses (a diagonally-elongated chart otherwise wastes
	//    a large AABB). Rotation preserves texel density/distortion; baked into
	//    faceTexcoords in place. Matches xatlas's rotateChartsToAxis.
	// ------------------------------------------------------------------
	if (params.orientCharts && numCharts > 0) {
		// Collect the convex-hull point set per chart. A chart-interior vertex's UV is
		// written once per incident face (~6× at average valence) and ConvexHull2D
		// dedups internally anyway, so DEDUP the corner stream up-front (by exact UV
		// value — never by global vertex id, which cut-to-disk charts map to two UVs) to
		// cut the per-chart hull sort/allocation ~6×. The pre-orientation CENTROID stays
		// the mean over the FULL corner stream, so the baked rotation is bitwise
		// identical to the un-deduped version.
		std::vector<std::vector<Eigen::Vector2d>> cpts(numCharts);
		std::vector<Eigen::Vector2d> csum(numCharts, Eigen::Vector2d::Zero());
		std::vector<double> ccnt(numCharts, 0.0);
		std::vector<std::unordered_set<uint64_t>> seen(numCharts);
		const auto uvkey = [](float x, float y) {
			uint32_t xi, yi;
			std::memcpy(&xi, &x, sizeof xi);
			std::memcpy(&yi, &y, sizeof yi);
			return (static_cast<uint64_t>(xi) << 32) | yi;
		};
		for (size_t fi = 0; fi < nf; ++fi) {
			const unsigned c = faceChart[fi];
			if (c >= numCharts)
				continue;
			for (int k = 0; k < 3; ++k) {
				const TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
				csum[c].x() += uv.x();
				csum[c].y() += uv.y();
				ccnt[c] += 1.0;
				if (seen[c].insert(uvkey(uv.x(), uv.y())).second)
					cpts[c].emplace_back(uv.x(), uv.y());
			}
		}
		std::vector<float> rc(numCharts, 1.f), rs(numCharts, 0.f);
		std::vector<Eigen::Vector2d> cen(numCharts, Eigen::Vector2d::Zero());
		for (unsigned c = 0; c < numCharts; ++c) {
			if (ccnt[c] < 3.0) // < 1 face (matches the old cpts.size()<3 guard)
				continue;
			cen[c] = csum[c] / ccnt[c];
			const double th = MinAreaRectAngle(cpts[c]);
			rc[c] = static_cast<float>(std::cos(-th));
			rs[c] = static_cast<float>(std::sin(-th));
		}
		for (size_t fi = 0; fi < nf; ++fi) {
			const unsigned c = faceChart[fi];
			if (c >= numCharts)
				continue;
			for (int k = 0; k < 3; ++k) {
				TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
				const float dx = uv.x() - static_cast<float>(cen[c].x());
				const float dy = uv.y() - static_cast<float>(cen[c].y());
				uv.x() = static_cast<float>(cen[c].x()) + rc[c] * dx - rs[c] * dy;
				uv.y() = static_cast<float>(cen[c].y()) + rs[c] * dx + rc[c] * dy;
			}
		}
	}

	// ------------------------------------------------------------------
	// 1. Compute per-chart bounding rect from current (normalized) UVs.
	//    NormalizeChartDensity guarantees bbox-min is at origin, so max == size.
	// ------------------------------------------------------------------
	const unsigned pad = params.padding;
	std::vector<ChartRect> crects(numCharts);
	for (unsigned c = 0; c < numCharts; ++c) {
		crects[c].uvMinX = std::numeric_limits<float>::max();
		crects[c].uvMinY = std::numeric_limits<float>::max();
		crects[c].uvMaxX = std::numeric_limits<float>::lowest();
		crects[c].uvMaxY = std::numeric_limits<float>::lowest();
	}

	for (size_t fi = 0; fi < nf; ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;
		ChartRect& cr = crects[cid];
		for (int k = 0; k < 3; ++k) {
			const TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
			cr.uvMinX = std::min(cr.uvMinX, uv.x());
			cr.uvMinY = std::min(cr.uvMinY, uv.y());
			cr.uvMaxX = std::max(cr.uvMaxX, uv.x());
			cr.uvMaxY = std::max(cr.uvMaxY, uv.y());
		}
	}
	for (unsigned c = 0; c < numCharts; ++c) {
		ChartRect& cr = crects[c];
		if (cr.uvMaxX < cr.uvMinX) {
			cr.w = cr.h = 0.f; // chart with no faces
			cr.uvMinX = cr.uvMinY = 0.f;
		} else {
			cr.w = cr.uvMaxX - cr.uvMinX;
			cr.h = cr.uvMaxY - cr.uvMinY;
		}
		// A zero-UV-area chart (a sliver the flattener collapsed to a point/segment, or
		// an empty chart) otherwise took a degenerate branch that stacked it at the
		// page-0 origin with unnormalized UVs — overlapping the chart legitimately packed
		// there (texel bleed) and possibly exceeding [0,1]. Flag it, clamp every rect to
		// a ≥1-texel slot (so grow-to-fit and the packer see identical dimensions), and
		// collapse its UVs to the slot centre in step 5.
		cr.degenerate = (cr.w <= 0.f || cr.h <= 0.f);
		cr.w = std::max(cr.w, 1.f);
		cr.h = std::max(cr.h, 1.f);
	}

	// ------------------------------------------------------------------
	// 1.5. Fit-to-resolution: globally rescale so the total PADDED chart area
	//      is ≈ one page, so the atlas fills a single `resolution`² page at high
	//      occupancy instead of spilling many small (padded) charts across
	//      several pages. Solve for the scale k that makes
	//        Σ (k·w + 2·pad)(k·h + 2·pad) = targetFill · resolution²
	//      (a quadratic in k), then scale UVs + chart rects by k.
	// ------------------------------------------------------------------
	if (params.fitToResolution && numCharts > 0) {
		double sumWh = 0.0, sumWph = 0.0;
		unsigned ncnt = 0;
		for (unsigned c = 0; c < numCharts; ++c) {
			if (crects[c].degenerate)
				continue; // degenerate charts stay a fixed 1-texel slot, unscaled
			sumWh += static_cast<double>(crects[c].w) * crects[c].h;
			sumWph += static_cast<double>(crects[c].w) + crects[c].h;
			++ncnt;
		}
		const double R = static_cast<double>(params.resolution);
		const double targetFill = 0.82; // leave slack for packing gaps → ~1 page
		const double a = sumWh;
		const double b = 2.0 * static_cast<double>(pad) * sumWph;
		const double cc = 4.0 * static_cast<double>(pad) * pad * ncnt - targetFill * R * R;
		double k = 0.0;
		if (a > 1e-12) {
			const double disc = b * b - 4.0 * a * cc;
			if (disc >= 0.0)
				k = (-b + std::sqrt(disc)) / (2.0 * a);
		} else if (b > 1e-12) {
			k = -cc / b; // all-zero-area edge case (charts collapse to points)
		}
		if (k > 0.0 && std::isfinite(k)) {
			// Clamp k so the LONGEST chart side (plus padding) fits one page:
			// PackRects grows a page to swallow any oversized rect, so without
			// this a single extreme-aspect chart made the "one page" far larger
			// than params.resolution — silently violating the documented one
			// resolution² page contract (the shrink loop below only tested the
			// page COUNT). Rotation cannot help: the long side must fit either way.
			double maxDim = 0.0;
			for (unsigned c = 0; c < numCharts; ++c) {
				if (crects[c].degenerate)
					continue;
				maxDim = std::max({maxDim, static_cast<double>(crects[c].w), static_cast<double>(crects[c].h)});
			}
			if (maxDim > 0.0 && R > 2.0 * pad)
				k = std::min(k, (R - 2.0 * pad) / maxDim);
			// The single 0.82-fill solve is open-loop: if actual skyline waste exceeds
			// ~18% (elongated / high-aspect charts) the pack overflows to a nearly-empty
			// SECOND page at the same density, doubling texture memory instead of fitting
			// the requested resolution. Iterate — probe a rect-only pack (no UV writes),
			// and while it needs >1 page OR overflows the page dimensions shrink k
			// analytically (proportional to overflow, bounded) and repack — then
			// apply the final k to the UVs and rects once. Repacks touch only numCharts
			// rects, so cost is negligible.
			double kf = k;
			std::vector<ChartRect> trial(crects);
			std::vector<Placement> probe;
			unsigned probePages = 0, probePw = 0, probePh = 0;
			float probeArea = 0.f;
			unsigned attempts = 0;
			for (int attempt = 0; attempt < 8; ++attempt) {
				++attempts;
				const float kk = static_cast<float>(kf);
				for (unsigned c = 0; c < numCharts; ++c) {
					if (crects[c].degenerate)
						continue;
					trial[c].w = crects[c].w * kk;
					trial[c].h = crects[c].h * kk;
				}
				PackRects(trial, numCharts, params, pad, probe, probePages, probePw, probePh, probeArea);
				if (probePages <= 1 && probePw <= params.resolution && probePh <= params.resolution)
					break;
				// Analytic shrink: the probe placed `probeArea` padded texels
				// against a one-page budget of targetFill·R². Step k by the
				// square root of the area ratio — proportional to the actual
				// overflow — instead of a blind ×0.95. Upper clamp 0.95 keeps
				// waste-driven overflows (area under budget, layout still >1
				// page) converging at least as fast as the old ladder; lower
				// clamp 0.80 stops one noisy probe from collapsing the scale.
				const double budgetArea = targetFill * R * R;
				double shrink = std::sqrt(budgetArea / std::max(static_cast<double>(probeArea), 1.0));
				shrink = std::clamp(shrink, 0.80, 0.95);
				kf *= shrink;
			}
			result.fitAttempts = attempts;
			const float kfinal = static_cast<float>(kf);
			for (size_t fi = 0; fi < nf; ++fi) {
				const unsigned cid = faceChart[fi];
				if (cid >= numCharts || crects[cid].degenerate)
					continue;
				for (int kk = 0; kk < 3; ++kk) {
					TexCoord& uv = mesh.faceTexcoords[fi * 3 + kk];
					uv.x() *= kfinal;
					uv.y() *= kfinal;
				}
			}
			for (unsigned c = 0; c < numCharts; ++c) {
				if (crects[c].degenerate)
					continue;
				crects[c].w *= kfinal;
				crects[c].h *= kfinal;
				crects[c].uvMinX *= kfinal;
				crects[c].uvMinY *= kfinal;
				crects[c].uvMaxX *= kfinal;
				crects[c].uvMaxY *= kfinal;
			}
		}
	}

	// ------------------------------------------------------------------
	// 2-4. Sort by area, size the page, and place with first-fit-decreasing over a
	//      growing set of skyline (min-waste) bins (see PackRects).
	// ------------------------------------------------------------------
	result.chartPage.resize(numCharts, 0u);
	std::vector<Placement> placements;
	unsigned numPages = 0, pageW = 0, pageH = 0;
	float packedAreaTotal = 0.f;
	PackRects(crects, numCharts, params, pad, placements, numPages, pageW, pageH, packedAreaTotal);
	for (unsigned c = 0; c < numCharts; ++c)
		result.chartPage[c] = placements[c].page;

	result.numPages = numPages;
	result.width = pageW;
	result.height = pageH;
	result.faceChart = faceChart; // copy so callers can verify per-face layout

	const float totalAtlasArea =
	    static_cast<float>(result.numPages) * static_cast<float>(pageW) * static_cast<float>(pageH);
	// Both packedAreaTotal (Σ placed.w*placed.h) and totalAtlasArea use the same
	// "texel area including padding" basis, so occupancy is a true fill fraction in
	// [0,1].  The clamp is a safety net; disjoint placed rects cannot exceed the page.
	result.occupancy = (totalAtlasArea > 0.f)
	                       ? std::min(1.f, packedAreaTotal / totalAtlasArea)
	                       : 0.f;

	// ------------------------------------------------------------------
	// 5. Rewrite faceTexcoords to normalized atlas-space [0,1].
	//    For a non-rotated chart placed at (px, py):
	//      atlas_uv = (local_uv - bboxMin + (px, py)) / (pageW, pageH)
	//    For a 90°-rotated chart (true rotation, det=+1, into an h×w footprint):
	//      atlas_uv.x = (local_uv.y - bbox_min_y + px) / pageW
	//      atlas_uv.y = (chart_w - (local_uv.x - bbox_min_x) + py) / pageH
	//    A transpose (ly, lx) would fit the same footprint but MIRRORS the
	//    chart (det=-1), flipping UV winding and breaking tangent-space bakes.
	// ------------------------------------------------------------------
	const float invPw = 1.f / static_cast<float>(pageW);
	const float invPh = 1.f / static_cast<float>(pageH);

	for (size_t fi = 0; fi < nf; ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;
		const Placement& pl = placements[cid];
		const ChartRect& cr = crects[cid];

		if (cr.degenerate) {
			// Zero-area chart: collapse every corner to its ≥1-texel slot centre so its
			// raw, unnormalized extent cannot bleed over neighbours or exceed [0,1].
			// Clamped below with the general case: accumulated float rounding
			// (placement offset + invPw/invPh multiply) can still push this a
			// sub-ULP amount outside [0,1] at extreme page-edge placements.
			const float ax = (pl.x + 0.5f * cr.w) * invPw;
			const float ay = (pl.y + 0.5f * cr.h) * invPh;
			const float cx = std::clamp(ax, 0.f, 1.f);
			const float cy = std::clamp(ay, 0.f, 1.f);
			for (int k = 0; k < 3; ++k) {
				TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
				uv.x() = cx;
				uv.y() = cy;
			}
			continue;
		}

		for (int k = 0; k < 3; ++k) {
			TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
			float lx = uv.x() - cr.uvMinX; // local coords (bbox-min origin)
			float ly = uv.y() - cr.uvMinY;

			float ax, ay;
			if (pl.rotated) {
				// 90° rotation preserving winding: (lx,ly) → (ly, w - lx)
				ax = (ly + pl.x) * invPw;
				ay = (cr.w - lx + pl.y) * invPh;
			} else {
				ax = (lx + pl.x) * invPw;
				ay = (ly + pl.y) * invPh;
			}
			// Accumulated float rounding (bbox-subtraction + placement offset +
			// invPw/invPh multiply) can overshoot the page bound by a sub-ULP
			// amount; measured 3 ULPs at 2048 (2048.00073 vs H=2048, deterministic)
			// on truck_textured.glb. The [0,1] contract above is documented but
			// wasn't enforced — clamp here at the source.
			uv.x() = std::clamp(ax, 0.f, 1.f);
			uv.y() = std::clamp(ay, 0.f, 1.f);
		}
	}

	// True triangle coverage of the final map (see AtlasResult::coverage).
	double triArea = 0.0;
	for (size_t fi = 0; fi < nf; ++fi) {
		if (faceChart[fi] >= numCharts)
			continue;
		const TexCoord& t0 = mesh.faceTexcoords[fi * 3 + 0];
		const TexCoord& t1 = mesh.faceTexcoords[fi * 3 + 1];
		const TexCoord& t2 = mesh.faceTexcoords[fi * 3 + 2];
		triArea += 0.5 * std::abs(static_cast<double>(t1.x() - t0.x()) * (t2.y() - t0.y())
		                          - static_cast<double>(t2.x() - t0.x()) * (t1.y() - t0.y()));
	}
	result.coverage = (numPages > 0)
	                      ? std::min(1.f, static_cast<float>(triArea / numPages))
	                      : 0.f;

	return result;
}

} // namespace halfmesh
