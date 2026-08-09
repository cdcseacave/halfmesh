/*
* Raster.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// CPU UV-space triangle rasterizer + gutter dilation for texture baking.
//
// RasterizeTriangleBary visits every pixel whose centre lies inside the 2D
// triangle and hands the caller affine barycentric coordinates — exactly what
// is needed to reconstruct any per-vertex attribute (3D position, normal, source
// UV) across an atlas chart. The inside test is on the sign of the barycentric
// weights, so it is winding-agnostic: mirrored charts (a packing may flip a
// chart) raster identically without culling.
//
// Dilate grows valid texels into the surrounding invalid border, filling the
// inter-chart gutter so bilinear/anisotropic GPU filtering of the baked atlas
// does not bleed the background across seams.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <halfmesh/Util/PixelTraits.h>

namespace halfmesh {

// Signed twice-area / edge function of the 2D triangle (a, b, c):
//   (c - a) x (b - a)
// Positive for one winding, negative for the other; zero when collinear.
template <typename T>
inline T EdgeFunction(const Eigen::Matrix<T, 2, 1>& a,
                      const Eigen::Matrix<T, 2, 1>& b,
                      const Eigen::Matrix<T, 2, 1>& c)
{
	return (c.x() - a.x()) * (b.y() - a.y()) - (c.y() - a.y()) * (b.x() - a.x());
}

// Rasterize the triangle (v1, v2, v3) over the [0,width) x [0,height) image and
// invoke cb(int x, int y, const Eigen::Matrix<T,3,1>& bary) for each covered
// pixel. bary are the affine barycentric weights of (v1, v2, v3) at the pixel
// centre, are all >= 0 inside, and sum to one.
//
// cull=false (default) rasters both windings; cull=true skips back-facing
// (non-positive signed area) triangles.
//
// [yLo, yHi) clips the row range in ADDITION to the image bounds, so a caller
// that owns a row band (the deterministic parallel bake) can rasterize only its
// own rows without scanning — and rejecting inside the callback — the rest. The
// visited-texel set for a given band is identical to the unclamped raster
// filtered to that band, so determinism is preserved. Default is the full image.
//
// conservative=false (default) is the exact centre-inside rule: bary >= 0, and
// the raw affine weights are passed through — the documented contract, unchanged.
// conservative=true additionally visits texels whose centre lies within half a
// pixel of an edge (the standard conservative-raster offset, using |invArea| so
// mirrored charts expand rather than shrink), clamps the negative weight(s) to
// zero and renormalizes, so the callback still receives non-negative weights that
// sum to one. Used to resample the ring of chart-boundary texels that GPU
// bilinear filtering reads, instead of leaving them to gutter dilation.
template <typename T, typename Callback>
void RasterizeTriangleBary(const Eigen::Matrix<T, 2, 1>& v1,
                           const Eigen::Matrix<T, 2, 1>& v2,
                           const Eigen::Matrix<T, 2, 1>& v3,
                           int width, int height,
                           Callback&& cb,
                           bool cull = false,
                           int yLo = 0,
                           int yHi = std::numeric_limits<int>::max(),
                           bool conservative = false)
{
	const T area = EdgeFunction(v1, v2, v3);
	if (area == T(0))
		return; // degenerate (collinear) triangle
	if (cull && area <= T(0))
		return;
	const T invArea = T(1) / area;

	// Bounding box of the triangle, clipped to the image and the row band.
	const T minx = std::min({v1.x(), v2.x(), v3.x()});
	const T maxx = std::max({v1.x(), v2.x(), v3.x()});
	const T miny = std::min({v1.y(), v2.y(), v3.y()});
	const T maxy = std::max({v1.y(), v2.y(), v3.y()});
	const int x0 = std::max(0, static_cast<int>(std::floor(minx)));
	const int x1 = std::min(width - 1, static_cast<int>(std::ceil(maxx)));
	const int y0 = std::max({0, yLo, static_cast<int>(std::floor(miny))});
	const int y1 = std::min({height - 1, yHi - 1, static_cast<int>(std::ceil(maxy))});

	// Per-pixel weight b_i = E_i * invArea, where E_i is the UNNORMALIZED edge
	// function (affine in x, y). Evaluate E1,E2,E3 fresh at each row's x0, then
	// step by the exact per-column deltas dE_i/dx (3 adds/pixel) instead of three
	// EdgeFunction evals. For exactly-representable coordinates this reproduces the
	// per-pixel evaluation bit-for-bit; the fresh per-row restart keeps the result
	// independent of the row band, so parallel bands stay byte-identical to serial.
	const T dx1 = v3.y() - v2.y(); // d/dx EdgeFunction(v2, v3, p)
	const T dx2 = v1.y() - v3.y(); // d/dx EdgeFunction(v3, v1, p)
	const T dx3 = v2.y() - v1.y(); // d/dx EdgeFunction(v1, v2, p)

	// Conservative half-pixel edge offsets in barycentric space (0 when off, so the
	// default reject threshold is exactly b_i < 0). The y-deltas of E_i are the x
	// deltas above rotated: dE_i/dy = -(x_b - x_a). |invArea| keeps the offset
	// positive for either winding.
	T off1 = T(0), off2 = T(0), off3 = T(0);
	if (conservative) {
		const T aInv = std::abs(invArea);
		const T dy1 = -(v3.x() - v2.x());
		const T dy2 = -(v1.x() - v3.x());
		const T dy3 = -(v2.x() - v1.x());
		off1 = T(0.5) * (std::abs(dx1) + std::abs(dy1)) * aInv;
		off2 = T(0.5) * (std::abs(dx2) + std::abs(dy2)) * aInv;
		off3 = T(0.5) * (std::abs(dx3) + std::abs(dy3)) * aInv;
	}

	for (int y = y0; y <= y1; ++y) {
		const Eigen::Matrix<T, 2, 1> p0(static_cast<T>(x0), static_cast<T>(y));
		T e1 = EdgeFunction(v2, v3, p0);
		T e2 = EdgeFunction(v3, v1, p0);
		T e3 = EdgeFunction(v1, v2, p0);
		for (int x = x0; x <= x1; ++x, e1 += dx1, e2 += dx2, e3 += dx3) {
			// Reject pixels outside the (optionally half-pixel-expanded) edges;
			// with off_i==0 this is the exact centre-inside test and the survivors
			// lie in [0,1] and sum to one.
			const T b1 = e1 * invArea;
			if (b1 < -off1)
				continue;
			const T b2 = e2 * invArea;
			if (b2 < -off2)
				continue;
			const T b3 = e3 * invArea;
			if (b3 < -off3)
				continue;
			if (!conservative) {
				cb(x, y, Eigen::Matrix<T, 3, 1>(b1, b2, b3));
			} else {
				// Clamp the (at most two) negative boundary weights to zero and
				// renormalize so the callback still gets a convex combination.
				const T c1 = b1 > 0 ? b1 : T(0);
				const T c2 = b2 > 0 ? b2 : T(0);
				const T c3 = b3 > 0 ? b3 : T(0);
				const T is = T(1) / (c1 + c2 + c3); // sum >= 1, never zero
				cb(x, y, Eigen::Matrix<T, 3, 1>(c1 * is, c2 * is, c3 * is));
			}
		}
	}
}

// Grow valid texels into invalid neighbours. For `iterations` passes, every
// pixel with mask==0 that has at least one valid neighbour within the
// (2*halfSize+1)^2 window is set to the rounded mean of its valid neighbours
// and marked valid. Each pass reads the pre-pass state, so fill does not bleed
// within a single pass. image and mask are updated in place.
//
// Frontier formulation: only invalid pixels that border a valid one can fill, so
// a queue of just those pixels is processed each pass (O(filled) work) instead of
// clone-and-rescan-the-whole-image (O(iterations * W*H) work plus an image+mask
// clone per pass). Fills are computed against the still-committed pre-pass state
// and committed only after the pass, preserving the Jacobi (no intra-pass bleed)
// semantics — so the output is byte-identical to the full-scan double-buffer, in
// the same row-major fill order (verified against the old implementation and the
// bake goldens/tests).
template <typename T>
void Dilate(cv::Mat_<T>& image, cv::Mat_<uint8_t>& mask,
            int iterations = 1, int halfSize = 1)
{
	using Acc = typename detail::AccumPixel<T, double>::type;
	const int rows = image.rows, cols = image.cols;
	if (rows <= 0 || cols <= 0 || iterations <= 0)
		return;
	const auto lin = [cols](int r, int c) { return static_cast<size_t>(r) * cols + c; };

	// `queued[i]` is set once a pixel has ever entered the frontier; every frontier
	// pixel borders a valid pixel (n>0) so it always fills, hence queued==1 means
	// "is or will be valid" — using it to dedup the next frontier never blocks a
	// still-fillable pixel.
	std::vector<uint8_t> queued(static_cast<size_t>(rows) * cols, 0);
	std::vector<int> frontier;
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			if (mask(r, c))
				continue;
			bool border = false;
			for (int i = -halfSize; i <= halfSize && !border; ++i) {
				const int rr = r + i;
				if (rr < 0 || rr >= rows)
					continue;
				for (int j = -halfSize; j <= halfSize; ++j) {
					if (i == 0 && j == 0)
						continue;
					const int cc = c + j;
					if (cc < 0 || cc >= cols)
						continue;
					if (mask(rr, cc)) {
						border = true;
						break;
					}
				}
			}
			if (border) {
				frontier.push_back(static_cast<int>(lin(r, c)));
				queued[lin(r, c)] = 1;
			}
		}
	}

	std::vector<std::pair<int, T>> fills;
	std::vector<int> next;
	for (int it = 0; it < iterations && !frontier.empty(); ++it) {
		// Compute each frontier pixel's fill from the still-committed pre-pass state.
		fills.clear();
		for (const int id : frontier) {
			const int r = id / cols, c = id % cols;
			Acc sum = detail::AccumZero<Acc>();
			int n = 0;
			for (int i = -halfSize; i <= halfSize; ++i) {
				const int rr = r + i;
				if (rr < 0 || rr >= rows)
					continue;
				for (int j = -halfSize; j <= halfSize; ++j) {
					if (i == 0 && j == 0)
						continue;
					const int cc = c + j;
					if (cc < 0 || cc >= cols)
						continue;
					if (!mask(rr, cc))
						continue;
					sum += detail::AccumCast<Acc>(image(rr, cc));
					++n;
				}
			}
			fills.emplace_back(id, detail::StoreCast<T>(sum * (1.0 / n)));
		}
		// Commit values + mask together (deferred, so the pass read only pre-pass state).
		for (const auto& f : fills) {
			image(f.first / cols, f.first % cols) = f.second;
			mask(f.first / cols, f.first % cols) = 255;
		}
		// Next frontier: invalid, not-yet-queued neighbours of just-filled pixels.
		next.clear();
		for (const auto& f : fills) {
			const int r = f.first / cols, c = f.first % cols;
			for (int i = -halfSize; i <= halfSize; ++i) {
				const int rr = r + i;
				if (rr < 0 || rr >= rows)
					continue;
				for (int j = -halfSize; j <= halfSize; ++j) {
					if (i == 0 && j == 0)
						continue;
					const int cc = c + j;
					if (cc < 0 || cc >= cols)
						continue;
					const size_t nid = lin(rr, cc);
					if (mask(rr, cc) || queued[nid])
						continue;
					queued[nid] = 1;
					next.push_back(static_cast<int>(nid));
				}
			}
		}
		frontier.swap(next);
	}
}

} // namespace halfmesh
