/*
* Sampler.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Generic 2D image sampler with selectable interpolation (Linear / Cubic),
// working over any cv::Mat_<T> pixel type (scalar or Eigen vector). Used by the
// texture-bake engine to resample source textures of differing pixel types
// (diffuse BGR uint8, float normal maps, scalar masks) through a single API.
//
// Conventions:
//   - Integer coordinate (x, y) addresses the centre of pixel (col=x, row=y);
//     image(row, col) is the element accessor (cv::Mat_ order).
//   - Out-of-bounds taps and validate-rejected samples are dropped and the
//     remaining weights renormalised, so border/masked omissions do not bias
//     the result toward zero (interior samples are unchanged: Linear and
//     Catmull-Rom weights already sum to one).

#pragma once

#include <cmath>
#include <type_traits>

#include <Eigen/Core>

#include <halfmesh/Util/PixelTraits.h>

namespace halfmesh {

// -------------------------------------------------------------------------
// Interpolation weight policies.
//
// Each policy maps a fractional offset x in [0,1] to `width` per-tap weights
// straddling the sampling position, and declares its `halfWidth` footprint.
// -------------------------------------------------------------------------

// Linear (== bilinear in 2D): two taps A,B around x.
//   A ... x ..... B
//   w[0]          w[1]
template <typename T = double>
struct LinearInterp
{
	using Type = T;
	static constexpr int halfWidth = 1;
	static constexpr int width = halfWidth * 2;

	void operator()(T x, T* w) const
	{
		w[0] = T(1) - x;
		w[1] = x;
	}
};

// Cubic convolution (Keys, "Cubic Convolution Interpolation for Digital Image
// Processing", eq. 4): four taps A,B,C,D around x. sharpness in [0.5, 0.75];
// 0.5 is the Catmull-Rom kernel (reproduces linear/quadratic exactly).
//   A ...... B ...x.. C ...... D
//   w[0]     w[1]     w[2]     w[3]
template <typename T = double>
struct CubicInterp
{
	using Type = T;
	static constexpr int halfWidth = 2;
	static constexpr int width = halfWidth * 2;

	T sharpness = T(0.5);

	void operator()(T x, T* w) const
	{
		w[0] = inter12(T(1) + x);
		w[1] = inter01(x);
		w[2] = inter01(T(1) - x);
		w[3] = inter12(T(2) - x);
	}

	// Central cubic segment for x in [0,1].
	T inter01(T x) const
	{
		return ((T(2) - sharpness) * x - (T(3) - sharpness)) * x * x + T(1);
	}
	// Tail cubic segment for x in [1,2].
	T inter12(T x) const
	{
		return (((T(5) - x) * x - T(8)) * x + T(4)) * sharpness;
	}
};

// -------------------------------------------------------------------------
// SampleImage — interpolate `image` at fractional position `pt`.
//
//   Interp   : LinearInterp<> (default) or CubicInterp<>.
//   Image    : any cv::Mat_<T> (defines value_type, rows, cols, operator(r,c)).
//   validate : optional predicate(const T&) -> bool; rejected taps are dropped.
//
// Returns the floating accumulator pixel (scalar for scalar images, an Eigen
// vector for vector pixels). Returns zero if no valid tap was found.
// -------------------------------------------------------------------------
template <typename Interp = LinearInterp<>, typename Image,
          typename Validate = detail::AlwaysValid>
typename detail::AccumPixel<typename Image::value_type, typename Interp::Type>::type
SampleImage(const Image& image,
            const Eigen::Matrix<typename Interp::Type, 2, 1>& pt,
            Validate validate = Validate{})
{
	using Type = typename Interp::Type;
	using Acc = typename detail::AccumPixel<typename Image::value_type, Type>::type;

	const Interp interp;
	const int bx = static_cast<int>(std::floor(pt.x()));
	const int by = static_cast<int>(std::floor(pt.y()));

	Type cx[Interp::width];
	Type cy[Interp::width];
	interp(pt.x() - static_cast<Type>(bx), cx);
	interp(pt.y() - static_cast<Type>(by), cy);

	Acc res = detail::AccumZero<Acc>();
	Type wsum = Type(0);

	// Interior fast path: when no tap can be rejected (the validator is the
	// no-op AlwaysValid, decided at compile time) and the whole footprint is
	// in-bounds (one range check, not per-tap), run a tight loop with no per-tap
	// bounds test or validate call. Same accumulation order and final wsum divide,
	// so the result is bit-identical to the general path.
	if constexpr (std::is_same_v<Validate, detail::AlwaysValid>) {
		const int r0 = by + 1 - Interp::halfWidth;
		const int c0 = bx + 1 - Interp::halfWidth;
		if (r0 >= 0 && r0 + Interp::width <= image.rows && c0 >= 0 && c0 + Interp::width <= image.cols) {
			for (int r = 0; r < Interp::width; ++r) {
				const int i = r0 + r;
				for (int c = 0; c < Interp::width; ++c) {
					const Type w = cx[c] * cy[r];
					res += detail::AccumCast<Acc>(image(i, c0 + c)) * w;
					wsum += w;
				}
			}
			return res * (Type(1) / wsum);
		}
	}

	for (int r = 0; r < Interp::width; ++r) {
		const int i = by + r + 1 - Interp::halfWidth;
		if (i < 0 || i >= image.rows)
			continue;
		for (int c = 0; c < Interp::width; ++c) {
			const int j = bx + c + 1 - Interp::halfWidth;
			if (j < 0 || j >= image.cols)
				continue;
			const auto& s = image(i, j);
			if (!validate(s))
				continue;
			const Type w = cx[c] * cy[r];
			res += detail::AccumCast<Acc>(s) * w;
			wsum += w;
		}
	}
	if (wsum == Type(0))
		return detail::AccumZero<Acc>();
	return res * (Type(1) / wsum);
}

} // namespace halfmesh
