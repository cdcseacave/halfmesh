/*
* PixelTraits.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Pixel accumulation helpers shared by the image sampler and the texture-bake
// engine. They let a single generic routine operate on any cv::Mat_<T> pixel
// type — scalar (float, uint8) or Eigen vector (e.g. Pixel = Vector3<uint8_t>) —
// by accumulating in a floating type and rounding/clamping back on store.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Core>

namespace halfmesh::detail {

// AccumPixel<S, Type>: floating accumulator type for a stored pixel S.
//   arithmetic S       -> Type                          (e.g. float -> double)
//   Eigen::Matrix<...> -> Eigen::Matrix<Type, R, C, ...> (e.g. Pixel -> Vector3d)
template <typename S, typename Type>
struct AccumPixel
{
	using type = Type;
};

template <typename Scalar, int R, int C, int O, int MR, int MC, typename Type>
struct AccumPixel<Eigen::Matrix<Scalar, R, C, O, MR, MC>, Type>
{
	using type = Eigen::Matrix<Type, R, C, O, MR, MC>;
};

// Zero of an accumulator (scalar 0 or Eigen zero vector).
template <typename Acc>
inline Acc AccumZero()
{
	if constexpr (std::is_arithmetic_v<Acc>)
		return Acc(0);
	else
		return Acc::Zero();
}

// Promote a stored pixel S to an accumulator Acc.
template <typename Acc, typename S>
inline Acc AccumCast(const S& s)
{
	if constexpr (std::is_arithmetic_v<S>)
		return static_cast<Acc>(s);
	else
		return s.template cast<typename Acc::Scalar>();
}

// Round-and-clamp an accumulator back to a stored pixel T. Integral channels are
// rounded to nearest and clamped to their representable range (no wrap); floating
// channels pass through.
template <typename Ch, typename V>
inline Ch StoreChannel(V v)
{
	if constexpr (std::is_integral_v<Ch>) {
		const double r = std::round(static_cast<double>(v));
		return static_cast<Ch>(std::clamp(
		    r,
		    static_cast<double>(std::numeric_limits<Ch>::lowest()),
		    static_cast<double>(std::numeric_limits<Ch>::max())));
	} else {
		return static_cast<Ch>(v);
	}
}

template <typename T, typename Acc>
inline T StoreCast(const Acc& acc)
{
	if constexpr (std::is_arithmetic_v<T>) {
		return StoreChannel<T>(acc);
	} else {
		T out;
		for (int k = 0; k < T::RowsAtCompileTime; ++k)
			out[k] = StoreChannel<typename T::Scalar>(acc[k]);
		return out;
	}
}

// Default predicate: every sample is valid.
struct AlwaysValid
{
	template <typename S>
	bool operator()(const S&) const { return true; }
};

} // namespace halfmesh::detail
