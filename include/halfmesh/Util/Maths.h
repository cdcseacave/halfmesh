/*
* Maths.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Small typed math helpers (SQUARE, CLAMP, D2R, R2D). They are function
// templates, never macros: templates are typed, scoped and debuggable, and the
// result type follows the argument type.
//
// Interop note (openMVS): nothing special is needed here. openMVS defines
// SQUARE, CLAMP, D2R and R2D as namespaced (SEACAVE::) function templates, so
// they cannot collide with these — unlike a macro, which would ignore
// namespaces and textually mangle the declarations below.

// Provides SQUARE, CLAMP, D2R, R2D, IsZero, AreEqual, MILLI_TOL, MICRO_TOL, NANO_TOL.
#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

// M_PI is POSIX/glibc, not standard C++; MSVC's <cmath> only defines it when
// _USE_MATH_DEFINES is set before the TU's FIRST <cmath>/<math.h> include,
// which is fragile under header reordering.  This ifndef fallback is
// order-independent instead: on MSVC without _USE_MATH_DEFINES, M_PI is never
// defined by the standard header regardless of include order, so this always
// fires and supplies it; on platforms that already define it (e.g. glibc),
// it's a no-op.  Value matches glibc's M_PI bit-for-bit (verified 2026-07).
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif

namespace halfmesh {

// -------------------------------------------------------------------------
// Tolerance constants.
// -------------------------------------------------------------------------
inline constexpr double MILLI_TOL = 1e-3;
inline constexpr double MICRO_TOL = 1e-6;
inline constexpr double NANO_TOL = 1e-9;

// -------------------------------------------------------------------------
// SQUARE — square of a value (a*a).
// -------------------------------------------------------------------------
template <typename T>
constexpr T SQUARE(const T& a)
{
	return a * a;
}

// -------------------------------------------------------------------------
// D2R / R2D — degree/radian conversion. Templates rather than macros so the
// result type follows the argument type: D2R(20.f) computes in float, D2R(20.0)
// in double. Requires a non-integral type (asserted at compile time).
// -------------------------------------------------------------------------
template <typename T>
constexpr T D2R(T deg)
{
	static_assert(!std::is_integral<T>::value,
	              "D2R requires a non-integer (floating-point) type");
	return deg * static_cast<T>(M_PI / 180.0);
}

template <typename T>
constexpr T R2D(T rad)
{
	static_assert(!std::is_integral<T>::value,
	              "R2D requires a non-integer (floating-point) type");
	return rad * static_cast<T>(180.0 / M_PI);
}

// -------------------------------------------------------------------------
// IsZero — zero check.
// Integral overload: exact zero check.
// Floating-point overload: |v| <= epsilon*1000.
// -------------------------------------------------------------------------
template <typename Scalar,
          std::enable_if_t<std::is_integral<Scalar>::value, bool> = true>
constexpr bool IsZero(Scalar v)
{
	return v == Scalar(0);
}

template <typename Scalar,
          std::enable_if_t<std::is_floating_point<Scalar>::value, bool> = true>
inline bool IsZero(Scalar v,
                   Scalar epsilon = std::numeric_limits<Scalar>::epsilon() * Scalar(1000))
{
	return std::abs(v) <= epsilon;
}

// -------------------------------------------------------------------------
// CLAMP — clamp x into [lo, hi].
// -------------------------------------------------------------------------
template <typename T, typename Tl>
constexpr T CLAMP(T x, Tl lo, Tl hi)
{
	return (x < static_cast<T>(lo))   ? static_cast<T>(lo)
	       : (x > static_cast<T>(hi)) ? static_cast<T>(hi)
	                                  : x;
}

// -------------------------------------------------------------------------
// AreEqual — approximate equality.
// -------------------------------------------------------------------------
template <typename Scalar,
          std::enable_if_t<std::is_integral<Scalar>::value, bool> = true>
constexpr bool AreEqual(Scalar v0, Scalar v1) { return v0 == v1; }

template <typename Scalar,
          std::enable_if_t<std::is_floating_point<Scalar>::value, bool> = true>
inline bool AreEqual(Scalar v0, Scalar v1,
                     Scalar epsilon = std::numeric_limits<Scalar>::epsilon() * Scalar(1000))
{
	return std::abs(v0 - v1) <= epsilon;
}

// -------------------------------------------------------------------------
// RoundCast — round-nearest cast (away from zero at halfway).
// Semantics:
//   RoundCast<int>(0.7)  -> 1
//   RoundCast<int>(0.5)  -> 1
//   RoundCast<int>(-0.5) -> -1
//   RoundCast<int>(-1.4) -> -1
// For non-integral T_out: compiles to a static_cast (no rounding needed).
// For integral T_out: rounds via std::round then static_casts.
// -------------------------------------------------------------------------
template <typename T_out, typename T_in>
inline typename std::enable_if<!std::is_integral<T_out>::value, T_out>::type
RoundCast(T_in t)
{
	return static_cast<T_out>(t);
}

template <typename T_out, typename T_in>
inline typename std::enable_if<std::is_integral<T_out>::value, T_out>::type
RoundCast(T_in t)
{
	return static_cast<T_out>(std::round(static_cast<double>(t)));
}

} // namespace halfmesh

// Internal shim: Geometry.h lives in namespace math and calls math::AreEqual,
// math::IsZero, math::CLAMP, math::RoundCast, and uses NANO_TOL via math::.
// These are NOT global — they are scoped to namespace math.
namespace math {
using halfmesh::SQUARE;
using halfmesh::IsZero;
using halfmesh::AreEqual;
using halfmesh::CLAMP;
using halfmesh::RoundCast;
using halfmesh::NANO_TOL;
using halfmesh::MILLI_TOL;
using halfmesh::MICRO_TOL;
} // namespace math
