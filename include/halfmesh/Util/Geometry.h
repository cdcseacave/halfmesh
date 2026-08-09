/*
* Geometry.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Geometry utilities:
//   DistanceBetweenLineSegmentAndPoint, DistanceBetweenTriangleAndPoint,
//   IsTriangleBarycentricValid, RayTriangleIntersectBarycentric, RayTriangleIntersect,
//   RayBoxIntersect
#pragma once

#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Maths.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <limits>

namespace math {

// -------------------------------------------------------------------------
// CosAngleBetweenRays / AngleBetweenRays
// -------------------------------------------------------------------------
template <typename Derived1, typename Derived2>
inline auto CosAngleBetweenRays(const Eigen::MatrixBase<Derived1>& v1,
                                const Eigen::MatrixBase<Derived2>& v2)
{
	static_assert(Eigen::MatrixBase<Derived1>::IsVectorAtCompileTime, "");
	static_assert(Eigen::MatrixBase<Derived2>::IsVectorAtCompileTime, "");
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	const Scalar norms = v1.squaredNorm() * v2.squaredNorm();
	// L^4 in ray magnitude: overflows to inf for float rays ~2^32, and inf
	// passes a bare '> 0'. Debug-only catch; release still returns the clamped
	// cos of inf/inf-contaminated math (accepted asymmetry).
	ASSERT(std::isfinite(norms) && norms > Scalar(0));
	const Scalar cosine = v1.dot(v2) / std::sqrt(norms);
	return CLAMP(cosine, Scalar(-1), Scalar(1));
}

template <typename Derived1, typename Derived2>
inline auto AngleBetweenRays(const Eigen::MatrixBase<Derived1>& v1,
                             const Eigen::MatrixBase<Derived2>& v2)
{
	return std::acos(CosAngleBetweenRays(v1, v2));
}

// -------------------------------------------------------------------------
// CotAngleBetweenRays
// Cotangent of the angle between v1 and v2: cot = cos/sin = dot / |cross|.
// One cross + one dot + one sqrt + one divide — no transcendental (acos/tan),
// and more accurate than acos near cos = +/-1. Guarded like the BuildSizingField
// cotan helper so a degenerate (zero cross) pair yields 0, never inf/NaN.
// -------------------------------------------------------------------------
template <typename Derived1, typename Derived2>
inline auto CotAngleBetweenRays(const Eigen::MatrixBase<Derived1>& v1,
                                const Eigen::MatrixBase<Derived2>& v2)
{
	static_assert(Eigen::MatrixBase<Derived1>::IsVectorAtCompileTime, "");
	static_assert(Eigen::MatrixBase<Derived2>::IsVectorAtCompileTime, "");
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	const Scalar cr = v1.cross(v2).norm();
	return cr > Scalar(0) ? static_cast<Scalar>(v1.dot(v2) / cr) : Scalar(0);
}

// -------------------------------------------------------------------------
// DistanceBetweenLineSegmentAndPointSquared
// Returns the SQUARED distance from `point` to the closest point on the segment
// [lineStart, lineEnd].  If projectedPoint != nullptr, *projectedPoint
// is set to that closest point.  Skips the sqrt so callers doing squared-space
// comparisons (nearest-point search) pay it once, at the end, not per candidate.
// -------------------------------------------------------------------------
template <typename Derived>
inline auto DistanceBetweenLineSegmentAndPointSquared(
    const Eigen::MatrixBase<Derived>& lineStart,
    const Eigen::MatrixBase<Derived>& lineEnd,
    const Eigen::MatrixBase<Derived>& point,
    Eigen::MatrixBase<Derived>* projectedPoint = nullptr)
{
	static_assert(Eigen::MatrixBase<Derived>::IsVectorAtCompileTime, "");
	using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;

	const auto lineStartTPoint = point - lineStart;
	const auto lineStartTLineEnd = lineEnd - lineStart;

	const Scalar normSquared = lineStartTLineEnd.squaredNorm();
	if (normSquared < static_cast<Scalar>(NANO_TOL)) {
		// Degenerate segment: the closest point is the (coincident) endpoint.
		// Define *projectedPoint so callers never observe indeterminate memory.
		if (projectedPoint) {
			*projectedPoint = lineStart;
		}
		return (point - lineStart).squaredNorm();
	}

	Scalar t = lineStartTPoint.dot(lineStartTLineEnd) / normSquared;
	t = CLAMP(t, Scalar(0), Scalar(1));
	const auto p = lineStart + t * lineStartTLineEnd;
	if (projectedPoint) {
		*projectedPoint = p;
	}
	return (point - p).squaredNorm();
}

// -------------------------------------------------------------------------
// DistanceBetweenLineSegmentAndPoint
// Returns the distance from `point` to the closest point on the segment
// [lineStart, lineEnd].  If projectedPoint != nullptr, *projectedPoint
// is set to that closest point.
// -------------------------------------------------------------------------
template <typename Derived>
inline auto DistanceBetweenLineSegmentAndPoint(
    const Eigen::MatrixBase<Derived>& lineStart,
    const Eigen::MatrixBase<Derived>& lineEnd,
    const Eigen::MatrixBase<Derived>& point,
    Eigen::MatrixBase<Derived>* projectedPoint = nullptr)
{
	using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
	return static_cast<Scalar>(std::sqrt(
	    DistanceBetweenLineSegmentAndPointSquared(lineStart, lineEnd, point, projectedPoint)));
}

// -------------------------------------------------------------------------
// DistanceBetweenTriangleAndPointSquared
// Compute the SQUARED distance of a point p to the triangle (v0, v1, v2).
// Skips the final sqrt so nearest-point search compares squared distances and
// pays a single sqrt per query rather than one per candidate triangle.
// -------------------------------------------------------------------------
template <typename Scalar>
Scalar DistanceBetweenTriangleAndPointSquared(
    const Eigen::Matrix<Scalar, 3, 1>& v0,
    const Eigen::Matrix<Scalar, 3, 1>& v1,
    const Eigen::Matrix<Scalar, 3, 1>& v2,
    const Eigen::Matrix<Scalar, 3, 1>& p,
    Eigen::Matrix<Scalar, 3, 1>* nearestPoint)
{
	Eigen::Matrix<Scalar, 3, 1> v0v1 = v1 - v0;
	Eigen::Matrix<Scalar, 3, 1> v0v2 = v2 - v0;
	Eigen::Matrix<Scalar, 3, 1> n = v0v1.cross(v0v2);
	const Scalar d = n.squaredNorm();

	// Degenerate triangle (zero area) — or d overflowed: |e1 x e2|^2 is L^4 in
	// edge length, inf for float edges >= ~2^32 ((2^32)^4 = 2^128 > FLT_MAX).
	// inf slips past a zero test (invD = 1/inf = 0 then silently returns
	// distance 0, a false nearest-hit that wins every comparison), so route it
	// to the segment fallback below: L^2-safe, exact when the nearest point is
	// on the boundary, and a conservative OVERestimate when p projects into the
	// interior — an error that can only lose comparisons, never falsely win.
	// For every finite d this condition is exactly equivalent to d == 0.
	if (!(d > Scalar(0)) || !std::isfinite(d)) {
		// Point-to-edge distance for each of the three edges. The signature is
		// (lineStart, lineEnd, point): pass the two triangle vertices as the
		// segment and p as the query point (all three calls consistent).
		Eigen::Matrix<Scalar, 3, 1> q, qq;
		Scalar dd = DistanceBetweenLineSegmentAndPointSquared(v0, v1, p, &qq);
		Scalar dd2 = DistanceBetweenLineSegmentAndPointSquared(v1, v2, p, &q);
		if (dd2 < dd) {
			dd = dd2;
			qq = q;
		}
		dd2 = DistanceBetweenLineSegmentAndPointSquared(v2, v0, p, &q);
		if (dd2 < dd) {
			dd = dd2;
			qq = q;
		}
		if (nearestPoint) {
			*nearestPoint = qq;
		}
		return dd;
	}

	const Scalar invD = Scalar(1) / d;
	Eigen::Matrix<Scalar, 3, 1> v1v2 = v2 - v1;
	Eigen::Matrix<Scalar, 3, 1> v0p = p - v0;
	Eigen::Matrix<Scalar, 3, 1> t = v0p.cross(n);
	Scalar a = t.dot(v0v2) * -invD;
	Scalar b = t.dot(v0v1) * invD;
	Scalar s01, s02, s12;

	if (a < Scalar(0)) {
		s02 = v0v2.dot(v0p) / v0v2.squaredNorm();
		if (s02 < Scalar(0)) {
			s01 = v0v1.dot(v0p) / v0v1.squaredNorm();
			if (s01 <= Scalar(0)) {
				v0p = v0;
			} else if (s01 >= Scalar(1)) {
				v0p = v1;
			} else {
				(v0p = v0) += (v0v1 *= s01);
			}
		} else if (s02 > Scalar(1)) {
			s12 = v1v2.dot(p - v1) / v1v2.squaredNorm();
			if (s12 >= Scalar(1)) {
				v0p = v2;
			} else if (s12 <= Scalar(0)) {
				v0p = v1;
			} else {
				(v0p = v1) += (v1v2 *= s12);
			}
		} else {
			(v0p = v0) += (v0v2 *= s02);
		}
	} else if (b < Scalar(0)) {
		s01 = v0v1.dot(v0p) / v0v1.squaredNorm();
		if (s01 < Scalar(0)) {
			s02 = v0v2.dot(v0p) / v0v2.squaredNorm();
			if (s02 <= Scalar(0)) {
				v0p = v0;
			} else if (s02 >= Scalar(1)) {
				v0p = v2;
			} else {
				(v0p = v0) += (v0v2 *= s02);
			}
		} else if (s01 > Scalar(1)) {
			s12 = v1v2.dot(p - v1) / v1v2.squaredNorm();
			if (s12 >= Scalar(1)) {
				v0p = v2;
			} else if (s12 <= Scalar(0)) {
				v0p = v1;
			} else {
				(v0p = v1) += (v1v2 *= s12);
			}
		} else {
			(v0p = v0) += (v0v1 *= s01);
		}
	} else if (a + b > Scalar(1)) {
		s12 = v1v2.dot(p - v1) / v1v2.squaredNorm();
		if (s12 >= Scalar(1)) {
			s02 = v0v2.dot(v0p) / v0v2.squaredNorm();
			if (s02 <= Scalar(0)) {
				v0p = v0;
			} else if (s02 >= Scalar(1)) {
				v0p = v2;
			} else {
				(v0p = v0) += (v0v2 *= s02);
			}
		} else if (s12 <= Scalar(0)) {
			s01 = v0v1.dot(v0p) / v0v1.squaredNorm();
			if (s01 <= Scalar(0)) {
				v0p = v0;
			} else if (s01 >= Scalar(1)) {
				v0p = v1;
			} else {
				(v0p = v0) += (v0v1 *= s01);
			}
		} else {
			(v0p = v1) += (v1v2 *= s12);
		}
	} else {
		// interior point
		(v0p = p) -= n * (n.dot(v0p) * invD);
	}

	if (nearestPoint) {
		*nearestPoint = v0p;
	}
	v0p -= p;
	return v0p.squaredNorm();
}

// -------------------------------------------------------------------------
// DistanceBetweenTriangleAndPoint
// Compute the distance of a point p to the triangle (v0, v1, v2).
// -------------------------------------------------------------------------
template <typename Scalar>
Scalar DistanceBetweenTriangleAndPoint(
    const Eigen::Matrix<Scalar, 3, 1>& v0,
    const Eigen::Matrix<Scalar, 3, 1>& v1,
    const Eigen::Matrix<Scalar, 3, 1>& v2,
    const Eigen::Matrix<Scalar, 3, 1>& p,
    Eigen::Matrix<Scalar, 3, 1>* nearestPoint)
{
	return std::sqrt(DistanceBetweenTriangleAndPointSquared(v0, v1, v2, p, nearestPoint));
}

// -------------------------------------------------------------------------
// IsTriangleBarycentricValid
// -------------------------------------------------------------------------
template <typename Scalar>
bool IsTriangleBarycentricValid(
    const Eigen::Matrix<Scalar, 3, 1>& bary,
    const Scalar eps = std::numeric_limits<Scalar>::epsilon() * Scalar(100))
{
	ASSERT(math::AreEqual(bary.array().sum(), Scalar(1)));
	return -eps < bary[0] && bary[0] < Scalar(1) + eps && -eps < bary[1] && bary[1] < Scalar(1) + eps && -eps < bary[2] && bary[2] < Scalar(1) + eps;
}

// -------------------------------------------------------------------------
// RayTriangleIntersectBarycentric
// Compute ray-triangle intersection as barycentric coordinates via the
// Moller-Trumbore algorithm (1997): no normalization (sqrt), no separate
// barycentric solve — the triple products give t and the barycentrics directly.
//
// Back-face culling is preserved by the determinant sign: det = e1 . (dir x e2)
// = -normal.dot(dir), so a front-facing hit has det > 0 (exactly the old
// cosine <= -eps convention). The degeneracy/parallel cull is scale-invariant:
// reject when det <= eps * |e1| * |e2| (i.e. sin(wedge) or grazing angle below
// eps), so valid tiny triangles on small-unit meshes are not rejected by an
// absolute area epsilon while true slivers still are. det is the sole
// denominator, so the cancellation-prone Gram determinant is gone.
//
// *pt is written before the t-range check (the original contract used by kd
// pruning). The half-open [minT, maxT) window is preserved.
// -------------------------------------------------------------------------
template <typename Scalar>
bool RayTriangleIntersectBarycentric(
    const Eigen::ParametrizedLine<Scalar, 3>& ray,
    const Eigen::Matrix<Scalar, 3, 1>& v0,
    const Eigen::Matrix<Scalar, 3, 1>& v1,
    const Eigen::Matrix<Scalar, 3, 1>& v2,
    Eigen::Matrix<Scalar, 3, 1>& bary,
    const Scalar minT = Scalar(0),
    const Scalar maxT = std::numeric_limits<Scalar>::max(),
    Scalar* pt = nullptr,
    const Scalar baryEps = std::numeric_limits<Scalar>::epsilon() * Scalar(100))
{
	const Eigen::Matrix<Scalar, 3, 1> e1 = v1 - v0;
	const Eigen::Matrix<Scalar, 3, 1> e2 = v2 - v0;
	const Eigen::Matrix<Scalar, 3, 1> pvec = ray.direction().cross(e2);
	const Scalar det = e1.dot(pvec);

	// Back-face cull (det > 0) + scale-invariant degeneracy/parallel cull:
	// reject unless det > eps * |e1| * |e2| (squared to avoid the sqrt).
	const Scalar eps = std::numeric_limits<Scalar>::epsilon();
	if (det <= Scalar(0) || SQUARE(det) <= SQUARE(eps) * e1.squaredNorm() * e2.squaredNorm())
		return false;

	const Scalar invDet = Scalar(1) / det;
	const Eigen::Matrix<Scalar, 3, 1> tvec = ray.origin() - v0;
	const Scalar u = tvec.dot(pvec) * invDet;
	const Eigen::Matrix<Scalar, 3, 1> qvec = tvec.cross(e1);
	const Scalar v = ray.direction().dot(qvec) * invDet;
	const Scalar t = e2.dot(qvec) * invDet;
	if (pt)
		*pt = t;
	if (t < minT || t >= maxT)
		return false;

	// Reject a plane hit that lands outside the triangle BEFORE forming
	// bary[0] = 1-u-v (the standard Moller-Trumbore early-out).  u and v are exact
	// here, but 1-u-v is not once |u|,|v| >> 1: the subtraction cancels and the
	// sum-to-one property of the returned barycentrics is destroyed (at |u|,|v| ~
	// 1e4 the reconstructed sum already drifts ~1e-3, far past AreEqual's absolute
	// tolerance).  Such barycentrics are pure noise -- a ray whose plane hit is far
	// outside the triangle -- so emitting them and leaving the caller to notice was
	// what made IsTriangleBarycentricValid's post-condition unverifiable.
	// The window is deliberately NO STRICTER than the one IsTriangleBarycentricValid
	// applies, so nothing that used to be accepted is rejected here: this only
	// converts "return true with meaningless barycentrics" into "return false".
	if (u < -baryEps || u > Scalar(1) + baryEps || v < -baryEps || u + v > Scalar(1) + baryEps)
		return false;

	bary[0] = Scalar(1) - u - v;
	bary[1] = u;
	bary[2] = v;
	return true;
}

// -------------------------------------------------------------------------
// RayTriangleIntersect
// -------------------------------------------------------------------------
template <typename Scalar>
bool RayTriangleIntersect(
    const Eigen::ParametrizedLine<Scalar, 3>& ray,
    const Eigen::Matrix<Scalar, 3, 1>& v0,
    const Eigen::Matrix<Scalar, 3, 1>& v1,
    const Eigen::Matrix<Scalar, 3, 1>& v2,
    const Scalar minT = Scalar(0),
    const Scalar maxT = std::numeric_limits<Scalar>::max(),
    Scalar* pt = nullptr,
    Eigen::Matrix<Scalar, 3, 1>* pbary = nullptr,
    const Scalar eps = std::numeric_limits<Scalar>::epsilon() * Scalar(100))
{
	Eigen::Matrix<Scalar, 3, 1> bary;
	if (!RayTriangleIntersectBarycentric(ray, v0, v1, v2, bary, minT, maxT, pt, eps))
		return false;
	// NOTE: matches original — pbary is accepted but intentionally not written.
	return IsTriangleBarycentricValid(bary, eps);
}

// -------------------------------------------------------------------------
// RayBoxIntersect (precomputed reciprocal direction)
// Test ray-AABB intersection.  invDir = 1/ray.direction() (component-wise);
// hoist it once per ray so a full traversal multiplies instead of dividing.
// Algorithm: the boundary-robust slab variant, tavianator.com/2022/ray_box_boundary.html
// The known-non-NaN accumulator is passed as the FIRST std::min/max operand so
// a NaN from an axis-parallel ray grazing a slab face (0*inf) is dropped instead
// of poisoning the interval; `minT <= maxT` admits flat boxes and grazing hits
// (matching the BVH's own inclusive RayAABB).
// -------------------------------------------------------------------------
template <typename Scalar>
bool RayBoxIntersect(
    const Eigen::ParametrizedLine<Scalar, 3>& ray,
    const Eigen::AlignedBox<Scalar, 3>& aabb,
    const Eigen::Matrix<Scalar, 3, 1>& invDir,
    Scalar* pt = nullptr)
{
	Scalar minT(0), maxT(std::numeric_limits<Scalar>::infinity());
	for (int i = 0; i < 3; ++i) {
		const Scalar t1 = (aabb.min()[i] - ray.origin()[i]) * invDir[i];
		const Scalar t2 = (aabb.max()[i] - ray.origin()[i]) * invDir[i];
		minT = std::min(std::max(minT, t1), std::max(minT, t2));
		maxT = std::max(std::min(maxT, t1), std::min(maxT, t2));
	}
	if (pt)
		*pt = minT;
	return minT <= maxT;
}

// -------------------------------------------------------------------------
// RayBoxIntersect
// Convenience overload computing the reciprocal direction per call.
// -------------------------------------------------------------------------
template <typename Scalar>
bool RayBoxIntersect(
    const Eigen::ParametrizedLine<Scalar, 3>& ray,
    const Eigen::AlignedBox<Scalar, 3>& aabb,
    Scalar* pt = nullptr)
{
	const Eigen::Matrix<Scalar, 3, 1> invDir = ray.direction().cwiseInverse();
	return RayBoxIntersect(ray, aabb, invDir, pt);
}

} // namespace math
