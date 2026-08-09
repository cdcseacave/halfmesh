/*
* Quadric.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <halfmesh/Types.h>
#include <halfmesh/Util/Maths.h>

#include <limits>

namespace halfmesh {

// Store and use a quadric as half symmetric 4x4 matrix.
template <typename Scalar>
class TQuadric
{
	public:
	typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
	typedef Eigen::Matrix<Scalar, 4, 1> Point4;
	typedef Eigen::Matrix<Scalar, 3, 1> Point3;

	public:
	TQuadric() { Clear(); }

	// construct quadric from upper triangle of symmetric 4x4 matrix
	TQuadric(Scalar a, Scalar b, Scalar c, Scalar d,
	         Scalar e, Scalar f, Scalar g,
	         Scalar h, Scalar i,
	         Scalar j) :
	    a(a), b(b), c(c), d(d),
	    e(e), f(f), g(g),
	    h(h), i(i),
	    j(j)
	{
	}

	// construct quadric from a plane equation: ax+by+cz+d=0
	TQuadric(Scalar a, Scalar b, Scalar c, Scalar d) :
	    a(a * a), b(a * b), c(a * c), d(a * d),
	    e(b * b), f(b * c), g(b * d),
	    h(c * c), i(c * d),
	    j(d * d)
	{
	}

	// construct from a normal and point defining the plane
	TQuadric(const Point3& n, const Point3& p) :
	    TQuadric(n.x(), n.y(), n.z(), -n.dot(p))
	{
	}

	// set matrix to zero
	void Clear() { a = b = c = d = e = f = g = h = i = j = 0; }

	// add two quadrics
	TQuadric operator+(const TQuadric& q) const
	{
		return TQuadric(
		    a + q.a, b + q.b, c + q.c, d + q.d,
		    e + q.e, f + q.f, g + q.g,
		    h + q.h, i + q.i,
		    j + q.j);
	}
	TQuadric& operator+=(const TQuadric& q)
	{
		a += q.a;
		b += q.b;
		c += q.c;
		d += q.d;
		e += q.e;
		f += q.f;
		g += q.g;
		h += q.h;
		i += q.i;
		j += q.j;
		return *this;
	}

	// multiply quadric by a scalar
	TQuadric& operator*=(Scalar s)
	{
		a *= s;
		b *= s;
		c *= s;
		d *= s;
		e *= s;
		f *= s;
		g *= s;
		h *= s;
		i *= s;
		j *= s;
		return *this;
	}

	// evaluate quadric Q at position p by computing (p^T * Q * p)
	Scalar operator*(const Point3& p) const
	{
		const Scalar x(p(0)), y(p(1)), z(p(2));
		return a * x * x + 2 * b * x * y + 2 * c * x * z + 2 * d * x + e * y * y + 2 * f * y * z + 2 * g * y + h * z * z + 2 * i * z + j;
	}

	// estimate the best point p replacing the given edge (p0, p1)
	// such that it minimizes p' Q p
	Point3 ComputeOptimalPoint(const Point3& p0, const Point3& p1) const
	{
		const Scalar det = Determinant<0, 1, 2, 1, 4, 5, 2, 5, 7>();
		// Scale-relative conditioning test, |det| > sqrt(eps)*||A||_F^3: for a
		// well-formed corner |det| ~ ||A||^3, while an absolute epsilon admits
		// nearly-coplanar plane sets whose Cramer solution lies arbitrarily far
		// from the edge (spike vertices); those fall through to the segment
		// parabola below, which is inherently edge-bounded.
		const Scalar norm2 = a * a + e * e + h * h + 2 * (b * b + c * c + f * f);
		if (det * det > std::numeric_limits<Scalar>::epsilon() * norm2 * norm2 * norm2) {
			// invertible (and well-conditioned)
			return Point3(
			    -Determinant<1, 2, 3, 4, 5, 6, 5, 7, 8>() / det,
			    Determinant<0, 2, 3, 1, 5, 6, 2, 7, 8>() / det,
			    -Determinant<0, 1, 3, 1, 4, 6, 2, 5, 8>() / det);
		}
		// not invertible: minimize E(t) = (p0 + t*dir)^T Q (p0 + t*dir) along the
		// segment; the direction is homogenized with w=0 (a vector, not a point)
		const Matrix4 q = *this;
		const Point3 dir = p1 - p0;
		const Point4 dirH(dir.x(), dir.y(), dir.z(), Scalar(0));
		const Point4 qDir = q * dirH;
		const Scalar a = dirH.dot(qDir);
		const Scalar b = p0.homogeneous().dot(qDir) * 2;
		if (a == 0) {
			if (b < 0)
				return p1;
			if (b == 0)
				return (p0 + p1) * Scalar(0.5);
			return p0;
		}
		const Scalar extreme = -b / (2 * a);
		if (extreme < 0 || extreme > 1 || a < 0) {
			// the end-points with minimum error
			const Scalar p0Cost = p0.homogeneous().dot(q * p0.homogeneous());
			const Scalar p1Cost = p1.homogeneous().dot(q * p1.homogeneous());
			return (p0Cost > p1Cost ? p1 : p0);
		}
		// parabola extreme
		return p0 + dir * extreme;
	}

	// compute the error introduced by replacing the given edge (p0, p1) with
	// the best point p, where p minimizes p' Q p
	Scalar ComputeError(const Point3& p0, const Point3& p1) const
	{
		return operator*(ComputeOptimalPoint(p0, p1));
	}

	operator Matrix4() const
	{
		Matrix4 m4;
		m4 << a, b, c, d,
		    b, e, f, g,
		    c, f, h, i,
		    d, g, i, j;
		return m4;
	}
	Scalar operator[](int c) const { return m[c]; }

	private:
	// compose axis determinant
	template <
	    int a11, int a12, int a13,
	    int a21, int a22, int a23,
	    int a31, int a32, int a33>
	Scalar Determinant() const
	{
		return m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] + m[a12] * m[a23] * m[a31]
		       - m[a13] * m[a22] * m[a31] - m[a11] * m[a23] * m[a32] - m[a12] * m[a21] * m[a33];
	}

	private:
	union {
		Scalar m[10];
		struct
		{
			Scalar a, b, c, d,
			    e, f, g,
			    h, i,
			    j;
		};
	};
};

// Convenience alias using the project-wide real scalar.
using Quadric = TQuadric<real>;

} // namespace halfmesh
