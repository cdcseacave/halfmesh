/*
* GeometryTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Geometry utility tests (DistanceBetweenTriangleAndPoint).
#include <gtest/gtest.h>

#include <halfmesh/Util/Geometry.h>
#include <halfmesh/Util/Maths.h>

#include <cmath>
#include <cstdlib>

using halfmesh::CLAMP;
using halfmesh::D2R;
using halfmesh::MILLI_TOL;

TEST(GeometryTest, DistanceBetweenTriangleAndPoint)
{
	std::srand(42);
	const unsigned iters = 100;

	for (unsigned iter = 0; iter < iters; ++iter) {
		const Eigen::Vector3d v0(-1.0, 0.0, -0.5);
		const Eigen::Vector3d v1(4.0, 1.0, 0.5);
		const Eigen::Vector3d v2(-2.0, 3.0, 1.0);

		const Eigen::Vector3d dir12 = (v2 - v1).normalized();
		const Eigen::Vector3d normal = dir12.cross(v0 - v1).normalized();
		const Eigen::Vector3d normalOrthoDir = dir12.cross(normal).normalized();

		Eigen::Vector3d orthoDir, pointOnV1v2, intersectionPoint;

		// Case 1: point projects onto a triangle line segment
		pointOnV1v2 = v1 + dir12 * CLAMP(static_cast<double>(std::rand()) / RAND_MAX, 0.0 + MILLI_TOL, 1.0 - MILLI_TOL);

		do {
			orthoDir = dir12.cross(
			                    pointOnV1v2 + Eigen::Vector3d::Random() * 10.0)
			               .normalized();
		} while (normalOrthoDir.dot(orthoDir) < std::cos(D2R(10.0)));

		double distance = 10.0 * static_cast<double>(std::rand()) / RAND_MAX;
		Eigen::Vector3d point = pointOnV1v2 + orthoDir * distance;

		EXPECT_TRUE(math::AreEqual(
		    math::DistanceBetweenTriangleAndPoint(v0, v1, v2, point, &intersectionPoint),
		    distance))
		    << "iter=" << iter << " distance mismatch (case 1)";
		EXPECT_TRUE(math::IsZero((pointOnV1v2 - intersectionPoint).norm()))
		    << "iter=" << iter << " intersection_point mismatch (case 1)";

		// Case 2: point projects onto a vertex
		pointOnV1v2 = v1 - dir12 * static_cast<double>(std::rand()) / RAND_MAX;
		distance = 10.0 * static_cast<double>(std::rand()) / RAND_MAX;
		point = v1 + orthoDir * distance;

		EXPECT_TRUE(math::AreEqual(
		    math::DistanceBetweenTriangleAndPoint(v0, v1, v2, point, &intersectionPoint),
		    distance))
		    << "iter=" << iter << " distance mismatch (case 2)";
		EXPECT_TRUE(math::IsZero((v1 - intersectionPoint).norm()))
		    << "iter=" << iter << " intersection_point mismatch (case 2)";
	}
}

// ---------------------------------------------------------------------------
// Degenerate-triangle fallback: point-to-edge distance (argument order) and a
// defined nearest point even when all segments collapse (projectedPoint init).
// ---------------------------------------------------------------------------
TEST(GeometryTest, DegenerateTriangleFullyCollapsed)
{
	// v0==v1==v2 at the origin, p=(1,0,0): distance is |p|=1, nearest is origin.
	const Eigen::Vector3d o(0.0, 0.0, 0.0);
	const Eigen::Vector3d p(1.0, 0.0, 0.0);
	Eigen::Vector3d nearest(9.0, 9.0, 9.0); // poison to catch an unwritten result
	const double d = math::DistanceBetweenTriangleAndPoint(o, o, o, p, &nearest);
	EXPECT_NEAR(d, 1.0, 1e-12) << "collapsed triangle: distance to the coincident vertex";
	EXPECT_NEAR((nearest - o).norm(), 0.0, 1e-12) << "nearest must be the (defined) vertex";
}

TEST(GeometryTest, DegenerateTriangleCollinear)
{
	// Collinear (zero-area) triangle along x; nearest edge point is the far end.
	const Eigen::Vector3d v0(0.0, 0.0, 0.0), v1(1.0, 0.0, 0.0), v2(2.0, 0.0, 0.0);
	const Eigen::Vector3d p(10.0, 5.0, 0.0);
	Eigen::Vector3d nearest;
	const double d = math::DistanceBetweenTriangleAndPoint(v0, v1, v2, p, &nearest);
	EXPECT_NEAR(d, std::sqrt(89.0), 1e-9) << "distance to (2,0,0)";
	EXPECT_NEAR((nearest - v2).norm(), 0.0, 1e-9) << "nearest is the (2,0,0) endpoint";
}

TEST(GeometryTest, DegenerateTriangleFuzzVsBruteForce)
{
	// Fuzz collinear/degenerate triangles against a min-of-three-segments oracle.
	std::srand(1234);
	auto seg = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& q) {
		const Eigen::Vector3d ab = b - a;
		const double L2 = ab.squaredNorm();
		double t = L2 > 0.0 ? (q - a).dot(ab) / L2 : 0.0;
		t = CLAMP(t, 0.0, 1.0);
		return (q - (a + t * ab)).norm();
	};
	for (int i = 0; i < 500; ++i) {
		auto rnd = [] { return static_cast<double>(std::rand()) / RAND_MAX * 4.0 - 2.0; };
		// build a collinear triangle: base point + scalar multiples of a direction
		const Eigen::Vector3d base(rnd(), rnd(), rnd());
		Eigen::Vector3d dir(rnd(), rnd(), rnd());
		if (dir.norm() < 1e-6)
			dir = Eigen::Vector3d(1.0, 0.0, 0.0);
		const Eigen::Vector3d v0 = base;
		const Eigen::Vector3d v1 = base + dir * rnd();
		const Eigen::Vector3d v2 = base + dir * rnd();
		const Eigen::Vector3d p(rnd(), rnd(), rnd());
		Eigen::Vector3d nearest;
		const double d = math::DistanceBetweenTriangleAndPoint(v0, v1, v2, p, &nearest);
		const double ref = std::min({seg(v0, v1, p), seg(v1, v2, p), seg(v2, v0, p)});
		EXPECT_NEAR(d, ref, 1e-9) << "i=" << i;
	}
}

// ---------------------------------------------------------------------------
// CotAngleBetweenRays — cot(angle) via dot/cross, parity with tan(pi/2 - acos).
// ---------------------------------------------------------------------------
TEST(GeometryTest, CotAngleBetweenRaysParity)
{
	std::srand(321);
	for (int i = 0; i < 500; ++i) {
		// Angle in [1deg, 179deg]; random magnitudes; both rays in the xy-plane.
		const double ang = D2R(1.0 + static_cast<double>(std::rand()) / RAND_MAX * 178.0);
		const double m1 = 0.1 + static_cast<double>(std::rand()) / RAND_MAX * 5.0;
		const double m2 = 0.1 + static_cast<double>(std::rand()) / RAND_MAX * 5.0;
		const Eigen::Vector3d v1(m1, 0.0, 0.0);
		const Eigen::Vector3d v2(m2 * std::cos(ang), m2 * std::sin(ang), 0.0);
		const double cot = math::CotAngleBetweenRays(v1, v2);
		const double ref = std::tan(M_PI * 0.5 - math::AngleBetweenRays(v1, v2));
		EXPECT_NEAR(cot, ref, 1e-6 * (1.0 + std::abs(ref))) << "i=" << i << " ang=" << ang;
	}
}

TEST(GeometryTest, CotAngleBetweenRaysDegenerateGuard)
{
	// Parallel rays => zero cross norm => guarded to 0 (no inf/NaN from tan/divide).
	const Eigen::Vector3d a(1.0, 0.0, 0.0), b(2.0, 0.0, 0.0);
	EXPECT_EQ(math::CotAngleBetweenRays(a, b), 0.0);
}

// ---------------------------------------------------------------------------
// Ray-triangle (Moller-Trumbore): basic hit / barycentrics, back-face culling,
// and scale-invariant sliver rejection (tiny valid triangles must still hit).
// ---------------------------------------------------------------------------
TEST(GeometryTest, RayTriangleHitBaryAndBackface)
{
	using V = Eigen::Vector3f;
	const V v0(0.f, 0.f, 0.f), v1(1.f, 0.f, 0.f), v2(0.f, 1.f, 0.f); // normal +z
	const float maxT = std::numeric_limits<float>::max();

	// Front hit toward (0.25, 0.25): t=2, bary = (0.5, 0.25, 0.25).
	{
		const Eigen::ParametrizedLine<float, 3> ray(V(0.25f, 0.25f, 2.f), V(0.f, 0.f, -1.f));
		float t = -1.f;
		V bary;
		ASSERT_TRUE(math::RayTriangleIntersectBarycentric(ray, v0, v1, v2, bary, 0.f, maxT, &t));
		EXPECT_NEAR(t, 2.f, 1e-5f);
		EXPECT_NEAR(bary[0], 0.5f, 1e-5f);
		EXPECT_NEAR(bary[1], 0.25f, 1e-5f);
		EXPECT_NEAR(bary[2], 0.25f, 1e-5f);
	}
	// Back-face (ray from -z going +z) must be culled.
	{
		const Eigen::ParametrizedLine<float, 3> ray(V(0.25f, 0.25f, -2.f), V(0.f, 0.f, 1.f));
		float t = -1.f;
		EXPECT_FALSE(math::RayTriangleIntersect(ray, v0, v1, v2, 0.f, maxT, &t));
	}
	// Ray passing outside the triangle misses.
	{
		const Eigen::ParametrizedLine<float, 3> ray(V(2.f, 2.f, 2.f), V(0.f, 0.f, -1.f));
		float t = -1.f;
		EXPECT_FALSE(math::RayTriangleIntersect(ray, v0, v1, v2, 0.f, maxT, &t));
	}
}

TEST(GeometryTest, RayTriangleTinyTriangleScaleInvariant)
{
	using V = Eigen::Vector3f;
	const float maxT = std::numeric_limits<float>::max();
	// A valid tiny triangle (edge ~1e-4) hit by a centroid-directed ray must hit.
	// The old absolute-area cull (nrm < machine-eps) rejected it as degenerate.
	for (float s : {1.f, 1e-2f, 1e-4f}) {
		const V v0(0.f, 0.f, 0.f), v1(s, 0.f, 0.f), v2(0.f, s, 0.f);
		const V centroid = (v0 + v1 + v2) / 3.f;
		const Eigen::ParametrizedLine<float, 3> ray(centroid + V(0.f, 0.f, 1.f), V(0.f, 0.f, -1.f));
		float t = -1.f;
		EXPECT_TRUE(math::RayTriangleIntersect(ray, v0, v1, v2, 0.f, maxT, &t))
		    << "scale s=" << s << " must hit";
		EXPECT_NEAR(t, 1.f, 1e-4f) << "scale s=" << s;
	}
}

// Squared-distance variant must agree with the sqrt'd form (value and nearest).
TEST(GeometryTest, TriangleDistanceSquaredParity)
{
	std::srand(555);
	auto rnd = [] { return static_cast<double>(std::rand()) / RAND_MAX * 6.0 - 3.0; };
	for (int i = 0; i < 1000; ++i) {
		const Eigen::Vector3d v0(rnd(), rnd(), rnd());
		const Eigen::Vector3d v1(rnd(), rnd(), rnd());
		const Eigen::Vector3d v2(rnd(), rnd(), rnd());
		const Eigen::Vector3d p(rnd(), rnd(), rnd());
		Eigen::Vector3d n1, n2;
		const double d = math::DistanceBetweenTriangleAndPoint(v0, v1, v2, p, &n1);
		const double d2 = math::DistanceBetweenTriangleAndPointSquared(v0, v1, v2, p, &n2);
		EXPECT_NEAR(d * d, d2, 1e-9 * (1.0 + d2)) << "i=" << i;
		EXPECT_NEAR((n1 - n2).norm(), 0.0, 1e-12) << "i=" << i << " nearest mismatch";
	}
}

// ---------------------------------------------------------------------------
// Large-magnitude overflow guard: d = |e1 x e2|^2 is L^4 in edge length, so
// float overflows at edges ~2^32 ((2^32)^4 = 2^128 > FLT_MAX). Below the
// threshold the kernel is exact and must be untouched; at/above it the guard
// must route into the (L^2-safe) segment fallback and return a finite,
// non-zero, conservative boundary distance — never the pre-guard silent 0
// that wins every nearest-face comparison. The fallback measures distance to
// the triangle BOUNDARY, an overestimate for an interior projection (here
// exactly 1.0625*s^2), so pin direction and sanity, not exact equality.
// ---------------------------------------------------------------------------
TEST(GeometryTest, TriangleDistanceLargeEdgeOverflowGuard)
{
	using V = Eigen::Vector3f;
	for (const int o : {31, 32, 33}) {
		const float s = std::ldexp(1.f, o); // edge length 2^o
		const V v0(0.f, 0.f, 0.f), v1(s, 0.f, 0.f), v2(0.f, s, 0.f);
		// p projects into the triangle interior (bary a=b=0.25) at height s:
		// true distance^2 = s^2, exact in float (all powers of two).
		const V p(0.25f * s, 0.25f * s, s);
		V nearest;
		const float d2 = math::DistanceBetweenTriangleAndPointSquared(v0, v1, v2, p, &nearest);
		const float trueD2 = s * s;
		EXPECT_TRUE(std::isfinite(d2)) << "octave " << o;
		EXPECT_GT(d2, 0.f) << "octave " << o << ": the old silent-0 false hit";
		EXPECT_TRUE(nearest.allFinite()) << "octave " << o;
		if (o == 31) {
			// |e1 x e2|^2 = 2^124 < FLT_MAX: exact interior path, untouched.
			EXPECT_FLOAT_EQ(d2, trueD2) << "octave 31 must remain exact";
		} else {
			// Overflow: conservative (>= exact) and sane (well under 2x here).
			EXPECT_GE(d2, trueD2) << "octave " << o << ": must be conservative";
			EXPECT_LE(d2, 2.f * trueD2) << "octave " << o << ": sane overestimate";
		}
	}
}

// ---------------------------------------------------------------------------
// RayBoxIntersect — boundary-robust slab test (tavianator 2022 variant).
// A flat AABB and a ray whose origin lies on a slab face (parallel direction,
// NaN case) must still register a hit.
// ---------------------------------------------------------------------------
TEST(GeometryTest, RayBoxFlatBoxPerpendicularRayHits)
{
	using V = Eigen::Vector3f;
	// Zero-thickness box in z (planar mesh node box).
	const Eigen::AlignedBox<float, 3> box(V(0.f, 0.f, 0.f), V(1.f, 1.f, 0.f));
	const Eigen::ParametrizedLine<float, 3> ray(V(0.5f, 0.5f, 1.f), V(0.f, 0.f, -1.f));
	float t = -1.f;
	EXPECT_TRUE(math::RayBoxIntersect(ray, box, &t)) << "flat box must be hit (entry==exit)";
	EXPECT_NEAR(t, 1.f, 1e-5f);
}

TEST(GeometryTest, RayBoxOriginOnMaxFaceParallelRayHits)
{
	using V = Eigen::Vector3f;
	const Eigen::AlignedBox<float, 3> box(V(0.f, 0.f, 0.f), V(1.f, 1.f, 1.f));
	// Origin exactly on the x==1 (max) face; direction has x==0 (0/0 => NaN on x).
	const Eigen::ParametrizedLine<float, 3> ray(V(1.f, 0.5f, 0.5f), V(0.f, 0.f, -1.f));
	float t = -1.f;
	EXPECT_TRUE(math::RayBoxIntersect(ray, box, &t)) << "max-face parallel ray must hit (NaN slab)";
}

// Precomputed-reciprocal overload must agree with the base overload — AND both
// must match independently hand-computed hit/t values.  RayBoxIntersect(ray,
// aabb,pt) is a thin wrapper that forwards to the invDir overload (same core),
// so agreement between the two alone cannot catch a shared-core regression;
// the fixed cases below are the independent oracle.  All boxes/rays use exact
// powers of two (0, 0.5, 1, 2) so the slab arithmetic is exact in float.
TEST(GeometryTest, RayBoxInvDirOverloadMatchesBase)
{
	using V = Eigen::Vector3f;
	const Eigen::AlignedBox<float, 3> unitBox(V(0.f, 0.f, 0.f), V(1.f, 1.f, 1.f));

	// Checks both overloads against an independently derived (expectHit, expectT)
	// pair, then keeps the original cross-overload agreement as a supplementary check.
	auto check = [](const Eigen::AlignedBox<float, 3>& box, const V& origin, const V& dir,
	                bool expectHit, float expectT, const char* label) {
		const Eigen::ParametrizedLine<float, 3> ray(origin, dir);
		const V invDir = dir.cwiseInverse();
		float ta = -1.f, tb = -1.f;
		const bool ha = math::RayBoxIntersect(ray, box, &ta);
		const bool hb = math::RayBoxIntersect(ray, box, invDir, &tb);
		EXPECT_EQ(ha, expectHit) << label << ": base overload hit mismatch";
		EXPECT_EQ(hb, expectHit) << label << ": inv_dir overload hit mismatch";
		if (expectHit) {
			EXPECT_FLOAT_EQ(ta, expectT) << label << ": base overload t mismatch";
			EXPECT_FLOAT_EQ(tb, expectT) << label << ": inv_dir overload t mismatch";
		}
		// Supplementary (not the point of this test, see comment above): the two
		// overloads must still agree with each other.
		EXPECT_EQ(ha, hb) << label << ": overloads disagree on hit";
		if (ha && hb)
			EXPECT_FLOAT_EQ(ta, tb) << label << ": overloads disagree on t";
	};

	// Clean hit: ray outside the box along +x hits the near face at x=0, exits
	// at x=1; entry t = (0 - (-2)) / 1 = 2.
	check(unitBox, V(-2.f, 0.5f, 0.5f), V(1.f, 0.f, 0.f), true, 2.f, "clean hit");

	// Miss: ray parallel to +x, offset entirely outside the box's y range
	// ([0,1] vs y=2) — the ray can never reach the box.
	check(unitBox, V(-2.f, 2.f, 0.5f), V(1.f, 0.f, 0.f), false, 0.f, "miss");

	// Ray origin already inside the box: entry t is clamped to 0 (the slab
	// test's minT starts at 0 and cannot go negative).
	check(unitBox, V(0.5f, 0.5f, 0.5f), V(1.f, 0.f, 0.f), true, 0.f, "origin inside");

	// Axis-parallel ray (dir=(0,0,-1)) hitting a zero-thickness box exactly on
	// its z-slab boundary (min_z == max_z == 0): entry==exit==1 pins the
	// boundary-robust "minT <= maxT" acceptance — a strict '<' would reject
	// this single-point graze as a miss.
	const Eigen::AlignedBox<float, 3> flatBox(V(0.f, 0.f, 0.f), V(1.f, 1.f, 0.f));
	check(flatBox, V(0.5f, 0.5f, 1.f), V(0.f, 0.f, -1.f), true, 1.f, "flat-box boundary graze");

	// Randomized consistency sweep: broad supplementary coverage on top of the
	// independent fixed cases above.
	std::srand(99);
	for (int i = 0; i < 200; ++i) {
		auto rnd = [] { return static_cast<float>(std::rand()) / RAND_MAX * 4.f - 2.f; };
		const V lo(rnd(), rnd(), rnd());
		const V hi = lo + V(std::abs(rnd()) + 0.1f, std::abs(rnd()) + 0.1f, std::abs(rnd()) + 0.1f);
		const Eigen::AlignedBox<float, 3> box(lo, hi);
		const V origin(rnd(), rnd(), rnd());
		V dir(rnd(), rnd(), rnd());
		if (dir.norm() < 1e-3f)
			dir = V(1.f, 0.f, 0.f);
		dir.normalize();
		const Eigen::ParametrizedLine<float, 3> ray(origin, dir);
		const V invDir = dir.cwiseInverse();
		float ta = -1.f, tb = -1.f;
		const bool ha = math::RayBoxIntersect(ray, box, &ta);
		const bool hb = math::RayBoxIntersect(ray, box, invDir, &tb);
		EXPECT_EQ(ha, hb) << "i=" << i;
		if (ha && hb)
			EXPECT_NEAR(ta, tb, 1e-4f * (1.f + std::abs(ta))) << "i=" << i;
	}
}
