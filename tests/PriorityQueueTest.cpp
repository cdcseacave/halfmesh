/*
* PriorityQueueTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for PriorityQueue.h and Quadric.h (TQuadric).

#include <gtest/gtest.h>
#include <cstdlib>
#include <climits>

#include <halfmesh/PriorityQueue.h>
#include <halfmesh/Quadric.h>

// ---------------------------------------------------------------------------
// Priority queue test helpers
// ---------------------------------------------------------------------------

namespace {

inline uint32_t RandomMax(uint32_t m)
{
	return static_cast<uint32_t>(static_cast<uint64_t>(m) * std::rand() / RAND_MAX);
}

} // namespace

// ---------------------------------------------------------------------------
// TPriorityQueue — ascending order
// ---------------------------------------------------------------------------

TEST(PriorityQueueTest, AscendingOrdering)
{
	std::srand(42);
	const unsigned iters = 1000;
	for (unsigned iter = 0; iter < iters; ++iter) {
		uint32_t n = 5 + RandomMax(5555);
		halfmesh::TPriorityQueue<uint32_t, uint32_t, true> queue;
		for (uint32_t i = 0; i < n; ++i)
			queue.emplace(i, RandomMax(999999));
		for (uint32_t i = 0; i < 9; ++i)
			queue.update(RandomMax(n - 1), RandomMax(9999999));
		for (uint32_t i = 0; i < 3; ++i) {
			const uint32_t key = RandomMax(--n);
			queue.pop(key);
			queue.move(queue.size(), key);
		}
		ASSERT_TRUE(queue.IsFaultless());
		uint32_t m = 0;
		while (!queue.empty()) {
			const uint32_t p = queue.peek().priority;
			EXPECT_LE(m, p) << "Ascending order violated at iter " << iter;
			m = p;
			queue.pop();
		}
		ASSERT_TRUE(queue.IsFaultless());
	}
}

// ---------------------------------------------------------------------------
// TPriorityQueue — descending order
// ---------------------------------------------------------------------------

TEST(PriorityQueueTest, DescendingOrdering)
{
	std::srand(42);
	const unsigned iters = 1000;
	for (unsigned iter = 0; iter < iters; ++iter) {
		uint32_t n = 5 + RandomMax(5555);
		halfmesh::TPriorityQueue<uint32_t, uint32_t, false> queue;
		for (uint32_t i = 0; i < n; ++i)
			queue.emplace(i, RandomMax(999999));
		for (uint32_t i = 0; i < 9; ++i)
			queue.update(RandomMax(n - 1), RandomMax(9999999));
		for (uint32_t i = 0; i < 3; ++i) {
			const uint32_t key = RandomMax(--n);
			queue.pop(key);
			queue.move(queue.size(), key);
		}
		ASSERT_TRUE(queue.IsFaultless());
		uint32_t m = UINT_MAX;
		while (!queue.empty()) {
			const uint32_t p = queue.peek().priority;
			EXPECT_GE(m, p) << "Descending order violated at iter " << iter;
			m = p;
			queue.pop();
		}
		ASSERT_TRUE(queue.IsFaultless());
	}
}

// ---------------------------------------------------------------------------
// TPriorityQueue — small deterministic correctness checks
// ---------------------------------------------------------------------------

TEST(PriorityQueueTest, EmplacePopOrder)
{
	halfmesh::TPriorityQueue<uint32_t, uint32_t, true> q;
	q.emplace(0, 30);
	q.emplace(1, 10);
	q.emplace(2, 20);
	ASSERT_EQ(q.size(), 3u);
	EXPECT_EQ(q.peek().priority, 10u);
	q.pop();
	EXPECT_EQ(q.peek().priority, 20u);
	q.pop();
	EXPECT_EQ(q.peek().priority, 30u);
	q.pop();
	EXPECT_TRUE(q.empty());
}

TEST(PriorityQueueTest, UpdateChangesOrder)
{
	halfmesh::TPriorityQueue<uint32_t, uint32_t, true> q;
	q.emplace(0, 100);
	q.emplace(1, 200);
	// Update key 0 to a higher priority (larger value in ascending = lower rank)
	q.update(0, 300);
	EXPECT_EQ(q.peek().key, 1u); // key 1 with priority 200 is now top
	EXPECT_TRUE(q.IsFaultless());
}

TEST(PriorityQueueTest, PopByKey)
{
	halfmesh::TPriorityQueue<uint32_t, uint32_t, true> q;
	q.emplace(0, 5);
	q.emplace(1, 3);
	q.emplace(2, 7);
	EXPECT_TRUE(q.pop(1u)); // remove the top element
	EXPECT_TRUE(q.IsFaultless());
	EXPECT_EQ(q.size(), 2u);
	EXPECT_EQ(q.peek().priority, 5u);
}

TEST(PriorityQueueTest, PeekByKey)
{
	halfmesh::TPriorityQueue<uint32_t, uint32_t, true> q;
	q.emplace(0, 42);
	const auto* node = q.peek(0u);
	ASSERT_NE(node, nullptr);
	EXPECT_EQ(node->priority, 42u);
	const auto* missing = q.peek(0u); // same key, valid
	EXPECT_NE(missing, nullptr);
}

// ---------------------------------------------------------------------------
// TQuadric tests
// ---------------------------------------------------------------------------

TEST(QuadricTest, OnPlanePointHasZeroError)
{
	// plane: z = 0, normal = (0,0,1), through origin
	// equation: 0*x + 0*y + 1*z + 0 = 0
	halfmesh::TQuadric<double> q(0.0, 0.0, 1.0, 0.0);
	Eigen::Vector3d pOn(1.0, 2.0, 0.0);
	EXPECT_NEAR(q * pOn, 0.0, 1e-12);
}

TEST(QuadricTest, OffPlanePointHasPositiveError)
{
	// plane: z = 0 → normal (0,0,1), through origin
	halfmesh::TQuadric<double> q(0.0, 0.0, 1.0, 0.0);
	Eigen::Vector3d pOff(1.0, 2.0, 3.0);
	// error = z^2 = 9
	EXPECT_NEAR(q * pOff, 9.0, 1e-12);
}

TEST(QuadricTest, PlaneFromNormalAndPoint)
{
	// plane z=1: normal=(0,0,1), point=(0,0,1) → d = -n.dot(p) = -1
	// equation 0*x + 0*y + 1*z - 1 = 0
	Eigen::Vector3d n(0.0, 0.0, 1.0);
	Eigen::Vector3d p(0.0, 0.0, 1.0);
	halfmesh::TQuadric<double> q(n, p);
	// on-plane point
	EXPECT_NEAR(q * Eigen::Vector3d(5.0, -3.0, 1.0), 0.0, 1e-12);
	// off-plane by 2 units: z=3, error=(3-1)^2=4
	EXPECT_NEAR(q * Eigen::Vector3d(0.0, 0.0, 3.0), 4.0, 1e-12);
}

TEST(QuadricTest, Accumulation)
{
	// Two planes: z=0 and z=1 (normal=(0,0,1)), sum their quadrics
	halfmesh::TQuadric<double> q0(0.0, 0.0, 1.0, 0.0); // z=0
	halfmesh::TQuadric<double> q1(0.0, 0.0, 1.0, -1.0); // z=1
	halfmesh::TQuadric<double> sum = q0 + q1;
	// at z=0.5 each plane contributes (0.5)^2=0.25, sum=0.5
	EXPECT_NEAR(sum * Eigen::Vector3d(0.0, 0.0, 0.5), 0.5, 1e-12);
}

TEST(QuadricTest, ComputeOptimalPointOnEdge)
{
	// z=0 plane; optimal point on edge from (0,0,0) to (0,0,1)
	// minimizes z^2, so optimal is (0,0,0)
	halfmesh::TQuadric<double> q(0.0, 0.0, 1.0, 0.0);
	Eigen::Vector3d p0(0.0, 0.0, 0.0);
	Eigen::Vector3d p1(0.0, 0.0, 1.0);
	Eigen::Vector3d opt = q.ComputeOptimalPoint(p0, p1);
	EXPECT_NEAR(q * opt, 0.0, 1e-10);
}

TEST(QuadricTest, ComputeOptimalPointSingularFallbackParabola)
{
	// Two parallel planes z=1 and z=3: the 3x3 block is singular (rank 1), so
	// ComputeOptimalPoint must take the 1-D parabola fallback along the segment.
	// E(p) = (z-1)^2 + (z-3)^2 is minimized at z=2.
	halfmesh::TQuadric<double> q1(0.0, 0.0, 1.0, -1.0); // z=1
	halfmesh::TQuadric<double> q3(0.0, 0.0, 1.0, -3.0); // z=3
	halfmesh::TQuadric<double> q = q1 + q3;
	Eigen::Vector3d p0(0.0, 0.0, 0.0);
	Eigen::Vector3d p1(0.0, 0.0, 4.0);
	Eigen::Vector3d opt = q.ComputeOptimalPoint(p0, p1);
	EXPECT_NEAR(opt.z(), 2.0, 1e-9);
	// the returned point must not cost more than any segment sample
	const double optCost = q * opt;
	for (double t = 0.0; t <= 1.0; t += 0.125)
		EXPECT_LE(optCost, q * Eigen::Vector3d(0.0, 0.0, 4.0 * t) + 1e-12);
}

TEST(QuadricTest, ComputeOptimalPointInPlaneSegmentIsMidpoint)
{
	// Planar (rank-1) quadric from plane z=1 NOT through the origin; a segment
	// lying in the plane has E(t) == 0 everywhere → tie broken at the midpoint.
	Eigen::Vector3d n(0.0, 0.0, 1.0);
	halfmesh::TQuadric<double> q(n, Eigen::Vector3d(0.0, 0.0, 1.0));
	Eigen::Vector3d p0(0.0, 0.0, 1.0);
	Eigen::Vector3d p1(4.0, 0.0, 1.0);
	Eigen::Vector3d opt = q.ComputeOptimalPoint(p0, p1);
	EXPECT_NEAR((opt - Eigen::Vector3d(2.0, 0.0, 1.0)).norm(), 0.0, 1e-9);
}

TEST(QuadricTest, ComputeOptimalPointIllConditionedStaysNearEdge)
{
	// Three nearly-coplanar planes (normals tilted by 1e-3, offsets 0.1 apart):
	// the 3x3 system has det ~ 1e-12 — numerically "invertible" for an absolute
	// epsilon test — but its Cramer solution is the 3-plane intersection at
	// roughly (100, -100, 1), a spike ~1400 segment-lengths away. The optimal
	// point must stay near the edge instead.
	const Eigen::Vector3d n1(0.0, 0.0, 1.0);
	const Eigen::Vector3d n2 = Eigen::Vector3d(1e-3, 0.0, 1.0).normalized();
	const Eigen::Vector3d n3 = Eigen::Vector3d(0.0, 1e-3, 1.0).normalized();
	halfmesh::TQuadric<double> q =
	    halfmesh::TQuadric<double>(n1, Eigen::Vector3d(0.0, 0.0, 1.0)) + halfmesh::TQuadric<double>(n2, Eigen::Vector3d(0.0, 0.0, 1.1)) + halfmesh::TQuadric<double>(n3, Eigen::Vector3d(0.0, 0.0, 0.9));
	const Eigen::Vector3d p0(0.0, 0.0, 0.95);
	const Eigen::Vector3d p1(0.0, 0.0, 1.05);
	const Eigen::Vector3d opt = q.ComputeOptimalPoint(p0, p1);
	EXPECT_LE((opt - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1.0)
	    << "ill-conditioned Cramer solve produced a spike at " << opt.transpose();
}

TEST(QuadricTest, DefaultConstructedIsZero)
{
	halfmesh::TQuadric<double> q;
	EXPECT_NEAR(q * Eigen::Vector3d(999.0, -999.0, 999.0), 0.0, 1e-12);
}

TEST(QuadricTest, ClearSetsZeroError)
{
	halfmesh::TQuadric<double> q(1.0, 0.0, 0.0, 0.0);
	q.Clear();
	EXPECT_NEAR(q * Eigen::Vector3d(999.0, 999.0, 999.0), 0.0, 1e-12);
}

TEST(QuadricTest, ScalarMultiply)
{
	halfmesh::TQuadric<double> q(0.0, 0.0, 1.0, 0.0); // z=0 plane
	double baseErr = q * Eigen::Vector3d(0.0, 0.0, 2.0); // = 4
	q *= 2.0;
	double scaledErr = q * Eigen::Vector3d(0.0, 0.0, 2.0); // = 8
	EXPECT_NEAR(baseErr * 2.0, scaledErr, 1e-12);
}

TEST(QuadricTest, MatrixConversionSymmetric)
{
	halfmesh::TQuadric<double> q(1.0, 2.0, 3.0, 4.0,
	                             5.0, 6.0, 7.0,
	                             8.0, 9.0,
	                             10.0);
	Eigen::Matrix4d m = q;
	// Symmetric: m(i,j) == m(j,i)
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			EXPECT_DOUBLE_EQ(m(r, c), m(c, r));
}
