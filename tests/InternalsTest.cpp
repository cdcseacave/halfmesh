/*
* InternalsTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for src/-internal units (halfmesh::detail) that public headers do not
// expose: FairMesh (MeshRemeshShared.h) and ParallelForPool (ParallelFor.h).
// The target gets src/ + the BS thread-pool dir on its include path — see
// tests/CMakeLists.txt (internals_test).
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "MeshRemeshShared.h"
#include "ParallelFor.h"

namespace {

// FairMesh must never write a non-finite solve back into the caller's points:
// LDLT reports Success for NaN-contaminated inputs, and before this guard the
// solution was committed unchecked — the hole-filler's default doFair path
// baked NaN into the mesh permanently. Contract under test is FairMesh's own
// doc: "Returns false
// (positions left unchanged)".
TEST(InternalsTest, FairMeshRejectsNonFiniteSolve)
{
	const double quietNaN = std::numeric_limits<double>::quiet_NaN();
	// 4-vertex ring around one free centre vertex; one locked ring vertex is NaN.
	std::vector<Eigen::Matrix<double, 3, 1>> points = {
	    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {quietNaN, quietNaN, quietNaN}, // locked, poisons the rhs
	    {0.5, 0.5, 1.0}, // free centre (index 4)
	};
	const std::vector<Eigen::Matrix<int, 3, 1>> tris = {
	    {4, 0, 1}, {4, 1, 2}, {4, 2, 3}, {4, 3, 0}};
	std::vector<bool> locked = {true, true, true, true, false};
	const Eigen::Matrix<double, 3, 1> centreBefore = points[4];
	const bool ok = halfmesh::detail::FairMesh(points, tris, locked, /*k=*/1);
	EXPECT_FALSE(ok) << "a non-finite solve must be reported as failure";
	EXPECT_TRUE(points[4].allFinite()) << "failed solve must not touch points";
	// (Eigen operator== is coefficient-wise — compare via norm, not EXPECT_EQ)
	EXPECT_EQ((points[4] - centreBefore).norm(), 0.0);
}

// A worker exception must reach the ParallelForPool caller: the vendored
// BS::light_thread_pool swallows task exceptions (empty catch(...)), so
// before this guard a throwing fn silently dropped the rest of its block's
// indices and wait() returned as if complete — silent partial results
TEST(InternalsTest, ParallelForPoolRethrowsWorkerException)
{
	BS::light_thread_pool pool;
	std::atomic<int> ran{0};
	EXPECT_THROW(
	    halfmesh::detail::ParallelForPool(pool, std::size_t(100), [&](std::size_t i) {
		    if (i == 50)
			    throw std::runtime_error("worker failure");
		    ran.fetch_add(1, std::memory_order_relaxed);
	    }),
	    std::runtime_error);
	// the pool must be reusable and complete after a failed region
	std::atomic<int> ran2{0};
	halfmesh::detail::ParallelForPool(pool, std::size_t(100), [&](std::size_t) {
		ran2.fetch_add(1, std::memory_order_relaxed);
	});
	EXPECT_EQ(ran2.load(), 100);
}

} // namespace
