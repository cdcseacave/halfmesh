/*
* UtilTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Tests for the Util/ helper layer: each test pins the exact documented
// semantics of one helper.
#include <gtest/gtest.h>

#include <halfmesh/Types.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Hash.h>
#include <halfmesh/Util/Log.h>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Maths.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

using halfmesh::SQUARE;
using halfmesh::D2R;

// ---------------------------------------------------------------------------
// Types.h — real, NO_ID, IIndex, Pixel, Image3u
// ---------------------------------------------------------------------------

TEST(Types, RealIsDouble)
{
	static_assert(std::is_same_v<halfmesh::real, double>,
	              "halfmesh::real must be double");
	SUCCEED();
}

TEST(Types, NO_ID_Value)
{
	// Sentinel must equal 0xFFFFFFFF (uint32 max)
	constexpr uint32_t expected = 0xFFFFFFFFu;
	static_assert(math::NO_ID == expected, "NO_ID must equal 0xFFFFFFFF");
	EXPECT_EQ(math::NO_ID, expected);
	EXPECT_EQ(math::NO_ID, std::numeric_limits<uint32_t>::max());
}

TEST(Types, NO_ID_IsUint32Sentinel)
{
	// NO_ID must be the uint32_t max sentinel (not INT_MAX, not SIZE_MAX)
	EXPECT_EQ(sizeof(math::NO_ID), sizeof(uint32_t));
	// Incrementing wraps to 0 (standard unsigned overflow)
	const uint32_t wrapped = math::NO_ID + 1u;
	EXPECT_EQ(wrapped, 0u);
}

TEST(Types, IIndexIsUint32)
{
	static_assert(std::is_same_v<halfmesh::IIndex, uint32_t>,
	              "IIndex must be uint32_t");
	SUCCEED();
}

TEST(Types, PixelIsBGRUint8x3)
{
	// Pixel == TPoint3<uint8_t> == Eigen::Matrix<uint8_t,3,1>
	static_assert(std::is_same_v<halfmesh::Pixel, halfmesh::TPoint3<uint8_t>>,
	              "Pixel must be TPoint3<uint8_t>");
	static_assert(std::is_same_v<halfmesh::Pixel::Scalar, uint8_t>,
	              "Pixel::Scalar must be uint8_t");
	static_assert(halfmesh::Pixel::RowsAtCompileTime == 3,
	              "Pixel must have 3 channels");
	SUCCEED();
}

TEST(Types, Image3uPixelRoundtrip)
{
	// Construct a 1x1 Image3u, write a BGR pixel, read it back.
	halfmesh::Image3u img(1, 1);
	halfmesh::Pixel pix;
	pix[0] = 10u; // B
	pix[1] = 20u; // G
	pix[2] = 30u; // R
	img(0, 0) = pix;

	const halfmesh::Pixel& got = img(0, 0);
	EXPECT_EQ(got[0], 10u);
	EXPECT_EQ(got[1], 20u);
	EXPECT_EQ(got[2], 30u);
}

// ---------------------------------------------------------------------------
// halfmesh/Util/Maths.h — SQUARE, D2R, IsZero
// ---------------------------------------------------------------------------

TEST(Math, SquarePositive)
{
	EXPECT_DOUBLE_EQ(SQUARE(3.0), 9.0);
	EXPECT_DOUBLE_EQ(SQUARE(0.5), 0.25);
}

TEST(Math, SquareNegative)
{
	EXPECT_DOUBLE_EQ(SQUARE(-5.0), 25.0);
}

TEST(Math, SquareFloat)
{
	EXPECT_FLOAT_EQ(SQUARE(2.0f), 4.0f);
}

TEST(Math, D2R180)
{
	// 180 degrees → π, must match within 1e-12 (formula: deg * (M_PI/180.0))
	const double result = D2R(180.0);
	EXPECT_NEAR(result, M_PI, 1e-12);
}

TEST(Math, D2R90)
{
	const double result = D2R(90.0);
	EXPECT_NEAR(result, M_PI / 2.0, 1e-12);
}

TEST(Math, D2RFloat)
{
	// Float version — less precision, but still round-trips cleanly for small angles
	const float result = D2R(180.0f);
	EXPECT_NEAR(result, static_cast<float>(M_PI), 1e-5f);
}

TEST(Math, IsZeroIntegralExact)
{
	EXPECT_TRUE(halfmesh::IsZero(0));
	EXPECT_FALSE(halfmesh::IsZero(1));
	EXPECT_FALSE(halfmesh::IsZero(-1));
}

TEST(Math, IsZeroFloatJustInside)
{
	// Original: epsilon = std::numeric_limits<double>::epsilon() * 1000
	// A value exactly equal to the threshold must be considered zero.
	const double eps = std::numeric_limits<double>::epsilon() * 1000.0;
	EXPECT_TRUE(halfmesh::IsZero(eps)); // == threshold → zero
	EXPECT_TRUE(halfmesh::IsZero(-eps)); // negative threshold
	EXPECT_TRUE(halfmesh::IsZero(0.0));
}

TEST(Math, IsZeroFloatJustOutside)
{
	// A value slightly beyond the threshold must NOT be considered zero.
	const double eps = std::numeric_limits<double>::epsilon() * 1000.0;
	// eps * (1 + small) > threshold
	const double justOutside = eps * (1.0 + 1e-9);
	EXPECT_FALSE(halfmesh::IsZero(justOutside));
	EXPECT_FALSE(halfmesh::IsZero(-justOutside));
}

TEST(Math, IsZeroCustomEpsilon)
{
	EXPECT_TRUE(halfmesh::IsZero(0.01, 0.02));
	EXPECT_FALSE(halfmesh::IsZero(0.03, 0.02));
}

// ---------------------------------------------------------------------------
// Util/Hash.h — pair and tuple hashing
// ---------------------------------------------------------------------------

TEST(Hash, PairEqualInputsHashEqual)
{
	using P = std::pair<uint32_t, uint32_t>;
	P a{42u, 7u};
	P b{42u, 7u};
	EXPECT_EQ(std::hash<P>{}(a), std::hash<P>{}(b));
}

TEST(Hash, PairDifferentInputsHashDiffer)
{
	using P = std::pair<uint32_t, uint32_t>;
	P a{42u, 7u};
	P b{7u, 42u}; // swapped
	// Not guaranteed distinct in general, but these specific values differ
	EXPECT_NE(std::hash<P>{}(a), std::hash<P>{}(b));
}

TEST(Hash, PairKnownValue)
{
	// Verify the exact Boost 1.81-style mixing formula on (1u, 2u):
	// seed += 0x9e3779b97f4a7c15 + hash(v); then a murmur3/splitmix64 finalizer.
	// This test verifies the formula is implemented correctly (not just "it hashes").
	using P = std::pair<uint32_t, uint32_t>;
	P p{1u, 2u};
	const size_t h = std::hash<P>{}(p);
	// Compute expected manually using the same formula
	auto hashCombineFn = [](size_t seed, uint32_t v) -> size_t {
		size_t x = seed + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + std::hash<uint32_t>{}(v);
		x ^= x >> 33;
		x *= static_cast<size_t>(0xff51afd7ed558ccdULL);
		x ^= x >> 33;
		x *= static_cast<size_t>(0xc4ceb9fe1a85ec53ULL);
		x ^= x >> 33;
		return x;
	};
	size_t expected = 0;
	expected = hashCombineFn(expected, 1u);
	expected = hashCombineFn(expected, 2u);
	EXPECT_EQ(h, expected);
}

TEST(Hash, PairUsableInUnorderedMap)
{
	// The primary use-case in src/HalfMesh.cpp: unordered_map keyed on pair/tuple of VIndex
	using P = std::pair<uint32_t, uint32_t>;
	std::unordered_map<P, uint32_t> m;
	P k1{1u, 2u};
	P k2{3u, 4u};
	m[k1] = 100u;
	m[k2] = 200u;
	EXPECT_EQ(m[k1], 100u);
	EXPECT_EQ(m[k2], 200u);
}

TEST(Hash, TupleEqualInputsHashEqual)
{
	using T = std::tuple<uint32_t, uint32_t>;
	T a{5u, 6u};
	T b{5u, 6u};
	EXPECT_EQ(std::hash<T>{}(a), std::hash<T>{}(b));
}

TEST(Hash, TupleUsableInUnorderedMap)
{
	// src/HalfMesh.cpp uses unordered_map keyed on std::tuple<VIndex,VIndex>
	using T = std::tuple<uint32_t, uint32_t>;
	std::unordered_map<T, uint32_t> m;
	T k{0u, 1u};
	m[k] = 42u;
	EXPECT_EQ(m[k], 42u);
}

// ---------------------------------------------------------------------------
// Util/Loop.h — FOREACH, FOREACHIDX, RFOREACH, RFOREACHIDX, FOREACHPTR, RFOREACHPTR
// ---------------------------------------------------------------------------

TEST(Loop, FOREACH_ForwardOrder)
{
	const std::vector<int> v{10, 20, 30};
	std::vector<int> result;
	FOREACH (i, v) {
		result.push_back(v[i]);
	}
	const std::vector<int> expected{10, 20, 30};
	EXPECT_EQ(result, expected);
}

TEST(Loop, RFOREACH_ReverseOrder)
{
	const std::vector<int> v{10, 20, 30};
	std::vector<int> result;
	RFOREACH (i, v) {
		result.push_back(v[i]);
	}
	const std::vector<int> expected{30, 20, 10};
	EXPECT_EQ(result, expected);
}

TEST(Loop, FOREACHIDX_TypedIndex)
{
	const std::vector<int> v{1, 2, 3};
	uint32_t sum = 0;
	FOREACHIDX (uint32_t, i, v) {
		sum += i;
	}
	EXPECT_EQ(sum, 0u + 1u + 2u);
}

TEST(Loop, RFOREACHIDX_TypedReverse)
{
	const std::vector<int> v{1, 2, 3};
	std::vector<uint32_t> indices;
	RFOREACHIDX (uint32_t, i, v) {
		indices.push_back(i);
	}
	const std::vector<uint32_t> expected{2u, 1u, 0u};
	EXPECT_EQ(indices, expected);
}

TEST(Loop, FOREACHPTR_ForwardPointers)
{
	std::vector<int> v{5, 6, 7};
	std::vector<int> result;
	FOREACHPTR (p, v) {
		result.push_back(*p);
	}
	const std::vector<int> expected{5, 6, 7};
	EXPECT_EQ(result, expected);
}

TEST(Loop, RFOREACHPTR_ReversePointers)
{
	std::vector<int> v{5, 6, 7};
	std::vector<int> result;
	RFOREACHPTR (p, v) {
		result.push_back(*p);
	}
	const std::vector<int> expected{7, 6, 5};
	EXPECT_EQ(result, expected);
}

TEST(Loop, FOREACH_EmptyVector)
{
	const std::vector<int> v;
	int count = 0;
	FOREACH (i, v) {
		++count;
	}
	EXPECT_EQ(count, 0);
}

// ---------------------------------------------------------------------------
// Util/Assert.h — ASSERT (runtime behavior in debug builds)
// ---------------------------------------------------------------------------

TEST(Assert, ASSERT_TrueDoesNotAbort)
{
	// Should not throw/abort
	EXPECT_NO_FATAL_FAILURE(ASSERT(1 == 1));
}

TEST(Assert, ASSERT_EQ_Equal)
{
	EXPECT_NO_FATAL_FAILURE(ASSERT_EQ(1, 1));
}

TEST(Assert, ASSERT_NE_Different)
{
	EXPECT_NO_FATAL_FAILURE(ASSERT_NE(1, 2));
}

// Runtime control of the library's REPORT_STATUS* progress logs (2026-08
// review: they were unconditionally on stdout with no consumer-side control).
TEST(Log, StatusLogRedirectAndSilence)
{
	std::ostringstream oss;
	halfmesh::SetStatusLog(&oss);
	REPORT_STATUS("hello {}", 42);
	EXPECT_NE(oss.str().find("hello 42"), std::string::npos);

	halfmesh::SetStatusLog(nullptr); // silence
	REPORT_STATUS("dropped");
	EXPECT_EQ(oss.str().find("dropped"), std::string::npos);

	halfmesh::SetStatusLog(&std::cout); // restore the default for other tests
	EXPECT_EQ(halfmesh::GetStatusLog(), &std::cout);
}
