/*
* AccumulatorTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// WeightedAccumulator tests; expected values are hand-computed.
#include <gtest/gtest.h>

#include <halfmesh/Util/Accumulator.h>

TEST(WeightedAccumulator, WeightedMean)
{
	math::WeightedAccumulator<double> acc;
	EXPECT_TRUE(acc.Empty());

	acc.Add(10.0, 2.0f); // accumulated value = 20, weight = 2
	acc.Add(4.0, 3.0f); // accumulated value = 32, weight = 5
	// weighted mean = 32 / 5 = 6.4
	EXPECT_NEAR(acc.Normalized(), 6.4, 1e-12);
	EXPECT_NEAR(static_cast<double>(acc.Weight()), 5.0, 1e-6);
	EXPECT_FALSE(acc.Empty());
}

TEST(WeightedAccumulator, SubAndOperators)
{
	math::WeightedAccumulator<double> a(0.0, 0.0f);
	a.Add(6.0, 3.0f); // value = 18, weight = 3

	math::WeightedAccumulator<double> b(0.0, 0.0f);
	b.Add(2.0, 1.0f); // value = 2, weight = 1

	const auto c = a + b; // value = 20, weight = 4 -> mean = 5
	EXPECT_NEAR(c.Normalized(), 5.0, 1e-12);

	const auto d = a - b; // value = 16, weight = 2 -> mean = 8
	EXPECT_NEAR(d.Normalized(), 8.0, 1e-12);
}

TEST(WeightedAccumulator, DefaultCtorIsEmpty)
{
	math::WeightedAccumulator<float> acc;
	EXPECT_TRUE(acc.Empty());
	EXPECT_EQ(acc.Weight(), 0.0f);
}
