/*
* Accumulator.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Maths.h>

#include <Eigen/Dense>

namespace halfmesh {

// -------------------------------------------------------------------------
// InitToZero — helper to produce a zero value for scalar or Eigen matrix types.
// -------------------------------------------------------------------------
template <typename Scalar>
inline Scalar InitToZero(Scalar*) { return Scalar(0); }

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
InitToZero(Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>*)
{
	return Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Zero();
}

// -------------------------------------------------------------------------
// WeightedAccumulator — basic weighted accumulator for arbitrary types.
// Note: Normalize() returns `*this` (WeightedAccumulator&), not `this`.
// -------------------------------------------------------------------------
template <typename TYPE, typename ACCUMTYPE = TYPE, typename WEIGHTTYPE = float>
class WeightedAccumulator
{
	public:
	typedef TYPE Type;
	typedef ACCUMTYPE AccumType;
	typedef WEIGHTTYPE WeightType;

	WeightedAccumulator() :
	    value(InitToZero(static_cast<Type*>(NULL))), weight(0) {}
	explicit WeightedAccumulator(const Type& v, WeightType w = WeightType(0)) :
	    value(v), weight(w) {}

	bool Empty() const
	{
		return weight <= WeightType(0);
	}

	// accumulate the given weighted value
	void Add(const Type& v, WeightType w = WeightType(1))
	{
		value += v * w;
		weight += w;
	}
	// subtract the given weighted value
	void Sub(const Type& v, WeightType w = WeightType(1))
	{
		value -= v * w;
		weight -= w;
	}

	WeightedAccumulator operator+(const WeightedAccumulator& rhs) const
	{
		return WeightedAccumulator(value + rhs.value, weight + rhs.weight);
	}
	WeightedAccumulator& operator+=(const WeightedAccumulator& rhs)
	{
		value += rhs.value;
		weight += rhs.weight;
		return *this;
	}
	WeightedAccumulator operator-(const WeightedAccumulator& rhs) const
	{
		return WeightedAccumulator(value - rhs.value, weight - rhs.weight);
	}
	WeightedAccumulator& operator-=(const WeightedAccumulator& rhs)
	{
		value -= rhs.value;
		weight -= rhs.weight;
		return *this;
	}

	// normalize accumulated value
	AccumType Normalized() const
	{
		ASSERT(weight > WeightType(0));
		return value / weight;
	}
	WeightedAccumulator& Normalize()
	{
		ASSERT(weight > WeightType(0));
		value /= weight;
		weight = WeightType(1);
		return *this;
	}

	// get value/weight
	AccumType Value() const { return value; }
	WeightType Weight() const { return weight; }

	private:
	AccumType value;
	WeightType weight;
};

} // namespace halfmesh

// -------------------------------------------------------------------------
// Expose in namespace math so call sites can use math::WeightedAccumulator.
// -------------------------------------------------------------------------
namespace math {
using halfmesh::WeightedAccumulator;
} // namespace math
