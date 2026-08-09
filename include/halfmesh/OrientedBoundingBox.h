/*
* OrientedBoundingBox.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <halfmesh/Types.h>
#include <halfmesh/Util/Assert.h>

#include <Eigen/Dense>
#include <array>

namespace halfmesh {

// -------------------------------------------------------------------------
// OBB — Oriented Bounding-Box.
// `rot` transforms a world-space point into OBB coordinate space.
// `box` is the axis-aligned box expressed in OBB coordinate space.
// -------------------------------------------------------------------------
struct OBB
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	enum : int { numCorners = 8 }; // 2^DIMS

	// empty constructor
	OBB() { Reset(); }

	// construct from min/max points in OBB coordinate space + rotation
	OBB(const Vector3& minPoint, const Vector3& maxPoint, const Matrix3& rotation) :
	    rot(rotation), box(minPoint, maxPoint) {}

	// construct this OBB from an axis-aligned bounding-box
	explicit OBB(const AABB3& aabb) { *this = aabb; }
	OBB& operator=(const AABB3& aabb)
	{
		rot.setIdentity();
		box = aabb;
		return *this;
	}

	// reset the OBB such that IsEmpty() returns true
	OBB& Reset()
	{
		rot.setIdentity();
		box.setEmpty();
		return *this;
	}

	// check if the OBB is empty
	bool IsEmpty() const { return box.isEmpty(); }

	// add points to this OBB
	template <typename Derived>
	OBB& Extend(const Eigen::MatrixBase<Derived>& p)
	{
		box.extend(rot * p);
		return *this;
	}

	// get the center of the OBB in OBB coordinate space
	Vector3 Center() const { return box.center(); }

	// get the size of the OBB in OBB coordinate space
	Vector3 Size() const { return box.sizes(); }

	// get the 8 corners of the OBB in world space
	std::array<Vector3, numCorners> GetCorners() const
	{
		std::array<Vector3, numCorners> corners;
		for (int i = 0; i < numCorners; ++i) {
			corners[i] = rot.transpose() * box.corner(static_cast<AABB3::CornerType>(i));
		}
		return corners;
	}

	// compute the minimum AABB that contains this OBB
	AABB3 GetAABB() const
	{
		const std::array<Vector3, numCorners> corners = GetCorners();
		AABB3 aabb;
		for (const Vector3& corner : corners) {
			aabb.extend(corner);
		}
		return aabb;
	}

	// expand the box symmetrically by an absolute value d
	void Expand(real d)
	{
		ASSERT(d >= 0);
		box.min() -= Vector3(d, d, d);
		box.max() += Vector3(d, d, d);
	}

	// expand the box symmetrically in each dimension by the vector d
	void Expand(Vector3 d)
	{
		ASSERT(d(0) >= 0 && d(1) >= 0 && d(2) >= 0);
		box.min() -= d;
		box.max() += d;
	}

	// transform the box by the given transform applied as:
	//   transformed_X = rotation * (X * scale) + translation
	void Transform(const Matrix3& rotation, const Point3& translation, real scale = 1)
	{
		rot *= rotation.transpose();
		const Point3 pos = rot * translation;
		box.min() = box.min() * scale + pos;
		box.max() = box.max() * scale + pos;
	}

	// get the volume of the box
	real Volume() const { return box.volume(); }

	// returns true if the point p is inside the OBB
	template <typename Derived>
	bool Contains(const Eigen::MatrixBase<Derived>& p) const
	{
		return box.contains(rot * p);
	}

	Matrix3 rot; // rotation from world to OBB coordinate space
	AABB3 box; // min/max represented in OBB coordinate space
};

} // namespace halfmesh
