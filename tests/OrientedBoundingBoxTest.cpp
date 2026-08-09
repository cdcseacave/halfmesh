/*
* OrientedBoundingBoxTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// OrientedBoundingBox tests (Contains + Transform).
#include <gtest/gtest.h>

#include <halfmesh/OrientedBoundingBox.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

double RandomNear(double mean, double sigma)
{
	return mean + sigma * (1.0 - 2.0 * (static_cast<double>(std::rand()) / RAND_MAX));
}

bool TestOBB(unsigned iters)
{
	using halfmesh::OBB;
	using Point3 = halfmesh::Point3;
	using Matrix3 = halfmesh::Matrix3;
	using real = halfmesh::real;

	const Point3 center = Point3::Random() * RandomNear(15.0, 3.0);
	const Point3 extent = Point3::Random().array().abs() * RandomNear(10.0, 3.0);

	OBB obb;
	obb.box.min() = center - extent;
	obb.box.max() = center + extent;
	obb.rot = Matrix3::Identity();

	std::vector<Point3> points;
	std::vector<bool> contains;
	points.reserve(iters);
	contains.reserve(iters);

	for (unsigned iter = 0; iter < iters; ++iter) {
		const Point3 delta = extent * RandomNear(0.0, 2.0);
		const Point3 point = center + delta;
		points.push_back(point);
		contains.push_back(obb.Contains(point));
	}

	// first transform
	{
		const Matrix3 rotation = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
		const Point3 translation = Point3::Random() * RandomNear(30.0, 10.0);
		const real scale = static_cast<real>(RandomNear(3.0, 1.0));
		obb.Transform(rotation, translation, scale);
		for (unsigned iter = 0; iter < iters; ++iter) {
			Point3& p = points[iter];
			p = rotation * (p * scale) + translation;
			if (obb.Contains(p) != contains[iter])
				return false;
		}
	}

	// second transform
	{
		const Matrix3 rotation = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
		const Point3 translation = Point3::Random() * RandomNear(30.0, 10.0);
		const real scale = static_cast<real>(RandomNear(3.0, 1.0));
		obb.Transform(rotation, translation, scale);
		for (unsigned iter = 0; iter < iters; ++iter) {
			const Point3 p = rotation * (points[iter] * scale) + translation;
			if (obb.Contains(p) != contains[iter])
				return false;
		}
	}
	return true;
}

} // anonymous namespace

TEST(OBBTest, ContainsAndTransform)
{
	std::srand(42);
	EXPECT_TRUE(TestOBB(100));
}
