/*
* Types.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <cstdint>
#include <limits>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Maths.h>

namespace halfmesh {

// -------------------------------------------------------------------------
// Fundamental scalar type used for area/quadric/accumulation computations.
// double (not float) — used for area/quadric/accumulation precision.
// -------------------------------------------------------------------------
using real = double;

// -------------------------------------------------------------------------
// Generic column-vector aliases (Eigen).
// -------------------------------------------------------------------------
template <typename T>
using TPoint3 = Eigen::Matrix<T, 3, 1>;

// -------------------------------------------------------------------------
// Concrete matrix / point / AABB aliases using the `real` scalar type.
// -------------------------------------------------------------------------
using Matrix3 = Eigen::Matrix<real, 3, 3>;
using Vector3 = Eigen::Matrix<real, 3, 1>;
using Point3 = Eigen::Matrix<real, 3, 1>;
using Point2 = Eigen::Matrix<real, 2, 1>;
using AABB3 = Eigen::AlignedBox<real, 3>;
using AABB2 = Eigen::AlignedBox<real, 2>;

// -------------------------------------------------------------------------
// Pixel — 3-channel uint8 in BGR order (matches OpenCV convention).
// -------------------------------------------------------------------------
using Pixel = TPoint3<uint8_t>;

// -------------------------------------------------------------------------
// Image3u — BGR image, one Pixel per element.
// -------------------------------------------------------------------------
using Image3u = cv::Mat_<Pixel>;

// -------------------------------------------------------------------------
// Index types
// -------------------------------------------------------------------------
using IIndex = uint32_t;

} // namespace halfmesh

// -------------------------------------------------------------------------
// NO_ID lives in namespace math so the mesh sources can access it as
// math::NO_ID (after "using namespace math;").
// -------------------------------------------------------------------------
namespace math {

constexpr uint32_t NO_ID = std::numeric_limits<uint32_t>::max(); // 0xFFFFFFFF

} // namespace math

// -------------------------------------------------------------------------
// Register halfmesh::Pixel as a valid cv::Mat_ element type.
// Eigen::Matrix<uint8_t,3,1> is not a native OpenCV type; we provide the
// specializations that cv::Mat_<Pixel> requires.
// -------------------------------------------------------------------------
namespace cv {

template <>
class DataDepth<halfmesh::Pixel::Scalar>
{
	public:
	enum {
		value = CV_8U,
		fmt = static_cast<int>('u')
	};
};

template <>
class DataType<halfmesh::Pixel>
{
	public:
	typedef halfmesh::Pixel value_type;
	typedef value_type work_type;
	typedef halfmesh::Pixel::Scalar channel_type;
	typedef value_type vec_type;
	enum {
		generic_type = 0,
		depth = DataDepth<channel_type>::value,
		channels = static_cast<int>(sizeof(value_type) / sizeof(channel_type)), // 3
		fmt = DataDepth<channel_type>::fmt,
		type = CV_MAKETYPE(depth, channels)
	};
};

} // namespace cv
