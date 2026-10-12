/*
* RasterTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Unit tests for the UV-space triangle rasterizer + gutter dilation
// (halfmesh/Util/Raster.h).

#include <gtest/gtest.h>

#include <map>
#include <utility>

#include <opencv2/core.hpp>

#include <halfmesh/Types.h>
#include <halfmesh/Util/Raster.h>

using halfmesh::Pixel;
using halfmesh::Image3u;
using halfmesh::Point2;
using halfmesh::Vector3;
using halfmesh::RasterizeTriangleBary;
using halfmesh::Dilate;
using halfmesh::PushPullFill;

namespace {

Pixel MakePixel(int b, int g, int r)
{
	return Pixel(static_cast<uint8_t>(b), static_cast<uint8_t>(g), static_cast<uint8_t>(r));
}

} // namespace

// The rasterizer must visit exactly the pixels whose centres lie inside the
// triangle — independent of winding (UV charts may be mirrored after packing).
TEST(RasterTest, CoversInteriorNotExterior)
{
	std::map<std::pair<int, int>, Vector3> hit;
	auto cb = [&](int x, int y, const Vector3& bary) {
		hit[{x, y}] = bary;
	};
	// Right triangle (clockwise winding -> negative signed area).
	RasterizeTriangleBary<double>(Point2(0, 0), Point2(4, 0), Point2(0, 4), 5, 5, cb);

	EXPECT_TRUE(hit.count({0, 0})); // vertex
	EXPECT_TRUE(hit.count({1, 1})); // interior (1+1 <= 4)
	EXPECT_FALSE(hit.count({3, 3})); // exterior (3+3 > 4)
}

// Barycentric weights at a vertex are the unit basis; at an interior point they
// are non-negative and sum to one.
TEST(RasterTest, BarycentricAtVerticesAndInterior)
{
	std::map<std::pair<int, int>, Vector3> hit;
	auto cb = [&](int x, int y, const Vector3& bary) { hit[{x, y}] = bary; };
	RasterizeTriangleBary<double>(Point2(0, 0), Point2(4, 0), Point2(0, 4), 5, 5, cb);

	const Vector3& atV1 = hit[{0, 0}];
	EXPECT_NEAR(atV1.x(), 1.0, 1e-9);
	EXPECT_NEAR(atV1.y(), 0.0, 1e-9);
	EXPECT_NEAR(atV1.z(), 0.0, 1e-9);

	const Vector3& atV2 = hit[{4, 0}];
	EXPECT_NEAR(atV2.y(), 1.0, 1e-9);

	const Vector3& atV3 = hit[{0, 4}];
	EXPECT_NEAR(atV3.z(), 1.0, 1e-9);

	const Vector3& atI = hit[{1, 1}];
	EXPECT_GE(atI.x(), 0.0);
	EXPECT_GE(atI.y(), 0.0);
	EXPECT_GE(atI.z(), 0.0);
	EXPECT_NEAR(atI.x() + atI.y() + atI.z(), 1.0, 1e-9);
}

// A triangle entirely outside the image rasters nothing (no out-of-bounds access).
TEST(RasterTest, TriangleOutsideImageRastersNothing)
{
	int count = 0;
	auto cb = [&](int, int, const Vector3&) { ++count; };
	RasterizeTriangleBary<double>(Point2(10, 10), Point2(14, 10), Point2(10, 14), 5, 5, cb);
	EXPECT_EQ(count, 0);
}

// Gutter dilation fills an invalid pixel with the mean of its valid neighbours
// (and marks it valid), exercising accumulate->store rounding.
TEST(RasterTest, DilateFillsWithNeighbourMean)
{
	Image3u img(1, 3);
	cv::Mat_<uint8_t> mask(1, 3);
	img(0, 0) = MakePixel(10, 10, 10);
	mask(0, 0) = 255;
	img(0, 1) = MakePixel(0, 0, 0);
	mask(0, 1) = 0; // invalid gutter pixel
	img(0, 2) = MakePixel(30, 30, 30);
	mask(0, 2) = 255;

	Dilate(img, mask, /*iterations=*/1, /*halfSize=*/1);

	EXPECT_EQ(mask(0, 1), 255);
	const Pixel& p = img(0, 1);
	EXPECT_EQ(static_cast<int>(p.x()), 20); // (10+30)/2
	EXPECT_EQ(static_cast<int>(p.y()), 20);
	EXPECT_EQ(static_cast<int>(p.z()), 20);
}

// Dilation must not overwrite already-valid pixels.
TEST(RasterTest, DilateLeavesValidPixelsUntouched)
{
	Image3u img(1, 3);
	cv::Mat_<uint8_t> mask(1, 3);
	img(0, 0) = MakePixel(10, 10, 10);
	mask(0, 0) = 255;
	img(0, 1) = MakePixel(0, 0, 0);
	mask(0, 1) = 0;
	img(0, 2) = MakePixel(30, 30, 30);
	mask(0, 2) = 255;

	Dilate(img, mask, 1, 1);

	EXPECT_EQ(static_cast<int>(img(0, 0).x()), 10);
	EXPECT_EQ(static_cast<int>(img(0, 2).x()), 30);
}

TEST(RasterTest, PushPullFillFillsEveryTexelFromTheKeptOnes)
{
	cv::Mat img(16, 24, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat mask(16, 24, CV_8UC1, cv::Scalar(0));
	// two kept regions of different colors, far apart
	img(cv::Rect(0, 0, 4, 4)).setTo(cv::Scalar(200, 100, 50));
	mask(cv::Rect(0, 0, 4, 4)).setTo(255);
	img(cv::Rect(20, 12, 4, 4)).setTo(cv::Scalar(10, 20, 30));
	mask(cv::Rect(20, 12, 4, 4)).setTo(255);
	PushPullFill(img, mask);
	// kept texels are untouched
	EXPECT_EQ(img.at<cv::Vec3b>(1, 1), cv::Vec3b(200, 100, 50));
	EXPECT_EQ(img.at<cv::Vec3b>(13, 21), cv::Vec3b(10, 20, 30));
	// every empty texel got a color between the two, the nearest region's next to it
	for (int r = 0; r < img.rows; ++r)
		for (int c = 0; c < img.cols; ++c) {
			const cv::Vec3b px = img.at<cv::Vec3b>(r, c);
			EXPECT_TRUE(px[0] >= 10 && px[0] <= 200 && px[2] >= 30 && px[2] <= 50) << r << "," << c;
		}
	EXPECT_EQ(img.at<cv::Vec3b>(4, 4), cv::Vec3b(200, 100, 50));
	EXPECT_EQ(img.at<cv::Vec3b>(11, 19), cv::Vec3b(10, 20, 30));
}
