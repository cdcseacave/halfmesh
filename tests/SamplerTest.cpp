/*
* SamplerTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Unit tests for the generic image sampler (halfmesh/Util/Sampler.h):
// Linear/Cubic interpolation policies over arbitrary cv::Mat_<T> pixel types,
// bounds-safe with an optional validate predicate.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <halfmesh/Types.h>
#include <halfmesh/Util/Sampler.h>

using halfmesh::Pixel;
using halfmesh::Image3u;
using halfmesh::Point2;
using halfmesh::Vector3;
using halfmesh::LinearInterp;
using halfmesh::CubicInterp;
using halfmesh::SampleImage;

namespace {

Pixel MakePixel(int b, int g, int r)
{
	return Pixel(static_cast<uint8_t>(b), static_cast<uint8_t>(g), static_cast<uint8_t>(r));
}

} // namespace

// At an exact integer coordinate the linear sampler must return that pixel
// untouched (no neighbour bleeding).
TEST(SamplerTest, LinearAtIntegerCoordReturnsExactPixel)
{
	Image3u img(2, 2);
	img(0, 0) = MakePixel(10, 20, 30);
	img(0, 1) = MakePixel(40, 50, 60);
	img(1, 0) = MakePixel(70, 80, 90);
	img(1, 1) = MakePixel(100, 110, 120);

	const Vector3 s = SampleImage<LinearInterp<>>(img, Point2(1.0, 0.0));
	EXPECT_NEAR(s.x(), 40.0, 1e-9);
	EXPECT_NEAR(s.y(), 50.0, 1e-9);
	EXPECT_NEAR(s.z(), 60.0, 1e-9);
}

// Sampling halfway between two horizontal neighbours averages them.
TEST(SamplerTest, LinearMidpointAveragesTwoPixels)
{
	Image3u img(1, 2);
	img(0, 0) = MakePixel(0, 0, 0);
	img(0, 1) = MakePixel(100, 200, 40);

	const Vector3 s = SampleImage<LinearInterp<>>(img, Point2(0.5, 0.0));
	EXPECT_NEAR(s.x(), 50.0, 1e-9);
	EXPECT_NEAR(s.y(), 100.0, 1e-9);
	EXPECT_NEAR(s.z(), 20.0, 1e-9);
}

// Bilinear: the centre of a 2x2 block is the mean of its four pixels.
TEST(SamplerTest, LinearCenterAveragesFourPixels)
{
	Image3u img(2, 2);
	img(0, 0) = MakePixel(0, 0, 0);
	img(0, 1) = MakePixel(40, 0, 0);
	img(1, 0) = MakePixel(0, 80, 0);
	img(1, 1) = MakePixel(40, 80, 120);

	const Vector3 s = SampleImage<LinearInterp<>>(img, Point2(0.5, 0.5));
	EXPECT_NEAR(s.x(), 20.0, 1e-9); // (0+40+0+40)/4
	EXPECT_NEAR(s.y(), 40.0, 1e-9); // (0+0+80+80)/4
	EXPECT_NEAR(s.z(), 30.0, 1e-9); // (0+0+0+120)/4
}

// Cubic convolution reproduces a linear function exactly: sample a horizontal
// ramp at a fractional interior coordinate and expect the analytic value.
TEST(SamplerTest, CubicReproducesLinearRamp)
{
	cv::Mat_<float> ramp(1, 6);
	for (int x = 0; x < 6; ++x)
		ramp(0, x) = 2.0f * x + 1.0f; // value = 2x + 1

	// Interior fractional sample (needs 2 taps each side: x in [1, cols-2]).
	const double v = SampleImage<CubicInterp<>>(ramp, Point2(2.25, 0.0));
	EXPECT_NEAR(v, 2.0 * 2.25 + 1.0, 1e-6);
}

// The sampler is generic over the pixel type: works on a scalar float image.
TEST(SamplerTest, LinearOnScalarFloatImage)
{
	cv::Mat_<float> img(1, 2);
	img(0, 0) = 10.0f;
	img(0, 1) = 30.0f;

	const double v = SampleImage<LinearInterp<>>(img, Point2(0.5, 0.0));
	EXPECT_NEAR(v, 20.0, 1e-6);
}

// The validate predicate excludes masked samples; remaining weights renormalise
// so a masked corner does not darken the result.
TEST(SamplerTest, ValidateSkipsMaskedPixels)
{
	Image3u img(2, 2);
	img(0, 0) = MakePixel(30, 30, 30);
	img(0, 1) = MakePixel(60, 60, 60);
	img(1, 0) = MakePixel(90, 90, 90);
	img(1, 1) = MakePixel(0, 0, 0); // masked-out sentinel

	const auto valid = [](const Pixel& p) { return !(p.x() == 0 && p.y() == 0 && p.z() == 0); };
	const Vector3 s = SampleImage<LinearInterp<>>(img, Point2(0.5, 0.5), valid);
	EXPECT_NEAR(s.x(), 60.0, 1e-9); // (30+60+90)/3
	EXPECT_NEAR(s.y(), 60.0, 1e-9);
	EXPECT_NEAR(s.z(), 60.0, 1e-9);
}
