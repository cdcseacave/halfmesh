/*
* TextureBakeTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Unit tests for the texture-bake engine + resolvers (halfmesh/TextureBake.h).
//
// The core correctness property is the *identity self-bake*: when the target is
// the source, with the same UV layout, the baked atlas must reproduce the source
// texture. This validates rasterize -> resolve -> sample -> write end to end for
// both the SameUV (defrag) and Raycast (rebake) resolvers.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include <halfmesh/Types.h>
#include <halfmesh/Mesh.h>
#include <halfmesh/TextureBake.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>

#include "Corpus.h"
#include <halfmesh/Util/PixelTraits.h>
#include <halfmesh/Util/Raster.h>

using halfmesh::Mesh;
using halfmesh::Pixel;
using halfmesh::Image3u;
using halfmesh::BakeParams;
using halfmesh::BakeResult;
using halfmesh::Correspondence;
using halfmesh::BakeAtlas;
using halfmesh::SameUVResolver;
using halfmesh::RaycastResolver;
using halfmesh::AutoAtlasResolution;
using halfmesh::UVBlobsAreNormalized;

namespace {

// A flat unit-square mesh (two triangles in the z=0 plane) whose UVs map the
// quad onto the whole [0,size] x [0,size] atlas (absolute pixel coordinates).
Mesh MakeQuad(int size)
{
	Mesh m;
	m.vertices = {
	    Mesh::Vertex(0, 0, 0),
	    Mesh::Vertex(1, 0, 0),
	    Mesh::Vertex(1, 1, 0),
	    Mesh::Vertex(0, 1, 0),
	};
	m.faces = {Mesh::Face(0, 1, 2), Mesh::Face(0, 2, 3)};
	const float s = static_cast<float>(size);
	const Mesh::TexCoord uv0(0, 0), uv1(s, 0), uv2(s, s), uv3(0, s);
	m.faceTexcoords = {uv0, uv1, uv2, uv0, uv2, uv3};
	return m;
}

Image3u MakeGradient(int size)
{
	Image3u img(size, size);
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c)
			img(r, c) = Pixel(static_cast<uint8_t>(c * 10),
			                  static_cast<uint8_t>(r * 10),
			                  static_cast<uint8_t>(50));
	return img;
}

// Real textured fixture: 300k faces, one 4096x4096 texture. Gitignored
// (tests/data/*.glb) — tests using it must GTEST_SKIP when absent, so CI and
// fresh clones stay green without it.
std::string TruckMeshPath()
{
	return (std::filesystem::path(__FILE__).parent_path() / "data" / "truck_textured.glb").string();
}

} // namespace

// A flat (cells x cells)-quad grid over the unit square, with identity UVs into
// a texsize^2 atlas and a gradient texture whose BLUE channel is constant (50) —
// so any correct sample of it has z == 50 regardless of the new layout.
Mesh MakeGridTextured(int cells, int texsize)
{
	Mesh m;
	const int n = cells + 1;
	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i)
			m.vertices.emplace_back(static_cast<float>(i) / cells,
			                        static_cast<float>(j) / cells, 0.f);
	auto vid = [n](int i, int j) { return static_cast<Mesh::VIndex>(j * n + i); };
	const float s = static_cast<float>(texsize);
	auto uv = [&](int i, int j) {
		return Mesh::TexCoord(static_cast<float>(i) / cells * s,
		                      static_cast<float>(j) / cells * s);
	};
	for (int j = 0; j < cells; ++j)
		for (int i = 0; i < cells; ++i) {
			m.faces.emplace_back(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1));
			m.faceTexcoords.push_back(uv(i, j));
			m.faceTexcoords.push_back(uv(i + 1, j));
			m.faceTexcoords.push_back(uv(i + 1, j + 1));
			m.faces.emplace_back(vid(i, j), vid(i + 1, j + 1), vid(i, j + 1));
			m.faceTexcoords.push_back(uv(i, j));
			m.faceTexcoords.push_back(uv(i + 1, j + 1));
			m.faceTexcoords.push_back(uv(i, j + 1));
		}
	Image3u tex(texsize, texsize);
	const int scale = 200 / texsize;
	for (int r = 0; r < texsize; ++r)
		for (int c = 0; c < texsize; ++c)
			tex(r, c) = Pixel(static_cast<uint8_t>(c * scale),
			                  static_cast<uint8_t>(r * scale), 50);
	m.texturesDiffuse = {tex};
	return m;
}

// `n` disconnected unit quads (separate UV islands => `n` charts), each mapped to
// the same [0,texsize] square of one constant-blue (z=50) texture. Used to force
// multi-page packing: at density texsize each chart fills a texsize-page.
Mesh MakeMultiChartTextured(int n, int texsize)
{
	Mesh m;
	const float s = static_cast<float>(texsize);
	const Mesh::TexCoord uv0(0, 0), uv1(s, 0), uv2(s, s), uv3(0, s);
	for (int q = 0; q < n; ++q) {
		const float x = static_cast<float>(q) * 2.f; // offset so quads are disjoint
		const auto base = static_cast<Mesh::VIndex>(m.vertices.size());
		m.vertices.emplace_back(x, 0, 0);
		m.vertices.emplace_back(x + 1, 0, 0);
		m.vertices.emplace_back(x + 1, 1, 0);
		m.vertices.emplace_back(x, 1, 0);
		m.faces.emplace_back(base, base + 1, base + 2);
		m.faces.emplace_back(base, base + 2, base + 3);
		m.faceTexcoords.push_back(uv0);
		m.faceTexcoords.push_back(uv1);
		m.faceTexcoords.push_back(uv2);
		m.faceTexcoords.push_back(uv0);
		m.faceTexcoords.push_back(uv2);
		m.faceTexcoords.push_back(uv3);
	}
	m.texturesDiffuse = {MakeGradient(texsize)};
	return m;
}

// Count filled (non-black) texels across all pages and assert every one carries
// the source's constant blue channel (z == 50) — a layout-invariant proof that
// the bake sampled the source and not the background.
int CheckFilledHaveConstantBlue(const Mesh& m)
{
	int filled = 0;
	for (const Image3u& atlas : m.texturesDiffuse)
		for (int r = 0; r < atlas.rows; ++r)
			for (int c = 0; c < atlas.cols; ++c) {
				const Pixel& p = atlas(r, c);
				if (p.x() || p.y() || p.z()) {
					++filled;
					EXPECT_EQ(static_cast<int>(p.z()), 50);
				}
			}
	return filled;
}

// SameUV resolver: identity self-bake reproduces the source texture exactly.
TEST(TextureBakeTest, SameUVIdentityReproducesSource)
{
	const int size = 4;
	Mesh m = MakeQuad(size);
	const Image3u src = MakeGradient(size);

	SameUVResolver resolver(m.faceTexcoords, m.faceTexblobs,
	                        {Eigen::Vector2i(size, size)});

	BakeParams params;
	params.resolution = size;
	params.padding = 0;

	const BakeResult res = BakeAtlas(m, {src}, resolver, params);

	ASSERT_EQ(res.numPages, 1u);
	ASSERT_EQ(m.texturesDiffuse.size(), 1u);
	EXPECT_FLOAT_EQ(res.emptyTexelRatio, 0.0f);

	const Image3u& atlas = m.texturesDiffuse[0];
	ASSERT_EQ(atlas.rows, size);
	ASSERT_EQ(atlas.cols, size);
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c) {
			EXPECT_EQ(atlas(r, c), src(r, c)) << "mismatch at (" << r << "," << c << ")";
		}
}

// Supersampling must not lose edge coverage: a texel whose sub-samples all fall
// outside the triangle falls back to its (always-inside) centre, so coverage
// matches single-sample baking.
TEST(TextureBakeTest, SuperSampleKeepsFullCoverage)
{
	const int size = 4;
	Mesh m = MakeQuad(size);
	const Image3u src = MakeGradient(size);
	SameUVResolver resolver(m.faceTexcoords, m.faceTexblobs,
	                        {Eigen::Vector2i(size, size)});

	BakeParams params;
	params.resolution = size;
	params.padding = 0;
	params.supersample = 2;

	const BakeResult res = BakeAtlas(m, {src}, resolver, params);
	EXPECT_FLOAT_EQ(res.emptyTexelRatio, 0.0f); // every covered texel filled
}

// SameUV resolver with a NORMALIZED source layout (as real loaded meshes store
// it): the resolver must scale source UVs by the image size AND subtract the
// half-texel offset before sampling. Normalized UVs follow the GPU/glTF
// convention where texel i's centre is (i + 0.5)/W (what
// Mesh::FTexcoordsNormalize produces), while the sampler treats INTEGER
// coordinates as pixel centres (Sampler.h), so pixel = u*W - 0.5, never plain
// u*W. Target layout is absolute pixels (as the packer produces).
TEST(TextureBakeTest, SameUVNormalizedSourceReproduces)
{
	const int size = 4;
	Mesh m = MakeQuad(size); // target UVs absolute [0,size]
	const Image3u src = MakeGradient(size);

	// Snapshot source layout in NORMALIZED centre convention: pixel coord t
	// maps to (t + 0.5)/size, mirroring Mesh::FTexcoordsNormalize.
	std::vector<Mesh::TexCoord> normUv;
	for (const Mesh::TexCoord& t : m.faceTexcoords)
		normUv.emplace_back((t.x() + 0.5f) / size, (t.y() + 0.5f) / size);
	const std::vector<Eigen::Vector2i> dims = {Eigen::Vector2i(size, size)};

	SameUVResolver resolver(normUv, m.faceTexblobs, dims);

	BakeParams params;
	params.resolution = size;
	params.padding = 0;

	BakeAtlas(m, {src}, resolver, params);
	const Image3u& atlas = m.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c)
			EXPECT_EQ(atlas(r, c), src(r, c)) << "at (" << r << "," << c << ")";
}

// Raycast resolver with a NORMALIZED source layout: the same half-texel rule
// applies when the hit face's UVs are normalized and get scaled to pixel space
// before sampling. Source UVs use the centre convention (t + 0.5)/size, so an
// identity self-bake must reproduce the texture; without the -0.5 every
// interior texel blends its right/bottom neighbours (off by ~5 on a
// 10-per-texel gradient).
TEST(TextureBakeTest, RaycastNormalizedSourceReproduces)
{
	const int size = 4;
	Mesh source = MakeQuad(size);
	for (Mesh::TexCoord& t : source.faceTexcoords)
		t = Mesh::TexCoord((t.x() + 0.5f) / size, (t.y() + 0.5f) / size);
	source.texturesDiffuse = {MakeGradient(size)};

	// Target is a geometric copy with absolute-pixel UVs (packer convention).
	Mesh target = MakeQuad(size);

	RaycastResolver resolver(source, Correspondence::Nearest);

	BakeParams params;
	params.resolution = size;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;

	BakeAtlas(target, source.texturesDiffuse, resolver, params);

	ASSERT_EQ(target.texturesDiffuse.size(), 1u);
	const Image3u& atlas = target.texturesDiffuse[0];
	const Image3u& src = source.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c) {
			const Pixel& a = atlas(r, c);
			const Pixel& b = src(r, c);
			EXPECT_NEAR(a.x(), b.x(), 1) << "at (" << r << "," << c << ")";
			EXPECT_NEAR(a.y(), b.y(), 1);
			EXPECT_NEAR(a.z(), b.z(), 1);
		}
}

// Normalized source UVs outside [0,1] (here tiled by -1, GL_REPEAT) must wrap
// into [0,1) before scaling, recovering the same texel — so the bake reproduces
// the source instead of scaling to a pixel coordinate outside the image and
// baking solid black. (-1 keeps |UV| <= 2 so the coords still read as normalized.)
TEST(TextureBakeTest, WrapNormalizedTiledSourceReproduces)
{
	const int size = 4;
	Mesh m = MakeQuad(size);
	const Image3u src = MakeGradient(size);

	std::vector<Mesh::TexCoord> normUv;
	for (const Mesh::TexCoord& t : m.faceTexcoords)
		normUv.emplace_back((t.x() + 0.5f) / size - 1.0f, (t.y() + 0.5f) / size - 1.0f);
	const std::vector<Eigen::Vector2i> dims = {Eigen::Vector2i(size, size)};

	SameUVResolver resolver(normUv, m.faceTexblobs, dims);
	BakeParams params;
	params.resolution = size;
	params.padding = 0;

	BakeAtlas(m, {src}, resolver, params);
	const Image3u& atlas = m.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c)
			EXPECT_EQ(atlas(r, c), src(r, c)) << "at (" << r << "," << c << ")";
}

// A sub-pixel-thin UV triangle covers no pixel centre. Its chart must still be
// resolved rather than left to gutter dilation from a neighbour: the conservative
// pass resolves the sliver's half-pixel boundary ring, and the centroid fallback is
// the safety net for a sliver too thin for even that to reach a texel. Either way
// the thin chart contributes at least one texel. The sliver below spans x in
// [1.1,1.4] (no integer column), so it has no centre-inside coverage.
TEST(TextureBakeTest, ThinTriangleContributesCentroidTexel)
{
	Mesh m;
	m.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0), Mesh::Vertex(0.5f, 1, 0)};
	m.faces = {Mesh::Face(0, 1, 2)};
	m.faceTexcoords = {Mesh::TexCoord(1.1f, 1.1f), Mesh::TexCoord(1.4f, 1.1f),
	                   Mesh::TexCoord(1.25f, 4.9f)};

	Image3u src(8, 8);
	src.setTo(cv::Scalar(20, 40, 60)); // solid, non-black
	SameUVResolver resolver(m.faceTexcoords, m.faceTexblobs, {Eigen::Vector2i(8, 8)});
	BakeParams params;
	params.resolution = 8;
	params.padding = 0;

	BakeAtlas(m, {src}, resolver, params);
	const Image3u& atlas = m.texturesDiffuse[0];
	int filled = 0;
	for (int r = 0; r < atlas.rows; ++r)
		for (int c = 0; c < atlas.cols; ++c)
			if (atlas(r, c).x() || atlas(r, c).y() || atlas(r, c).z())
				++filled;
	// Coverage superset of the old "exactly the centroid fallback texel": the
	// conservative boundary ring now also resolves the sliver, so filled >= 1.
	EXPECT_GE(filled, 1);
}

// Near-degenerate (sliver) source triangles must not extrapolate a UV outside the
// hit face's chart: the clamp/renormalize onto the simplex keeps the resolved UV
// inside the convex hull of the face's three texcoords, whatever the query point.
TEST(TextureBakeTest, SliverSourceUVStaysInChart)
{
	Mesh source;
	source.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0),
	                   Mesh::Vertex(0.5f, 1e-6f, 0)};
	source.faces = {Mesh::Face(0, 1, 2)};
	source.faceTexcoords = {Mesh::TexCoord(2, 2), Mesh::TexCoord(8, 2), Mesh::TexCoord(5, 8)};
	Image3u tex(16, 16);
	tex.setTo(cv::Scalar(10, 20, 30));
	source.texturesDiffuse = {tex};

	RaycastResolver resolver(source, Correspondence::Nearest);
	const halfmesh::Vector3 dummy(1.0 / 3, 1.0 / 3, 1.0 / 3);
	const Mesh::Normal nrm(0, 0, 1);
	for (double qy = -0.5; qy <= 0.5; qy += 0.1)
		for (double qx = -0.2; qx <= 1.2; qx += 0.1) {
			unsigned img;
			Mesh::TexCoord uv;
			Mesh::FIndex hint = std::numeric_limits<Mesh::FIndex>::max();
			ASSERT_TRUE(resolver.Resolve(0, dummy, halfmesh::Vector3(qx, qy, 0.3), nrm, img, uv, hint));
			// Convex hull of {(2,2),(8,2),(5,8)} is contained in [2,8]x[2,8].
			EXPECT_GE(uv.x(), 2.f - 1e-3f);
			EXPECT_LE(uv.x(), 8.f + 1e-3f);
			EXPECT_GE(uv.y(), 2.f - 1e-3f);
			EXPECT_LE(uv.y(), 8.f + 1e-3f);
		}
}

// Large-coordinate accuracy: an identity self-bake translated to georeferenced
// coordinates (1e6) must reproduce the source exactly, because the barycentrics
// are recomputed in double on the winning face rather than decomposed from the
// float-quantized query hit (which jitters several texels at 1e6).
TEST(TextureBakeTest, RaycastIdentityAtLargeCoordinatesReproducesSource)
{
	const int size = 4;
	const float shift = 1e6f;
	Mesh source = MakeQuad(size);
	for (Mesh::Vertex& v : source.vertices)
		v += Mesh::Vertex(shift, shift, 0);
	source.texturesDiffuse = {MakeGradient(size)};

	Mesh target = MakeQuad(size);
	for (Mesh::Vertex& v : target.vertices)
		v += Mesh::Vertex(shift, shift, 0);

	RaycastResolver resolver(source, Correspondence::Nearest);
	BakeParams params;
	params.resolution = size;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;

	BakeAtlas(target, source.texturesDiffuse, resolver, params);
	const Image3u& atlas = target.texturesDiffuse[0];
	const Image3u& src = source.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c) {
			EXPECT_NEAR(atlas(r, c).x(), src(r, c).x(), 1) << "at (" << r << "," << c << ")";
			EXPECT_NEAR(atlas(r, c).y(), src(r, c).y(), 1);
			EXPECT_NEAR(atlas(r, c).z(), src(r, c).z(), 1);
		}
}

// Raycast resolver (nearest mode): baking the source onto itself reproduces the
// source texture (correspondence + barycentric-from-hit are correct).
TEST(TextureBakeTest, RaycastIdentityReproducesSource)
{
	const int size = 4;
	Mesh source = MakeQuad(size);
	source.texturesDiffuse = {MakeGradient(size)};

	// Target is a geometric+UV copy of the source.
	Mesh target = MakeQuad(size);

	RaycastResolver resolver(source, Correspondence::Nearest);

	BakeParams params;
	params.resolution = size;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;

	const BakeResult res = BakeAtlas(target, source.texturesDiffuse, resolver, params);

	ASSERT_EQ(target.texturesDiffuse.size(), 1u);
	EXPECT_FLOAT_EQ(res.emptyTexelRatio, 0.0f);

	const Image3u& atlas = target.texturesDiffuse[0];
	const Image3u& src = source.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c) {
			const Pixel& a = atlas(r, c);
			const Pixel& b = src(r, c);
			EXPECT_NEAR(a.x(), b.x(), 1) << "at (" << r << "," << c << ")";
			EXPECT_NEAR(a.y(), b.y(), 1);
			EXPECT_NEAR(a.z(), b.z(), 1);
		}
}

// Parallel baking over faces must be deterministic and byte-identical to the
// single-threaded result (each output texel is owned by exactly one thread).
TEST(TextureBakeTest, ParallelMatchesSerial)
{
	Mesh m = MakeGridTextured(/*cells=*/20, /*texsize=*/64);
	SameUVResolver resolver(m.faceTexcoords, m.faceTexblobs,
	                        {Eigen::Vector2i(64, 64)});

	BakeParams base;
	base.resolution = 64;
	base.padding = 2;

	Mesh serial = m;
	Mesh parallel = m;
	BakeParams p1 = base;
	p1.numThreads = 1;
	BakeParams pN = base;
	pN.numThreads = 8;

	const BakeResult rs = BakeAtlas(serial, m.texturesDiffuse, resolver, p1);
	const BakeResult rp = BakeAtlas(parallel, m.texturesDiffuse, resolver, pN);

	ASSERT_EQ(rs.numPages, rp.numPages);
	ASSERT_EQ(serial.texturesDiffuse.size(), parallel.texturesDiffuse.size());
	for (size_t p = 0; p < serial.texturesDiffuse.size(); ++p) {
		cv::Mat diff;
		cv::absdiff(serial.texturesDiffuse[p], parallel.texturesDiffuse[p], diff);
		EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0) << "page " << p << " differs";
	}
}

// The warm-started nearest query chains the hint face differently in serial (one
// chain over all faces) vs parallel (one chain per row band), yet the resolved
// nearest point — hence the baked atlas — must be byte-identical either way. This
// pins the warm-start + bounded-ray optimizations as output-preserving.
TEST(TextureBakeTest, RaycastNearestParallelMatchesSerial)
{
	const Mesh src = MakeGridTextured(/*cells=*/20, /*texsize=*/64);
	RaycastResolver resolver(src, Correspondence::Nearest);

	BakeParams base;
	base.resolution = 64;
	base.padding = 2;
	base.correspondence = Correspondence::Nearest;

	Mesh serial = MakeGridTextured(20, 64);
	Mesh parallel = MakeGridTextured(20, 64);
	BakeParams p1 = base;
	p1.numThreads = 1;
	BakeParams pN = base;
	pN.numThreads = 8;

	const BakeResult rs = BakeAtlas(serial, src.texturesDiffuse, resolver, p1);
	const BakeResult rp = BakeAtlas(parallel, src.texturesDiffuse, resolver, pN);

	ASSERT_EQ(rs.numPages, rp.numPages);
	for (size_t p = 0; p < serial.texturesDiffuse.size(); ++p) {
		cv::Mat diff;
		cv::absdiff(serial.texturesDiffuse[p], parallel.texturesDiffuse[p], diff);
		EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0) << "page " << p << " differs";
	}
}

// Defragment: existing single chart is repacked into one filled page and the
// texture is resampled losslessly (constant blue channel preserved everywhere).
TEST(TextureBakeTest, DefragProducesSingleFilledPage)
{
	Mesh m = MakeGridTextured(/*cells=*/4, /*texsize=*/16);

	BakeParams params;
	params.resolution = 16;
	params.padding = 0;

	const BakeResult res = DefragmentTexture(m, params);

	EXPECT_EQ(res.numPages, 1u);
	ASSERT_EQ(m.texturesDiffuse.size(), 1u);
	EXPECT_LT(res.emptyTexelRatio, 0.05f);

	const int filled = CheckFilledHaveConstantBlue(m);
	const int total = m.texturesDiffuse[0].rows * m.texturesDiffuse[0].cols;
	EXPECT_GT(filled, total / 2);
}

// Rebake: generate a fresh atlas on a UV-less target and bake the source's
// texture onto it via surface queries; every baked texel must carry the source's
// constant blue channel.
TEST(TextureBakeTest, RebakeOntoFreshTargetSamplesSource)
{
	const Mesh source = MakeGridTextured(4, 16);

	// Target: same geometry, no UVs/textures (rebake regenerates them).
	Mesh target = MakeGridTextured(4, 16);
	target.faceTexcoords.clear();
	target.texturesDiffuse.clear();

	BakeParams params;
	params.resolution = 32;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;

	const BakeResult res = RebakeTexture(source, target, params);

	ASSERT_FALSE(target.texturesDiffuse.empty());
	EXPECT_GE(res.numPages, 1u); // packer may use multiple charts/pages
	EXPECT_EQ(target.texturesDiffuse.size(), res.numPages);

	const int filled = CheckFilledHaveConstantBlue(target);
	EXPECT_GT(filled, 0);
}

// AutoAtlasResolution sizes a single page to hold the source's texel detail,
// clamped to [256, maxResolution] and rounded to a power of two.
TEST(TextureBakeTest, AutoAtlasResolutionScalesWithSource)
{
	EXPECT_EQ(AutoAtlasResolution(MakeQuad(2048), 8192), 4096u); // ~2261 -> 4096
	EXPECT_EQ(AutoAtlasResolution(MakeQuad(16), 8192), 256u); // tiny -> min clamp
	EXPECT_EQ(AutoAtlasResolution(MakeQuad(8192), 8192), 8192u); // huge -> max clamp
}

// Multi-page mode preserves the source texel density and overflows into multiple
// pages; the engine fills every page.
TEST(TextureBakeTest, DefragMultiPagePreservesBudget)
{
	Mesh m = MakeMultiChartTextured(/*n=*/2, /*texsize=*/32); // two 32x32 charts

	BakeParams params;
	params.resolution = 32; // one chart per 32x32 page -> two pages
	params.padding = 0;
	params.multiPage = true;

	const BakeResult res = DefragmentTexture(m, params);

	EXPECT_GT(res.numPages, 1u);
	EXPECT_EQ(m.texturesDiffuse.size(), res.numPages);
	EXPECT_GT(CheckFilledHaveConstantBlue(m), 0);
}

// Auto texture-size: when the ideal single page exceeds maxResolution, it must
// auto-switch to multi-page (preserve density) even without multiPage set.
TEST(TextureBakeTest, AutoSwitchesToMultiPageBeyondCap)
{
	Mesh m = MakeMultiChartTextured(/*n=*/8, /*texsize=*/16); // eight 16x16 charts

	BakeParams params;
	params.resolution = 0; // auto
	params.maxResolution = 32; // ideal (~50) exceeds cap -> auto-switch
	params.padding = 0;
	params.multiPage = false; // not explicitly requested

	const BakeResult res = DefragmentTexture(m, params);
	EXPECT_GT(res.numPages, 1u);
}

namespace {

// Reference (pre-optimization) gutter dilation: clone-and-full-scan double buffer,
// exactly the algorithm the frontier version replaced. Used to prove byte-identity.
void DilateReference(Image3u& image, cv::Mat_<uint8_t>& mask, int iterations, int halfSize)
{
	using halfmesh::detail::AccumPixel;
	using Acc = AccumPixel<Pixel, double>::type;
	for (int it = 0; it < iterations; ++it) {
		Image3u out = image.clone();
		cv::Mat_<uint8_t> newmask = mask.clone();
		bool filledAny = false;
		for (int r = 0; r < image.rows; ++r)
			for (int c = 0; c < image.cols; ++c) {
				if (mask(r, c))
					continue;
				Acc sum = halfmesh::detail::AccumZero<Acc>();
				int n = 0;
				for (int i = -halfSize; i <= halfSize; ++i) {
					const int rr = r + i;
					if (rr < 0 || rr >= image.rows)
						continue;
					for (int j = -halfSize; j <= halfSize; ++j) {
						if (i == 0 && j == 0)
							continue;
						const int cc = c + j;
						if (cc < 0 || cc >= image.cols)
							continue;
						if (!mask(rr, cc))
							continue;
						sum += halfmesh::detail::AccumCast<Acc>(image(rr, cc));
						++n;
					}
				}
				if (n > 0) {
					out(r, c) = halfmesh::detail::StoreCast<Pixel>(sum * (1.0 / n));
					newmask(r, c) = 255;
					filledAny = true;
				}
			}
		image = out;
		mask = newmask;
		if (!filledAny)
			break;
	}
}

} // namespace

// Opt-in conservative rasterization must (a) leave the default centre-inside
// coverage untouched and (b) as a superset, additionally visit the ring of
// boundary texels within half a pixel of an edge, handing back clamped-and-
// renormalized non-negative barycentrics that still sum to one.
TEST(TextureBakeTest, ConservativeRasterCoversBoundarySuperset)
{
	using halfmesh::RasterizeTriangleBary;
	using halfmesh::Point2;
	using halfmesh::Vector3;

	// Triangle with edges biting mid-texel so a half-pixel expansion adds coverage.
	const Point2 a(1.5, 1.5), b(6.5, 1.5), c(1.5, 6.5);

	std::set<std::pair<int, int>> centre, conservative;
	auto collect = [](std::set<std::pair<int, int>>& s) {
		return [&s](int x, int y, const Vector3& bary) {
			EXPECT_GE(bary.x(), 0.0);
			EXPECT_GE(bary.y(), 0.0);
			EXPECT_GE(bary.z(), 0.0);
			EXPECT_NEAR(bary.x() + bary.y() + bary.z(), 1.0, 1e-9);
			s.insert({x, y});
		};
	};
	RasterizeTriangleBary<double>(a, b, c, 8, 8, collect(centre), false, 0,
	                              std::numeric_limits<int>::max(), /*conservative=*/false);
	RasterizeTriangleBary<double>(a, b, c, 8, 8, collect(conservative), false, 0,
	                              std::numeric_limits<int>::max(), /*conservative=*/true);

	ASSERT_FALSE(centre.empty());
	for (const auto& p : centre)
		EXPECT_TRUE(conservative.count(p)); // superset
	EXPECT_GT(conservative.size(), centre.size()); // strictly more boundary texels
}

// The frontier-based in-place Dilate must be byte-identical to the old full-scan
// double-buffer implementation over many passes and a scattered mask.
TEST(TextureBakeTest, DilateMatchesReferenceByteForByte)
{
	const int rows = 37, cols = 53;
	uint32_t seed = 0x1234567u;
	auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return seed; };

	Image3u img(rows, cols), imgRef(rows, cols);
	cv::Mat_<uint8_t> mask(rows, cols), maskRef(rows, cols);
	for (int r = 0; r < rows; ++r)
		for (int c = 0; c < cols; ++c) {
			const Pixel p(static_cast<uint8_t>(rnd() % 256), static_cast<uint8_t>(rnd() % 256),
			              static_cast<uint8_t>(rnd() % 256));
			// ~30% valid seeds, scattered, so several dilation passes are needed.
			const uint8_t m = (rnd() % 100 < 30) ? 255 : 0;
			img(r, c) = p;
			imgRef(r, c) = p;
			mask(r, c) = m;
			maskRef(r, c) = m;
		}

	halfmesh::Dilate(img, mask, /*iterations=*/6, /*halfSize=*/1);
	DilateReference(imgRef, maskRef, 6, 1);

	cv::Mat diff, mdiff;
	cv::absdiff(img, imgRef, diff);
	cv::absdiff(mask, maskRef, mdiff);
	EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
	EXPECT_EQ(cv::countNonZero(mdiff), 0);
}

// Item A: BakeAtlas must resolve chart-boundary texels (centre OUTSIDE the chart,
// triangle overlaps within half a pixel) via the conservative clamped-UV sample,
// not leave them to gutter dilation. Identity self-bake of a single triangle over
// a linear gradient: bilinear interpolation of the linear ramp img(r,c)=(c*10,r*10,50)
// is exact in the interior, and the clamped bary is a convex combination of the
// triangle corners, so the boundary texel's clamped-UV colour is ANALYTIC
// (u*10, v*10, 50) — computed here independently of the engine's sample path.
TEST(TextureBakeTest, ConservativeBoundaryTexelGetsClampedUVColor)
{
	using halfmesh::RasterizeTriangleBary;
	using halfmesh::Point2;
	using halfmesh::Vector3;

	const int size = 16;
	// Single-triangle chart in absolute pixel UVs; edges bite mid-texel so the
	// half-pixel conservative expansion adds a boundary ring inside the image.
	const Mesh::TexCoord t0(2.5f, 2.5f), t1(12.5f, 2.5f), t2(2.5f, 12.5f);
	const Point2 a(t0.x(), t0.y()), b(t1.x(), t1.y()), c(t2.x(), t2.y());

	// Enumerate centre-inside vs conservative coverage exactly as BakeAtlas will,
	// and capture the clamped-renormalized bary handed back for one conservative-
	// ONLY texel (centre outside the chart) — the reference the engine must match.
	std::set<std::pair<int, int>> centre;
	RasterizeTriangleBary<double>(
	    a, b, c, size, size, [&](int x, int y, const Vector3&) { centre.insert({x, y}); },
	    false, 0, std::numeric_limits<int>::max(), /*conservative=*/false);

	int tx = -1, ty = -1;
	Vector3 clampedBary = Vector3::Zero();
	RasterizeTriangleBary<double>(
	    a, b, c, size, size,
	    [&](int x, int y, const Vector3& bary) {
		    if (tx < 0 && !centre.count({x, y})) {
			    tx = x;
			    ty = y;
			    clampedBary = bary;
		    }
	    },
	    false, 0, std::numeric_limits<int>::max(), /*conservative=*/true);
	ASSERT_GE(tx, 0) << "expected at least one conservative-only boundary texel";

	// Analytic reference: clamped bary -> source UV (inside the chart) -> gradient.
	const double u = clampedBary[0] * t0.x() + clampedBary[1] * t1.x() + clampedBary[2] * t2.x();
	const double v = clampedBary[0] * t0.y() + clampedBary[1] * t1.y() + clampedBary[2] * t2.y();
	const int ex = static_cast<int>(std::lround(u * 10.0));
	const int ey = static_cast<int>(std::lround(v * 10.0));

	Mesh m;
	m.vertices = {Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0), Mesh::Vertex(0, 1, 0)};
	m.faces = {Mesh::Face(0, 1, 2)};
	m.faceTexcoords = {t0, t1, t2};
	SameUVResolver resolver(m.faceTexcoords, m.faceTexblobs, {Eigen::Vector2i(size, size)});
	BakeParams params;
	params.resolution = size;
	params.padding = 0; // no gutter dilation: conservative resolve is the ONLY colour source

	BakeAtlas(m, {MakeGradient(size)}, resolver, params);
	const Pixel& got = m.texturesDiffuse[0](ty, tx);
	// tol=2/255 absorbs uint8 StoreCast rounding of the two bilinear taps; the ramp
	// is otherwise reproduced exactly in the interior. Pre-fix the texel is (0,0,0).
	EXPECT_NEAR(static_cast<int>(got.x()), ex, 2) << "boundary texel (" << tx << "," << ty << ")";
	EXPECT_NEAR(static_cast<int>(got.y()), ey, 2);
	EXPECT_EQ(static_cast<int>(got.z()), 50);
}

// Many small disjoint quads on a coarse grid with fractional edges and inter-chart
// gaps, sized to trigger the parallel (multi-band) bake. Fractional origins make
// each quad's edges bite mid-texel, so the conservative pass produces a real
// boundary ring at internal chart seams; the row spacing (14) is deliberately not a
// multiple of the band height (128/8 = 16), so several quads straddle band
// boundaries and their conservative texels are resolved across bands.
Mesh MakeScatteredCharts(int page)
{
	Mesh m;
	const int cols = 8, rows = 9; // 72 quads => 144 faces (>=64 => parallel path)
	const double spacing = 14.0, quad = 12.0; // ~2-texel gaps between charts
	for (int j = 0; j < rows; ++j)
		for (int i = 0; i < cols; ++i) {
			const double ox = i * spacing + 0.7; // fractional => mid-texel edges
			const double oy = j * spacing + 0.3;
			const float x0 = static_cast<float>(ox), y0 = static_cast<float>(oy);
			const float x1 = static_cast<float>(ox + quad), y1 = static_cast<float>(oy + quad);
			const auto base = static_cast<Mesh::VIndex>(m.vertices.size());
			m.vertices.emplace_back(x0, y0, 0);
			m.vertices.emplace_back(x1, y0, 0);
			m.vertices.emplace_back(x1, y1, 0);
			m.vertices.emplace_back(x0, y1, 0);
			m.faces.emplace_back(base, base + 1, base + 2);
			m.faces.emplace_back(base, base + 2, base + 3);
			const Mesh::TexCoord uv0(x0, y0), uv1(x1, y0), uv2(x1, y1), uv3(x0, y1);
			m.faceTexcoords.push_back(uv0);
			m.faceTexcoords.push_back(uv1);
			m.faceTexcoords.push_back(uv2);
			m.faceTexcoords.push_back(uv0);
			m.faceTexcoords.push_back(uv2);
			m.faceTexcoords.push_back(uv3);
		}
	m.texturesDiffuse = {MakeGradient(page)};
	return m;
}

// Parallel baking WITH conservative boundary resolve must stay byte-identical to
// serial. The centre-priority tie rule + fixed face-order winner make each texel a
// pure function of the faces that cover it, never of the row-band decomposition, so
// thread count cannot change a byte. Extends ParallelMatchesSerial to the
// conservative texels the wiring adds (and proves they are actually produced).
TEST(TextureBakeTest, ConservativeParallelMatchesSerial)
{
	using halfmesh::RasterizeTriangleBary;
	using halfmesh::Point2;
	using halfmesh::Vector3;

	const int size = 128;
	Mesh m = MakeScatteredCharts(size);
	const Image3u src = MakeGradient(size);
	SameUVResolver resolver(m.faceTexcoords, m.faceTexblobs, {Eigen::Vector2i(size, size)});

	BakeParams base;
	base.resolution = size;
	base.padding = 0; // no dilation: filled-outside-centre proves the conservative pass fired

	Mesh serial = m;
	Mesh parallel = m;
	BakeParams p1 = base;
	p1.numThreads = 1;
	BakeParams pN = base;
	pN.numThreads = 8;

	BakeAtlas(serial, {src}, resolver, p1);
	BakeAtlas(parallel, {src}, resolver, pN);

	cv::Mat diff;
	cv::absdiff(serial.texturesDiffuse[0], parallel.texturesDiffuse[0], diff);
	ASSERT_EQ(cv::countNonZero(diff.reshape(1)), 0) << "serial vs parallel bake differ";

	// Non-vacuous: count filled texels whose centre lies outside EVERY chart. With
	// padding=0 and no thin faces these are exactly the conservative boundary texels
	// (the source blue is 50 everywhere, so any resolved texel is non-black).
	std::set<std::pair<int, int>> centre;
	for (size_t f = 0; f < m.faces.size(); ++f) {
		const Point2 a(m.faceTexcoords[f * 3 + 0].x(), m.faceTexcoords[f * 3 + 0].y());
		const Point2 b(m.faceTexcoords[f * 3 + 1].x(), m.faceTexcoords[f * 3 + 1].y());
		const Point2 c(m.faceTexcoords[f * 3 + 2].x(), m.faceTexcoords[f * 3 + 2].y());
		RasterizeTriangleBary<double>(
		    a, b, c, size, size, [&](int x, int y, const Vector3&) { centre.insert({x, y}); },
		    false, 0, std::numeric_limits<int>::max(), /*conservative=*/false);
	}
	int conservativeFilled = 0;
	const Image3u& atlas = serial.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int cc = 0; cc < size; ++cc) {
			const Pixel& p = atlas(r, cc);
			if ((p.x() || p.y() || p.z()) && !centre.count({cc, r}))
				++conservativeFilled;
		}
	EXPECT_GT(conservativeFilled, 0) << "conservative boundary ring was not resolved";
}

// Real textured mesh through the full GenerateAtlas -> ApplyPackedLayout ->
// BakeAtlas composition (via RebakeTexture), forced multi-page so the per-face
// page assignment and the normalized->pixel UV scale — glue with no type-level
// protection — run on genuine multi-page GenerateAtlas output. The source is
// single-texture (~16.78M texels), so DecideLayout alone would stay single-page
// (ideal side ~4523 < maxResolution); resolution=2048 + multiPage forces the
// explicit branch, and RebakeTexture preserves source density in that mode, so
// the fixed texel budget partitions into ~5 pages. Expected runtime ~10-30s in
// Release (GLB load + 300k-face atlas + ~16.8M-texel raycast bake); the
// fixture is gitignored, so this skips in CI. Exact texel values are NOT
// asserted: RebakeTexture forces Correspondence::Nearest, and only a constant
// channel survives raycast resampling exactly — a real texture has none.
TEST(TextureBakeTest, RebakeTruckMultiPageAssignsPagesAndFillsThem)
{
	Mesh source;
	if (!std::filesystem::exists(TruckMeshPath()) || !source.Load(TruckMeshPath()))
		GTEST_SKIP() << "truck_textured.glb not available";
	ASSERT_FALSE(source.texturesDiffuse.empty());
	ASSERT_FALSE(source.faceTexcoords.empty());

	Mesh target = source; // same geometry
	target.faceTexcoords.clear(); // rebake regenerates the layout
	target.faceTexblobs.clear();
	target.texturesDiffuse.clear();

	BakeParams params;
	params.resolution = 2048; // ~5 pages via DecideLayout's explicit branch
	params.multiPage = true;
	params.padding = 0;

	const BakeResult res = RebakeTexture(source, target, params);

	// The multi-page branch was actually reached — everything below proves
	// nothing if the atlas landed on a single page.
	ASSERT_GT(res.numPages, 1u) << "expected forced multi-page layout";

	// One baked image per page.
	ASSERT_EQ(target.texturesDiffuse.size(), res.numPages);

	// Per-face page assignment exists and stays in range.
	ASSERT_EQ(target.faceTexblobs.size(), target.faces.size());
	size_t badBlob = 0;
	for (const Mesh::FIndex b : target.faceTexblobs)
		if (b >= res.numPages)
			++badBlob;
	EXPECT_EQ(badBlob, 0u) << "face_texblobs out of page range";

	// ApplyPackedLayout's normalized->pixel scale step: every UV must land in
	// [0,width] x [0,height] absolute pixel space.
	const float W = static_cast<float>(res.width);
	const float H = static_cast<float>(res.height);
	size_t out_of_range = 0;
	for (const Mesh::TexCoord& uv : target.faceTexcoords)
		if (!(uv.x() >= 0.f && uv.x() <= W && uv.y() >= 0.f && uv.y() <= H))
			++out_of_range;
	EXPECT_EQ(out_of_range, 0u) << "faceTexcoords outside pixel bounds";

	// EVERY page has baked texels — the assertion that pins the per-face page
	// assignment (faceTexblobs[f] = chartPage[faceChart[f]]). The classic
	// failure is every face landing on blob 0, leaving later pages empty —
	// which an aggregate filled>0 check would happily pass.
	for (unsigned p = 0; p < res.numPages; ++p)
		EXPECT_GT(cv::countNonZero(target.texturesDiffuse[p].reshape(1)), 0)
		    << "page " << p << " baked no texels";
}

// Normalized source UVs tiled BEYOND the old fixed threshold (integer offset
// +2 -> |uv| up to ~3) must still classify as normalized: the fixed 2.0 cutoff
// misread genuinely tiled assets as pixel-space, skipping the wrap+unnormalize
// and sampling garbage. Under the
// per-blob dimension-aware classifier (threshold max(2, min_dim/2) = 8 for a
// 16px texture) the tiled coords stay normalized; GL_REPEAT wrapping recovers
// the exact texel, so the bake must reproduce the source exactly.
TEST(TextureBakeTest, TiledBeyondTwoNormalizedSourceReproduces)
{
	const int size = 16;
	Mesh m = MakeQuad(size);
	const Image3u src = MakeGradient(size);

	std::vector<Mesh::TexCoord> normUv;
	for (const Mesh::TexCoord& t : m.faceTexcoords)
		normUv.emplace_back((t.x() + 0.5f) / size + 2.0f, (t.y() + 0.5f) / size + 2.0f);
	const std::vector<Eigen::Vector2i> dims = {Eigen::Vector2i(size, size)};

	SameUVResolver resolver(normUv, m.faceTexblobs, dims);
	BakeParams params;
	params.resolution = size;
	params.padding = 0;

	BakeAtlas(m, {src}, resolver, params);
	const Image3u& atlas = m.texturesDiffuse[0];
	for (int r = 0; r < size; ++r)
		for (int c = 0; c < size; ++c)
			EXPECT_EQ(atlas(r, c), src(r, c)) << "at (" << r << "," << c << ")";
}

// Two blobs in DIFFERENT UV regimes: blob 0 normalized on an 8px texture,
// blob 1 absolute-pixel on a 256px texture. The old single global flag was
// dominated by blob 1's magnitudes (~256 -> "pixel"), so blob 0's [0,1]
// coords were read as pixel positions and its faces sampled the top-left
// texel region instead of the whole image. Per-blob classification must
// reproduce blob 0's source exactly
// while blob 1 keeps its correct pixel-space read.
TEST(TextureBakeTest, MixedRegimeBlobsClassifiedPerBlob)
{
	Mesh m;
	m.vertices = {
	    Mesh::Vertex(0, 0, 0), Mesh::Vertex(1, 0, 0), Mesh::Vertex(1, 1, 0), Mesh::Vertex(0, 1, 0),
	    Mesh::Vertex(2, 0, 0), Mesh::Vertex(3, 0, 0), Mesh::Vertex(3, 1, 0), Mesh::Vertex(2, 1, 0)};
	m.faces = {Mesh::Face(0, 1, 2), Mesh::Face(0, 2, 3),
	           Mesh::Face(4, 5, 6), Mesh::Face(4, 6, 7)};
	m.faceTexblobs = {0, 0, 1, 1};
	// target atlas layout (absolute pixels, 16x16 page): quad 0 -> [0,8]^2,
	// quad 1 -> [8,16]^2
	m.faceTexcoords = {
	    Mesh::TexCoord(0, 0), Mesh::TexCoord(8, 0), Mesh::TexCoord(8, 8),
	    Mesh::TexCoord(0, 0), Mesh::TexCoord(8, 8), Mesh::TexCoord(0, 8),
	    Mesh::TexCoord(8, 8), Mesh::TexCoord(16, 8), Mesh::TexCoord(16, 16),
	    Mesh::TexCoord(8, 8), Mesh::TexCoord(16, 16), Mesh::TexCoord(8, 16)};

	const Image3u srcA = MakeGradient(8);
	Image3u srcB(256, 256);
	srcB.setTo(cv::Scalar(20, 40, 60));

	// snapshot: blob-0 corners normalized (centre convention) on the 8px
	// texture; blob-1 corners absolute pixels spanning the 256px texture
	std::vector<Mesh::TexCoord> snap;
	for (size_t i = 0; i < 6; ++i)
		snap.emplace_back((m.faceTexcoords[i].x() + 0.5f) / 8.f,
		                  (m.faceTexcoords[i].y() + 0.5f) / 8.f);
	for (size_t i = 6; i < 12; ++i)
		snap.emplace_back((m.faceTexcoords[i].x() - 8.f) * 32.f,
		                  (m.faceTexcoords[i].y() - 8.f) * 32.f);

	SameUVResolver resolver(snap, m.faceTexblobs,
	                        {Eigen::Vector2i(8, 8), Eigen::Vector2i(256, 256)});
	BakeParams params;
	params.resolution = 16;
	params.padding = 0;

	BakeAtlas(m, {srcA, srcB}, resolver, params);
	// faceTexblobs doubles as the TARGET's per-face output page (BakeAtlas: one
	// page per faceTexblobs entry), so blob 0's faces land on page 0 and blob 1's
	// on page 1 -- each checked on its own page.
	const Image3u& atlas0 = m.texturesDiffuse[0];
	const Image3u& atlas1 = m.texturesDiffuse[1];
	// blob-0 region must reproduce its gradient exactly
	for (int r = 0; r < 8; ++r)
		for (int c = 0; c < 8; ++c)
			EXPECT_EQ(atlas0(r, c), srcA(r, c)) << "at (" << r << "," << c << ")";
	// blob-1 region stays correctly pixel-space (solid source color)
	EXPECT_EQ(atlas1(12, 12), Pixel(20, 40, 60));
}

// A rogue (out-of-range) faceTexblobs entry must be IGNORED by the per-blob
// UV-regime classifier, not folded into blob 0's max: one bad face's huge
// pixel-space UVs would otherwise flip a genuinely-normalized blob 0 to
// pixel-space (2026-07-18 final review, follow-up 2).
TEST(TextureBakeTest, ClassifierSkipsOutOfRangeBlobIndices)
{
	// face 0 -> blob 0, normalized-range UVs; face 1 -> rogue blob 7, huge UVs
	const std::vector<Mesh::TexCoord> uv = {
	    Mesh::TexCoord(0.1f, 0.2f), Mesh::TexCoord(0.9f, 0.1f), Mesh::TexCoord(0.5f, 0.8f),
	    Mesh::TexCoord(300, 300), Mesh::TexCoord(400, 300), Mesh::TexCoord(350, 400)};
	const std::vector<Mesh::TexIndex> blobs = {0, 7};
	const std::vector<Eigen::Vector2i> dims = {Eigen::Vector2i(256, 256)};
	const std::vector<bool> norm = UVBlobsAreNormalized(uv, blobs, dims);
	ASSERT_EQ(norm.size(), 1u);
	// fold-to-0 pollutes mx[0] with 400 (>= threshold 128) -> false; skip keeps
	// mx[0] at 0.9 -> true
	EXPECT_TRUE(norm[0]);
}

// A per-face blob list of the WRONG SIZE (not one id per face) carries no
// usable face->blob mapping and previously caused an out-of-bounds read of
// faceBlobs[f]. It must degrade to the single-blob convention (all faces ->
// blob 0), same as an empty list. NOTE: the pre-fix behavior here is undefined
// (OOB read), so this test pins the new defined degrade rather than a
// deterministic RED.
TEST(TextureBakeTest, ClassifierGuardsMismatchedBlobListSize)
{
	// two faces, but only ONE blob entry
	const std::vector<Mesh::TexCoord> uv = {
	    Mesh::TexCoord(0.1f, 0.2f), Mesh::TexCoord(0.9f, 0.1f), Mesh::TexCoord(0.5f, 0.8f),
	    Mesh::TexCoord(0.2f, 0.3f), Mesh::TexCoord(0.8f, 0.2f), Mesh::TexCoord(0.4f, 0.7f)};
	const std::vector<Mesh::TexIndex> blobs = {1};
	const std::vector<Eigen::Vector2i> dims = {Eigen::Vector2i(64, 64),
	                                           Eigen::Vector2i(64, 64)};
	const std::vector<bool> norm = UVBlobsAreNormalized(uv, blobs, dims);
	ASSERT_EQ(norm.size(), 2u);
	EXPECT_TRUE(norm[0]); // all faces fold to blob 0: mx 0.9 < threshold 32
	EXPECT_TRUE(norm[1]); // nothing accumulated: mx 0 < threshold 32
}

// Fail-fast preconditions (2026-08 review): input without per-corner UVs or
// textures must return a default (numPages == 0) result in every build mode —
// the downstream size invariants are Debug-only ASSERTs, so Release used to
// read out of bounds instead of diagnosing.
TEST(TextureBakeTest, DefragUntexturedMeshFailsFast)
{
	BakeParams bp;
	bp.resolution = 64;

	Mesh noTex = MakeMultiChartTextured(2, 16);
	noTex.texturesDiffuse.clear(); // UVs but no texture
	EXPECT_EQ(DefragmentTexture(noTex, bp).numPages, 0u);

	Mesh noUv = MakeMultiChartTextured(2, 16);
	noUv.faceTexcoords.clear(); // texture but no UVs
	EXPECT_EQ(DefragmentTexture(noUv, bp).numPages, 0u);
}

TEST(TextureBakeTest, RebakeUntexturedSourceFailsFast)
{
	BakeParams bp;
	bp.resolution = 64;
	Mesh source = MakeMultiChartTextured(2, 16);
	source.texturesDiffuse.clear();
	Mesh target = MakeMultiChartTextured(2, 16);
	EXPECT_EQ(RebakeTexture(source, target, bp).numPages, 0u);
}

// End-to-end rebake quality floor (2026-08 review: the suite had no fidelity
// bound at all, which is how the chart-foldover bug shipped undetected).
// Atlas a UVSphere, paint its texture with a signal that is smooth ON THE
// SURFACE (linear in 3D position — reproducible exactly by a well-registered
// bake, unlike a texture-space gradient which is discontinuous at UV seams),
// rebake onto the same geometry, and demand the result reproduce the analytic
// signal to >= 40 dB (measured headroom: a healthy pipeline lands well above;
// gross misregistration or a folded chart drops tens of dB).
TEST(TextureBakeTest, RebakeFidelityFloorSurfaceSmooth)
{
	Mesh src = hmtest::corpus::UVSphere(24, 32);
	halfmesh::ParametrizeParams pp;
	halfmesh::AtlasParams ap;
	ap.resolution = 512;
	ap.padding = 2;
	const halfmesh::AtlasResult atlas = halfmesh::GenerateAtlas(src, pp, ap);
	ASSERT_GE(atlas.numPages, 1u);
	ASSERT_EQ(src.faceTexcoords.size(), src.faces.size() * 3);

	// paint color = linear function of 3D position at each texel a face covers
	// (texel centres sit at integer UV coords — Util/Raster.h convention)
	const auto box = src.ComputeAABBox();
	const Eigen::Vector3d bmin = box.min().cast<double>();
	const Eigen::Vector3d ext = (box.max() - box.min()).cast<double>().cwiseMax(1e-12);
	const auto analytic = [&](const Eigen::Vector3d& p) {
		const Eigen::Vector3d t = (p - bmin).cwiseQuotient(ext) * 255.0;
		return Eigen::Vector3d(std::clamp(t[0], 0.0, 255.0), std::clamp(t[1], 0.0, 255.0),
		                       std::clamp(t[2], 0.0, 255.0));
	};
	// GenerateAtlas returns NORMALIZED [0,1] UVs (ApplyPackedLayout inside
	// RebakeTexture is what converts to absolute pixels) — paint and read the
	// source through the normalized convention (u*size - 0.5, texel centres at
	// integer coords; same mapping the bake resolvers use).
	Image3u tex(static_cast<int>(atlas.height), static_cast<int>(atlas.width));
	tex.setTo(cv::Scalar(0, 0, 0));
	const auto srcPx = [&](size_t corner) {
		return Eigen::Vector2d(src.faceTexcoords[corner].x() * tex.cols - 0.5,
		                       src.faceTexcoords[corner].y() * tex.rows - 0.5);
	};
	for (size_t f = 0; f < src.faces.size(); ++f) {
		const Eigen::Vector2d a = srcPx(f * 3 + 0);
		const Eigen::Vector2d b = srcPx(f * 3 + 1);
		const Eigen::Vector2d c = srcPx(f * 3 + 2);
		const Eigen::Vector3d p0 = src.vertices[src.faces[f][0]].cast<double>();
		const Eigen::Vector3d p1 = src.vertices[src.faces[f][1]].cast<double>();
		const Eigen::Vector3d p2 = src.vertices[src.faces[f][2]].cast<double>();
		halfmesh::RasterizeTriangleBary<double>(
		    a, b, c, tex.cols, tex.rows,
		    [&](int x, int y, const Eigen::Vector3d& w) {
			    const Eigen::Vector3d col = analytic(w[0] * p0 + w[1] * p1 + w[2] * p2);
			    tex(y, x) = Pixel(static_cast<uint8_t>(col[0]), static_cast<uint8_t>(col[1]),
			                      static_cast<uint8_t>(col[2]));
		    },
		    /*cull=*/false, 0, std::numeric_limits<int>::max(), /*conservative=*/true);
	}
	src.texturesDiffuse = {tex};

	Mesh target;
	target.vertices = src.vertices;
	target.faces = src.faces;
	BakeParams bp;
	bp.resolution = 512;
	// Nearest correspondence: source and target surfaces coincide (identity
	// rebake), exactly the case TextureBake.h documents Nearest for — the
	// SameUV default would upgrade to Raycast, whose normal-direction rays are
	// the wrong tool on coincident surfaces.
	bp.correspondence = Correspondence::Nearest;
	const BakeResult res = RebakeTexture(src, target, bp);
	ASSERT_GE(res.numPages, 1u);
	ASSERT_EQ(target.faceTexcoords.size(), target.faces.size() * 3);
	ASSERT_FALSE(target.texturesDiffuse.empty());

	// deterministic surface sampling: face centroid + corner-biased points,
	// nearest-texel readback in the target atlas, error vs the ANALYTIC signal
	double sse = 0.0;
	long ns = 0;
	const double bw[4][3] = {{1 / 3.0, 1 / 3.0, 1 / 3.0},
	                         {0.6, 0.2, 0.2},
	                         {0.2, 0.6, 0.2},
	                         {0.2, 0.2, 0.6}};
	for (size_t f = 0; f < target.faces.size(); ++f) {
		const unsigned blob = static_cast<unsigned>(target.FTexblob(static_cast<Mesh::FIndex>(f)));
		ASSERT_LT(blob, target.texturesDiffuse.size());
		const Image3u& img = target.texturesDiffuse[blob];
		for (const auto& w : bw) {
			Eigen::Vector2d uv = Eigen::Vector2d::Zero();
			Eigen::Vector3d p = Eigen::Vector3d::Zero();
			for (int k = 0; k < 3; ++k) {
				uv += w[k] * Eigen::Vector2d(target.faceTexcoords[f * 3 + k].x(), target.faceTexcoords[f * 3 + k].y());
				p += w[k] * target.vertices[target.faces[f][k]].cast<double>();
			}
			const int x = std::clamp(static_cast<int>(std::lround(uv.x())), 0, img.cols - 1);
			const int y = std::clamp(static_cast<int>(std::lround(uv.y())), 0, img.rows - 1);
			const Pixel& px = img(y, x);
			const Eigen::Vector3d ref = analytic(p);
			for (int ch = 0; ch < 3; ++ch) {
				const double d = static_cast<double>(px[ch]) - ref[ch];
				sse += d * d;
			}
			++ns;
		}
	}
	const double mse = sse / (3.0 * static_cast<double>(ns));
	const double psnr = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
	RecordProperty("psnr_db", static_cast<int>(psnr));
	EXPECT_GE(psnr, 40.0) << "MSE=" << mse << " over " << ns << " samples";
}

// Defrag keeps every source UV patch as its own chart, and the skyline repack
// is superlinear in patch count (measured: a 128,748-patch CAD atlas spun for
// >40 minutes). Atlases beyond the documented 65,536-patch cap must be refused
// fast — the guard fires right after patch counting.
TEST(TextureBakeTest, DefragOverFragmentedAtlasRefused)
{
	Mesh m = MakeMultiChartTextured(65537, 4);
	BakeParams bp;
	bp.resolution = 256;
	EXPECT_EQ(DefragmentTexture(m, bp).numPages, 0u);
}

// The patch-count guard is a parameter: a small limit refuses a mildly
// fragmented atlas, and 0 disables the guard entirely for the same input.
TEST(TextureBakeTest, DefragPatchLimitIsTunable)
{
	Mesh m = MakeMultiChartTextured(20, 4);
	BakeParams bp;
	bp.resolution = 256;
	bp.maxDefragPatches = 16;
	EXPECT_EQ(DefragmentTexture(m, bp).numPages, 0u);

	Mesh m2 = MakeMultiChartTextured(20, 4);
	bp.maxDefragPatches = 0; // unlimited
	const BakeResult res = DefragmentTexture(m2, bp);
	EXPECT_GE(res.numPages, 1u);
}

// A solid-color page the size of `m`'s atlas, so any texel the bake did not
// touch is recognizable by its sentinel value afterwards.
void SetSentinelTexture(Mesh& m, int size, uint8_t v)
{
	Image3u tex(size, size);
	tex.setTo(cv::Scalar(v, v, v));
	m.texturesDiffuse = {tex};
}

// Count texels carrying the source's constant blue (z == 50) and, separately,
// the sentinel the target went in with.
void CountBakedAndSentinel(const Mesh& m, uint8_t sentinel, int& baked, int& kept)
{
	baked = kept = 0;
	for (const Image3u& atlas : m.texturesDiffuse)
		for (int r = 0; r < atlas.rows; ++r)
			for (int c = 0; c < atlas.cols; ++c) {
				const Pixel& p = atlas(r, c);
				if (p.z() == 50)
					++baked;
				else if (p.x() == sentinel && p.y() == sentinel && p.z() == sentinel)
					++kept;
			}
}

// BakeOntoAtlas is RebakeTexture's counterpart for a target whose UV layout must
// survive: the UVs must come back byte-identical, and the texels under them must
// carry the source's color rather than the target's own.
TEST(TextureBakeTest, BakeOntoAtlasKeepsTargetUVsAndSamplesSource)
{
	const Mesh source = MakeGridTextured(4, 64);

	Mesh target = MakeGridTextured(4, 64);
	SetSentinelTexture(target, 64, 9);
	const std::vector<Mesh::TexCoord> uvBefore = target.faceTexcoords;

	BakeParams params;
	params.resolution = 64;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;

	const BakeResult res = BakeOntoAtlas(source, target, params);

	EXPECT_EQ(res.numPages, 1u);
	EXPECT_EQ(target.faceTexcoords, uvBefore) << "the target's authored UVs must survive";

	int baked = 0, kept = 0;
	CountBakedAndSentinel(target, 9, baked, kept);
	EXPECT_GT(baked, 0);
	EXPECT_EQ(kept, 0) << "an unmasked bake covers the whole atlas";
}

// Under a face mask the bake is an in-place edit: the selected faces' texels get
// the source color, and every other texel keeps exactly what the target carried
// (the page is seeded from target.texturesDiffuse instead of black).
TEST(TextureBakeTest, BakeOntoAtlasMaskLeavesUnselectedTexelsAlone)
{
	const Mesh source = MakeGridTextured(4, 64);

	Mesh target = MakeGridTextured(4, 64);
	SetSentinelTexture(target, 64, 9);

	std::vector<bool> mask(target.faces.size(), false);
	mask[0] = true;

	BakeParams params;
	params.resolution = 64;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;
	params.faceMask = &mask;

	const BakeResult res = BakeOntoAtlas(source, target, params);

	EXPECT_EQ(res.numPages, 1u);
	int baked = 0, kept = 0;
	CountBakedAndSentinel(target, 9, baked, kept);
	EXPECT_GT(baked, 0) << "the selected face must have been rasterized";
	EXPECT_GT(kept, 0) << "the other 31 faces' texels must be untouched";
	EXPECT_LT(baked, 64 * 64 / 4) << "one of 32 faces cannot cover a quarter of the atlas";
}

// A mask that does not match the target's face count is a caller bug that would
// silently bake the wrong faces; it is refused and the full bake runs instead.
TEST(TextureBakeTest, BakeOntoAtlasWrongSizedMaskIsIgnored)
{
	const Mesh source = MakeGridTextured(4, 64);

	Mesh target = MakeGridTextured(4, 64);
	SetSentinelTexture(target, 64, 9);
	std::vector<bool> mask(target.faces.size() + 1, false);

	BakeParams params;
	params.resolution = 64;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;
	params.faceMask = &mask;

	EXPECT_EQ(BakeOntoAtlas(source, target, params).numPages, 1u);
	int baked = 0, kept = 0;
	CountBakedAndSentinel(target, 9, baked, kept);
	EXPECT_GT(baked, 0);
	EXPECT_EQ(kept, 0);
}

// The target's own texture size is the only evidence of the pixel space its UVs
// live in, so baking at a different resolution is refused rather than writing to
// texels the UVs never addressed.
TEST(TextureBakeTest, BakeOntoAtlasRefusesResolutionMismatch)
{
	const Mesh source = MakeGridTextured(4, 64);

	Mesh target = MakeGridTextured(4, 64); // UVs and texture in 64-pixel space
	const std::vector<Mesh::TexCoord> uvBefore = target.faceTexcoords;

	BakeParams params;
	params.resolution = 128; // ... but asked for a 128 atlas
	params.padding = 0;

	EXPECT_EQ(BakeOntoAtlas(source, target, params).numPages, 0u);
	EXPECT_EQ(target.faceTexcoords, uvBefore) << "a refused bake must not touch the target";
	EXPECT_EQ(target.texturesDiffuse.front().rows, 64);
}

// RebakeTexture generates a fresh layout, in which the unselected texels of the
// old one mean nothing -- so it ignores faceMask instead of leaving most of the
// new atlas black (or seeded from a stale page that happened to match in size).
TEST(TextureBakeTest, RebakeTextureIgnoresFaceMask)
{
	const Mesh source = MakeGridTextured(4, 64);

	BakeParams params;
	params.resolution = 64;
	params.padding = 0;
	params.correspondence = Correspondence::Nearest;

	Mesh plain = MakeGridTextured(4, 64);
	RebakeTexture(source, plain, params);

	Mesh masked = MakeGridTextured(4, 64);
	std::vector<bool> mask(masked.faces.size(), false);
	mask[0] = true;
	params.faceMask = &mask;
	RebakeTexture(source, masked, params);

	ASSERT_EQ(plain.texturesDiffuse.size(), masked.texturesDiffuse.size());
	for (size_t i = 0; i < plain.texturesDiffuse.size(); ++i) {
		const Image3u& a = plain.texturesDiffuse[i];
		const Image3u& b = masked.texturesDiffuse[i];
		ASSERT_EQ(a.size(), b.size());
		int diff = 0;
		for (int r = 0; r < a.rows; ++r)
			for (int c = 0; c < a.cols; ++c)
				if (a(r, c) != b(r, c))
					++diff;
		EXPECT_EQ(diff, 0) << "page " << i;
	}
}
