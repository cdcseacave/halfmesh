/*
* TextureBakeTool.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// examples/texturebake — rebake / defragment a textured mesh, and measure
// texture-transfer fidelity. Exercises the public TextureBake API and doubles as
// a comparison harness against other implementations on real meshes.
//
// Usage:
//   texturebake rebake   --input X --output Y [--decimation 0.3] [--texture-size 4096]
//                        [--multi-page] [--max-resolution 8192] [--threads 0] [--cubic] [--measure]
//                        [--smooth | --surface-smooth]  (fidelity diagnostics, see below)
//   texturebake defrag   --input X --output Y [--texture-size 4096] [--multi-page] [--weld]
//                        [--max-resolution 8192] [--threads 0] [--measure]
//   texturebake fidelity --ref R --test T [--samples 200000]
//
// rebake generates a fresh atlas on a decimated copy of the input and bakes the
// full-res source onto it. defrag repacks the existing atlas. --texture-size 0
// auto-sizes the atlas from the mesh's texel budget; --multi-page preserves the
// source density across as many pages as needed. --measure prints the in-memory
// transfer fidelity (PSNR) right after a rebake.
//
// fidelity samples random surface points on --ref, reads their colour, finds the
// nearest surface point on --test and reads ITS colour, and reports PSNR — a
// layout-convention-robust measure of how well --test reproduces --ref.

#include <halfmesh/Mesh.h>
#include <halfmesh/TextureBake.h>
#include <halfmesh/TriangleKDTree.h>
#include <halfmesh/Util/Raster.h>
#include <halfmesh/Util/Sampler.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace halfmesh;
using Clock = std::chrono::steady_clock;

namespace {

double Seconds(Clock::time_point a, Clock::time_point b)
{
	return std::chrono::duration<double>(b - a).count();
}

std::string Arg(int argc, char** argv, const std::string& key, const std::string& def)
{
	for (int i = 0; i < argc - 1; ++i)
		if (key == argv[i])
			return argv[i + 1];
	return def;
}

bool Flag(int argc, char** argv, const std::string& key)
{
	for (int i = 0; i < argc; ++i)
		if (key == argv[i])
			return true;
	return false;
}

// Colour of textured mesh `m` at face f, barycentric weights (v0,v1,v2).
// `normalized` is the library's per-blob UV-regime classification
// (UVBlobsAreNormalized): entry FTexblob(f) says whether this face's UVs are
// normalized [0,1] or already absolute pixels.
Vector3 ColorAt(const Mesh& m, const std::vector<bool>& normalized, Mesh::FIndex f, const Vector3& bary)
{
	const Mesh::TexCoord* t = &m.faceTexcoords[static_cast<size_t>(f) * 3];
	const Eigen::Vector2d uv = t[0].cast<double>() * bary[0] + t[1].cast<double>() * bary[1] + t[2].cast<double>() * bary[2];
	const unsigned blob = static_cast<unsigned>(m.FTexblob(f));
	// Rogue blob id (no matching image): nothing to sample -- return black
	// rather than reading out of bounds (diagnostic tool, keep it defined).
	if (blob >= m.texturesDiffuse.size())
		return Vector3(0, 0, 0);
	const Image3u& img = m.texturesDiffuse[blob];
	// Integer pixel coordinates are texel CENTERS (Sampler.h), and the library
	// maps a normalized UV to pixel space as uv*size - 0.5 (FTexcoordsUnNormalize).
	// Omitting the -0.5 half-texel offset misregisters the sample by half a texel
	// (a texel-center UV would return a 50/50 blend of neighbours instead of the
	// exact texel).  No Y-flip: normalized UVs that reach here follow the no-flip
	// FTexcoordsNormalize convention (LoadPLY already undid the PLY V-flip).
	const bool norm = blob < normalized.size() && normalized[blob];
	const Point2 px = norm ? Point2(uv.x() * img.cols - 0.5, uv.y() * img.rows - 0.5)
	                       : Point2(uv.x(), uv.y());
	return SampleImage<LinearInterp<>>(img, px);
}

// Barycentric weights of p w.r.t. triangle (v0,v1,v2). weights[0]->v0, etc.
Vector3 Barycentric3D(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& p)
{
	const Vector3 e0 = v1 - v0, e1 = v2 - v0, e2 = p - v0;
	const double d00 = e0.dot(e0), d01 = e0.dot(e1), d11 = e1.dot(e1);
	const double d20 = e2.dot(e0), d21 = e2.dot(e1);
	const double denom = d00 * d11 - d01 * d01;
	if (denom == 0.0)
		return Vector3(1, 0, 0);
	const double inv = 1.0 / denom;
	const double b1 = (d11 * d20 - d01 * d21) * inv;
	const double b2 = (d00 * d21 - d01 * d20) * inv;
	return Vector3(1.0 - b1 - b2, b1, b2);
}

// PSNR of how well `test` reproduces `ref`'s textured appearance: for random
// area-weighted surface points on ref, compare ref's colour with the colour at
// the nearest point on test.
int DoFidelity(const Mesh& ref, const Mesh& test, int samples)
{
	if (!ref.HasTexture() || !test.HasTexture()) {
		std::cerr << "fidelity: both meshes must be textured\n";
		return EXIT_FAILURE;
	}
	// Per-blob UV regime via the library's shared classifier — the same rules
	// the bake itself uses (fixed threshold was the BAKE-1/2 defect class).
	const std::vector<bool> refN = UVBlobsAreNormalized(ref.faceTexcoords, ref.faceTexblobs, BlobDims(ref));
	const std::vector<bool> testN = UVBlobsAreNormalized(test.faceTexcoords, test.faceTexblobs, BlobDims(test));
	const TriangleKdTree kd(test);

	std::vector<double> cum(ref.faces.size() + 1, 0.0);
	for (size_t f = 0; f < ref.faces.size(); ++f)
		cum[f + 1] = cum[f] + 0.5 * ref.ComputeFaceDoubleArea(static_cast<Mesh::FIndex>(f));
	const double total = cum.back();

	std::mt19937 rng(12345);
	std::uniform_real_distribution<double> U(0.0, 1.0);
	double sse = 0.0;
	int n = 0;
	for (int s = 0; s < samples; ++s) {
		const double r = U(rng) * total;
		size_t f = std::upper_bound(cum.begin(), cum.end(), r) - cum.begin() - 1;
		f = std::min(f, ref.faces.size() - 1);
		double u = U(rng), v = U(rng);
		if (u + v > 1.0) {
			u = 1.0 - u;
			v = 1.0 - v;
		}
		const Vector3 bary(1.0 - u - v, u, v);
		const Mesh::Face& face = ref.faces[f];
		const Vector3 p = bary[0] * ref.vertices[face[0]].cast<double>() + bary[1] * ref.vertices[face[1]].cast<double>() + bary[2] * ref.vertices[face[2]].cast<double>();
		const Vector3 cref = ColorAt(ref, refN, static_cast<Mesh::FIndex>(f), bary);

		const auto nn = kd.NearestPoint(p.cast<float>());
		if (!nn.IsValid())
			continue;
		const Mesh::Face& tf = test.faces[nn.idxFace];
		const Vector3 tb = Barycentric3D(test.vertices[tf[0]].cast<double>(),
		                                 test.vertices[tf[1]].cast<double>(),
		                                 test.vertices[tf[2]].cast<double>(),
		                                 nn.nearest.cast<double>());
		const Vector3 ctest = ColorAt(test, testN, nn.idxFace, tb);
		sse += (cref - ctest).squaredNorm();
		++n;
	}
	const double mse = sse / (3.0 * std::max(1, n));
	const double psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
	std::cout << "fidelity: " << n << " samples, MSE=" << mse << ", PSNR=" << psnr << " dB\n";
	return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: texturebake <rebake|defrag|fidelity> ...\n";
		return EXIT_FAILURE;
	}
	const std::string cmd = argv[1];
	const unsigned tex = static_cast<unsigned>(std::stoul(Arg(argc, argv, "--texture-size", "4096")));
	const unsigned threads = static_cast<unsigned>(std::stoul(Arg(argc, argv, "--threads", "0")));
	const unsigned maxRes = static_cast<unsigned>(std::stoul(Arg(argc, argv, "--max-resolution", "8192")));

	if (cmd == "fidelity") {
		Mesh ref, test;
		if (!ref.Load(Arg(argc, argv, "--ref", "")) || !test.Load(Arg(argc, argv, "--test", ""))) {
			std::cerr << "fidelity: failed to load --ref/--test\n";
			return EXIT_FAILURE;
		}
		const int samples = std::stoi(Arg(argc, argv, "--samples", "200000"));
		return DoFidelity(ref, test, samples);
	}

	const std::string in = Arg(argc, argv, "--input", "");
	const std::string out = Arg(argc, argv, "--output", "");
	Mesh mesh;
	if (!mesh.Load(in)) {
		std::cerr << "failed to load '" << in << "'\n";
		return EXIT_FAILURE;
	}
	std::cout << "loaded " << mesh.vertices.size() << " verts, " << mesh.faces.size()
	          << " faces, textured=" << mesh.HasTexture() << "\n";

	// Diagnostic: replace the source texture(s) with a TEXTURE-SPACE gradient.
	// CAUTION reading its fidelity: a texture-space gradient is DISCONTINUOUS on
	// the surface at every source UV seam (measured on the truck fixture: 22.6%
	// of interior edges are seams, 21.7% of surface area within one source texel
	// of one), so no resampler can reproduce it and PSNR bottoms out in the
	// ~25-27 dB range even for a perfectly registered bake. It still exposes
	// gross misregistration; for a signal that isolates true registration /
	// injectivity error, use --surface-smooth below.
	if (Flag(argc, argv, "--smooth")) {
		for (Image3u& img : mesh.texturesDiffuse)
			for (int r = 0; r < img.rows; ++r)
				for (int c = 0; c < img.cols; ++c)
					img(r, c) = Pixel(static_cast<uint8_t>(c * 255 / std::max(1, img.cols - 1)),
					                  static_cast<uint8_t>(r * 255 / std::max(1, img.rows - 1)),
					                  128);
		std::cout << "(replaced source texture with texture-space gradient)\n";
	}

	// Diagnostic: paint the source texture(s) with a signal that is smooth ON
	// THE SURFACE (a linear function of 3D position). This survives rebake
	// resampling exactly, so residual infidelity is genuine registration /
	// chart-overlap error (a well-registered identity rebake measures 50+ dB
	// on scan meshes; the texture-space --smooth gradient cannot exceed ~27).
	if (Flag(argc, argv, "--surface-smooth")) {
		const auto box = mesh.ComputeAABBox();
		const Vector3 bmin = box.min().cast<real>(), bmax = box.max().cast<real>();
		const Vector3 ext = (bmax - bmin).cwiseMax(1e-12);
		std::vector<Eigen::Vector2i> blobDims;
		for (const Image3u& img : mesh.texturesDiffuse)
			blobDims.emplace_back(img.cols, img.rows);
		const std::vector<bool> normalized =
		    UVBlobsAreNormalized(mesh.faceTexcoords, mesh.faceTexblobs, blobDims);
		std::vector<cv::Mat_<uint8_t>> masks(mesh.texturesDiffuse.size());
		for (size_t b = 0; b < mesh.texturesDiffuse.size(); ++b) {
			mesh.texturesDiffuse[b].setTo(cv::Scalar(0, 0, 0));
			masks[b] = cv::Mat_<uint8_t>::zeros(mesh.texturesDiffuse[b].rows,
			                                    mesh.texturesDiffuse[b].cols);
		}
		for (size_t f = 0; f < mesh.faces.size(); ++f) {
			const unsigned b = static_cast<unsigned>(mesh.FTexblob(static_cast<Mesh::FIndex>(f)));
			if (b >= mesh.texturesDiffuse.size())
				continue;
			Image3u& img = mesh.texturesDiffuse[b];
			const bool nrm = b < normalized.size() && normalized[b];
			const auto px = [&](const Mesh::TexCoord& t) {
				return nrm ? Point2(t.x() * img.cols - 0.5, t.y() * img.rows - 0.5)
				           : Point2(t.x(), t.y());
			};
			const Point2 a = px(mesh.faceTexcoords[f * 3 + 0]);
			const Point2 b2 = px(mesh.faceTexcoords[f * 3 + 1]);
			const Point2 c2 = px(mesh.faceTexcoords[f * 3 + 2]);
			const Mesh::Face& fc = mesh.faces[f];
			const Vector3 p0 = mesh.vertices[fc[0]].cast<real>();
			const Vector3 p1 = mesh.vertices[fc[1]].cast<real>();
			const Vector3 p2 = mesh.vertices[fc[2]].cast<real>();
			cv::Mat_<uint8_t>& mask = masks[b];
			RasterizeTriangleBary<real>(
			    a, b2, c2, img.cols, img.rows,
			    [&](int x, int y, const Vector3& w) {
				    const Vector3 p = w[0] * p0 + w[1] * p1 + w[2] * p2;
				    const Vector3 t = (p - bmin).cwiseQuotient(ext) * 255.0;
				    img(y, x) = Pixel(static_cast<uint8_t>(std::clamp(t[0], 0.0, 255.0)),
				                      static_cast<uint8_t>(std::clamp(t[1], 0.0, 255.0)),
				                      static_cast<uint8_t>(std::clamp(t[2], 0.0, 255.0)));
				    mask(y, x) = 255;
			    },
			    /*cull=*/false, 0, std::numeric_limits<int>::max(), /*conservative=*/true);
		}
		for (size_t b = 0; b < mesh.texturesDiffuse.size(); ++b)
			Dilate(mesh.texturesDiffuse[b], masks[b], 8, 1);
		std::cout << "(replaced source texture with surface-smooth gradient)\n";
	}

	BakeParams bp;
	bp.resolution = tex; // --texture-size 0 ⇒ auto-size from the mesh
	bp.maxResolution = maxRes;
	bp.multiPage = Flag(argc, argv, "--multi-page");
	bp.padding = 4;
	bp.numThreads = threads;
	bp.supersample = static_cast<unsigned>(std::stoul(Arg(argc, argv, "--supersample", "1")));
	bp.interp = Flag(argc, argv, "--cubic") ? Interpolation::Cubic : Interpolation::Linear;
	bp.correspondence = Flag(argc, argv, "--raycast") ? Correspondence::Raycast
	                                                  : Correspondence::Nearest;
	bp.accelerator = Flag(argc, argv, "--kdtree") ? Accelerator::KdTree : Accelerator::BVH;

	if (cmd == "rebake") {
		const double dec = std::stod(Arg(argc, argv, "--decimation", "0.3"));
		// Decimate a copy and bake from the full-res source. glTF stores textured
		// meshes unwelded (one vertex per corner); weld + clean so the target has
		// the shared connectivity decimation and charting need.
		Mesh target = mesh;
		target.faceTexcoords.clear();
		target.faceTexblobs.clear();
		target.texturesDiffuse.clear();
		target.RemoveDuplicateVertices(0);
		target.RemoveDegenerateFaces();
		target.RemoveUnreferencedVertices();
		const auto td0 = Clock::now();
		target.Simplify(static_cast<float>(dec));
		const auto td1 = Clock::now();
		std::cout << "decimated to " << target.faces.size() << " faces ("
		          << Seconds(td0, td1) << " s)\n";

		const auto tb0 = Clock::now();
		const BakeResult r = RebakeTexture(mesh, target, bp);
		const auto tb1 = Clock::now();
		std::cout << "rebake: " << r.numPages << " page(s) " << r.width << "x" << r.height
		          << ", empty_texel_ratio=" << r.emptyTexelRatio << " ("
		          << Seconds(tb0, tb1) << " s)\n";

		if (Flag(argc, argv, "--measure")) {
			std::cout << "[in-memory] ";
			DoFidelity(mesh, target, 100000);
		}
		if (!out.empty() && target.Save(out))
			std::cout << "saved " << out << "\n";
	} else if (cmd == "defrag") {
		// glTF meshes are unwelded; --weld recovers shared connectivity (needed for
		// the half-edge build) while keeping per-corner UVs + textures.
		if (Flag(argc, argv, "--weld")) {
			mesh.RemoveDuplicateVertices(0);
			mesh.RemoveUnreferencedVertices();
			std::cout << "welded to " << mesh.vertices.size() << " verts\n";
		}
		const bool measure = Flag(argc, argv, "--measure");
		Mesh ref;
		if (measure)
			ref = mesh; // snapshot original layout for fidelity
		const auto tb0 = Clock::now();
		const BakeResult r = DefragmentTexture(mesh, bp);
		const auto tb1 = Clock::now();
		std::cout << "defrag: " << r.numPages << " page(s) " << r.width << "x" << r.height
		          << ", empty_texel_ratio=" << r.emptyTexelRatio << " ("
		          << Seconds(tb0, tb1) << " s)\n";
		if (measure) {
			std::cout << "[in-memory] ";
			DoFidelity(ref, mesh, 100000);
		}
		if (!out.empty() && mesh.Save(out))
			std::cout << "saved " << out << "\n";
	} else {
		std::cerr << "unknown command '" << cmd << "'\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
