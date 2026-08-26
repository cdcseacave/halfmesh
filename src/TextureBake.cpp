/*
* TextureBake.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/TextureBake.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>
#include <halfmesh/Parametrize.h>
#include <halfmesh/TriangleBVH.h>
#include <halfmesh/TriangleKDTree.h>
#include <halfmesh/Util/Geometry.h>
#include <halfmesh/Util/Log.h>
#include <halfmesh/Util/PixelTraits.h>
#include <halfmesh/Util/Raster.h>
#include <halfmesh/Util/Sampler.h>

namespace halfmesh {

namespace {

// Barycentric weights of point p w.r.t. triangle (v0, v1, v2), all in 3D.
// weights[0]->v0, [1]->v1, [2]->v2. Returns (1,0,0) for a degenerate triangle.
Vector3 Barycentric3D(const Vector3& v0, const Vector3& v1, const Vector3& v2,
                      const Vector3& p)
{
	const Vector3 e0 = v1 - v0, e1 = v2 - v0, e2 = p - v0;
	const double d00 = e0.dot(e0), d01 = e0.dot(e1), d11 = e1.dot(e1);
	const double d20 = e2.dot(e0), d21 = e2.dot(e1);
	const double denom = d00 * d11 - d01 * d01;
	// Relative degeneracy test: denom = |e0 x e1|^2 and d00*d11 = |e0|^2 |e1|^2
	// bounds it (Cauchy-Schwarz), so a near-collinear sliver has denom << d00*d11
	// long before it underflows to exactly 0. An exact-zero test lets such slivers
	// through, where catastrophic cancellation drives the barycentrics far outside
	// [0,1] and InterpUV then extrapolates a UV into an unrelated chart.
	if (!(denom > 1e-12 * d00 * d11))
		return Vector3(1, 0, 0);
	const double inv = 1.0 / denom;
	const double b1 = (d11 * d20 - d01 * d21) * inv;
	const double b2 = (d00 * d21 - d01 * d20) * inv;
	return Vector3(1.0 - b1 - b2, b1, b2);
}

// Interpolate the three per-corner texcoords of a face by barycentric weights.
Mesh::TexCoord InterpUV(const Mesh::TexCoord& t0, const Mesh::TexCoord& t1,
                        const Mesh::TexCoord& t2, const Vector3& bary)
{
	const Eigen::Vector2d u = t0.cast<double>() * bary[0] + t1.cast<double>() * bary[1] + t2.cast<double>() * bary[2];
	return u.cast<float>();
}

} // namespace

// Texture UVs are stored either normalized [0,1] (loaded meshes) or in absolute
// pixel space (our packed layouts). Classify PER BLOB against that blob's own
// texture dimensions: normalized UVs stay small (1 plus the tiling repeat
// count) while pixel UVs reach the blob's dimensions. A fixed global threshold
// gets BOTH regimes wrong: it misreads a >2x-tiled normalized asset as
// pixel-space, and one pixel-space blob drags every other blob's classification
// with it. Half the blob's
// smaller side separates the regimes wherever they are separable at all; the
// floor of 2.0 applies to blobs with no/degenerate texture dims.
// The comparison is a STRICT less-than: a tie at exactly the threshold goes to
// pixel-space. Pixel UVs legitimately reach exact fractions of the texture
// dims — a chart occupying exactly half the atlas maxes out at precisely
// min_dim/2 — while a normalized UV tiled to exactly half the texture
// dimension (repeat count 0.5x) is a pathological case that does not arise in
// practice. `<=` would misclassify the former as normalized on the exact tie
// (caught by SliverSourceUVStaysInChart: mx=8 on a 16px texture, threshold
// max(2,8)=8).
std::vector<bool> UVBlobsAreNormalized(const std::vector<Mesh::TexCoord>& uv,
                                       const std::vector<Mesh::TexIndex>& faceBlobs,
                                       const std::vector<Eigen::Vector2i>& blobDims)
{
	const size_t numBlobs = std::max<size_t>(blobDims.size(), 1);
	std::vector<float> mx(numBlobs, 0.f);
	const size_t numFaces = uv.size() / 3;
	// Trust the per-face blob list only when it has exactly one id per face: a
	// wrong-sized list carries no usable face->blob mapping (and indexing it
	// would read out of bounds), so it degrades to the single-blob convention,
	// same as an empty list.
	const bool haveBlobs = faceBlobs.size() == numFaces;
	for (size_t f = 0; f < numFaces; ++f) {
		const size_t b = haveBlobs ? static_cast<size_t>(faceBlobs[f]) : 0;
		if (b >= numBlobs)
			continue; // rogue blob id: must not pollute another blob's max
		for (int k = 0; k < 3; ++k) {
			const Mesh::TexCoord& t = uv[f * 3 + k];
			mx[b] = std::max(mx[b], std::max(std::abs(t.x()), std::abs(t.y())));
		}
	}
	std::vector<bool> normalized(numBlobs);
	for (size_t b = 0; b < numBlobs; ++b) {
		float threshold = 2.0f;
		if (b < blobDims.size())
			threshold = std::max(
			    threshold, 0.5f * static_cast<float>(std::min(blobDims[b].x(), blobDims[b].y())));
		normalized[b] = mx[b] < threshold;
	}
	return normalized;
}

// (cols,rows) of each diffuse texture: the per-blob dims the classifier needs.
std::vector<Eigen::Vector2i> BlobDims(const Mesh& m)
{
	std::vector<Eigen::Vector2i> dims;
	dims.reserve(m.texturesDiffuse.size());
	for (const Image3u& im : m.texturesDiffuse)
		dims.emplace_back(im.cols, im.rows);
	return dims;
}

namespace {

// Total texel count of a textured mesh = Σ per-face UV-triangle area in PIXELS
// (scaled by the face's blob dimensions when UVs are stored normalized).
double SourceTexelCount(const Mesh& m)
{
	if (m.faceTexcoords.size() != m.faces.size() * 3)
		return 0.0;
	const std::vector<bool> norm = UVBlobsAreNormalized(m.faceTexcoords, m.faceTexblobs, BlobDims(m));
	double total = 0.0;
	for (size_t f = 0; f < m.faces.size(); ++f) {
		const Mesh::TexCoord& t0 = m.faceTexcoords[f * 3 + 0];
		const Mesh::TexCoord& t1 = m.faceTexcoords[f * 3 + 1];
		const Mesh::TexCoord& t2 = m.faceTexcoords[f * 3 + 2];
		double a = 0.5 * std::abs(static_cast<double>(Mesh::ComputeTriangleDoubleArea2D(t0, t1, t2)));
		const unsigned blob = static_cast<unsigned>(m.FTexblob(static_cast<Mesh::FIndex>(f)));
		if (blob < m.texturesDiffuse.size() && norm[blob]) {
			const Image3u& im = m.texturesDiffuse[blob];
			a *= static_cast<double>(im.cols) * static_cast<double>(im.rows);
		}
		total += a;
	}
	return total;
}

// Source texel density (texels per world unit) = sqrt(texel_count / worldArea).
double SourceDensity(const Mesh& m)
{
	const double wa = m.ComputeArea();
	return wa > 0.0 ? std::sqrt(SourceTexelCount(m) / wa) : 0.0;
}

// Fraction of an atlas page the skyline packer fills in practice (charts plus
// gutters). Used to size a page from a texel budget: a page of side S holds
// roughly ATLAS_PACKING_FILL * S^2 useful texels once packed.
constexpr double ATLAS_PACKING_FILL = 0.82;

// Side (texels) of the single page that would hold the mesh's entire texel
// budget at the packer's typical fill ratio.
double IdealSinglePageSide(const Mesh& m)
{
	return std::sqrt(SourceTexelCount(m) / ATLAS_PACKING_FILL);
}

// Smallest power of two >= side, clamped to [lo, hi].
unsigned NextPow2Clamp(double side, unsigned lo, unsigned hi)
{
	if (lo > hi)
		lo = hi;
	if (side <= static_cast<double>(lo))
		return lo;
	unsigned p = lo;
	while (p < hi && static_cast<double>(p) < side)
		p <<= 1;
	return std::min(p, hi);
}

// Resolve atlas page size, multi-page flag and (for multi-page) the density to
// preserve, from the source mesh and the bake params. resolution==0 auto-sizes
// from the source texel budget; when a single page would exceed maxResolution
// it auto-switches to multi-page to keep the full density.
void DecideLayout(const Mesh& src, const BakeParams& p,
                  unsigned& page, bool& multi, double& density)
{
	density = SourceDensity(src);
	const double ideal = IdealSinglePageSide(src);
	const unsigned lo = std::min(256u, p.maxResolution);
	if (p.resolution > 0) {
		page = p.resolution;
		multi = p.multiPage;
	} else if (ideal <= static_cast<double>(p.maxResolution)) {
		page = NextPow2Clamp(ideal, lo, p.maxResolution);
		multi = p.multiPage;
	} else {
		page = p.maxResolution; // ideal exceeds cap
		multi = true; // auto-switch to multi-page to preserve density
	}
}

// Sample a source image at uv (absolute pixel coords) with the chosen kernel.
Vector3 SampleColor(const Image3u& img, const Mesh::TexCoord& uv, Interpolation interp)
{
	const Point2 p(uv.x(), uv.y());
	if (interp == Interpolation::Cubic)
		return SampleImage<CubicInterp<>>(img, p);
	return SampleImage<LinearInterp<>>(img, p);
}

// Wrap a normalized UV coordinate into [0,1) (GL_REPEAT), so wrapped/tiled or
// slightly-out-of-range source UVs sample the correct tiled content instead of
// scaling to a pixel coordinate outside the image (which samples solid black).
// Guards the float-rounding corner where u - floor(u) rounds up to exactly 1
// (e.g. u = -1e-8f), which would land one texel past the edge.
inline float WrapUnit(float u)
{
	float w = u - std::floor(u);
	if (w >= 1.0f)
		w = 0.0f;
	return w;
}

// Whether the UV triangle covers at least one pixel centre in [0,W)x[0,H), using
// the same inside test as RasterizeTriangleBary. Early-outs at the first covered
// pixel, so it is cheap for covered faces (the common case) and only scans the
// full — necessarily tiny — bbox of a sub-pixel-thin triangle that covers none.
bool TriangleCoversAnyPixel(const Point2& v1, const Point2& v2, const Point2& v3,
                            int width, int height)
{
	const double area = EdgeFunction<double>(v1, v2, v3);
	if (area == 0.0)
		return false;
	const double invArea = 1.0 / area;
	const double minx = std::min({v1.x(), v2.x(), v3.x()});
	const double maxx = std::max({v1.x(), v2.x(), v3.x()});
	const double miny = std::min({v1.y(), v2.y(), v3.y()});
	const double maxy = std::max({v1.y(), v2.y(), v3.y()});
	const int x0 = std::max(0, static_cast<int>(std::floor(minx)));
	const int x1 = std::min(width - 1, static_cast<int>(std::ceil(maxx)));
	const int y0 = std::max(0, static_cast<int>(std::floor(miny)));
	const int y1 = std::min(height - 1, static_cast<int>(std::ceil(maxy)));
	const double dx1 = v3.y() - v2.y(), dx2 = v1.y() - v3.y(), dx3 = v2.y() - v1.y();
	for (int y = y0; y <= y1; ++y) {
		const Point2 p0(static_cast<double>(x0), static_cast<double>(y));
		double e1 = EdgeFunction<double>(v2, v3, p0);
		double e2 = EdgeFunction<double>(v3, v1, p0);
		double e3 = EdgeFunction<double>(v1, v2, p0);
		for (int x = x0; x <= x1; ++x, e1 += dx1, e2 += dx2, e3 += dx3)
			if (e1 * invArea >= 0 && e2 * invArea >= 0 && e3 * invArea >= 0)
				return true;
	}
	return false;
}

} // namespace

// -------------------------------------------------------------------------
// SameUVResolver
// -------------------------------------------------------------------------
SameUVResolver::SameUVResolver(std::vector<Mesh::TexCoord> origTexcoords,
                               std::vector<Mesh::TexIndex> origBlobs,
                               std::vector<Eigen::Vector2i> blobDims) :
    texcoords(std::move(origTexcoords)), blobs(std::move(origBlobs)),
    dims(std::move(blobDims)), normalized(UVBlobsAreNormalized(texcoords, blobs, dims))
{
}

bool SameUVResolver::Resolve(Mesh::FIndex tgtFace, const Vector3& bary,
                             const Vector3& /*pos*/, const Mesh::Normal& /*nrm*/,
                             unsigned& srcImage, Mesh::TexCoord& srcUv,
                             Mesh::FIndex& /*hintFace*/) const
{
	const Mesh::TexCoord* t = &texcoords[tgtFace * 3];
	srcUv = InterpUV(t[0], t[1], t[2], bary);
	srcImage = blobs.empty() ? 0u : static_cast<unsigned>(blobs[tgtFace]);
	if (srcImage < dims.size() && normalized[srcImage]) {
		// Wrap into [0,1) (GL_REPEAT) so wrapped/tiled or slightly out-of-range UVs
		// sample real content instead of solid black, then unnormalize with the
		// GPU/glTF centre convention (texel i's centre at (i+0.5)/dim, mirror of
		// Mesh::FTexcoordsUnNormalize): the sampler reads integer coords as pixel
		// centres, so u*dim - 0.5, never a plain u*dim.
		srcUv.x() = WrapUnit(srcUv.x()) * dims[srcImage].x() - 0.5f;
		srcUv.y() = WrapUnit(srcUv.y()) * dims[srcImage].y() - 0.5f;
	}
	return true;
}

// -------------------------------------------------------------------------
// RaycastResolver
// -------------------------------------------------------------------------
RaycastResolver::RaycastResolver(const Mesh& source, Correspondence mode,
                                 float raySearchDist, Accelerator accel) :
    source(source), mode(mode), rayDist(raySearchDist),
    srcNormalized()
{
	source.SyncFacesConst();
	srcNormalized = UVBlobsAreNormalized(source.faceTexcoords, source.faceTexblobs, BlobDims(source));
	if (accel == Accelerator::KdTree)
		kdtree = std::make_unique<TriangleKdTree>(source);
	else
		bvh = std::make_unique<TriangleBVH>(source);
	if (rayDist <= 0.f) {
		const auto box = source.ComputeAABBox();
		rayDist = (box.max() - box.min()).norm() * 0.05f;
	}
}

RaycastResolver::~RaycastResolver() = default;

namespace {
// Uniform query result so the resolver is agnostic to the accelerator type.
struct SurfaceHit
{
	float dist;
	Mesh::Vertex nearest;
	Mesh::FIndex idxFace;
	bool valid;
};
} // namespace

bool RaycastResolver::Resolve(Mesh::FIndex /*tgtFace*/, const Vector3& /*bary*/,
                              const Vector3& pos, const Mesh::Normal& nrm,
                              unsigned& srcImage, Mesh::TexCoord& srcUv,
                              Mesh::FIndex& hintFace) const
{
	// The accelerator pipeline is float (Mesh::Type); query with the float cast of
	// the double sample point, but keep `pos` in double for the leaf barycentric.
	const Mesh::Vertex posf = pos.cast<float>();
	// Dispatch to whichever accelerator was built (BVH or kd-tree). The nearest
	// query is warm-started with the previous texel's resolved face (BVH only):
	// scanline-adjacent texels almost always land on the same/adjacent source
	// triangle, so seeding the search bound with the hint prunes most of the
	// descent. The hint only tightens the bound, so the winning nearest point is
	// unchanged (output byte-identical to the cold query).
	auto nearest = [&](const Mesh::Vertex& p) -> SurfaceHit {
		if (bvh) {
			const auto n = bvh->NearestPoint(
			    p, std::numeric_limits<float>::max(), hintFace);
			return {n.dist, n.nearest, n.idxFace, n.IsValid()};
		}
		const auto n = kdtree->NearestPoint(p);
		return {n.dist, n.nearest, n.idxFace, n.IsValid()};
	};
	// Ray queries are bounded by the search distance DURING traversal: subtrees
	// beyond it are pruned instead of finding a far hit that the post-filter then
	// discards. The bound is inflated one step above rayDist so a boundary hit
	// the (unchanged) Euclidean post-filter would accept is never pruned first.
	const float rayTmax = std::nextafter(rayDist, std::numeric_limits<float>::max());
	auto intersect = [&](const Eigen::ParametrizedLine<float, 3>& r) -> SurfaceHit {
		if (bvh) {
			const auto n = bvh->IntersectedPoint(r, rayTmax);
			return {n.dist, n.nearest, n.idxFace, n.IsValid()};
		}
		const auto n = kdtree->IntersectedPoint(r, rayTmax);
		return {n.dist, n.nearest, n.idxFace, n.IsValid()};
	};

	SurfaceHit nn{};
	if (mode == Correspondence::Nearest) {
		nn = nearest(posf);
	} else {
		// Cast along +n and -n; keep the closer hit within the search distance.
		const SurfaceHit a = intersect(Eigen::ParametrizedLine<float, 3>(posf, nrm));
		const SurfaceHit b = intersect(Eigen::ParametrizedLine<float, 3>(posf, (-nrm).eval()));
		const float da = a.valid ? (a.nearest - posf).norm() : 0.f;
		const float db = b.valid ? (b.nearest - posf).norm() : 0.f;
		const bool hitA = a.valid && da <= rayDist;
		const bool hitB = b.valid && db <= rayDist;
		if (hitA && hitB)
			nn = da <= db ? a : b;
		else if (hitA)
			nn = a;
		else if (hitB)
			nn = b;
		else
			nn = nearest(posf); // fallback when the ray misses
	}
	if (!nn.valid)
		return false;
	// Warm-start the next coherent query with the face we just resolved.
	hintFace = nn.idxFace;

	const Mesh::Face& sf = source.faces[nn.idxFace];
	const Vector3 v0 = source.vertices[sf[0]].cast<double>();
	const Vector3 v1 = source.vertices[sf[1]].cast<double>();
	const Vector3 v2 = source.vertices[sf[2]].cast<double>();
	// Recompute the closest point on the winning face in DOUBLE from the double
	// sample position, rather than decomposing the float-quantized query hit point.
	// The accelerator (float) already chose the face; promoting only this leaf
	// evaluation keeps barycentrics accurate at large/georeferenced coordinates,
	// where a float ULP is several texels of UV error.
	Vector3 nearestD;
	math::DistanceBetweenTriangleAndPointSquared<double>(v0, v1, v2, pos, &nearestD);
	Vector3 bary = Barycentric3D(v0, v1, v2, nearestD);
	// Clamp/renormalize onto the simplex before InterpUV: nearestD lies on the
	// face, so valid barycentrics are already in [0,1] (this is a no-op); only a
	// sliver's cancellation-blown weights are corrected, preventing a UV that
	// extrapolates outside the face's chart and samples an unrelated one.
	if (bary.minCoeff() < 0.0 || bary.maxCoeff() > 1.0) {
		bary = bary.cwiseMax(0.0);
		const double s = bary.sum();
		if (s > 0.0)
			bary /= s;
	}

	const Mesh::TexCoord* t = &source.faceTexcoords[nn.idxFace * 3];
	srcUv = InterpUV(t[0], t[1], t[2], bary);
	srcImage = static_cast<unsigned>(source.FTexblob(nn.idxFace));
	if (srcImage < source.texturesDiffuse.size() && srcNormalized[srcImage]) {
		// Wrap into [0,1) (GL_REPEAT) then half-texel unnormalize, mirror of
		// Mesh::FTexcoordsUnNormalize (integer coords are pixel centres): u*dim - 0.5.
		const Image3u& im = source.texturesDiffuse[srcImage];
		srcUv.x() = WrapUnit(srcUv.x()) * im.cols - 0.5f;
		srcUv.y() = WrapUnit(srcUv.y()) * im.rows - 0.5f;
	}
	return true;
}

// -------------------------------------------------------------------------
// BakeAtlas — shared rasterize -> resolve -> sample -> dilate engine.
// -------------------------------------------------------------------------
BakeResult BakeAtlas(Mesh& target, const std::vector<Image3u>& sourceImages,
                     const SourceResolver& resolver, const BakeParams& params)
{
	target.SyncFaces();
	BakeResult res;
	const int W = static_cast<int>(params.resolution);
	const int H = static_cast<int>(params.resolution);
	res.width = params.resolution;
	res.height = params.resolution;
	const size_t nf = target.faces.size();
	ASSERT(target.faceTexcoords.size() == nf * 3);

	unsigned numPages = 1;
	for (const Mesh::FIndex b : target.faceTexblobs)
		numPages = std::max(numPages, static_cast<unsigned>(b) + 1u);
	res.numPages = numPages;

	// Area-weighted per-vertex normals (drive the ray-based resolver; harmless
	// otherwise). Self-contained so a bare vertices/faces/uv mesh can be baked.
	std::vector<Vector3> vnorm(target.vertices.size(), Vector3::Zero());
	for (const Mesh::Face& f : target.faces) {
		const Vector3 v0 = target.vertices[f[0]].cast<double>();
		const Vector3 v1 = target.vertices[f[1]].cast<double>();
		const Vector3 v2 = target.vertices[f[2]].cast<double>();
		const Vector3 fn = (v1 - v0).cross(v2 - v0); // area-weighted
		vnorm[f[0]] += fn;
		vnorm[f[1]] += fn;
		vnorm[f[2]] += fn;
	}
	for (Vector3& n : vnorm) {
		const double l = n.norm();
		if (l > 0)
			n /= l;
	}

	std::vector<Image3u> atlases(numPages);
	std::vector<cv::Mat_<uint8_t>> masks(numPages);
	// Per-texel coverage class, driving centre-priority conservative resolve:
	//   0 = unclaimed, 1 = centre-inside covered, 2 = conservative-only claimed.
	// A centre-inside sample always wins (writes 1, overriding any prior 2); among
	// conservative-only coverers the first in face order wins (claims 2, later ones
	// skip). Kept separate from `masks` (which marks only texels that got a source
	// HIT): a centre-inside texel whose sample missed still holds state 1, so it is
	// never re-filled by the conservative pass — its miss is left to gutter dilation
	// exactly as before, and the reported emptyTexelRatio is unchanged.
	std::vector<cv::Mat_<uint8_t>> cover(numPages);
	for (unsigned p = 0; p < numPages; ++p) {
		atlases[p].create(H, W);
		atlases[p].setTo(cv::Scalar(0, 0, 0));
		masks[p] = cv::Mat_<uint8_t>::zeros(H, W);
		cover[p] = cv::Mat_<uint8_t>::zeros(H, W);
	}

	const int ss = std::max(1, static_cast<int>(params.supersample));
	// Defrag (SameUV) resolvers ignore the interpolated position/normal, so the
	// engine can skip that per-sample geometry work (and the normal's sqrt).
	const bool needsGeom = resolver.NeedsGeometry();
	const Mesh::FIndex noHint = std::numeric_limits<Mesh::FIndex>::max();
	size_t covered = 0, filled = 0;

	// Bake one face into rows [yLo, yHi). RasterizeTriangleBary is clipped to the
	// band so only owned rows are scanned; the in-callback guard is kept as cheap
	// defense. Each output texel belongs to exactly one row-band, so threads owning
	// disjoint bands never write the same pixel and the result equals serial.
	// `hint` is per-band coherent-query state threaded through the resolver.
	auto processFace = [&](size_t fi, int yLo, int yHi, size_t& cov, size_t& fil,
	                       Mesh::FIndex& hint) {
		const Mesh::Face& f = target.faces[fi];
		const unsigned page = target.faceTexblobs.empty()
		                          ? 0u
		                          : static_cast<unsigned>(target.faceTexblobs[fi]);
		const Mesh::TexCoord& t0 = target.faceTexcoords[fi * 3 + 0];
		const Mesh::TexCoord& t1 = target.faceTexcoords[fi * 3 + 1];
		const Mesh::TexCoord& t2 = target.faceTexcoords[fi * 3 + 2];
		const Point2 uv0(t0.x(), t0.y()), uv1(t1.x(), t1.y()), uv2(t2.x(), t2.y());
		const Vector3 v0 = target.vertices[f[0]].cast<double>();
		const Vector3 v1 = target.vertices[f[1]].cast<double>();
		const Vector3 v2 = target.vertices[f[2]].cast<double>();
		const Vector3& n0 = vnorm[f[0]];
		const Vector3& n1 = vnorm[f[1]];
		const Vector3& n2 = vnorm[f[2]];
		Image3u& atlas = atlases[page];
		cv::Mat_<uint8_t>& mask = masks[page];
		cv::Mat_<uint8_t>& cstate = cover[page];

		// Sub-sample barycentrics are affine, so hoist the triangle area + its
		// reciprocal out of the ss^2 inner loop (Barycentric2D recomputes both per
		// call). inv reuses the single 1/area value, so each sub-sample's weights
		// are bit-identical to Barycentric2D; the degenerate guard is redundant
		// because RasterizeTriangleBary already rejected area==0 to reach here.
		const double inv2d = ss > 1 ? 1.0 / EdgeFunction<double>(uv0, uv1, uv2) : 0.0;
		auto bary2d = [&](double px, double py) -> Vector3 {
			const Point2 p(px, py);
			return Vector3(EdgeFunction<double>(uv1, uv2, p) * inv2d,
			               EdgeFunction<double>(uv2, uv0, p) * inv2d,
			               EdgeFunction<double>(uv0, uv1, p) * inv2d);
		};

		// Resolve+sample one barycentric position; returns true and the source colour
		// on a hit. Skips the position/normal interpolation for geometry-free resolvers.
		auto resolveSample = [&](const Vector3& b, Vector3& color) -> bool {
			Vector3 pos = Vector3::Zero();
			Vector3 nrm = Vector3::Zero();
			if (needsGeom) {
				pos = b[0] * v0 + b[1] * v1 + b[2] * v2;
				nrm = b[0] * n0 + b[1] * n1 + b[2] * n2;
				const double nl = nrm.norm();
				if (nl > 0)
					nrm /= nl;
			}
			unsigned img;
			Mesh::TexCoord uv;
			if (resolver.Resolve(static_cast<Mesh::FIndex>(fi), b, pos, nrm.cast<float>(),
			                     img, uv, hint)
			    && img < sourceImages.size()) {
				color = SampleColor(sourceImages[img], uv, params.interp);
				return true;
			}
			return false;
		};

		// Multisample + resolve one texel; returns whether it got a source colour and
		// (on a hit) the averaged colour. Shared by the centre-inside and conservative
		// passes so both resolve identically; only the coverage bookkeeping differs.
		auto sampleTexel = [&](int x, int y, const Vector3& baryCenter, Vector3& out) -> bool {
			Vector3 acc = Vector3::Zero();
			int hits = 0;
			Vector3 color;
			if (ss == 1) {
				if (resolveSample(baryCenter, color)) {
					acc += color;
					++hits;
				}
			} else {
				for (int sy = 0; sy < ss; ++sy) {
					for (int sx = 0; sx < ss; ++sx) {
						const double ox = (sx + 0.5) / ss - 0.5;
						const double oy = (sy + 0.5) / ss - 0.5;
						const Vector3 b = bary2d(x + ox, y + oy);
						if (b[0] >= 0 && b[1] >= 0 && b[2] >= 0 && resolveSample(b, color)) {
							acc += color; // sub-sample inside the triangle, hit
							++hits;
						}
					}
				}
				if (hits == 0 && resolveSample(baryCenter, color)) {
					acc += color; // all sub-samples missed; centre is inside
					++hits;
				}
			}
			if (hits > 0) {
				out = (acc / static_cast<double>(hits)).eval();
				return true;
			}
			return false;
		};

		// Per-row warm-start reset: clear the BVH hint at the first texel of every row
		// so a row's hint chain is a pure function of that row (its own columns), never
		// of the rows/faces/bands processed before it. Row-band boundaries depend on the
		// thread count, so without this the hint entering a band's first row differs by
		// thread count; on an exact nearest-distance seam tie the warm-started query
		// could then resolve to a different face (hence colour) in parallel than serial.
		// Resetting makes serial==parallel STRUCTURAL rather than accidental. The
		// cross-column warm start within a row — the bulk of the coherence win — is
		// preserved; the cost is one cold query per row. (SameUV ignores the hint, so
		// this only affects the Raycast/Nearest resolvers.)
		int hintRow = -1;
		auto rowReset = [&](int y) {
			if (y != hintRow) {
				hint = noHint;
				hintRow = y;
			}
		};

		// Pass 1 — centre-inside coverage (the documented, byte-identical contract):
		// every texel whose centre lies inside the chart. Marks cstate=1 (hit or miss)
		// so the conservative pass never touches it.
		size_t faceCov = 0; // texels this face covered in this band (0 => maybe thin)
		RasterizeTriangleBary<double>(
		    uv0, uv1, uv2, W, H,
		    [&](int x, int y, const Vector3& baryCenter) {
			    if (y < yLo || y >= yHi)
				    return; // texel owned by a different row-band
			    rowReset(y);
			    ++cov;
			    ++faceCov;
			    cstate(y, x) = 1; // centre-inside coverage wins over conservative
			    Vector3 color;
			    if (sampleTexel(x, y, baryCenter, color)) {
				    atlas(y, x) = detail::StoreCast<Pixel>(color);
				    mask(y, x) = 255;
				    ++fil;
			    }
		    },
		    /*cull=*/false, yLo, yHi);

		// Pass 2 — conservative boundary ring. Resolve texels whose centre lies just
		// outside the chart (within half a pixel of an edge) that NO centre-inside
		// sample claimed, so the ring GPU bilinear filtering reads first carries a
		// true clamped-UV source sample instead of a gutter-dilation neighbour mean.
		// Runs after pass 1: this face's own centre-inside texels are already cstate==1
		// and are skipped here (so they are never re-resolved with renormalized
		// barycentrics — the centre-inside output stays byte-identical). cstate makes
		// the winner deterministic: centre beats conservative, and among conservative
		// coverers the first in face order wins. Because every face whose conservative
		// footprint reaches row y is binned into (and only writes) the single band that
		// owns row y, and faces run in ascending index within a band, "first in face
		// order" == lowest face index for that texel, independent of the thread count.
		// Conservative texels are deliberately excluded from cov/fil: emptyTexelRatio
		// keeps measuring the chart's centre-inside coverage, unchanged from before.
		hintRow = -1; // restart per-row hint tracking for the conservative scan
		RasterizeTriangleBary<double>(
		    uv0, uv1, uv2, W, H,
		    [&](int x, int y, const Vector3& baryClamped) {
			    if (y < yLo || y >= yHi)
				    return; // texel owned by a different row-band
			    rowReset(y);
			    if (cstate(y, x) != 0)
				    return; // centre-covered (1) or already conservative-claimed (2)
			    cstate(y, x) = 2;
			    Vector3 color;
			    if (sampleTexel(x, y, baryClamped, color)) {
				    atlas(y, x) = detail::StoreCast<Pixel>(color);
				    mask(y, x) = 255;
			    }
		    },
		    /*cull=*/false, yLo, yHi, /*conservative=*/true);

		// Thin-triangle fallback: a UV triangle thinner than a texel covers no pixel
		// centre and would be left to gutter dilation from a neighbouring chart. Give
		// it one texel at its UV centroid (bary = 1/3,1/3,1/3) so it contributes its
		// own colour. The decision is GLOBAL coverage (not this band): when the face
		// covered nothing in the band owning its centroid row, confirm it covers no
		// pixel anywhere, then write exactly one texel — identical in serial and
		// parallel, since only the centroid-owning band takes this path.
		if (faceCov == 0) {
			const double cxu = (uv0.x() + uv1.x() + uv2.x()) / 3.0;
			const double cyu = (uv0.y() + uv1.y() + uv2.y()) / 3.0;
			const int cx = std::min(W - 1, std::max(0, static_cast<int>(std::floor(cxu))));
			const int cy = std::min(H - 1, std::max(0, static_cast<int>(std::floor(cyu))));
			// Skip when the conservative pass already resolved the centroid texel
			// (cstate!=0): the sliver's own half-pixel ring usually covers it now, so
			// the fallback is only the safety net for a sliver too thin for even the
			// conservative footprint to reach a texel. Only the centroid-owning band
			// runs this, so serial and parallel stay identical.
			if (cy >= yLo && cy < yHi && cstate(cy, cx) == 0
			    && !TriangleCoversAnyPixel(uv0, uv1, uv2, W, H)) {
				++cov;
				cstate(cy, cx) = 2;
				hint = noHint; // cold query: the centroid resolve is a pure function of the face
				Vector3 color;
				if (resolveSample(Vector3(1.0 / 3, 1.0 / 3, 1.0 / 3), color)) {
					atlas(cy, cx) = detail::StoreCast<Pixel>(color.eval());
					mask(cy, cx) = 255;
					++fil;
				}
			}
		}
	};

	// Resolve thread count: serial for tiny meshes, else one band of rows each.
	unsigned nthreads = params.numThreads;
	if (nthreads == 0)
		nthreads = std::max(1u, std::thread::hardware_concurrency());
	nthreads = std::min<unsigned>(nthreads, static_cast<unsigned>(std::max(1, H)));
	if (nf < 64)
		nthreads = 1;

	if (nthreads <= 1) {
		Mesh::FIndex hint = noHint;
		for (size_t fi = 0; fi < nf; ++fi)
			processFace(fi, 0, H, covered, filled, hint);
	} else {
		// Contiguous row bands; bin each face into the bands its UV-bbox rows
		// overlap (a face spanning K bands is rasterized K times, but each pass
		// writes only its band's rows, so every texel is produced exactly once).
		std::vector<int> rowStart(nthreads + 1);
		for (unsigned b = 0; b <= nthreads; ++b)
			rowStart[b] = static_cast<int>(static_cast<long>(b) * H / nthreads);
		std::vector<std::vector<size_t>> bins(nthreads);
		for (size_t fi = 0; fi < nf; ++fi) {
			const Mesh::TexCoord& t0 = target.faceTexcoords[fi * 3 + 0];
			const Mesh::TexCoord& t1 = target.faceTexcoords[fi * 3 + 1];
			const Mesh::TexCoord& t2 = target.faceTexcoords[fi * 3 + 2];
			const double ymin = std::min({t0.y(), t1.y(), t2.y()});
			const double ymax = std::max({t0.y(), t1.y(), t2.y()});
			const int ry0 = std::max(0, static_cast<int>(std::floor(ymin)));
			const int ry1 = std::min(H - 1, static_cast<int>(std::ceil(ymax)));
			if (ry0 > ry1)
				continue; // face fully outside the page vertically
			// rowStart is sorted/contiguous, so the overlapped bands form one
			// interval [b0, b1] found by two binary searches (O(log T)) instead of
			// scanning all T bands; bin contents are identical to the linear scan.
			const int b0 = static_cast<int>(
			                   std::upper_bound(rowStart.begin(), rowStart.end(), ry0) - rowStart.begin())
			               - 1;
			const int b1 = static_cast<int>(
			                   std::upper_bound(rowStart.begin(), rowStart.end(), ry1) - rowStart.begin())
			               - 1;
			for (int b = b0; b <= b1; ++b)
				bins[b].push_back(fi);
		}
		std::vector<size_t> cov(nthreads, 0), fil(nthreads, 0);
		std::vector<std::thread> pool;
		pool.reserve(nthreads);
		for (unsigned b = 0; b < nthreads; ++b) {
			pool.emplace_back([&, b] {
				Mesh::FIndex hint = noHint; // per-band coherent-query state
				for (size_t fi : bins[b])
					processFace(fi, rowStart[b], rowStart[b + 1], cov[b], fil[b], hint);
			});
		}
		for (std::thread& th : pool)
			th.join();
		for (unsigned b = 0; b < nthreads; ++b) {
			covered += cov[b];
			filled += fil[b];
		}
	}

	if (params.padding > 0) {
		const int pad = static_cast<int>(params.padding);
		// Pages are independent, so dilate them in parallel (each page fully by one
		// thread, block-partitioned). Per-page output is deterministic and thread-
		// assignment does not affect it, so the result is independent of scheduling.
		if (numPages > 1 && nthreads > 1) {
			const unsigned dt = std::min<unsigned>(nthreads, numPages);
			std::vector<std::thread> dpool;
			dpool.reserve(dt);
			for (unsigned t = 0; t < dt; ++t) {
				const unsigned p0 = static_cast<unsigned>(static_cast<long>(t) * numPages / dt);
				const unsigned p1 = static_cast<unsigned>(static_cast<long>(t + 1) * numPages / dt);
				dpool.emplace_back([&, p0, p1] {
					for (unsigned p = p0; p < p1; ++p)
						Dilate(atlases[p], masks[p], pad, 1);
				});
			}
			for (std::thread& th : dpool)
				th.join();
		} else {
			for (unsigned p = 0; p < numPages; ++p)
				Dilate(atlases[p], masks[p], pad, 1);
		}
	}

	target.texturesDiffuse = std::move(atlases);
	res.emptyTexelRatio =
	    covered ? static_cast<float>(covered - filled) / static_cast<float>(covered) : 0.f;
	return res;
}

namespace {

// Turn a freshly packed [0,1] layout (AtlasResult) into the engine's input:
// absolute pixel coordinates in mesh.faceTexcoords + a per-face page index in
// mesh.faceTexblobs (cleared when single-page).
void ApplyPackedLayout(Mesh& mesh, const AtlasResult& atlas)
{
	const float W = static_cast<float>(atlas.width);
	const float H = static_cast<float>(atlas.height);
	for (Mesh::TexCoord& uv : mesh.faceTexcoords) {
		uv.x() *= W;
		uv.y() *= H;
	}
	if (atlas.numPages > 1 && atlas.faceChart.size() == mesh.faces.size()) {
		mesh.faceTexblobs.resize(mesh.faces.size());
		for (size_t f = 0; f < mesh.faces.size(); ++f)
			mesh.faceTexblobs[f] = atlas.chartPage[atlas.faceChart[f]];
	} else {
		mesh.faceTexblobs.clear();
	}
}

} // namespace

// -------------------------------------------------------------------------
// RebakeTexture
// -------------------------------------------------------------------------
BakeResult RebakeTexture(const Mesh& source, Mesh& target, const BakeParams& params)
{
	source.SyncFacesConst();
	target.SyncFaces();
	// Fail fast on violated preconditions (documented in TextureBake.h): the
	// size invariants downstream are Debug-only ASSERTs, so without this an
	// untextured source is an out-of-bounds read in Release, not a diagnostic.
	// A default-constructed result (numPages == 0) signals the no-op.
	if (source.faces.empty() || target.faces.empty() || source.faceTexcoords.size() != source.faces.size() * 3 || source.texturesDiffuse.empty()) {
		REPORT_WARNING("RebakeTexture: source must carry per-corner UVs ({} needed, {} present) and "
		               "textures ({} present), and both meshes need faces; nothing baked",
		               source.faces.size() * 3, source.faceTexcoords.size(), source.texturesDiffuse.size());
		return BakeResult{};
	}
	unsigned page;
	bool multi;
	double density;
	DecideLayout(source, params, page, multi, density);

	ParametrizeParams pp;
	AtlasParams ap;
	ap.resolution = page;
	ap.padding = params.padding;
	ap.square = true;
	if (multi) {
		ap.texelsPerUnit = static_cast<float>(density); // preserve source density
		ap.fitToResolution = false; // overflow into N pages
	} else {
		ap.texelsPerUnit = 0.f; // auto density, one page
		ap.fitToResolution = true;
	}
	const AtlasResult atlas = GenerateAtlas(target, pp, ap);
	ApplyPackedLayout(target, atlas);

	BakeParams bp = params;
	bp.resolution = atlas.width;
	if (bp.correspondence == Correspondence::SameUV)
		bp.correspondence = Correspondence::Nearest; // SameUV is undefined cross-mesh
	RaycastResolver resolver(source, bp.correspondence, params.raySearchDist,
	                         params.accelerator);
	return BakeAtlas(target, source.texturesDiffuse, resolver, bp);
}

// -------------------------------------------------------------------------
// DefragmentTexture (light)
// -------------------------------------------------------------------------
BakeResult DefragmentTexture(Mesh& mesh, const BakeParams& params)
{
	mesh.SyncFaces();
	// Fail fast on violated preconditions (documented in TextureBake.h) — the
	// downstream size invariants are Debug-only ASSERTs, so an untextured mesh
	// would be an out-of-bounds read in Release instead of a diagnostic.
	if (mesh.faces.empty() || mesh.faceTexcoords.size() != mesh.faces.size() * 3 || mesh.texturesDiffuse.empty()) {
		REPORT_WARNING("DefragmentTexture: mesh must carry per-corner UVs ({} needed, {} present) and "
		               "textures ({} present); nothing defragmented",
		               mesh.faces.size() * 3, mesh.faceTexcoords.size(), mesh.texturesDiffuse.size());
		return BakeResult{};
	}
	// Snapshot the original layout + textures BEFORE any modification, so the
	// SameUV resolver can map each (unchanged) face back to its original texture.
	std::vector<Mesh::TexCoord> origUv = mesh.faceTexcoords;
	std::vector<Mesh::TexIndex> origBlob = mesh.faceTexblobs;
	const std::vector<Image3u> origTex = mesh.texturesDiffuse;

	// Decide the layout from the ORIGINAL (pre-relayout) UV/texture budget.
	unsigned page;
	bool multi;
	double density;
	DecideLayout(mesh, params, page, multi, density);

	// Existing UV islands are the charts (light defrag: no re-parametrization).
	if (mesh.halfMesh.Empty())
		mesh.ListHalfEdges(); // NOTE: auto-repairs (manifoldizes) non-manifold
	// input in place — see Mesh::ListHalfEdges
	std::vector<unsigned> faceChart;
	const unsigned numCharts = mesh.ListTexPatchFaces(faceChart);
	// Guard against pathologically fragmented source atlases: the skyline
	// packer's cost is superlinear in the chart count, and defrag keeps every
	// source patch as its own chart. Measured (2026-08 review): a CAD glTF with
	// 128,748 patches (2.3 faces/patch) spun PackRects for >40 minutes, while
	// 35,897 patches packed in 81 s. Such atlases need re-charting, which is
	// exactly RebakeTexture's job — refuse instead of hanging.
	REPORT_STATUS("DefragmentTexture: {} source patches over {} faces", numCharts, mesh.faces.size());
	if (params.maxDefragPatches != 0 && numCharts > params.maxDefragPatches) {
		REPORT_WARNING("DefragmentTexture: source atlas has {} patches (> {} allowed by "
		               "BakeParams::maxDefragPatches); repacking this many rects is intractable — "
		               "use RebakeTexture to re-chart instead, or raise/zero the limit (note: "
		               "building the half-edge structure above may already have manifoldized "
		               "non-manifold input)",
		               numCharts, params.maxDefragPatches);
		return BakeResult{};
	}

	// Re-normalize per-chart density and repack (one page, or N pages preserving density).
	AtlasParams ap;
	ap.resolution = page;
	ap.padding = params.padding;
	ap.square = true;
	if (multi) {
		ap.texelsPerUnit = static_cast<float>(density);
		ap.fitToResolution = false;
	} else {
		ap.texelsPerUnit = 0.f;
		ap.fitToResolution = true;
	}
	NormalizeChartDensity(mesh, faceChart, numCharts, ap);
	const AtlasResult atlas = PackAtlas(mesh, faceChart, numCharts, ap);
	ApplyPackedLayout(mesh, atlas);

	std::vector<Eigen::Vector2i> dims;
	dims.reserve(origTex.size());
	for (const Image3u& im : origTex)
		dims.emplace_back(im.cols, im.rows);

	SameUVResolver resolver(std::move(origUv), std::move(origBlob), std::move(dims));
	BakeParams bp = params;
	bp.resolution = atlas.width;
	bp.correspondence = Correspondence::SameUV;
	return BakeAtlas(mesh, origTex, resolver, bp);
}

unsigned AutoAtlasResolution(const Mesh& source, unsigned maxResolution)
{
	source.SyncFacesConst();
	return NextPow2Clamp(IdealSinglePageSide(source), std::min(256u, maxResolution),
	                     maxResolution);
}

} // namespace halfmesh
