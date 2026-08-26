/*
* TextureBake.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Unified texture rebake + defragmentation.
//
// Both operations are the same atlas-resampling step — for each output texel,
// find a source location, sample it, write, then dilate the gutter — differing
// only in HOW the source location is found. That choice lives behind a single
// SourceResolver:
//   - SameUVResolver  (defrag): the source is the same texture; a target face
//     maps to its own ORIGINAL uv/blob (snapshot taken before the layout moved).
//   - RaycastResolver (rebake): the source is another mesh's surface, found by
//     ray-along-normal / nearest-point query, then barycentric -> source uv.
//
// The shared BakeAtlas engine rasterizes the target UV layout, resolves+samples
// each texel through the given resolver, and fills target.texturesDiffuse.

#pragma once

#include <memory>
#include <vector>

#include <halfmesh/Mesh.h>
#include <halfmesh/Types.h>

namespace halfmesh {

class TriangleKdTree;
class TriangleBVH;

// Surface acceleration structure used by the ray-based resolver. BVH (flattened,
// cache-friendly) is faster for the millions of queries a bake issues; KdTree is
// the simpler reference. Both are correct.
enum class Accelerator {
	BVH,
	KdTree,
};

// How a target texel's source location is found.
enum class Correspondence {
	SameUV, // defrag: same face -> its original uv (no geometry)
	Nearest, // rebake: nearest point on the source surface
	Raycast, // rebake: ray along the target normal, nearest-point fallback
};

// Source-texture interpolation kind (see Util/Sampler.h).
enum class Interpolation {
	Linear,
	Cubic,
};

struct BakeParams
{
	unsigned resolution = 4096; // atlas page size (texels); 0 => auto
	unsigned maxResolution = 8192; // per-page cap; auto overflows beyond this
	bool multiPage = false; // preserve source density across N pages
	unsigned padding = 4; // gutter texels to dilate
	unsigned supersample = 1; // NxN per-texel multisample
	Interpolation interp = Interpolation::Linear; // source sampling kernel
	Correspondence correspondence = Correspondence::Nearest; // used by front-ends
	    // (Nearest is faster + more
	    // accurate for coincident-surface
	    // rebake; Raycast suits offset cages)
	float raySearchDist = 0.f; // 0 => auto (5% src bbox diag)
	unsigned numThreads = 0; // 0 => auto (hardware concurrency)
	Accelerator accelerator = Accelerator::BVH; // source-surface query structure
	unsigned maxDefragPatches = 65536; // DefragmentTexture only: refuse source
	    // atlases with more UV patches than this (skyline repack cost is
	    // superlinear in patch count; such atlases need RebakeTexture's
	    // re-charting instead); 0 => no limit
};

struct BakeResult
{
	unsigned width = 0;
	unsigned height = 0;
	unsigned numPages = 0;
	float emptyTexelRatio = 0.f; // chart texels with no source hit / chart texels
};

// Per-blob UV-regime classification: entry b is true when blob b's UVs are
// stored normalized ([0,1] plus tiling repeats), false when they are absolute
// pixel coordinates. Classified against each blob's own texture dimensions:
// threshold max(2, min(cols,rows)/2), STRICT less-than (a tie goes to pixel
// space). `faceBlobs` maps each face (uv.size()/3 corner triples) to its
// blob; an empty or wrong-sized list means single-blob (all faces -> blob 0),
// and faces carrying an out-of-range blob id are ignored. Entries with no
// corresponding blobDims entry -- including the single entry produced for
// an empty blobDims -- are classified against the bare floor threshold of 2. Always returns max(blobDims.size(), 1) entries.
std::vector<bool> UVBlobsAreNormalized(const std::vector<Mesh::TexCoord>& uv,
                                       const std::vector<Mesh::TexIndex>& faceBlobs,
                                       const std::vector<Eigen::Vector2i>& blobDims);

// (cols,rows) of each m.texturesDiffuse image: the per-blob dims the
// classifier needs.
std::vector<Eigen::Vector2i> BlobDims(const Mesh& m);

// Maps a target surface sample to a source (image index, uv in that image's
// absolute pixel space). Returns false when no source exists for the sample.
class SourceResolver
{
	public:
	virtual ~SourceResolver() = default;
	// Resolve one target surface sample to a source (image, uv). `hintFace` is
	// coherent-query state threaded by the caller across scanline-adjacent texels:
	// resolvers may read it as a warm-start seed and MUST update it to the face
	// they resolved (an out-param), so the next call starts near this one. It is
	// per-thread state owned by the caller (never shared), so const Resolve stays
	// re-entrant. Resolvers that need no geometry (see NeedsGeometry) ignore pos/nrm.
	virtual bool Resolve(Mesh::FIndex tgtFace, const Vector3& bary,
	                     const Vector3& pos, const Mesh::Normal& nrm,
	                     unsigned& srcImage, Mesh::TexCoord& srcUv,
	                     Mesh::FIndex& hintFace) const = 0;

	// Whether Resolve reads the interpolated 3D position/normal. Defrag (SameUV)
	// does not, so the engine can skip per-sample position/normal interpolation
	// (and the normal's sqrt) for it. Query-based resolvers need the geometry.
	virtual bool NeedsGeometry() const { return true; }
};

// Defrag resolver: a target face maps to its own original uv/blob. Construct
// from a SNAPSHOT of the mesh's per-corner texcoords/blobs taken before the
// layout was changed; resamples from the original textures. blobDims are the
// (cols,rows) of each source image: if the snapshot UVs are normalized [0,1]
// (as loaded meshes store them) they are scaled to pixel space on resolve.
class SameUVResolver : public SourceResolver
{
	public:
	SameUVResolver(std::vector<Mesh::TexCoord> origTexcoords,
	               std::vector<Mesh::TexIndex> origBlobs,
	               std::vector<Eigen::Vector2i> blobDims);
	bool Resolve(Mesh::FIndex tgtFace, const Vector3& bary,
	             const Vector3& pos, const Mesh::Normal& nrm,
	             unsigned& srcImage, Mesh::TexCoord& srcUv,
	             Mesh::FIndex& hintFace) const override;
	bool NeedsGeometry() const override { return false; }

	private:
	std::vector<Mesh::TexCoord> texcoords; // per-corner (faces*3)
	std::vector<Mesh::TexIndex> blobs; // per-face, or empty (single blob)
	std::vector<Eigen::Vector2i> dims; // (cols,rows) per blob
	std::vector<bool> normalized; // per blob: snapshot UVs are [0,1]
};

// Rebake resolver: query the SOURCE surface to find the source face +
// barycentric, then read the source uv/blob. Owns a TriangleKdTree over source.
class RaycastResolver : public SourceResolver
{
	public:
	explicit RaycastResolver(const Mesh& source,
	                         Correspondence mode = Correspondence::Raycast,
	                         float raySearchDist = 0.f,
	                         Accelerator accel = Accelerator::BVH);
	~RaycastResolver() override;
	bool Resolve(Mesh::FIndex tgtFace, const Vector3& bary,
	             const Vector3& pos, const Mesh::Normal& nrm,
	             unsigned& srcImage, Mesh::TexCoord& srcUv,
	             Mesh::FIndex& hintFace) const override;

	private:
	const Mesh& source;
	Correspondence mode;
	float rayDist;
	std::vector<bool> srcNormalized; // per blob: source UVs are [0,1] -> scale to pixels on resolve
	std::unique_ptr<TriangleBVH> bvh; // one of these is built (per Accelerator)
	std::unique_ptr<TriangleKdTree> kdtree;
};

// Shared engine. target.faceTexcoords must already hold the final atlas layout
// in absolute pixel coordinates (one page per target.faceTexblobs entry, or a
// single page when empty). Fills target.texturesDiffuse (numPages images of
// `resolution` squared) by resampling sourceImages through `resolver`.
BakeResult BakeAtlas(Mesh& target,
                     const std::vector<Image3u>& sourceImages,
                     const SourceResolver& resolver,
                     const BakeParams& params);

// -------------------------------------------------------------------------
// Front-ends.
// -------------------------------------------------------------------------

// Atlas page side (texels) that holds `source`'s texel detail in a single page:
// next-power-of-two of sqrt(source_texel_count / packing_fill), clamped to
// [256, maxResolution]. Used when BakeParams.resolution == 0 (auto-size).
unsigned AutoAtlasResolution(const Mesh& source, unsigned maxResolution = 8192);

// Rebake: generate a fresh UV atlas for `target` (GenerateAtlas: D-Charts +
// SLIM + density-normalize + pack at params.resolution) and bake `source`'s
// textures onto it by querying the source surface. `source` and `target` may be
// different geometry (e.g. a decimated target). On return `target` carries the
// new UVs and baked textures. correspondence must be Nearest or Raycast (SameUV
// is upgraded to Raycast). Assumes faceTexcoords are absolute pixel coords.
// Preconditions are checked in every build mode: `source` must carry per-corner
// UVs and at least one texture, and both meshes need faces — otherwise a
// default-constructed result (numPages == 0) is returned and a warning logged.
BakeResult RebakeTexture(const Mesh& source, Mesh& target, const BakeParams& params);

// Defragment (light): keep the mesh's existing UV charts (no re-parametrization),
// repack them tightly into a single atlas at params.resolution, and resample the
// existing textures into the new layout (near-lossless rigid repack — reclaims
// wasted space and consolidates many source images into one). Mesh must be
// manifold and carry per-corner UVs in absolute pixel coordinates.
// Preconditions are checked in every build mode: per-corner UVs + at least one
// texture, else a default result (numPages == 0) is returned with a warning.
// Also refused: source atlases with more than 65,536 UV patches — the skyline
// repack is superlinear in patch count (a 128k-patch CAD atlas measured >40 min);
// such inputs need re-charting via RebakeTexture instead.
BakeResult DefragmentTexture(Mesh& mesh, const BakeParams& params);

} // namespace halfmesh
