/*
* InteropOpenMVS.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

// Optional interoperability helpers for using halfmesh alongside openMVS
// (https://github.com/cdcseacave/openMVS).  halfmesh and openMVS each define their
// own `Mesh` type in distinct namespaces (`halfmesh::Mesh` vs `MVS::Mesh`), so the
// two libraries can be included and linked together without clashing.  This header
// adds zero-friction conversion between the two mesh representations.
//
// It is header-only and entirely opt-in: the body is compiled only when the openMVS
// headers are reachable (i.e. `MVS/Mesh.h` is on the include path).  halfmesh has no
// build- or link-time dependency on openMVS; including this header in a project that
// does not have openMVS available is a no-op.  Because it is gated this way it is not
// exercised by halfmesh's own test suite — it is validated by the consuming project.
//
// Usage:
//     #include <MVS/Common.h>               // openMVS first: Common.h defines the
//     #include <MVS/Mesh.h>                 // types Mesh.h assumes are in scope
//     #include <halfmesh/InteropOpenMVS.h>
//     ...
//     MVS::Mesh mvsMesh = ...;
//     halfmesh::Mesh hm;
//     halfmesh::ConvertMesh(mvsMesh, hm);              // MVS  -> halfmesh (copies)
//     ...
//     MVS::Mesh out;
//     halfmesh::ConvertMesh(hm, out);                  // halfmesh -> MVS (copies)
//
// Each direction also has a consuming overload that frees the source array by
// array as it goes, so a large mesh never has to be resident in both
// representations at once:
//
//     halfmesh::ConvertMesh(std::move(mvsMesh), hm);   // mvsMesh is empty after
//     halfmesh::ConvertMesh(std::move(hm), out);       // hm is empty after

#if __has_include(<MVS/Mesh.h>)

	#include <MVS/Mesh.h>
	#include <halfmesh/Mesh.h>
	#include <halfmesh/Util/Log.h> // REPORT_WARNING (texture-count cap below)

	#include <vector> // ReleaseVector below

namespace halfmesh {

namespace detail {

// Shared bodies behind the four public ConvertMesh overloads.
//
// `Src` deduces to `const X` for the copying overload and to plain `X` for the
// consuming one, so the `if constexpr (kRelease)` arms -- the only code that
// mutates the source -- are instantiated solely for the non-const case.  That
// keeps one implementation per direction without resorting to a const_cast.
//
// With kRelease each array is freed the moment its last reader is done with it,
// so peak footprint is the larger of the two meshes plus one array rather than
// their sum.

// Frees a halfmesh array outright.  clear() would keep the capacity, which is
// precisely what a caller asking for a consuming conversion wants back.
template <typename T>
inline void ReleaseVector(std::vector<T>& v)
{
	std::vector<T>().swap(v);
}

template <bool kRelease, typename Src>
void ConvertFromMVS(Src& src, halfmesh::Mesh& dst)
{
	dst = halfmesh::Mesh{};

	// Capture the counts up front: with kRelease the geometry is freed as it is
	// consumed, so the "is this optional array full-length?" guards below can no
	// longer ask the source for its vertex/face count.
	const size_t numVertices = src.vertices.size();
	const size_t numFaces = src.faces.size();

	dst.vertices.resize(numVertices);
	for (size_t i = 0; i < numVertices; ++i) {
		const MVS::Mesh::Vertex& v = src.vertices[(MVS::Mesh::VIndex)i];
		dst.vertices[i] = halfmesh::Mesh::Vertex(v.x, v.y, v.z);
	}
	if constexpr (kRelease)
		src.vertices.Release();

	dst.faces.resize(numFaces);
	for (size_t i = 0; i < numFaces; ++i) {
		const MVS::Mesh::Face& f = src.faces[(MVS::Mesh::FIndex)i];
		dst.faces[i] = halfmesh::Mesh::Face(f.x, f.y, f.z);
	}
	if constexpr (kRelease)
		src.faces.Release();

	// Per-vertex colors: both sides are 3x uint8 with blue in the first byte under
	// the default openMVS build; copy by channel name so a _COLORMODE_RGB build of
	// openMVS still maps each channel to halfmesh's fixed BGR element order.
	// halfmesh's contract for optional per-element arrays is "empty or exactly
	// sized" (its swap-pop removal paths index them whenever non-empty, and
	// Mesh::Load clears mismatched arrays -- see src/MeshIO.cpp); only transfer a
	// partially-populated source array when it is full-length, otherwise leave
	// the destination empty (it already is, from the reset above).
	if (src.vertexColors.size() == numVertices) {
		dst.vertexColors.resize(src.vertexColors.size());
		for (size_t i = 0; i < dst.vertexColors.size(); ++i) {
			const MVS::Mesh::Color& c = src.vertexColors[(MVS::Mesh::VIndex)i];
			dst.vertexColors[i] = halfmesh::Mesh::Pixel(c.b, c.g, c.r);
		}
	}
	// Released whether or not it was transferred: a consuming conversion takes
	// ownership of the source either way.  Same for every optional array below.
	if constexpr (kRelease)
		src.vertexColors.Release();

	// Face normals are transferred as-is (see the size guard above) and trusted
	// as fresh by halfmesh, whose own freshness checks are size-only; if the
	// source normals may be stale relative to its geometry, call
	// ComputeFaceNormals() on the destination after conversion.
	if (src.faceNormals.size() == numFaces) {
		dst.faceNormals.resize(src.faceNormals.size());
		for (size_t i = 0; i < dst.faceNormals.size(); ++i) {
			const MVS::Mesh::Normal& n = src.faceNormals[(MVS::Mesh::FIndex)i];
			dst.faceNormals[i] = halfmesh::Mesh::Normal(n.x, n.y, n.z);
		}
	}
	if constexpr (kRelease)
		src.faceNormals.Release();

	// Texture coordinates: openMVS stores them per face-corner (3*faces) or per
	// vertex — the same two layouts halfmesh supports — so copy 1:1.
	dst.faceTexcoords.resize(src.faceTexcoords.size());
	for (size_t i = 0; i < dst.faceTexcoords.size(); ++i) {
		const MVS::Mesh::TexCoord& t = src.faceTexcoords[(MVS::Mesh::FIndex)i];
		dst.faceTexcoords[i] = halfmesh::Mesh::TexCoord(t.x, t.y);
	}
	if constexpr (kRelease)
		src.faceTexcoords.Release();

	// Per-face texture index.  halfmesh::Mesh::TexIndex is defined as uint8 to
	// match MVS::Mesh::TexIndex, so this is a straight copy with nothing to
	// widen or narrow.
	dst.faceTexblobs.resize(src.faceTexindices.size());
	for (size_t i = 0; i < dst.faceTexblobs.size(); ++i)
		dst.faceTexblobs[i] = (halfmesh::Mesh::TexIndex)src.faceTexindices[(MVS::Mesh::FIndex)i];
	if constexpr (kRelease)
		src.faceTexindices.Release();

	// Diffuse textures: both sides are CV_8UC3 (BGR) cv::Mat under the hood, so a
	// typed copy converts the element type (cv::Vec3b <-> halfmesh::Pixel) safely.
	// cv::Mat is refcounted, so where the element types already agree this shares
	// the pixel buffer instead of duplicating it — releasing the source then
	// simply drops openMVS's reference and the images are never paid for twice.
	dst.texturesDiffuse.resize(src.texturesDiffuse.size());
	for (size_t i = 0; i < dst.texturesDiffuse.size(); ++i)
		dst.texturesDiffuse[i] = halfmesh::Mesh::Image3u(src.texturesDiffuse[(MVS::Mesh::TexIndex)i]);
	if constexpr (kRelease)
		src.texturesDiffuse.Release();
}

template <bool kRelease, typename Src>
void ConvertToMVS(Src& src, MVS::Mesh& dst)
{
	src.SyncFacesConst();
	dst.Release();

	const size_t numVertices = src.vertices.size();
	const size_t numFaces = src.faces.size();

	dst.vertices.resize((MVS::Mesh::VIndex)numVertices);
	for (size_t i = 0; i < numVertices; ++i) {
		const halfmesh::Mesh::Vertex& v = src.vertices[i];
		dst.vertices[(MVS::Mesh::VIndex)i] = MVS::Mesh::Vertex(v[0], v[1], v[2]);
	}
	if constexpr (kRelease)
		ReleaseVector(src.vertices);

	dst.faces.resize((MVS::Mesh::FIndex)numFaces);
	for (size_t i = 0; i < numFaces; ++i) {
		const halfmesh::Mesh::Face& f = src.faces[i];
		dst.faces[(MVS::Mesh::FIndex)i] = MVS::Mesh::Face(f[0], f[1], f[2]);
	}
	if constexpr (kRelease) {
		ReleaseVector(src.faces);
		// The half-edge structure and the incident-face lists are derived data
		// that together usually outweigh the geometry; nothing below reads them.
		src.InvalidateHalfMesh();
		ReleaseVector(src.vertexFaces);
	}

	// Per-vertex colors: assign by channel name (see the MVS -> halfmesh note).
	// Only transfer when full-length -- see the size-invariant note above; a
	// partially-populated source array leaves the destination empty (it already
	// is, from the Release() above).
	if (src.vertexColors.size() == numVertices) {
		dst.vertexColors.resize((MVS::Mesh::VIndex)src.vertexColors.size());
		for (size_t i = 0; i < src.vertexColors.size(); ++i) {
			const halfmesh::Mesh::Pixel& p = src.vertexColors[i];
			MVS::Mesh::Color& c = dst.vertexColors[(MVS::Mesh::VIndex)i];
			c.b = p[0];
			c.g = p[1];
			c.r = p[2];
		}
	}
	if constexpr (kRelease)
		ReleaseVector(src.vertexColors);

	// Face normals are transferred as-is (see the size guard above) and trusted
	// as fresh by halfmesh, whose own freshness checks are size-only; if the
	// source normals may be stale relative to its geometry, call
	// ComputeFaceNormals() on the destination after conversion.
	if (src.faceNormals.size() == numFaces) {
		dst.faceNormals.resize((MVS::Mesh::FIndex)src.faceNormals.size());
		for (size_t i = 0; i < src.faceNormals.size(); ++i) {
			const halfmesh::Mesh::Normal& n = src.faceNormals[i];
			dst.faceNormals[(MVS::Mesh::FIndex)i] = MVS::Mesh::Normal(n[0], n[1], n[2]);
		}
	}
	if constexpr (kRelease)
		ReleaseVector(src.faceNormals);

	dst.faceTexcoords.resize((MVS::Mesh::FIndex)src.faceTexcoords.size());
	for (size_t i = 0; i < src.faceTexcoords.size(); ++i) {
		const halfmesh::Mesh::TexCoord& t = src.faceTexcoords[i];
		dst.faceTexcoords[(MVS::Mesh::FIndex)i] = MVS::Mesh::TexCoord(t[0], t[1]);
	}
	if constexpr (kRelease)
		ReleaseVector(src.faceTexcoords);

	// halfmesh::Mesh::MAX_TEXBLOBS is defined as openMVS's own limit: its texture
	// array is a cList indexed by uint8 TexIndex, so a 256th entry would wrap the
	// size field to 0 while the index loop below still ran the full range — an
	// out-of-bounds heap write.  halfmesh caps its own loaders at the same bound,
	// so this can now only trip on a mesh assembled in memory past it.
	if (src.texturesDiffuse.size() > halfmesh::Mesh::MAX_TEXBLOBS) {
		REPORT_WARNING("ConvertMesh: {} texture blobs exceed openMVS's uint8 texture-index limit ({}); "
		               "converting geometry and UVs WITHOUT textures",
		               src.texturesDiffuse.size(), halfmesh::Mesh::MAX_TEXBLOBS);
		dst.faceTexindices.Release();
		dst.texturesDiffuse.Release();
		if constexpr (kRelease) {
			ReleaseVector(src.faceTexblobs);
			ReleaseVector(src.texturesDiffuse);
		}
		return;
	}

	// Both sides are uint8 per face (see the note in the other direction).
	dst.faceTexindices.resize((MVS::Mesh::FIndex)src.faceTexblobs.size());
	for (size_t i = 0; i < src.faceTexblobs.size(); ++i)
		dst.faceTexindices[(MVS::Mesh::FIndex)i] = (MVS::Mesh::TexIndex)src.faceTexblobs[i];
	if constexpr (kRelease)
		ReleaseVector(src.faceTexblobs);

	// Image8U3 is a SEACAVE typedef (Common/Types.h), not a true member of
	// namespace MVS -- it's only visible unqualified here because
	// MVS/Common.h has a global `using namespace SEACAVE;`. Every openMVS
	// source spells it unqualified; `MVS::Image8U3` does not compile.
	dst.texturesDiffuse.resize((MVS::Mesh::TexIndex)src.texturesDiffuse.size());
	for (size_t i = 0; i < src.texturesDiffuse.size(); ++i)
		dst.texturesDiffuse[(MVS::Mesh::TexIndex)i] = Image8U3(src.texturesDiffuse[i]);
	if constexpr (kRelease)
		ReleaseVector(src.texturesDiffuse);
}

} // namespace detail

// Convert an openMVS mesh into a halfmesh mesh (geometry, per-vertex colors,
// per-face normals, per-corner UVs, and diffuse textures).  Only the fields
// halfmesh understands are transferred; openMVS extras (per-vertex normals,
// vertex/face adjacency caches, octree, etc.) are not copied — halfmesh::Mesh has
// no per-vertex-normal storage, and the adjacency caches are derived data: call
// the relevant halfmesh List*() builders afterwards if you need them.
inline void ConvertMesh(const MVS::Mesh& src, halfmesh::Mesh& dst)
{
	detail::ConvertFromMVS<false>(src, dst);
}

// Same conversion, but consuming: every source array is freed as soon as it has
// been copied, so a large mesh is never resident in both representations at
// once.  `src` is left empty — a valid mesh carrying no geometry.
//
//     halfmesh::ConvertMesh(std::move(mvsMesh), hm);
//
// Spelling this as an rvalue overload rather than a bool flag means the
// destructive version can never be selected by accident, and the std::move at
// the call site is the documentation.
inline void ConvertMesh(MVS::Mesh&& src, halfmesh::Mesh& dst)
{
	detail::ConvertFromMVS<true>(src, dst);
}

// Convert a halfmesh mesh into an openMVS mesh (geometry, per-vertex colors,
// per-face normals, per-corner UVs, and diffuse textures).
inline void ConvertMesh(const halfmesh::Mesh& src, MVS::Mesh& dst)
{
	detail::ConvertToMVS<false>(src, dst);
}

// Consuming counterpart — see the note on ConvertMesh(MVS::Mesh&&, ...).  This
// direction also drops the half-edge structure and the incident-face cache,
// which together usually outweigh the geometry itself.
inline void ConvertMesh(halfmesh::Mesh&& src, MVS::Mesh& dst)
{
	detail::ConvertToMVS<true>(src, dst);
}

} // namespace halfmesh

#endif // __has_include(<MVS/Mesh.h>)
