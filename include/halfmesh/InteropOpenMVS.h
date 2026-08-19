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
//     #include <MVS/Mesh.h>                 // openMVS first (defines MVS::Mesh)
//     #include <halfmesh/InteropOpenMVS.h>
//     ...
//     MVS::Mesh mvsMesh = ...;
//     halfmesh::Mesh hm;
//     halfmesh::ConvertMesh(mvsMesh, hm);   // MVS  -> halfmesh
//     ...
//     MVS::Mesh out;
//     halfmesh::ConvertMesh(hm, out);       // halfmesh -> MVS

#if __has_include(<MVS/Mesh.h>)

	#include <MVS/Mesh.h>
	#include <halfmesh/Mesh.h>
	#include <halfmesh/Util/Log.h> // REPORT_WARNING (texture-count cap below)

	#include <limits>

namespace halfmesh {

// Convert an openMVS mesh into a halfmesh mesh (geometry, per-vertex colors,
// per-face normals, per-corner UVs, and diffuse textures).  Only the fields
// halfmesh understands are transferred; openMVS extras (per-vertex normals,
// vertex/face adjacency caches, octree, etc.) are not copied — halfmesh::Mesh has
// no per-vertex-normal storage, and the adjacency caches are derived data: call
// the relevant halfmesh List*() builders afterwards if you need them.
inline void ConvertMesh(const MVS::Mesh& src, halfmesh::Mesh& dst)
{
	dst = halfmesh::Mesh{};

	dst.vertices.resize(src.vertices.size());
	for (size_t i = 0; i < src.vertices.size(); ++i) {
		const MVS::Mesh::Vertex& v = src.vertices[(MVS::Mesh::VIndex)i];
		dst.vertices[i] = halfmesh::Mesh::Vertex(v.x, v.y, v.z);
	}

	dst.faces.resize(src.faces.size());
	for (size_t i = 0; i < src.faces.size(); ++i) {
		const MVS::Mesh::Face& f = src.faces[(MVS::Mesh::FIndex)i];
		dst.faces[i] = halfmesh::Mesh::Face(f.x, f.y, f.z);
	}

	// Per-vertex colors: both sides are 3x uint8 with blue in the first byte under
	// the default openMVS build; copy by channel name so a _COLORMODE_RGB build of
	// openMVS still maps each channel to halfmesh's fixed BGR element order.
	// halfmesh's contract for optional per-element arrays is "empty or exactly
	// sized" (its swap-pop removal paths index them whenever non-empty, and
	// Mesh::Load clears mismatched arrays -- see src/MeshIO.cpp); only transfer a
	// partially-populated source array when it is full-length, otherwise leave
	// the destination empty (it already is, from the reset above).
	if (src.vertexColors.size() == src.vertices.size()) {
		dst.vertexColors.resize(src.vertexColors.size());
		for (size_t i = 0; i < src.vertexColors.size(); ++i) {
			const MVS::Mesh::Color& c = src.vertexColors[(MVS::Mesh::VIndex)i];
			dst.vertexColors[i] = halfmesh::Mesh::Pixel(c.b, c.g, c.r);
		}
	}

	// Face normals are transferred as-is (see the size guard above) and trusted
	// as fresh by halfmesh, whose own freshness checks are size-only; if the
	// source normals may be stale relative to its geometry, call
	// ComputeFaceNormals() on the destination after conversion.
	if (src.faceNormals.size() == src.faces.size()) {
		dst.faceNormals.resize(src.faceNormals.size());
		for (size_t i = 0; i < src.faceNormals.size(); ++i) {
			const MVS::Mesh::Normal& n = src.faceNormals[(MVS::Mesh::FIndex)i];
			dst.faceNormals[i] = halfmesh::Mesh::Normal(n.x, n.y, n.z);
		}
	}

	// Texture coordinates: openMVS stores them per face-corner (3*faces) or per
	// vertex — the same two layouts halfmesh supports — so copy 1:1.
	dst.faceTexcoords.resize(src.faceTexcoords.size());
	for (size_t i = 0; i < src.faceTexcoords.size(); ++i) {
		const MVS::Mesh::TexCoord& t = src.faceTexcoords[(MVS::Mesh::FIndex)i];
		dst.faceTexcoords[i] = halfmesh::Mesh::TexCoord(t.x, t.y);
	}

	// Per-face texture index (openMVS uint8) -> texture-blob id (halfmesh FIndex).
	dst.faceTexblobs.resize(src.faceTexindices.size());
	for (size_t i = 0; i < src.faceTexindices.size(); ++i)
		dst.faceTexblobs[i] = (halfmesh::Mesh::FIndex)src.faceTexindices[(MVS::Mesh::FIndex)i];

	// Diffuse textures: both sides are CV_8UC3 (BGR) cv::Mat under the hood, so a
	// typed copy converts the element type (cv::Vec3b <-> halfmesh::Pixel) safely.
	dst.texturesDiffuse.resize(src.texturesDiffuse.size());
	for (size_t i = 0; i < src.texturesDiffuse.size(); ++i)
		dst.texturesDiffuse[i] = halfmesh::Mesh::Image3u(src.texturesDiffuse[(MVS::Mesh::TexIndex)i]);
}

// Convert a halfmesh mesh into an openMVS mesh (geometry, per-vertex colors,
// per-face normals, per-corner UVs, and diffuse textures).
inline void ConvertMesh(const halfmesh::Mesh& src, MVS::Mesh& dst)
{
	dst.Release();

	dst.vertices.resize((MVS::Mesh::VIndex)src.vertices.size());
	for (size_t i = 0; i < src.vertices.size(); ++i) {
		const halfmesh::Mesh::Vertex& v = src.vertices[i];
		dst.vertices[(MVS::Mesh::VIndex)i] = MVS::Mesh::Vertex(v[0], v[1], v[2]);
	}

	dst.faces.resize((MVS::Mesh::FIndex)src.faces.size());
	for (size_t i = 0; i < src.faces.size(); ++i) {
		const halfmesh::Mesh::Face& f = src.faces[i];
		dst.faces[(MVS::Mesh::FIndex)i] = MVS::Mesh::Face(f[0], f[1], f[2]);
	}

	// Per-vertex colors: assign by channel name (see the MVS -> halfmesh note).
	// Only transfer when full-length -- see the size-invariant note above; a
	// partially-populated source array leaves the destination empty (it already
	// is, from the Release() above).
	if (src.vertexColors.size() == src.vertices.size()) {
		dst.vertexColors.resize((MVS::Mesh::VIndex)src.vertexColors.size());
		for (size_t i = 0; i < src.vertexColors.size(); ++i) {
			const halfmesh::Mesh::Pixel& p = src.vertexColors[i];
			MVS::Mesh::Color& c = dst.vertexColors[(MVS::Mesh::VIndex)i];
			c.b = p[0];
			c.g = p[1];
			c.r = p[2];
		}
	}

	// Face normals are transferred as-is (see the size guard above) and trusted
	// as fresh by halfmesh, whose own freshness checks are size-only; if the
	// source normals may be stale relative to its geometry, call
	// ComputeFaceNormals() on the destination after conversion.
	if (src.faceNormals.size() == src.faces.size()) {
		dst.faceNormals.resize((MVS::Mesh::FIndex)src.faceNormals.size());
		for (size_t i = 0; i < src.faceNormals.size(); ++i) {
			const halfmesh::Mesh::Normal& n = src.faceNormals[i];
			dst.faceNormals[(MVS::Mesh::FIndex)i] = MVS::Mesh::Normal(n[0], n[1], n[2]);
		}
	}

	dst.faceTexcoords.resize((MVS::Mesh::FIndex)src.faceTexcoords.size());
	for (size_t i = 0; i < src.faceTexcoords.size(); ++i) {
		const halfmesh::Mesh::TexCoord& t = src.faceTexcoords[i];
		dst.faceTexcoords[(MVS::Mesh::FIndex)i] = MVS::Mesh::TexCoord(t[0], t[1]);
	}

	// MVS::Mesh::TexIndex is uint8, and every size/index below passes through a
	// (TexIndex) cast: with more than 255 blobs the resize truncates to
	// size mod 256 while the index loop still runs the full range — an
	// out-of-bounds heap write — and blob ids >= 256 silently remap to the
	// wrong texture. 255 is the largest size for which BOTH the resize casts
	// and every id stay faithful. Refuse to convert the texture set beyond it
	// (geometry and UVs are unaffected by the cap).
	constexpr size_t maxTextures = std::numeric_limits<MVS::Mesh::TexIndex>::max(); // 255
	if (src.texturesDiffuse.size() > maxTextures) {
		REPORT_WARNING("ConvertMesh: {} texture blobs exceed openMVS's uint8 texture-index limit ({}); "
		               "converting geometry and UVs WITHOUT textures",
		               src.texturesDiffuse.size(), maxTextures);
		dst.faceTexindices.resize(0);
		dst.texturesDiffuse.resize(0);
		return;
	}

	dst.faceTexindices.resize((MVS::Mesh::FIndex)src.faceTexblobs.size());
	for (size_t i = 0; i < src.faceTexblobs.size(); ++i)
		dst.faceTexindices[(MVS::Mesh::FIndex)i] = (MVS::Mesh::TexIndex)src.faceTexblobs[i];

	// Image8U3 is a SEACAVE typedef (Common/Types.h), not a true member of
	// namespace MVS -- it's only visible unqualified here because
	// MVS/Common.h has a global `using namespace SEACAVE;`. Every openMVS
	// source spells it unqualified; `MVS::Image8U3` does not compile.
	dst.texturesDiffuse.resize((MVS::Mesh::TexIndex)src.texturesDiffuse.size());
	for (size_t i = 0; i < src.texturesDiffuse.size(); ++i)
		dst.texturesDiffuse[(MVS::Mesh::TexIndex)i] = Image8U3(src.texturesDiffuse[i]);
}

} // namespace halfmesh

#endif // __has_include(<MVS/Mesh.h>)
