/*
* MeshIO.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Implements: Mesh I/O — PLY/glTF load/save, texture handling, texcoord
// utilities, seam export, and textured-mesh transforms.

#include <halfmesh/Mesh.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tinyply.h>
#include <tiny_gltf.h>
#include <BS_thread_pool.hpp>

#include "ParallelFor.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stack>
#include <cstring>
#include <vector>

namespace halfmesh {

// Account for pixel center convention: OpenCV and DirectX 9 define the center
// of a pixel at integer coordinates (center at (0, 0), top-left at (-0.5, -0.5));
// DirectX 10+, OpenGL, and Vulkan define it at half coordinates (center at
// (0.5, 0.5), top-left at (0, 0)).
static const Mesh::TexCoord halfPixel(0.5f, 0.5f);

namespace {

using detail::ParallelForPool;

// Texture size for the given blob, falling back to 1x1 when the slot is missing
// OR the image failed to load (empty Mat).  Without this a missing/unreadable
// sidecar texture yields a 0x0 size and the UV converters divide/multiply by
// zero, collapsing every coordinate to a single point.
cv::Size TextureSizeOrUnit(const std::vector<Image3u>& textures, Mesh::FIndex idxTexblob)
{
	cv::Size sz = idxTexblob < textures.size() ? textures[idxTexblob].size() : cv::Size(1, 1);
	if (sz.area() <= 0)
		sz = cv::Size(1, 1);
	return sz;
}

// --- PLY buffer type conversion -------------------------------------------
// tinyply returns each property buffer in the FILE's native scalar type, not
// the requested one.  A blind memcpy into our fixed-layout arrays corrupts
// (and overflows) whenever the file type differs (e.g. double positions ->
// float slots).  These helpers keep the memcpy fast path for the expected type
// and narrow/widen the common alternatives, reporting anything unsupported.
enum class PlyCopy { Fast,
	                 Narrowed,
	                 Unsupported };

// Copy n scalars from a tinyply buffer into a float destination.
PlyCopy PlyCopyToFloat(const tinyply::PlyData& d, float* dst, size_t n)
{
	const uint8_t* src = d.buffer.get_const();
	switch (d.t) {
	case tinyply::Type::FLOAT32:
		std::memcpy(dst, src, n * sizeof(float));
		return PlyCopy::Fast;
	case tinyply::Type::FLOAT64: {
		const double* s = reinterpret_cast<const double*>(src);
		for (size_t i = 0; i < n; ++i)
			dst[i] = static_cast<float>(s[i]);
		return PlyCopy::Narrowed;
	}
	default:
		return PlyCopy::Unsupported;
	}
}

// Copy n scalars from a tinyply buffer into 32-bit indices (widening as needed).
PlyCopy PlyCopyToIndex(const tinyply::PlyData& d, uint32_t* dst, size_t n)
{
	const uint8_t* src = d.buffer.get_const();
	switch (d.t) {
	case tinyply::Type::UINT32:
	case tinyply::Type::INT32:
		std::memcpy(dst, src, n * sizeof(uint32_t));
		return PlyCopy::Fast;
	case tinyply::Type::UINT16:
		for (size_t i = 0; i < n; ++i)
			dst[i] = reinterpret_cast<const uint16_t*>(src)[i];
		return PlyCopy::Narrowed;
	case tinyply::Type::INT16:
		for (size_t i = 0; i < n; ++i)
			dst[i] = static_cast<uint32_t>(reinterpret_cast<const int16_t*>(src)[i]);
		return PlyCopy::Narrowed;
	case tinyply::Type::UINT8:
		for (size_t i = 0; i < n; ++i)
			dst[i] = reinterpret_cast<const uint8_t*>(src)[i];
		return PlyCopy::Narrowed;
	case tinyply::Type::INT8:
		for (size_t i = 0; i < n; ++i)
			dst[i] = static_cast<uint32_t>(reinterpret_cast<const int8_t*>(src)[i]);
		return PlyCopy::Narrowed;
	default:
		return PlyCopy::Unsupported;
	}
}

// Read the i-th value of a single-scalar PLY color channel as uint8.  Standard
// PLY colors are uchar (the fast path); other types are narrowed heuristically.
uint8_t PlyChannelU8(const tinyply::PlyData& d, size_t i, bool& narrowed)
{
	const uint8_t* src = d.buffer.get_const();
	switch (d.t) {
	case tinyply::Type::UINT8:
		return reinterpret_cast<const uint8_t*>(src)[i];
	case tinyply::Type::INT8:
		narrowed = true;
		return static_cast<uint8_t>(std::clamp<int>(reinterpret_cast<const int8_t*>(src)[i], 0, 255));
	case tinyply::Type::UINT16:
		narrowed = true;
		return static_cast<uint8_t>(reinterpret_cast<const uint16_t*>(src)[i] >> 8);
	case tinyply::Type::INT16:
		narrowed = true;
		return static_cast<uint8_t>(std::clamp<int>(reinterpret_cast<const int16_t*>(src)[i] >> 8, 0, 255));
	case tinyply::Type::FLOAT32:
		narrowed = true;
		return static_cast<uint8_t>(std::clamp(reinterpret_cast<const float*>(src)[i] * 255.f + 0.5f, 0.f, 255.f));
	case tinyply::Type::FLOAT64:
		narrowed = true;
		return static_cast<uint8_t>(std::clamp(reinterpret_cast<const double*>(src)[i] * 255.0 + 0.5, 0.0, 255.0));
	default:
		narrowed = true;
		return 0;
	}
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ToTexCoordPerVertex
// ---------------------------------------------------------------------------
Mesh Mesh::ToTexCoordPerVertex() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTexture());
	const size_t newNumVertices = vertices.size() * 4 / 3;
	Mesh mesh;
	mesh.vertices.reserve(newNumVertices);
	mesh.vertices = vertices;
	mesh.faces.resize(faces.size());
	mesh.faceTexcoords.reserve(newNumVertices);
	mesh.faceTexcoords.resize(vertices.size());
	mesh.faceTexblobs.reserve(newNumVertices);
	mesh.faceTexblobs.resize(vertices.size(), math::NO_ID);
	// mapVertices[i] is the next vertex in the linked list of vertices
	// sharing the same position but potentially different texture coordinates
	std::vector<VIndex> mapVertices(vertices.size(), math::NO_ID);
	FOREACHIDX (FIndex, idxF, faces) {
		const Face& face = faces[idxF];
		const FIndex tb = FTexblob(idxF);
		Face& newFace = mesh.faces[idxF];
		for (int i = 0; i < 3; ++i) {
			const TexCoord& tc = faceTexcoords[idxF * 3 + i];
			VIndex idxV(face[i]);
			while (true) {
				VIndex& idxVT = mapVertices[idxV];
				if (idxVT == math::NO_ID) {
					// vertex not seen yet, so use it directly
					newFace[i] = idxVT = idxV;
					mesh.faceTexblobs[idxV] = tb;
					mesh.faceTexcoords[idxV] = tc;
					break;
				}
				// vertex already seen in a previous face;
				// check if they share also the texture coordinates
				if (mesh.faceTexblobs[idxV] == tb && mesh.faceTexcoords[idxV] == tc) {
					// same texture coordinates, use it
					newFace[i] = idxV;
					break;
				}
				if (idxVT == idxV) {
					// duplicate vertex
					mapVertices.emplace_back(newFace[i] = idxVT = mesh.vertices.size());
					mesh.vertices.emplace_back(vertices[face[i]]);
					mesh.faceTexblobs.emplace_back(tb);
					mesh.faceTexcoords.emplace_back(tc);
					break;
				}
				// continue with the next linked vertex;
				// all share the same position, but different texture coordinates
				idxV = idxVT;
			}
		}
	}
	if (texturesDiffuse.size() == 1)
		mesh.faceTexblobs = std::vector<FIndex>();
	mesh.texturesDiffuse = texturesDiffuse;
	return mesh;
}

// ---------------------------------------------------------------------------
// ToOneMeshPerTexblob
// ---------------------------------------------------------------------------
std::vector<Mesh> Mesh::ToOneMeshPerTexblob() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTexture());
	ASSERT(vertices.size() == faceTexcoords.size());
	ASSERT(faceTexblobs.empty() || vertices.size() == faceTexblobs.size());
	ASSERT((texturesDiffuse.size() == 1 && faceTexblobs.empty()) || texturesDiffuse.size() == *std::max_element(faceTexblobs.begin(), faceTexblobs.end()) + 1);
	const size_t numMeshes(texturesDiffuse.size());
	if (numMeshes == 1)
		return std::vector<Mesh>{*this};
	std::vector<Mesh> meshes(numMeshes);
	// Per-vertex UV layout guarantees each source vertex is referenced only by
	// faces of its single blob (faceTexblobs is vertex-indexed, asserted below),
	// so one shared remap table suffices: no two blobs ever compete for the same
	// slot.  This replaces the O(B*V) per-blob NO_ID-filled tables with a single
	// O(V) table while keeping per-blob vertex ordering (first-encounter) and face
	// indices byte-identical.
	std::vector<VIndex> mapVertices(vertices.size(), math::NO_ID);
	for (const Face& face : faces) {
		ASSERT(faceTexblobs[face[0]] == faceTexblobs[face[1]] && faceTexblobs[face[1]] == faceTexblobs[face[2]]);
		const FIndex tb = faceTexblobs[face[0]];
		Mesh& mesh = meshes[tb];
		Face newFace;
		for (int v = 0; v < 3; ++v) {
			const VIndex idxV = face[v];
			VIndex& idxVT = mapVertices[idxV];
			if (idxVT == math::NO_ID) {
				// vertex not seen yet, fill it
				idxVT = mesh.vertices.size();
				mesh.vertices.emplace_back(vertices[idxV]);
				mesh.faceTexcoords.emplace_back(faceTexcoords[idxV]);
			}
			newFace[v] = idxVT;
		}
		mesh.faces.emplace_back(newFace);
	}
	FOREACH (i, meshes)
		meshes[i].texturesDiffuse.emplace_back(texturesDiffuse[i]);
	return meshes;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Load — dispatch by extension (mirrors Save)
// ---------------------------------------------------------------------------
bool Mesh::Load(const std::string& fileName)
{
	const std::string::size_type extPos = fileName.rfind('.');
	std::string ext(extPos != fileName.npos ? fileName.substr(extPos) : "");
	std::transform(ext.begin(), ext.end(), ext.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	if (ext == ".glb" || ext == ".gltf")
		return LoadGLTF(fileName);
	// default to PLY (also handles .ply and the extension-less case)
	return LoadPLY(fileName);
}

// ---------------------------------------------------------------------------
// LoadPLY
// ---------------------------------------------------------------------------
bool Mesh::LoadPLY(const std::string& fileName)
{
	std::ifstream fileStream(fileName, std::ios::binary);
	if (fileStream.fail()) {
		REPORT_WARNING("failed to open {}", fileName);
		return false;
	}

	tinyply::PlyFile file;
	try {
		// parse_header throws on non-PLY input (e.g. a .glb passed by mistake);
		// fail gracefully instead of letting the exception abort the process.
		file.parse_header(fileStream);
	} catch (const std::exception&) {
		REPORT_WARNING("not a valid PLY file (cannot parse header): {}", fileName);
		return false;
	}

	std::shared_ptr<tinyply::PlyData> vertices, faces, texcoords, texnumber;
	std::shared_ptr<tinyply::PlyData> red, green, blue;

	try {
		vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
	} catch (const std::exception&) {
	}
	// Colors: request each channel SEPARATELY.  tinyply fills a grouped-request
	// buffer in the FILE's property order (not the requested order), so a single
	// {blue,green,red} request silently swaps R/B on standard red,green,blue
	// files.  Per-channel requests each own their buffer, making the load
	// order-independent; we assemble the BGR Pixel explicitly below.
	const auto requestChannel = [&](const char* lname, const char* sname) -> std::shared_ptr<tinyply::PlyData> {
		std::shared_ptr<tinyply::PlyData> d;
		try {
			d = file.request_properties_from_element("vertex", {lname});
		} catch (const std::exception&) {
		}
		if (!d) {
			try {
				d = file.request_properties_from_element("vertex", {sname});
			} catch (const std::exception&) {
			}
		}
		return d;
	};
	red = requestChannel("red", "r");
	green = requestChannel("green", "g");
	blue = requestChannel("blue", "b");
	// NOTE: vertex normals (nx/ny/nz) are intentionally NOT requested: they were
	// only ever used to print a count and then discarded (tinyply would allocate
	// and copy a count*12-byte buffer for nothing).  The count is read from the
	// header metadata below instead.

#if HALFMESH_TRIS
	const uint32_t faceListHint = 3;
#else
	const uint32_t faceListHint = 0;
#endif
	try {
		faces = file.request_properties_from_element("face", {"vertex_indices"}, faceListHint);
	} catch (const std::exception&) {
	}
	try {
		texcoords = file.request_properties_from_element("face", {"texcoord"}, faceListHint * 2);
	} catch (const std::exception&) {
	}
	try {
		texnumber = file.request_properties_from_element("face", {"texnumber"});
	} catch (const std::exception&) {
	}

	try {
		// read() throws on truncated/malformed payloads and on face lists that
		// differ from the triangle size hint (e.g. quad PLYs) — same contract
		// as parse_header above: warn and fail, never abort the process.
		file.read(fileStream);
	} catch (const std::exception& e) {
		REPORT_WARNING("failed to read PLY data from {}: {}", fileName, e.what());
		return false;
	}
	if (!vertices || vertices->count <= 0 || !faces || faces->count <= 0) {
		REPORT_WARNING("invalid mesh file {}", fileName);
		return false;
	}
	// Report available vertex normals from header metadata (not buffered).
	for (const tinyply::PlyElement& el : file.get_elements()) {
		if (el.name != "vertex")
			continue;
		for (const tinyply::PlyProperty& p : el.properties)
			if (p.name == "nx") {
				REPORT_STATUS("Available {} vertex normals", el.size);
				break;
			}
	}

	// Convert to our mesh type.  tinyply hands back each buffer in the FILE's
	// native scalar type; narrow/widen where it differs from our fixed layout
	// (a blind memcpy of e.g. double positions into float slots corrupts the
	// mesh and overflows the heap).
	halfMesh.Clear();
	vertexFaces.clear();
	bool narrowed = false;
	{
		this->vertices.resize(vertices->count);
		const PlyCopy r = PlyCopyToFloat(*vertices, reinterpret_cast<float*>(this->vertices.data()), vertices->count * 3);
		if (r == PlyCopy::Unsupported) {
			REPORT_WARNING("unsupported vertex position type in {}", fileName);
			return false;
		}
		narrowed = narrowed || r == PlyCopy::Narrowed;
	}
	if (red && green && blue && red->count == vertices->count && green->count == vertices->count && blue->count == vertices->count) {
		vertexColors.resize(vertices->count);
		for (size_t i = 0; i < vertices->count; ++i)
			vertexColors[i] = Pixel(PlyChannelU8(*blue, i, narrowed),
			                        PlyChannelU8(*green, i, narrowed),
			                        PlyChannelU8(*red, i, narrowed));
	}
	{
		this->faces.resize(faces->count);
#if HALFMESH_TRIS
		const PlyCopy r = PlyCopyToIndex(*faces, reinterpret_cast<uint32_t*>(this->faces.data()), faces->count * 3);
		if (r == PlyCopy::Unsupported) {
			REPORT_WARNING("unsupported face index type in {}", fileName);
			return false;
		}
		narrowed = narrowed || r == PlyCopy::Narrowed;
#else
		// Polygon (non-triangle) build: widen every index into a scratch buffer
		// honoring the file's scalar type, then map row-major.
		const auto plyTypeSize = [](tinyply::Type t) -> size_t {
			switch (t) {
			case tinyply::Type::INT8:
			case tinyply::Type::UINT8: return 1;
			case tinyply::Type::INT16:
			case tinyply::Type::UINT16: return 2;
			case tinyply::Type::INT32:
			case tinyply::Type::UINT32:
			case tinyply::Type::FLOAT32: return 4;
			case tinyply::Type::FLOAT64: return 8;
			default: return 0;
			}
		};
		const size_t tsize = plyTypeSize(faces->t);
		if (tsize == 0) {
			REPORT_WARNING("unsupported face index type in {}", fileName);
			return false;
		}
		const size_t totalScalars = faces->buffer.size_bytes() / tsize;
		const size_t k = totalScalars / faces->count;
		std::vector<uint32_t> widened(totalScalars);
		const PlyCopy r = PlyCopyToIndex(*faces, widened.data(), totalScalars);
		if (r == PlyCopy::Unsupported) {
			REPORT_WARNING("unsupported face index type in {}", fileName);
			return false;
		}
		narrowed = narrowed || r == PlyCopy::Narrowed;
		Eigen::Map<const Eigen::Matrix<VIndex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
		    widened.data(), faces->count, k);
		FOREACH (i, this->faces) {
			this->faces[i] = map.row(i);
		}
#endif
	}
	if (this->vertices.empty() || this->faces.empty())
		return false;
	if (texcoords) {
		this->faceTexcoords.resize(texcoords->count * 3);
		const PlyCopy r = PlyCopyToFloat(*texcoords, reinterpret_cast<float*>(this->faceTexcoords.data()), texcoords->count * 6);
		if (r == PlyCopy::Unsupported) {
			REPORT_WARNING("unsupported texcoord type in {}; ignoring texture coordinates", fileName);
			this->faceTexcoords.clear();
		} else {
			narrowed = narrowed || r == PlyCopy::Narrowed;
			// Resolve sidecar texture paths first, then decode them in parallel
			// (independent files -> independent cv::Mat outputs), assigning by
			// index so texturesDiffuse order is deterministic.
			// error_code overload: canonical() throws on filesystem races (file
			// unlinked after open); fall back to the given path un-canonicalized.
			std::error_code canonEc;
			std::filesystem::path canon =
			    std::filesystem::canonical(std::filesystem::path(fileName), canonEc);
			if (canonEc)
				canon = std::filesystem::path(fileName);
			const std::filesystem::path path(canon.parent_path());
			std::vector<std::string> texFileNames;
			for (const std::string& comment : file.get_comments()) {
				if (comment.compare(0, 12, "TextureFile ") == 0) {
					std::filesystem::path texFileName(comment.substr(12));
					if (!texFileName.is_absolute())
						texFileName = path / texFileName;
					texFileNames.emplace_back(texFileName.string());
				}
			}
			this->texturesDiffuse.resize(texFileNames.size());
			{
				BS::light_thread_pool pool;
				ParallelForPool(pool, texFileNames.size(), [&](std::size_t i) {
					// cv::imread() can throw (e.g. a truncated/corrupt sidecar);
					// under ParallelForPool's rethrow-after-wait() contract that
					// would otherwise escape Mesh::Load's bool API. Catch here,
					// per item, so one bad texture cannot abort its siblings, and
					// leave the slot at its default-constructed empty cv::Mat --
					// the same "missing/unreadable texture" 1x1-fallback degrade
					// already used for a file that fails to open, reported by
					// the post-loop check below.
					try {
						this->texturesDiffuse[i] = cv::imread(texFileNames[i]);
					} catch (const std::exception&) {
						this->texturesDiffuse[i] = cv::Mat();
					}
				});
			}
			FOREACH (i, this->texturesDiffuse) {
				if (this->texturesDiffuse[i].empty())
					REPORT_WARNING("missing/unreadable texture '{}' for {}; UVs kept via 1x1 fallback size",
					               texFileNames[i], fileName);
			}
			this->faceTexcoords = FTexcoordsUnNormalizeFlipY();
		}
	}
	if (texnumber) {
		this->faceTexblobs.resize(texnumber->count);
		const PlyCopy r = PlyCopyToIndex(*texnumber, reinterpret_cast<uint32_t*>(this->faceTexblobs.data()), texnumber->count);
		if (r == PlyCopy::Unsupported) {
			REPORT_WARNING("unsupported texnumber type in {}; ignoring texture ids", fileName);
			this->faceTexblobs.clear();
		} else {
			narrowed = narrowed || r == PlyCopy::Narrowed;
		}
	}
	if (narrowed)
		REPORT_WARNING("{}: non-standard PLY property types were narrowed on load", fileName);
	REPORT_STATUS_NOW("Mesh loaded{}: {}",
	                  texturesDiffuse.empty() ? "" : HALFMESH_FORMAT(" ({} textures)", texturesDiffuse.size()),
	                  fileName);
	return true;
}

// ---------------------------------------------------------------------------
// SavePLY
// ---------------------------------------------------------------------------
bool Mesh::SavePLY(const std::string& fileName, bool binary) const
{
	const_cast<Mesh*>(this)->SyncFaces();
	tinyply::PlyFile file;
	file.add_properties_to_element("vertex", {"x", "y", "z"},
	                               tinyply::Type::FLOAT32, vertices.size(),
	                               reinterpret_cast<const uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
	if (!vertexColors.empty()) {
		file.add_properties_to_element("vertex", {"blue", "green", "red"},
		                               tinyply::Type::UINT8, vertexColors.size(),
		                               reinterpret_cast<const uint8_t*>(vertexColors.data()), tinyply::Type::INVALID, 0);
	}
	std::vector<Normal> vertexNormals;
	if (!faceNormals.empty()) {
		vertexNormals = const_cast<Mesh&>(*this).ComputeVertexNormals();
		file.add_properties_to_element("vertex", {"nx", "ny", "nz"},
		                               tinyply::Type::FLOAT32, vertexNormals.size(),
		                               reinterpret_cast<const uint8_t*>(vertexNormals.data()), tinyply::Type::INVALID, 0);
	}
	file.add_properties_to_element("face", {"vertex_indices"},
	                               tinyply::Type::UINT32, faces.size(),
	                               reinterpret_cast<const uint8_t*>(faces.data()), tinyply::Type::UINT8, 3);
	std::vector<TexCoord> normFaceTexcoords;
	if (HasTextureCoordinates()) {
		normFaceTexcoords = FTexcoordsNormalizeFlipY();
		file.add_properties_to_element("face", {"texcoord"},
		                               tinyply::Type::FLOAT32, faces.size(),
		                               reinterpret_cast<const uint8_t*>(normFaceTexcoords.data()), tinyply::Type::UINT8, 2 * 3);
		if (!faceTexblobs.empty()) {
			file.add_properties_to_element("face", {"texnumber"},
			                               tinyply::Type::INT32 /*UINT32 breaks Meshlab*/, faces.size(),
			                               reinterpret_cast<const uint8_t*>(faceTexblobs.data()), tinyply::Type::INVALID, 0);
		}
		const std::vector<int> codecParams{cv::IMWRITE_JPEG_QUALITY, 95};
		const std::filesystem::path path(
		    std::filesystem::weakly_canonical(std::filesystem::path(fileName)));
		// Append the TextureFile header comments serially (index order) BEFORE the
		// parallel region so header order stays deterministic; then encode/write
		// the independent sidecar JPEGs in parallel across blobs (same codec
		// params => byte-identical files, just off the critical path).
		std::vector<std::string> texFileNames(texturesDiffuse.size());
		FOREACH (i, texturesDiffuse) {
			const std::string texName = HALFMESH_FORMAT("{}_diffuse{:#02d}.jpg",
			                                            path.stem().string(), static_cast<int>(i));
			texFileNames[i] = (path.parent_path() / texName).string();
			file.get_comments().emplace_back("TextureFile " + texName);
		}
		BS::light_thread_pool pool;
		std::mutex failedTexMutex;
		std::vector<std::size_t> failedTexIndices;
		ParallelForPool(pool, texturesDiffuse.size(), [&](std::size_t i) {
			// cv::imwrite() can throw (e.g. an EMPTY cv::Mat, which LoadPLY leaves
			// in place for a missing/unreadable sidecar texture -- see the 1x1
			// fallback above); under ParallelForPool's rethrow-after-wait()
			// contract that would otherwise escape SavePLY's bool API. Its
			// success return is also unchecked below, so both failure modes are
			// folded into the same degrade path here. REPORT_WARNING streams to
			// std::cerr, which is not safe to call concurrently from pool
			// workers, so failures are only recorded here (mutex-guarded) and
			// reported serially, once per texture, after the parallel region --
			// matching LoadPLY's post-loop reporting above. The texture is
			// skipped; geometry still saves.
			bool ok = false;
			try {
				ok = cv::imwrite(texFileNames[i], texturesDiffuse[i], codecParams);
			} catch (const std::exception&) {
				ok = false;
			}
			if (!ok) {
				const std::lock_guard<std::mutex> lock(failedTexMutex);
				failedTexIndices.push_back(i);
			}
		});
		for (const std::size_t i : failedTexIndices)
			REPORT_WARNING("failed to write texture '{}' for {}; texture skipped, geometry still saved",
			               texFileNames[i], fileName);
	}
	std::filebuf fb;
	fb.open(fileName, binary ? std::ios::out | std::ios::binary : std::ios::out);
	if (!fb.is_open()) {
		REPORT_WARNING("Could not open file: {}", fileName);
		return false;
	}
	std::ostream outstream(&fb);
	if (outstream.fail())
		return false;
	file.write(outstream, binary);
	return true;
}

// ---------------------------------------------------------------------------
// LoadGLTF — helpers
// ---------------------------------------------------------------------------
namespace {

// Load a T from a possibly-misaligned buffer position.  glTF accessor byte
// offsets are not guaranteed to satisfy alignof(T) (e.g. an interleaved
// vertex layout can start a FLOAT/UNSIGNED_SHORT accessor at an odd byte);
// dereferencing a reinterpret_cast<const T*> there is UB on strict-alignment
// targets (and trips UBSan's alignment check even on x86).  memcpy sidesteps
// this and compiles to the same load as a direct deref on x86.
template <typename T>
inline T ReadUnaligned(const unsigned char* p)
{
	T v;
	std::memcpy(&v, p, sizeof(T));
	return v;
}

// Decode one component of a glTF accessor element to a double, covering the
// standard component types and de-normalizing integer types when flagged.
double ReadGltfComponent(const unsigned char* p, int componentType, bool normalized)
{
	switch (componentType) {
	case TINYGLTF_COMPONENT_TYPE_FLOAT:
		return static_cast<double>(ReadUnaligned<float>(p));
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
		// 1-byte type: alignof == 1, a direct deref can never be misaligned.
		const double v = *reinterpret_cast<const uint8_t*>(p);
		return normalized ? v / 255.0 : v;
	}
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
		const double v = ReadUnaligned<uint16_t>(p);
		return normalized ? v / 65535.0 : v;
	}
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
		return static_cast<double>(ReadUnaligned<uint32_t>(p));
	case TINYGLTF_COMPONENT_TYPE_BYTE: {
		// 1-byte type: alignof == 1, a direct deref can never be misaligned.
		const double v = *reinterpret_cast<const int8_t*>(p);
		return normalized ? std::max(v / 127.0, -1.0) : v;
	}
	case TINYGLTF_COMPONENT_TYPE_SHORT: {
		const double v = ReadUnaligned<int16_t>(p);
		return normalized ? std::max(v / 32767.0, -1.0) : v;
	}
	default:
		return 0.0;
	}
}

// Resolve an accessor's first byte + element stride, validating every index
// hop (accessor -> bufferView -> buffer).  Returns nullptr on inconsistency.
const unsigned char* AccessorData(const tinygltf::Model& model,
                                  const tinygltf::Accessor& acc, int& strideOut)
{
	if (acc.bufferView < 0 || acc.bufferView >= static_cast<int>(model.bufferViews.size()))
		return nullptr;
	const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
	if (bv.buffer < 0 || bv.buffer >= static_cast<int>(model.buffers.size()))
		return nullptr;
	const int stride = acc.ByteStride(bv);
	if (stride <= 0)
		return nullptr;
	const tinygltf::Buffer& buf = model.buffers[bv.buffer];
	// The bufferView itself must lie within its buffer (guards accessors that
	// would otherwise read adjacent, unrelated bytes past the view's extent).
	if (bv.byteOffset + bv.byteLength > buf.data.size())
		return nullptr;
	if (acc.count > 0) {
		// glTF spec bound: byteOffset + stride*(count-1) + elementSize must fit
		// within the bufferView.  This accepts spec-valid interleaved accessors
		// whose last element ends before a full trailing stride (the old
		// stride*count check over-rejected them and silently dropped geometry).
		const size_t elemSize =
		    static_cast<size_t>(tinygltf::GetNumComponentsInType(acc.type)) * static_cast<size_t>(tinygltf::GetComponentSizeInBytes(acc.componentType));
		const size_t last = acc.byteOffset + static_cast<size_t>(stride) * (acc.count - 1) + elemSize;
		if (last > bv.byteLength)
			return nullptr;
	}
	strideOut = stride;
	return buf.data.data() + bv.byteOffset + acc.byteOffset;
}

// Read a VEC{N} accessor into Eigen float vectors (missing components -> 0).
template <int N>
std::vector<Eigen::Matrix<float, N, 1>> ReadAccessorVec(const tinygltf::Model& model,
                                                        const tinygltf::Accessor& acc)
{
	std::vector<Eigen::Matrix<float, N, 1>> out;
	int stride = 0;
	const unsigned char* base = AccessorData(model, acc, stride);
	if (!base)
		return out;
	const int ncomp = tinygltf::GetNumComponentsInType(acc.type);
	const int csize = tinygltf::GetComponentSizeInBytes(acc.componentType);
	out.resize(acc.count);
	// Fast path for the overwhelmingly common tightly-packed non-normalized
	// FLOAT case: copy each row directly, skipping the per-component switch and
	// the float->double->float round trip (which is bit-exact, so this is
	// value-identical to the generic path below).
	if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && !acc.normalized && ncomp >= N) {
		for (size_t i = 0; i < acc.count; ++i) {
			float e[N];
			std::memcpy(e, base + i * stride, N * sizeof(float));
			Eigen::Matrix<float, N, 1> v;
			for (int c = 0; c < N; ++c)
				v[c] = e[c];
			out[i] = v;
		}
		return out;
	}
	for (size_t i = 0; i < acc.count; ++i) {
		const unsigned char* e = base + i * stride;
		Eigen::Matrix<float, N, 1> v = Eigen::Matrix<float, N, 1>::Zero();
		for (int c = 0; c < N && c < ncomp; ++c)
			v[c] = static_cast<float>(ReadGltfComponent(e + c * csize, acc.componentType, acc.normalized));
		out[i] = v;
	}
	return out;
}

// Read a SCALAR index accessor into 32-bit indices.
std::vector<uint32_t> ReadIndices(const tinygltf::Model& model, const tinygltf::Accessor& acc)
{
	std::vector<uint32_t> out;
	int stride = 0;
	const unsigned char* base = AccessorData(model, acc, stride);
	if (!base)
		return out;
	out.resize(acc.count);
	// Fast paths for the standard tightly-packed index types (value-identical to
	// the generic switch, which round-trips through double): bulk 32-bit copy and
	// a tight 16-bit widen.  Exact for uint16/uint32 (both integral in double).
	if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT && stride == static_cast<int>(sizeof(uint32_t))) {
		std::memcpy(out.data(), base, acc.count * sizeof(uint32_t));
		return out;
	}
	if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
		for (size_t i = 0; i < acc.count; ++i)
			out[i] = ReadUnaligned<uint16_t>(base + i * stride);
		return out;
	}
	for (size_t i = 0; i < acc.count; ++i)
		out[i] = static_cast<uint32_t>(ReadGltfComponent(base + i * stride, acc.componentType, false));
	return out;
}

// Node local transform: explicit column-major matrix, or TRS composition.
Eigen::Matrix4d NodeLocalMatrix(const tinygltf::Node& node)
{
	if (node.matrix.size() == 16) {
		Eigen::Matrix4d m;
		for (int col = 0; col < 4; ++col)
			for (int row = 0; row < 4; ++row)
				m(row, col) = node.matrix[col * 4 + row];
		return m;
	}
	Eigen::Vector3d t(0, 0, 0), s(1, 1, 1);
	Eigen::Quaterniond q(1, 0, 0, 0);
	if (node.translation.size() == 3)
		t = Eigen::Vector3d(node.translation[0], node.translation[1], node.translation[2]);
	if (node.rotation.size() == 4) // glTF quaternion component order is (x, y, z, w)
		q = Eigen::Quaterniond(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
	if (node.scale.size() == 3)
		s = Eigen::Vector3d(node.scale[0], node.scale[1], node.scale[2]);
	Eigen::Affine3d a = Eigen::Affine3d::Identity();
	a.translate(t).rotate(q).scale(s);
	return a.matrix();
}

// Convert a tinygltf (stb-decoded, 8-bit) image to a BGR cv::Mat; empty on failure.
cv::Mat GltfImageToBGR(const tinygltf::Image& gi)
{
	if (gi.image.empty() || gi.width <= 0 || gi.height <= 0 || gi.bits != 8)
		return cv::Mat();
	const cv::Mat src(gi.height, gi.width, CV_8UC(gi.component),
	                  const_cast<unsigned char*>(gi.image.data()));
	cv::Mat bgr;
	switch (gi.component) {
	case 1: cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR); break;
	case 3: cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR); break;
	case 4: cv::cvtColor(src, bgr, cv::COLOR_RGBA2BGR); break;
	default: return cv::Mat();
	}
	return bgr; // cvtColor allocates a fresh buffer, so this outlives `gi`
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// LoadGLTF — import a .glb/.gltf scene into our mesh representation.
//
// glTF stores per-vertex attributes (POSITION, TEXCOORD_0, COLOR_0) indexed by
// a triangle-index buffer, with geometry placed in a node hierarchy.  We flatten
// the node world transforms into the positions, concatenate every triangle
// primitive of every instanced mesh, and expand the per-vertex UVs into our
// per-face-corner `faceTexcoords` (faces.size()*3) layout.  Embedded base-color
// images are decoded (tinygltf links stb_image) and stored as BGR in
// texturesDiffuse; their UVs are converted from glTF-normalized back to our
// absolute-pixel convention (mirror of SaveGLTF's FTexcoordsNormalize()).
// ---------------------------------------------------------------------------
bool Mesh::LoadGLTF(const std::string& fileName)
{
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err, warn;

	std::string::size_type extPos = fileName.rfind('.');
	std::string ext(extPos != fileName.npos ? fileName.substr(extPos) : "");
	std::transform(ext.begin(), ext.end(), ext.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	const bool ok = (ext == ".gltf")
	                    ? loader.LoadASCIIFromFile(&model, &err, &warn, fileName)
	                    : loader.LoadBinaryFromFile(&model, &err, &warn, fileName);
	if (!warn.empty())
		REPORT_WARNING("glTF warning for {}: {}", fileName, warn);
	if (!ok) {
		REPORT_WARNING("failed to load glTF {}{}", fileName, err.empty() ? std::string() : ": " + err);
		return false;
	}

	halfMesh.Clear();
	vertexFaces.clear();
	vertices.clear();
	faces.clear();
	vertexColors.clear();
	faceTexcoords.clear();
	faceTexblobs.clear();
	texturesDiffuse.clear();

	// Flatten the node hierarchy into a list of (mesh, world-matrix) instances.
	struct Instance
	{
		int mesh;
		Eigen::Matrix4d world;
	};
	std::vector<Instance> instances;
	{
		std::vector<std::pair<int, Eigen::Matrix4d>> stack;
		const int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : 0;
		if (!model.scenes.empty() && sceneIdx < static_cast<int>(model.scenes.size()))
			for (int root : model.scenes[sceneIdx].nodes)
				stack.emplace_back(root, Eigen::Matrix4d::Identity());
		while (!stack.empty()) {
			const auto [ni, parent] = stack.back();
			stack.pop_back();
			if (ni < 0 || ni >= static_cast<int>(model.nodes.size()))
				continue;
			const tinygltf::Node& node = model.nodes[ni];
			const Eigen::Matrix4d world = parent * NodeLocalMatrix(node);
			if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size()))
				instances.push_back({node.mesh, world});
			for (int child : node.children)
				stack.emplace_back(child, world);
		}
		// No (usable) scene graph: treat every mesh as an identity-placed instance.
		if (instances.empty())
			for (size_t m = 0; m < model.meshes.size(); ++m)
				instances.push_back({static_cast<int>(m), Eigen::Matrix4d::Identity()});
	}

	bool anyTexcoords = false;
	std::vector<int> texblobOfFace; // texturesDiffuse slot per face (-1 = none)
	std::vector<int> gltfImageToSlot(model.images.size(), -1);

	auto toU8 = [](float v) {
		return static_cast<uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
	};

	for (const Instance& inst : instances) {
		const tinygltf::Mesh& gmesh = model.meshes[inst.mesh];
		for (const tinygltf::Primitive& prim : gmesh.primitives) {
			if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1)
				continue; // only triangle soup is supported
			const auto itPos = prim.attributes.find("POSITION");
			if (itPos == prim.attributes.end())
				continue;
			const std::vector<Eigen::Vector3f> P =
			    ReadAccessorVec<3>(model, model.accessors[itPos->second]);
			if (P.empty())
				continue;
			const VIndex base = static_cast<VIndex>(vertices.size());

			vertices.reserve(vertices.size() + P.size());
			for (const Eigen::Vector3f& p : P) {
				const Eigen::Vector4d w = inst.world * Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
				vertices.emplace_back(static_cast<float>(w.x()), static_cast<float>(w.y()),
				                      static_cast<float>(w.z()));
			}

			std::vector<Eigen::Vector2f> UV;
			if (const auto it = prim.attributes.find("TEXCOORD_0"); it != prim.attributes.end()) {
				UV = ReadAccessorVec<2>(model, model.accessors[it->second]);
				if (UV.size() != P.size())
					UV.clear();
			}

			if (const auto it = prim.attributes.find("COLOR_0"); it != prim.attributes.end()) {
				const std::vector<Eigen::Vector4f> C = ReadAccessorVec<4>(model, model.accessors[it->second]);
				if (C.size() == P.size()) {
					vertexColors.resize(vertices.size());
					for (size_t i = 0; i < C.size(); ++i) // glTF RGBA -> our BGR uint8
						vertexColors[base + i] = Pixel(toU8(C[i].z()), toU8(C[i].y()), toU8(C[i].x()));
				}
			}

			std::vector<uint32_t> idx;
			if (prim.indices >= 0 && prim.indices < static_cast<int>(model.accessors.size()))
				idx = ReadIndices(model, model.accessors[prim.indices]);
			else { // non-indexed: implicit 0,1,2,...
				idx.resize(P.size());
				for (uint32_t i = 0; i < idx.size(); ++i)
					idx[i] = i;
			}

			// File-supplied index values are untrusted: one out-of-range entry
			// would read past UV below (OOB) and plant a wild vertex index in
			// faces — validate before ANY use; skip the primitive instead.
			// (Its already-appended vertices stay as unreferenced points.)
			{
				bool idxOk = true;
				for (const uint32_t iv : idx)
					if (iv >= P.size()) {
						idxOk = false;
						break;
					}
				if (!idxOk) {
					REPORT_WARNING("glTF primitive has out-of-range indices; skipping: {}", fileName);
					continue;
				}
			}

			// Resolve this primitive's base-color image -> texturesDiffuse slot.
			int slot = -1;
			if (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size())) {
				const int tex = model.materials[prim.material].pbrMetallicRoughness.baseColorTexture.index;
				if (tex >= 0 && tex < static_cast<int>(model.textures.size())) {
					const int img = model.textures[tex].source;
					if (img >= 0 && img < static_cast<int>(model.images.size())) {
						if (gltfImageToSlot[img] < 0) {
							cv::Mat mat = GltfImageToBGR(model.images[img]);
							if (!mat.empty()) {
								gltfImageToSlot[img] = static_cast<int>(texturesDiffuse.size());
								texturesDiffuse.emplace_back(mat);
							}
						}
						slot = gltfImageToSlot[img];
					}
				}
			}

			// If this primitive carries UVs, backfill (0,0) for any faces from
			// earlier untextured primitives so the per-corner texcoord array stays
			// aligned with faces regardless of primitive order.  Without this, a
			// UV-less primitive processed first left faceTexcoords short and the
			// whole array was later discarded (all UVs lost).
			if (!UV.empty() && faceTexcoords.size() < faces.size() * 3)
				faceTexcoords.resize(faces.size() * 3, TexCoord(0, 0));

			const size_t ntri = idx.size() / 3;
			faces.reserve(faces.size() + ntri);
			for (size_t t = 0; t < ntri; ++t) {
				const uint32_t i0 = idx[t * 3 + 0], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
				faces.emplace_back(base + i0, base + i1, base + i2);
				if (!UV.empty()) {
					anyTexcoords = true;
					faceTexcoords.emplace_back(UV[i0].x(), UV[i0].y());
					faceTexcoords.emplace_back(UV[i1].x(), UV[i1].y());
					faceTexcoords.emplace_back(UV[i2].x(), UV[i2].y());
				} else if (anyTexcoords) { // keep per-corner array aligned with faces
					faceTexcoords.insert(faceTexcoords.end(), 3, TexCoord(0, 0));
				}
				texblobOfFace.push_back(slot);
			}
		}
	}

	if (vertices.empty() || faces.empty()) {
		REPORT_WARNING("glTF has no triangle geometry: {}", fileName);
		return false;
	}

	// Per-face texture id only when more than one image is actually used.
	int maxSlot = -1;
	for (const int s : texblobOfFace)
		maxSlot = std::max(maxSlot, s);
	if (maxSlot >= 1) {
		faceTexblobs.resize(faces.size());
		FOREACH (f, faces)
			faceTexblobs[f] = texblobOfFace[f] < 0 ? 0 : static_cast<FIndex>(texblobOfFace[f]);
	}

	if (anyTexcoords && faceTexcoords.size() == faces.size() * 3) {
		// Textured: glTF stored (pixel+0.5)/texSize (SaveGLTF's FTexcoordsNormalize),
		// so invert to our absolute-pixel convention.  UV-only atlas/unwrap meshes
		// keep their normalized [0,1] coords (matching SaveGLTF's UV-only path).
		if (!texturesDiffuse.empty())
			faceTexcoords = FTexcoordsUnNormalize();
	} else {
		faceTexcoords.clear();
	}
	if (!vertexColors.empty() && vertexColors.size() != vertices.size())
		vertexColors.clear();

	REPORT_STATUS_NOW("Mesh loaded{}: {}",
	                  texturesDiffuse.empty() ? "" : HALFMESH_FORMAT(" ({} textures)", texturesDiffuse.size()),
	                  fileName);
	return true;
}

// ---------------------------------------------------------------------------
// SaveGLTF — helper
// ---------------------------------------------------------------------------
namespace {
template <typename T>
void ExtendBufferGLTF(const std::vector<T>& src, tinygltf::Buffer& dst,
                      size_t& byteOffset, size_t& byteLength)
{
	byteOffset = dst.data.size();
	byteLength = sizeof(T) * src.size();
	byteLength = ((byteLength + 3) / 4) * 4;
	dst.data.resize(byteOffset + byteLength);
	memcpy(&dst.data[byteOffset], &src[0], byteLength);
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// ToTexCoordPerVertexUVOnly — UV-only variant of ToTexCoordPerVertex.
// Converts per-face-corner texture coordinates (faces.size()*3) to
// per-vertex layout by duplicating vertices at UV seams.  Unlike the full
// ToTexCoordPerVertex() this variant does NOT require texturesDiffuse to be
// present, so it works for atlas/unwrap meshes that have UVs but no image.
// ---------------------------------------------------------------------------
Mesh Mesh::ToTexCoordPerVertexUVOnly() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTextureCoordinates());
	ASSERT(faceTexcoords.size() == faces.size() * 3);
	const size_t newNumVertices = vertices.size() * 4 / 3;
	Mesh mesh;
	mesh.vertices.reserve(newNumVertices);
	mesh.vertices = vertices;
	mesh.faces.resize(faces.size());
	mesh.faceTexcoords.reserve(newNumVertices);
	mesh.faceTexcoords.resize(vertices.size());
	std::vector<VIndex> mapVertices(vertices.size(), math::NO_ID);
	FOREACHIDX (FIndex, idxF, faces) {
		const Face& face = faces[idxF];
		Face& newFace = mesh.faces[idxF];
		for (int i = 0; i < 3; ++i) {
			const TexCoord& tc = faceTexcoords[idxF * 3 + i];
			VIndex idxV(face[i]);
			while (true) {
				VIndex& idxVT = mapVertices[idxV];
				if (idxVT == math::NO_ID) {
					newFace[i] = idxVT = idxV;
					mesh.faceTexcoords[idxV] = tc;
					break;
				}
				if (mesh.faceTexcoords[idxV] == tc) {
					newFace[i] = idxV;
					break;
				}
				if (idxVT == idxV) {
					mapVertices.emplace_back(newFace[i] = idxVT = mesh.vertices.size());
					mesh.vertices.emplace_back(vertices[face[i]]);
					mesh.faceTexcoords.emplace_back(tc);
					break;
				}
				idxV = idxVT;
			}
		}
	}
	return mesh;
}

// ---------------------------------------------------------------------------
// SaveGLTF
// ---------------------------------------------------------------------------
bool Mesh::SaveGLTF(const std::string& fileName, bool binary) const
{
	const_cast<Mesh*>(this)->SyncFaces();
	std::vector<Mesh> meshes;
	if (HasTexture())
		meshes = ToTexCoordPerVertex().ToOneMeshPerTexblob();
	else if (HasTextureCoordinates())
		// UV-only path: convert per-face-corner UVs to per-vertex (no image)
		meshes.emplace_back(ToTexCoordPerVertexUVOnly());
	else
		meshes.emplace_back(*this);
	const bool doubleSided = true;
	tinygltf::Model gltfModel;
	tinygltf::Scene gltfScene;
	gltfScene.name = "scene";
	tinygltf::Mesh gltfMesh;
	gltfMesh.name = "mesh";
	tinygltf::Buffer gltfBuffer;

	// assign one-mesh properties per primitive
	for (const Mesh& mesh : meshes) {
		tinygltf::Primitive primitive;

		{
			static_assert(3 * sizeof(Vertex::Scalar) == sizeof(Vertex),
			              "Vertex expected to be continuous");
			primitive.attributes["POSITION"] = gltfModel.accessors.size();
			tinygltf::Accessor vertexPositionAccessor;
			vertexPositionAccessor.name = "vertex_position_accessor";
			vertexPositionAccessor.bufferView = gltfModel.bufferViews.size();
			vertexPositionAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
			vertexPositionAccessor.count = mesh.vertices.size();
			vertexPositionAccessor.type = TINYGLTF_TYPE_VEC3;
			Eigen::AlignedBox3f bb;
			for (const Vertex& v : mesh.vertices)
				bb.extend(v);
			const Eigen::Vector3f bbMin = bb.min();
			const Eigen::Vector3f bbMax = bb.max();
			vertexPositionAccessor.minValues = {bbMin.x(), bbMin.y(), bbMin.z()};
			vertexPositionAccessor.maxValues = {bbMax.x(), bbMax.y(), bbMax.z()};
			gltfModel.accessors.emplace_back(std::move(vertexPositionAccessor));

			tinygltf::BufferView vertexPositionBufferView;
			vertexPositionBufferView.name = "vertex_position_buffer_view";
			vertexPositionBufferView.buffer = gltfModel.buffers.size();
			ExtendBufferGLTF(mesh.vertices, gltfBuffer,
			                 vertexPositionBufferView.byteOffset,
			                 vertexPositionBufferView.byteLength);
			gltfModel.bufferViews.emplace_back(std::move(vertexPositionBufferView));
		}

		primitive.material = gltfModel.materials.size();
		tinygltf::Material material;
		material.name = "material";
		material.doubleSided = doubleSided;
		if (HasTexture()) {
			// Textured path: full material with embedded image + TEXCOORD_0
			material.emissiveFactor = std::vector<double>{0.0, 0.0, 0.0};
			material.pbrMetallicRoughness.baseColorTexture.index = gltfModel.textures.size();
			material.pbrMetallicRoughness.baseColorTexture.texCoord = 0;
			material.pbrMetallicRoughness.baseColorFactor = std::vector<double>{1.0, 1.0, 1.0, 1.0};
			material.pbrMetallicRoughness.metallicFactor = 0;
			material.pbrMetallicRoughness.roughnessFactor = 1;
			material.extensions = {{"KHR_materials_unlit", {}}};
			gltfModel.extensionsUsed = {"KHR_materials_unlit"};

			static_assert(2 * sizeof(TexCoord::Scalar) == sizeof(TexCoord),
			              "TexCoord expected to be continuous");
			ASSERT(mesh.vertices.size() == mesh.faceTexcoords.size());
			primitive.attributes["TEXCOORD_0"] = gltfModel.accessors.size();
			tinygltf::Accessor vertexTexcoordAccessor;
			vertexTexcoordAccessor.name = "vertex_texcoord_0_accessor";
			vertexTexcoordAccessor.bufferView = gltfModel.bufferViews.size();
			vertexTexcoordAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
			vertexTexcoordAccessor.count = mesh.faceTexcoords.size();
			vertexTexcoordAccessor.type = TINYGLTF_TYPE_VEC2;
			gltfModel.accessors.emplace_back(std::move(vertexTexcoordAccessor));

			tinygltf::BufferView vertexTexcoordBufferView;
			vertexTexcoordBufferView.name = "vertex_texcoord_0_buffer_view";
			vertexTexcoordBufferView.buffer = gltfModel.buffers.size();
			const std::vector<TexCoord> normFaceTexcoords = mesh.FTexcoordsNormalize();
			ExtendBufferGLTF(normFaceTexcoords, gltfBuffer,
			                 vertexTexcoordBufferView.byteOffset,
			                 vertexTexcoordBufferView.byteLength);
			gltfModel.bufferViews.emplace_back(std::move(vertexTexcoordBufferView));

			tinygltf::Texture gltfTexture;
			gltfTexture.name = "texture";
			gltfTexture.source = gltfModel.images.size();
			gltfTexture.sampler = gltfModel.samplers.size();
			gltfModel.textures.emplace_back(std::move(gltfTexture));

			ASSERT(mesh.texturesDiffuse.size() == 1);
			tinygltf::Image gltfImage;
			gltfImage.name = "image";
			gltfImage.width = mesh.texturesDiffuse[0].cols;
			gltfImage.height = mesh.texturesDiffuse[0].rows;
			gltfImage.component = 3;
			gltfImage.bits = 8;
			gltfImage.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
			gltfImage.mimeType = "image/jpeg";
			gltfImage.image.resize(mesh.texturesDiffuse[0].size().area() * 3);
			cv::cvtColor(mesh.texturesDiffuse[0],
			             cv::Mat(mesh.texturesDiffuse[0].size(), CV_8UC3, gltfImage.image.data()),
			             cv::COLOR_BGR2RGB);
			gltfModel.images.emplace_back(std::move(gltfImage));

			tinygltf::Sampler gltfSampler;
			gltfSampler.name = "sampler";
			gltfSampler.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
			gltfSampler.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
			gltfSampler.wrapS = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE;
			gltfSampler.wrapT = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE;
			gltfModel.samplers.emplace_back(std::move(gltfSampler));
		} else if (mesh.HasTextureCoordinates()) {
			// UV-only path: emit TEXCOORD_0 without a texture image.
			// This supports the atlas/unwrap pipeline which generates UVs in
			// faceTexcoords (already normalized [0,1]) without a diffuse image.
			// A TEXCOORD_0 accessor with no baseColorTexture is valid glTF 2.0.
			ASSERT(mesh.vertices.size() == mesh.faceTexcoords.size());
			static_assert(2 * sizeof(TexCoord::Scalar) == sizeof(TexCoord),
			              "TexCoord expected to be continuous");
			primitive.attributes["TEXCOORD_0"] = gltfModel.accessors.size();
			tinygltf::Accessor vertexTexcoordAccessor;
			vertexTexcoordAccessor.name = "vertex_texcoord_0_accessor";
			vertexTexcoordAccessor.bufferView = gltfModel.bufferViews.size();
			vertexTexcoordAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
			vertexTexcoordAccessor.count = mesh.faceTexcoords.size();
			vertexTexcoordAccessor.type = TINYGLTF_TYPE_VEC2;
			gltfModel.accessors.emplace_back(std::move(vertexTexcoordAccessor));

			tinygltf::BufferView vertexTexcoordBufferView;
			vertexTexcoordBufferView.name = "vertex_texcoord_0_buffer_view";
			vertexTexcoordBufferView.buffer = gltfModel.buffers.size();
			// faceTexcoords are already normalized [0,1] from the atlas pipeline
			ExtendBufferGLTF(mesh.faceTexcoords, gltfBuffer,
			                 vertexTexcoordBufferView.byteOffset,
			                 vertexTexcoordBufferView.byteLength);
			gltfModel.bufferViews.emplace_back(std::move(vertexTexcoordBufferView));
		}
		gltfModel.materials.emplace_back(std::move(material));

		{
			static_assert(3 * sizeof(Face::Scalar) == sizeof(Face),
			              "Face expected to be continuous");
			primitive.indices = gltfModel.accessors.size();
			tinygltf::Accessor triangleAccessor;
			triangleAccessor.name = "triangle_accessor";
			triangleAccessor.bufferView = gltfModel.bufferViews.size();
			triangleAccessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
			triangleAccessor.count = mesh.faces.size() * 3;
			triangleAccessor.type = TINYGLTF_TYPE_SCALAR;
			gltfModel.accessors.emplace_back(std::move(triangleAccessor));

			tinygltf::BufferView triangleBufferView;
			triangleBufferView.name = "triangle_buffer_view";
			triangleBufferView.buffer = gltfModel.buffers.size();
			ExtendBufferGLTF(mesh.faces, gltfBuffer,
			                 triangleBufferView.byteOffset, triangleBufferView.byteLength);
			gltfModel.bufferViews.emplace_back(std::move(triangleBufferView));

			primitive.mode = TINYGLTF_MODE_TRIANGLES;
		}

		gltfModel.buffers.emplace_back(std::move(gltfBuffer));
		gltfMesh.primitives.emplace_back(std::move(primitive));
	}

	tinygltf::Node node;
	node.name = "node";
	node.mesh = gltfModel.meshes.size();
	// https://github.com/CesiumGS/3d-tiles/tree/1.0/specification#gltf-transforms
	// Following "Implementation note" there:
	// The root node matrix specifies a column-major z-up to y-up transform.
	// This transforms the source data into a y-up coordinate system as required
	// by glTF.
	const bool applyYUpRotation = true;
	if (applyYUpRotation) {
		node.matrix = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};
	}
	gltfScene.nodes.emplace_back(gltfModel.nodes.size());
	gltfModel.nodes.emplace_back(std::move(node));
	gltfModel.meshes.emplace_back(std::move(gltfMesh));
	gltfModel.scenes.emplace_back(std::move(gltfScene));

	gltfModel.asset.generator = "halfmesh";
	gltfModel.asset.version = "2.0";
	gltfModel.defaultScene = 0;

	tinygltf::TinyGLTF gltf;
	constexpr bool embedImages = true;
	constexpr bool embedBuffers = true;
	constexpr bool prettyPrint = true;
	return gltf.WriteGltfSceneToFile(&gltfModel, fileName, embedImages, embedBuffers,
	                                 prettyPrint, binary);
}

// ---------------------------------------------------------------------------
// Save — dispatch by extension
// ---------------------------------------------------------------------------
bool Mesh::Save(const std::string& fileName, bool binary) const
{
	const_cast<Mesh*>(this)->SyncFaces();
	const std::string::size_type extPos = fileName.rfind('.');
	const std::string ext(extPos != fileName.npos ? fileName.substr(extPos) : "");
	if (ext == ".ply") {
		// save in PLY format
		if (!SavePLY(fileName, binary))
			return false;
	} else if (ext == ".glb") {
		// save in GLB format
		if (!SaveGLTF(fileName, true))
			return false;
	} else if (ext == ".gltf") {
		// save in GLTF format
		if (!SaveGLTF(fileName, false))
			return false;
	} else {
		return false;
	}
	REPORT_STATUS_NOW("Mesh saved{}: {}",
	                  texturesDiffuse.empty() ? "" : HALFMESH_FORMAT(" ({} textures)", texturesDiffuse.size()),
	                  fileName);
	return true;
}

// ---------------------------------------------------------------------------
// ExportSeamEdges (explicit seamEdges)
// ---------------------------------------------------------------------------
bool Mesh::ExportSeamEdges(std::vector<std::pair<VIndex, VIndex>> seamEdges,
                           const std::string& fileName, bool binary) const
{
	if (seamEdges.empty())
		return false;

	VIndex numVertices = 0;
	std::vector<VIndex> mapVertices(vertices.size(), math::NO_ID);
	for (const auto& edge : seamEdges) {
		if (mapVertices[edge.first] == math::NO_ID)
			mapVertices[edge.first] = numVertices++;
		if (mapVertices[edge.second] == math::NO_ID)
			mapVertices[edge.second] = numVertices++;
	}
	std::vector<Vertex> seamVertices(numVertices);
	for (VIndex iV = 0; iV < static_cast<VIndex>(mapVertices.size()); ++iV) {
		if (mapVertices[iV] == math::NO_ID)
			continue;
		seamVertices[mapVertices[iV]] = vertices[iV];
	}
	for (auto& edge : seamEdges) {
		edge.first = mapVertices[edge.first];
		edge.second = mapVertices[edge.second];
	}

	tinyply::PlyFile file;
	file.add_properties_to_element("vertex", {"x", "y", "z"},
	                               tinyply::Type::FLOAT32, seamVertices.size(),
	                               reinterpret_cast<const uint8_t*>(seamVertices.data()), tinyply::Type::INVALID, 0);
	file.add_properties_to_element("edge", {"vertex1", "vertex2"},
	                               tinyply::Type::UINT32, seamEdges.size(),
	                               reinterpret_cast<const uint8_t*>(seamEdges.data()), tinyply::Type::INVALID, 0);

	std::filebuf fb;
	fb.open(fileName, binary ? std::ios::out | std::ios::binary : std::ios::out);
	if (!fb.is_open()) {
		REPORT_WARNING("Could not open file: {}", fileName);
		return false;
	}
	std::ostream outstream(&fb);
	if (outstream.fail())
		return false;
	file.write(outstream, binary);
	return true;
}

// ---------------------------------------------------------------------------
// ExportSeamEdges (derive seam edges from texture patches)
// ---------------------------------------------------------------------------
bool Mesh::ExportSeamEdges(const std::string& fileName, bool binary) const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(halfMesh.vHalfedges.size() == vertices.size());
	std::vector<uint32_t> facePatchIds;
	ListTexPatchFaces(facePatchIds);
	std::vector<std::pair<Mesh::VIndex, Mesh::VIndex>> seamEdges;
	FOREACHIDX (Mesh::VIndex, iV0, vertices) {
		for (HIndex iHe : halfMesh.VOutgoingHalfedges(iV0)) {
			const Mesh::VIndex iV1 = halfMesh.HeHeadVertex(iHe);
			if (iV0 > iV1)
				continue;
			if (!halfMesh.EHeIsBoundary(iHe)) {
				const Mesh::FIndex iF0 = halfMesh.HeFace(iHe);
				const uint32_t label0 = facePatchIds[iF0];
				const Mesh::FIndex iF1 = halfMesh.HeFace(halfMesh.HeTwin(iHe));
				const uint32_t label1 = facePatchIds[iF1];
				if (label0 == label1)
					continue;
			}
			seamEdges.emplace_back(iV0, iV1);
		}
	}
	return ExportSeamEdges(seamEdges, fileName, binary);
}

// ---------------------------------------------------------------------------
// ListTexPatchFaces
// ---------------------------------------------------------------------------
uint32_t Mesh::ListTexPatchFaces(std::vector<uint32_t>& facePatchIds) const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(faceTexcoords.size() == faces.size() * 3);
	ASSERT(halfMesh.vHalfedges.size() == vertices.size());
	facePatchIds = std::vector<uint32_t>(faces.size(), math::NO_ID);
	std::stack<FIndex> stackFaces;
	uint32_t patchId = 0;
	FOREACHIDX (FIndex, idxFace, faces) {
		if (facePatchIds[idxFace] != math::NO_ID)
			continue;
		facePatchIds[idxFace] = patchId;
		stackFaces.push(idxFace);
		do {
			FIndex iF = stackFaces.top();
			stackFaces.pop();
			for (HIndex iHe : halfMesh.FAdjacentHalfedges(iF)) {
				const HIndex iHeAdj = halfMesh.HeTwin(iHe);
				const FIndex iFAdj = halfMesh.HeFace(iHeAdj);
				if (iFAdj == math::NO_ID || facePatchIds[iFAdj] != math::NO_ID)
					continue;
				const VIndex iV = halfMesh.FVertexIth(iHe, faces[iF]);
				const VIndex iVAdj = halfMesh.FVertexIth(iHeAdj, faces[iFAdj]);
				const TexCoord* pts = &faceTexcoords[iF * 3];
				const TexCoord* adjPts = &faceTexcoords[iFAdj * 3];
				if (pts[iV] == adjPts[(iVAdj + 1) % 3] && pts[(iV + 1) % 3] == adjPts[iVAdj]) {
					facePatchIds[iFAdj] = patchId;
					stackFaces.push(iFAdj);
				}
			}
		} while (!stackFaces.empty());
		++patchId;
	}
	return patchId;
}

// ---------------------------------------------------------------------------
// FTexcoordsNormalize
// ---------------------------------------------------------------------------
std::vector<Mesh::TexCoord> Mesh::FTexcoordsNormalize() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTextureCoordinates());
	std::vector<TexCoord> normFaceTexcoords(faceTexcoords.size());
	if (faceTexcoords.size() == faces.size() * 3) {
		FOREACHIDX (FIndex, iF, faces) {
			const FIndex idxTexblob = FTexblob(iF);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    1.f / static_cast<float>(textureSize.width),
			    1.f / static_cast<float>(textureSize.height));
			for (int v = 0; v < 3; ++v) {
				const TexCoord& coord = faceTexcoords[iF * 3 + v];
				normFaceTexcoords[iF * 3 + v] = (coord + halfPixel).cwiseProduct(invNorm);
			}
		}
	} else {
		FOREACHIDX (FIndex, iV, vertices) {
			const FIndex idxTexblob = FTexblob(iV);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    1.f / static_cast<float>(textureSize.width),
			    1.f / static_cast<float>(textureSize.height));
			const TexCoord& coord = faceTexcoords[iV];
			normFaceTexcoords[iV] = (coord + halfPixel).cwiseProduct(invNorm);
		}
	}
	return normFaceTexcoords;
}

// ---------------------------------------------------------------------------
// FTexcoordsNormalizeFlipY
// ---------------------------------------------------------------------------
std::vector<Mesh::TexCoord> Mesh::FTexcoordsNormalizeFlipY() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTextureCoordinates());
	std::vector<TexCoord> normFaceTexcoords(faceTexcoords.size());
	// Untextured UV-atlas meshes (GenerateAtlas output) already carry normalized
	// [0,1] coordinates; with no texture the pixel-space (coord+0.5)/size transform
	// would fall back to a 1x1 size and shift/flip them out of [0,1] — corrupting
	// the layout for any external PLY reader.  Pass them through with only the
	// image-space Y-flip (FTexcoordsUnNormalizeFlipY inverts exactly this).
	if (texturesDiffuse.empty()) {
		FOREACH (i, faceTexcoords)
			normFaceTexcoords[i] = TexCoord(faceTexcoords[i].x(), 1.f - faceTexcoords[i].y());
		return normFaceTexcoords;
	}
	if (faceTexcoords.size() == faces.size() * 3) {
		FOREACHIDX (FIndex, iF, faces) {
			const FIndex idxTexblob = FTexblob(iF);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    1.f / static_cast<float>(textureSize.width),
			    1.f / static_cast<float>(textureSize.height));
			for (int v = 0; v < 3; ++v) {
				const TexCoord& coord = faceTexcoords[iF * 3 + v];
				normFaceTexcoords[iF * 3 + v] = TexCoord(
				    (coord.x() + halfPixel.x()) * invNorm.x(),
				    1.f - (coord.y() + halfPixel.y()) * invNorm.y());
			}
		}
	} else {
		FOREACHIDX (FIndex, iV, vertices) {
			const FIndex idxTexblob = FTexblob(iV);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    1.f / static_cast<float>(textureSize.width),
			    1.f / static_cast<float>(textureSize.height));
			const TexCoord& coord = faceTexcoords[iV];
			normFaceTexcoords[iV] = TexCoord(
			    (coord.x() + halfPixel.x()) * invNorm.x(),
			    1.f - (coord.y() + halfPixel.y()) * invNorm.y());
		}
	}
	return normFaceTexcoords;
}

// ---------------------------------------------------------------------------
// FTexcoordsUnNormalize
// ---------------------------------------------------------------------------
std::vector<Mesh::TexCoord> Mesh::FTexcoordsUnNormalize() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTextureCoordinates());
	std::vector<TexCoord> normFaceTexcoords(faceTexcoords.size());
	if (faceTexcoords.size() == faces.size() * 3) {
		FOREACHIDX (FIndex, iF, faces) {
			const FIndex idxTexblob = FTexblob(iF);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    static_cast<float>(textureSize.width),
			    static_cast<float>(textureSize.height));
			for (int v = 0; v < 3; ++v) {
				const TexCoord& coord = faceTexcoords[iF * 3 + v];
				normFaceTexcoords[iF * 3 + v] = coord.cwiseProduct(invNorm) - halfPixel;
			}
		}
	} else {
		FOREACHIDX (FIndex, iV, vertices) {
			const FIndex idxTexblob = FTexblob(iV);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    static_cast<float>(textureSize.width),
			    static_cast<float>(textureSize.height));
			const TexCoord& coord = faceTexcoords[iV];
			normFaceTexcoords[iV] = coord.cwiseProduct(invNorm) - halfPixel;
		}
	}
	return normFaceTexcoords;
}

// ---------------------------------------------------------------------------
// FTexcoordsUnNormalizeFlipY
// ---------------------------------------------------------------------------
std::vector<Mesh::TexCoord> Mesh::FTexcoordsUnNormalizeFlipY() const
{
	const_cast<Mesh*>(this)->SyncFaces();
	ASSERT(HasTextureCoordinates());
	std::vector<TexCoord> normFaceTexcoords(faceTexcoords.size());
	// Exact inverse of FTexcoordsNormalizeFlipY's no-texture path: an untextured
	// mesh (e.g. an atlas PLY reloaded with no sidecar) keeps its normalized [0,1]
	// coordinates, undoing only the image-space Y-flip.
	if (texturesDiffuse.empty()) {
		FOREACH (i, faceTexcoords)
			normFaceTexcoords[i] = TexCoord(faceTexcoords[i].x(), 1.f - faceTexcoords[i].y());
		return normFaceTexcoords;
	}
	if (faceTexcoords.size() == faces.size() * 3) {
		FOREACHIDX (FIndex, iF, faces) {
			const FIndex idxTexblob = FTexblob(iF);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    static_cast<float>(textureSize.width),
			    static_cast<float>(textureSize.height));
			for (int v = 0; v < 3; ++v) {
				const TexCoord& coord = faceTexcoords[iF * 3 + v];
				normFaceTexcoords[iF * 3 + v] = TexCoord(
				    coord.x() * invNorm.x() - halfPixel.x(),
				    (1.f - coord.y()) * invNorm.y() - halfPixel.x());
			}
		}
	} else {
		FOREACHIDX (FIndex, iV, vertices) {
			const FIndex idxTexblob = FTexblob(iV);
			const cv::Size textureSize(TextureSizeOrUnit(texturesDiffuse, idxTexblob));
			const TexCoord invNorm(
			    static_cast<float>(textureSize.width),
			    static_cast<float>(textureSize.height));
			const TexCoord& coord = faceTexcoords[iV];
			normFaceTexcoords[iV] = TexCoord(
			    coord.x() * invNorm.x() - halfPixel.x(),
			    (1.f - coord.y()) * invNorm.y() - halfPixel.x());
		}
	}
	return normFaceTexcoords;
}

} // namespace halfmesh
