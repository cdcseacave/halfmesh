/*
* Mesh.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h>
#include <halfmesh/Util/Maths.h>
#include <halfmesh/Util/Sampler.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <vector>
#include <BS_thread_pool.hpp>

#include "ParallelFor.h"

using namespace math;

namespace halfmesh {
namespace {

using detail::ParallelForPool;

} // anonymous namespace

void Mesh::ReleaseOptional()
{
	vertexColors = std::vector<Pixel>();
	vertexNormals = std::vector<Normal>();
	faceNormals = std::vector<Normal>();
	faceTexcoords = std::vector<TexCoord>();
	faceTexblobs = std::vector<TexIndex>();
	texturesDiffuse = std::vector<Image3u>();
	vertexFaces = std::vector<VertexFaces>();
}

void Mesh::VertexAttributesSwapPop(VIndex idx)
{
	if (!vertexColors.empty()) {
		vertexColors[idx] = vertexColors.back();
		vertexColors.pop_back();
	}
	if (!vertexNormals.empty()) {
		vertexNormals[idx] = vertexNormals.back();
		vertexNormals.pop_back();
	}
}

void Mesh::VertexAttributesAppendFrom(VIndex source)
{
	if (!vertexColors.empty())
		vertexColors.emplace_back(vertexColors[source]);
	if (!vertexNormals.empty())
		vertexNormals.emplace_back(vertexNormals[source]);
}

void Mesh::InvalidateFaces()
{
	// Only AUTHORED data is worth a warning: faceNormals and vertexFaces are
	// derived caches every mutator is expected to drop, so including them would
	// make the texture-policy warning fire on untextured meshes.
	const bool droppedTexture = !faceTexcoords.empty() || !faceTexblobs.empty() || !texturesDiffuse.empty();
	faces.clear();
	faceTexcoords.clear();
	faceTexblobs.clear();
	faceNormals.clear();
	texturesDiffuse.clear();
	vertexFaces.clear();
	if (droppedTexture) {
		static std::once_flag warningFlag;
		std::call_once(warningFlag, []() {
			REPORT_WARNING("texture attributes dropped: processing methods expect untextured meshes");
		});
	}
}

void Mesh::SyncFaces()
{
	if (faces.empty() && !halfMesh.Empty())
		halfMesh.FFaces(faces);
	ASSERT(ValidateInvariants());
}

void Mesh::SyncFacesOnPublicExit()
{
	if (!deferFaceSync)
		SyncFaces();
}

void Mesh::BeginHalfEdgePipeline()
{
	ASSERT(!deferFaceSync);
	ListHalfEdges();
	if (!halfMesh.Empty())
		InvalidateFaces();
	deferFaceSync = true;
}

void Mesh::EndHalfEdgePipeline()
{
	ASSERT(deferFaceSync);
	deferFaceSync = false;
	SyncFaces();
}

void Mesh::InvalidateHalfMesh()
{
	halfMesh.Clear();
}

bool Mesh::ValidateInvariants() const
{
	return (faces.empty() || halfMesh.Empty() || faces.size() == halfMesh.FSize()) && (halfMesh.Empty() || vertices.size() == halfMesh.VSize())
	       && (vertexColors.empty() || vertexColors.size() == vertices.size())
	       && (vertexNormals.empty() || vertexNormals.size() == vertices.size());
}

bool Mesh::ValidateHalfMesh() const
{
	if (!ValidateInvariants())
		return false;
	if (halfMesh.Empty())
		return vertices.empty() && faces.empty();

	const HalfMesh& live = halfMesh;
	const std::size_t numHalfedges = live.heNexts.size();
	const VIndex numVertices = live.VSize();
	const FIndex numFaces = live.FSize();
	if (numHalfedges == 0 || (numHalfedges & 1u) != 0 || live.heVertices.size() != numHalfedges || live.heFaces.size() != numHalfedges || numVertices != vertices.size() || numFaces == 0)
		return false;

	std::vector<HIndex> outgoingCount(numVertices, 0);
	std::vector<bool> boundaryVertices(numVertices, false);
	for (HIndex iHe = 0; iHe < numHalfedges; ++iHe) {
		const HIndex next = live.heNexts[iHe];
		const VIndex vertex = live.heVertices[iHe];
		const FIndex face = live.heFaces[iHe];
		if (next >= numHalfedges || vertex >= numVertices || (face != NO_ID && face >= numFaces))
			return false;
		if (live.heVertices[live.HeTwin(iHe)] != live.heVertices[next])
			return false;
		++outgoingCount[vertex];
		if (face == NO_ID) {
			if ((iHe & 1u) == 0)
				return false;
			boundaryVertices[vertex] = true;
			boundaryVertices[live.HeVertex(live.HeTwin(iHe))] = true;
		}
	}

	std::vector<bool> visitedFaceHalfedges(numHalfedges, false);
	for (FIndex iF = 0; iF < numFaces; ++iF) {
		const HIndex start = live.fHalfedges[iF];
		if (start >= numHalfedges || live.heFaces[start] != iF)
			return false;
		HIndex current = start;
		unsigned degree = 0;
		do {
			if (current >= numHalfedges || live.heFaces[current] != iF || visitedFaceHalfedges[current])
				return false;
			visitedFaceHalfedges[current] = true;
			current = live.heNexts[current];
			if (++degree > numHalfedges)
				return false;
		} while (current != start);
#if HALFMESH_TRIS
		if (degree != 3)
			return false;
#endif
	}
	for (HIndex iHe = 0; iHe < numHalfedges; ++iHe)
		if (live.heFaces[iHe] != NO_ID && !visitedFaceHalfedges[iHe])
			return false;

	std::vector<bool> visitedBoundaryHalfedges(numHalfedges, false);
	for (HIndex iHe = 0; iHe < numHalfedges; ++iHe) {
		if (live.heFaces[iHe] != NO_ID || visitedBoundaryHalfedges[iHe])
			continue;
		const HIndex start = iHe;
		HIndex current = start;
		std::size_t steps = 0;
		do {
			if (current >= numHalfedges || live.heFaces[current] != NO_ID || visitedBoundaryHalfedges[current])
				return false;
			visitedBoundaryHalfedges[current] = true;
			current = live.heNexts[current];
			if (++steps > numHalfedges)
				return false;
		} while (current != start);
	}

	for (VIndex iV = 0; iV < numVertices; ++iV) {
		const HIndex start = live.vHalfedges[iV];
		if (start >= numHalfedges || live.heVertices[start] != iV || outgoingCount[iV] == 0)
			return false;
		if (live.alwaysEven && (start & 1u))
			return false;
		if (boundaryVertices[iV] && (live.heFaces[start] == NO_ID || live.heFaces[live.HeTwin(start)] != NO_ID))
			return false;
		HIndex current = start;
		HIndex reached = 0;
		do {
			if (current >= numHalfedges || live.heVertices[current] != iV)
				return false;
			current = live.HeNextOutgoingHalfedge(current);
			if (++reached > outgoingCount[iV])
				return false;
		} while (current != start);
		if (reached != outgoingCount[iV])
			return false;
	}

	std::vector<Face> harvestedFaces;
	live.FFacesForValidation(harvestedFaces);
	if (!faces.empty()) {
		if (faces.size() != harvestedFaces.size())
			return false;
		for (std::size_t i = 0; i < faces.size(); ++i)
			for (Eigen::Index v = 0; v < faces[i].rows(); ++v)
				if (faces[i][v] != harvestedFaces[i][v])
					return false;
	}

	HalfMesh rebuilt;
	if (!rebuilt.BuildForValidation(numVertices, harvestedFaces) || rebuilt.VSize() != numVertices || rebuilt.FSize() != numFaces || rebuilt.ESize() != live.ESize())
		return false;

	const auto SortedAdjacentVertices = [](const HalfMesh& mesh, VIndex vertex) {
		std::vector<VIndex> adjacent;
		for (VIndex neighbor : mesh.VAdjacentVertices(vertex))
			adjacent.emplace_back(neighbor);
		std::sort(adjacent.begin(), adjacent.end());
		return adjacent;
	};
	const auto SortedAdjacentFaces = [](const HalfMesh& mesh, VIndex vertex) {
		std::vector<FIndex> adjacent;
		for (FIndex face : mesh.VAdjacentFaces(vertex))
			adjacent.emplace_back(face);
		std::sort(adjacent.begin(), adjacent.end());
		return adjacent;
	};
	for (VIndex iV = 0; iV < numVertices; ++iV) {
		if (SortedAdjacentVertices(live, iV) != SortedAdjacentVertices(rebuilt, iV) || SortedAdjacentFaces(live, iV) != SortedAdjacentFaces(rebuilt, iV))
			return false;
	}

	std::vector<std::vector<VIndex>> liveHoles, rebuiltHoles;
	live.EnumerateHoles(liveHoles);
	rebuilt.EnumerateHoles(rebuiltHoles);
	const auto CanonicalizeHoles = [](std::vector<std::vector<VIndex>>& holes) {
		for (std::vector<VIndex>& hole : holes)
			std::sort(hole.begin(), hole.end());
		std::sort(holes.begin(), holes.end());
	};
	CanonicalizeHoles(liveHoles);
	CanonicalizeHoles(rebuiltHoles);
	return liveHoles == rebuiltHoles;
}

void Mesh::ComputeFaceNormals()
{
	SyncFaces();
	faceNormals.resize(faces.size());
	FOREACH (idxFace, faces) {
		// exactly-degenerate (collinear) face: .normalized() on the zero cross
		// product is 0/0 = NaN, which ComputeVertexNormals would then accumulate
		// into every touching vertex normal; a zero normal accumulates harmlessly
		const Normal n = ComputeFaceNormal(faces[idxFace]);
		faceNormals[idxFace] = n.squaredNorm() > Type(0) ? Normal(n.normalized()) : Normal(Normal::Zero());
	}
}

void Mesh::ComputeSmoothFaceNormals(float maxAngle, float currentNormalWeight, unsigned iterations)
{
	ListHalfEdges();
	SyncFaces();
	if (faceNormals.size() != faces.size())
		ComputeFaceNormals();
	const float cosMaxAngle = std::cos(D2R(maxAngle));
	// Jacobi double-buffering: hoist one scratch buffer out of the iteration loop
	// (swap each pass) instead of allocating a fresh F-sized vector every pass.
	// Each face writes only its own slot and reads only the previous buffer, so
	// the per-face loop parallelizes across the pool with bit-identical results
	// (the per-face neighbor summation order is fixed and unchanged).
	std::vector<Normal> newFaceNormals(faceNormals.size());
	BS::light_thread_pool pool;
	for (unsigned iter = 0; iter < iterations; ++iter) {
		ParallelForPool(pool, faces.size(), [&](std::size_t idxFace) {
			const Normal& currentNormal = faceNormals[idxFace];
			Normal avgNeighborsNormal = Normal::Zero();
			for (FIndex idxNeighborFace : halfMesh.FAdjacentFaces(static_cast<FIndex>(idxFace))) {
				const Normal& neighborNormal = faceNormals[idxNeighborFace];
				if (currentNormal.dot(neighborNormal) >= cosMaxAngle)
					avgNeighborsNormal += neighborNormal;
			}
			// no admissible neighbor (isolated face, or every neighbor beyond
			// maxAngle) leaves the average at zero — normalizing it would inject
			// NaN into the blend; keep the current normal instead. The final
			// blend can only be zero when currentNormal itself is zero (a
			// degenerate face): keep it as-is rather than normalize 0/0.
			if (avgNeighborsNormal.squaredNorm() > Type(0)) {
				avgNeighborsNormal.normalize();
				const Normal blended = currentNormal * currentNormalWeight + avgNeighborsNormal * (1.f - currentNormalWeight);
				newFaceNormals[idxFace] = blended.squaredNorm() > Type(0) ? Normal(blended.normalized()) : currentNormal;
			} else {
				newFaceNormals[idxFace] = currentNormal;
			}
		});
		newFaceNormals.swap(faceNormals);
	}
}

std::vector<Mesh::Normal> Mesh::ComputeVertexNormals()
{
	SyncFaces();
	if (faceNormals.size() != faces.size())
		ComputeFaceNormals();
	// Angle-weighted (Thurmer & Wuthrich 1998) pseudonormals: weight each
	// incident (possibly smoothed) face normal by the triangle's corner angle at
	// the vertex.  This is tessellation-independent -- unlike uniform averaging,
	// a fan of slivers on one side cannot bias the normal -- while still
	// honoring ComputeSmoothFaceNormals output (we scale the cached face normal,
	// not a fresh cross product).
	std::vector<Normal> normals(vertices.size(), Normal::Zero());
	FOREACH (idxFace, faces) {
		const Face& face = faces[idxFace];
		const Normal& faceNormal = faceNormals[idxFace];
		for (int i = 0; i < 3; ++i) {
			const Vertex& v = vertices[face[i]];
			const Normal e1 = vertices[face[(i + 1) % 3]] - v;
			const Normal e2 = vertices[face[(i + 2) % 3]] - v;
			const Type l1 = e1.norm();
			const Type l2 = e2.norm();
			if (l1 <= Type(0) || l2 <= Type(0))
				continue; // degenerate corner (zero-length edge): no angle weight
			const Type cosAngle = std::clamp(e1.dot(e2) / (l1 * l2), Type(-1), Type(1));
			normals[face[i]] += faceNormal * std::acos(cosAngle);
		}
	}
	for (Normal& normal : normals)
		normal.normalize();
	return normals;
}

real Mesh::ComputeArea() const
{
	SyncFacesConst();
	real area(0);
	for (const Face& face : faces)
		area += ComputeFaceDoubleArea(face);
	return area * real(0.5);
}

real Mesh::ComputeArea(const std::vector<FIndex>& indices) const
{
	SyncFacesConst();
	real area(0);
	for (FIndex idxFace : indices)
		area += ComputeFaceDoubleArea(idxFace);
	return area * real(0.5);
}

Mesh::Type Mesh::ComputeMeanEdgeLength()
{
	// A mesh with no connectivity has no edges to average; guard before
	// ListHalfEdges(), which requires faces when vertices are present.
	if (faces.empty() && halfMesh.Empty())
		return 0;
	ListHalfEdges();
	if (halfMesh.ESize() == 0)
		return 0;
	real sumLength(0);
	for (EIndex idxEdge = 0; idxEdge < halfMesh.ESize(); ++idxEdge) {
		const auto verts = halfMesh.EVertices(idxEdge);
		sumLength += (vertices[verts.first] - vertices[verts.second]).norm();
	}
	return Type(sumLength / halfMesh.ESize());
}

Eigen::AlignedBox<Mesh::Type, 3> Mesh::ComputeAABBox() const
{
	Eigen::AlignedBox<Type, 3> bbox;
	for (const Vertex& vert : vertices)
		bbox.extend(vert);
	return bbox;
}

bool Mesh::IsWatertight() const
{
	if (!halfMesh.Empty()) {
		// a half-edge structure is manifold and consistently oriented by
		// construction, so only a border half-edge can open it
		return halfMesh.FSize() != 0 && std::find(halfMesh.heFaces.begin(), halfMesh.heFaces.end(), math::NO_ID) == halfMesh.heFaces.end();
	}
	if (faces.empty())
		return false;
	// Bucket every directed edge under its smaller endpoint, as the larger one
	// shifted left with the direction in the low bit: a watertight edge is then
	// a pair of entries in its bucket, one each way, and nothing else. The
	// entries and offsets are 32-bit unless the mesh is too large for them.
	const auto check = [this]<typename Index>() -> bool {
		std::vector<Index> offsets(vertices.size() + 2, 0);
		for (const Face& face : faces) {
			for (int k = 0; k < 3; ++k) {
				const VIndex a = face[k], b = face[(k + 1) % 3];
				if (a == b)
					return false;
				++offsets[std::min(a, b) + 2];
			}
		}
		// shifted by one, so the fill can use offsets[v+1] as the cursor of bucket
		// v and leave bucket v spanning [offsets[v], offsets[v+1])
		std::partial_sum(offsets.begin() + 2, offsets.end(), offsets.begin() + 2);
		std::vector<Index> entries(faces.size() * 3);
		for (const Face& face : faces) {
			for (int k = 0; k < 3; ++k) {
				const VIndex a = face[k], b = face[(k + 1) % 3];
				entries[offsets[std::min(a, b) + 1]++] = (static_cast<Index>(std::max(a, b)) << 1) | static_cast<Index>(a > b);
			}
		}
		std::atomic<bool> watertight{true};
		const auto checkBuckets = [&](size_t begin, size_t end) {
			for (size_t v = begin; v < end && watertight.load(std::memory_order_relaxed); ++v) {
				Index* const first = entries.data() + offsets[v];
				Index* const last = entries.data() + offsets[v + 1];
				if ((last - first) & 1) {
					watertight = false;
					return;
				}
				std::sort(first, last);
				for (const Index* e = first; e != last; e += 2) {
					if ((e[0] & 1) != 0 || e[1] != (e[0] | 1) || (e + 2 != last && (e[2] >> 1) == (e[0] >> 1))) {
						watertight = false;
						return;
					}
				}
			}
		};
		if (faces.size() < (1u << 16)) {
			checkBuckets(0, vertices.size());
		} else {
			BS::light_thread_pool pool;
			pool.detach_blocks(size_t(0), vertices.size(), checkBuckets);
			pool.wait();
		}
		return watertight;
	};
	if (vertices.size() < (size_t(1) << 31) && faces.size() * 3 < (size_t(1) << 32))
		return check.template operator()<uint32_t>();
	return check.template operator()<uint64_t>();
}

namespace {
// The volume sums run over blocks of this many faces, each summed in order and
// the block sums added in order, so the result depends on the block size only,
// never on the thread count.
constexpr size_t VOLUME_BLOCK_FACES = size_t(1) << 16;

template <typename Sum, typename BlockFn>
std::vector<Sum> SumFaceBlocks(size_t numFaces, BlockFn&& sumBlock)
{
	std::vector<Sum> sums((numFaces + VOLUME_BLOCK_FACES - 1) / VOLUME_BLOCK_FACES);
	const auto run = [&](size_t block) {
		sums[block] = sumBlock(block * VOLUME_BLOCK_FACES, std::min(numFaces, (block + 1) * VOLUME_BLOCK_FACES));
	};
	if (sums.size() < 2) {
		if (!sums.empty())
			run(0);
	} else {
		BS::light_thread_pool pool;
		ParallelForPool(pool, sums.size(), run);
	}
	return sums;
}
} // anonymous namespace

real Mesh::ComputeVolume() const
{
	SyncFacesConst();
	if (faces.empty())
		return 0;
	// the tetrahedra span the faces with the bounding-box center, so every term
	// stays at the scale of the mesh, not of its distance to the origin; the edges
	// are differences of floats, exact in double
	const Eigen::Vector3d center = ComputeAABBox().center().cast<double>();
	const std::vector<double> sums = SumFaceBlocks<double>(faces.size(), [&](size_t begin, size_t end) {
		double sum = 0;
		for (size_t f = begin; f < end; ++f) {
			const Face& face = faces[f];
			const Eigen::Vector3d p0 = vertices[face[0]].cast<double>();
			const Eigen::Vector3d e1 = vertices[face[1]].cast<double>() - p0;
			const Eigen::Vector3d e2 = vertices[face[2]].cast<double>() - p0;
			sum += (p0 - center).dot(e1.cross(e2));
		}
		return sum;
	});
	return std::accumulate(sums.begin(), sums.end(), 0.0) / 6;
}

Mesh::PlaneVolume Mesh::ComputeVolume(const Plane& plane) const
{
	ASSERT(std::abs(plane.normal().norm() - 1) < 1e-6);
	SyncFacesConst();
	const std::vector<PlaneVolume> sums = SumFaceBlocks<PlaneVolume>(faces.size(), [&](size_t begin, size_t end) {
		PlaneVolume sum;
		for (size_t f = begin; f < end; ++f) {
			const Face& face = faces[f];
			const Eigen::Vector3d p[3] = {vertices[face[0]].cast<double>(), vertices[face[1]].cast<double>(), vertices[face[2]].cast<double>()};
			const double h[3] = {plane.signedDistance(p[0]), plane.signedDistance(p[1]), plane.signedDistance(p[2])};
			// the prism between the face and its projection: twice the signed
			// projected area times the mean height, over 2 (the height is linear)
			const double area2 = plane.normal().dot((p[1] - p[0]).cross(p[2] - p[0]));
			const double prism = area2 * (h[0] + h[1] + h[2]) / 6;
			const int numAbove = (h[0] > 0) + (h[1] > 0) + (h[2] > 0);
			const int numBelow = (h[0] < 0) + (h[1] < 0) + (h[2] < 0);
			if (numBelow == 0) {
				sum.above += prism;
			} else if (numAbove == 0) {
				sum.below += prism;
			} else {
				// the vertex alone on its side and the two points where its edges
				// cross the plane bound a corner triangle of t1*t2 the projected
				// area, at height h there and zero at the crossings; the height
				// being linear, the rest of the prism is the difference (a vertex
				// on the plane makes its crossing parameter 1)
				int lone = 0;
				while (numAbove == 1 ? !(h[lone] > 0) : !(h[lone] < 0))
					++lone;
				const int j = (lone + 1) % 3, k = (lone + 2) % 3;
				const double t1 = h[lone] / (h[lone] - h[j]);
				const double t2 = h[lone] / (h[lone] - h[k]);
				const double corner = area2 * t1 * t2 * h[lone] / 6;
				(numAbove == 1 ? sum.above : sum.below) += corner;
				(numAbove == 1 ? sum.below : sum.above) += prism - corner;
			}
		}
		return sum;
	});
	PlaneVolume volume;
	for (const PlaneVolume& sum : sums) {
		volume.above += sum.above;
		volume.below += sum.below;
	}
	return volume;
}

void Mesh::Join(const Mesh& other)
{
	SyncFaces();
	other.SyncFacesConst();
	if (other.vertices.empty())
		return;
	if (vertices.empty()) {
		*this = other;
		vertexFaces.clear();
		return;
	}
	const VIndex offsetV = static_cast<VIndex>(vertices.size());
	const FIndex offsetF = static_cast<FIndex>(faces.size());
	ASSERT(size_t(offsetV) + other.vertices.size() < math::NO_ID && size_t(offsetF) + other.faces.size() < math::NO_ID);
	// an attribute only one side carries would leave the array partial
	const auto joinAttribute = [](auto& array, const auto& otherArray) {
		if (array.empty() || otherArray.empty())
			array.clear();
		else
			array.insert(array.end(), otherArray.begin(), otherArray.end());
	};
	vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
	joinAttribute(vertexColors, other.vertexColors);
	joinAttribute(vertexNormals, other.vertexNormals);
	const bool hasTexcoords = !faceTexcoords.empty() && faceTexcoords.size() == faces.size() * 3;
	const bool otherHasTexcoords = !other.faceTexcoords.empty() && other.faceTexcoords.size() == other.faces.size() * 3;
	const size_t numTextures = texturesDiffuse.size();
	if (hasTexcoords && otherHasTexcoords && texturesDiffuse.empty() == other.texturesDiffuse.empty() && numTextures + other.texturesDiffuse.size() <= MAX_TEXBLOBS) {
		faceTexcoords.insert(faceTexcoords.end(), other.faceTexcoords.begin(), other.faceTexcoords.end());
		if (numTextures == 0) {
			joinAttribute(faceTexblobs, other.faceTexblobs);
		} else {
			if (faceTexblobs.empty())
				faceTexblobs.assign(offsetF, 0);
			if (other.faceTexblobs.empty())
				faceTexblobs.insert(faceTexblobs.end(), other.faces.size(), static_cast<TexIndex>(numTextures));
			else
				for (const TexIndex idxTexblob : other.faceTexblobs)
					faceTexblobs.push_back(static_cast<TexIndex>(idxTexblob + numTextures));
			texturesDiffuse.insert(texturesDiffuse.end(), other.texturesDiffuse.begin(), other.texturesDiffuse.end());
		}
	} else {
		if (hasTexcoords || otherHasTexcoords)
			REPORT_WARNING("Join: texture coordinates dropped, the two meshes do not carry compatible textures");
		faceTexcoords.clear();
		faceTexblobs.clear();
		texturesDiffuse.clear();
	}
	faces.reserve(faces.size() + other.faces.size());
	for (const Face& face : other.faces)
		faces.emplace_back(face[0] + offsetV, face[1] + offsetV, face[2] + offsetV);
	joinAttribute(faceNormals, other.faceNormals);
	vertexFaces.clear();
	if (halfMesh.Empty() || other.halfMesh.Empty()) {
		halfMesh.Clear();
		return;
	}
	// the half-edge count is even, so the shift keeps twins paired (h^1) and
	// every representative's parity
	const HIndex offsetH = static_cast<HIndex>(halfMesh.heNexts.size());
	ASSERT((offsetH & 1u) == 0);
	const auto appendShifted = [](std::vector<uint32_t>& array, const std::vector<uint32_t>& otherArray, uint32_t offset) {
		array.reserve(array.size() + otherArray.size());
		for (const uint32_t idx : otherArray)
			array.push_back(idx == math::NO_ID ? idx : idx + offset);
	};
	appendShifted(halfMesh.vHalfedges, other.halfMesh.vHalfedges, offsetH);
	appendShifted(halfMesh.fHalfedges, other.halfMesh.fHalfedges, offsetH);
	appendShifted(halfMesh.heNexts, other.halfMesh.heNexts, offsetH);
	appendShifted(halfMesh.heVertices, other.halfMesh.heVertices, offsetV);
	appendShifted(halfMesh.heFaces, other.halfMesh.heFaces, offsetF);
	halfMesh.alwaysEven = halfMesh.alwaysEven && other.halfMesh.alwaysEven;
	ASSERT(ValidateInvariants());
}

Mesh Mesh::SubMesh(std::span<const FIndex> faceIndices, std::vector<VIndex>* vertexMap) const
{
	SyncFacesConst();
	Mesh mesh;
	// the source vertex of each new one, in order of first reference
	std::vector<VIndex> sources;
	sources.reserve(std::min(faceIndices.size() * 3, vertices.size()));
	const size_t numCorners = faceIndices.size() * 3;
	std::vector<VIndex> denseMap, hashKeys, hashValues;
	int hashShift = 0;
	if (numCorners * 8 >= vertices.size()) {
		denseMap.assign(vertices.size(), math::NO_ID);
	} else {
		// open addressing at load factor <= 1/2, Fibonacci hashing, linear probing
		const size_t capacity = std::bit_ceil(std::max<size_t>(numCorners * 2, 16));
		hashShift = 64 - std::countr_zero(capacity);
		hashKeys.assign(capacity, math::NO_ID);
		hashValues.resize(capacity);
	}
	const auto mapVertex = [&](VIndex idxVertex) -> VIndex {
		ASSERT(idxVertex < vertices.size());
		if (!denseMap.empty()) {
			VIndex& idxNew = denseMap[idxVertex];
			if (idxNew == math::NO_ID) {
				idxNew = static_cast<VIndex>(sources.size());
				sources.push_back(idxVertex);
			}
			return idxNew;
		}
		const size_t mask = hashKeys.size() - 1;
		for (size_t slot = static_cast<size_t>((uint64_t(idxVertex) * 0x9E3779B97F4A7C15ull) >> hashShift);; slot = (slot + 1) & mask) {
			if (hashKeys[slot] == idxVertex)
				return hashValues[slot];
			if (hashKeys[slot] == math::NO_ID) {
				hashKeys[slot] = idxVertex;
				hashValues[slot] = static_cast<VIndex>(sources.size());
				sources.push_back(idxVertex);
				return hashValues[slot];
			}
		}
	};
	mesh.faces.reserve(faceIndices.size());
	for (const FIndex idxFace : faceIndices) {
		ASSERT(idxFace < faces.size());
		const Face& face = faces[idxFace];
		Face& newFace = mesh.faces.emplace_back();
		for (int k = 0; k < 3; ++k)
			newFace[k] = mapVertex(face[k]);
	}
	mesh.vertices.reserve(sources.size());
	for (const VIndex idxVertex : sources)
		mesh.vertices.push_back(vertices[idxVertex]);
	if (!vertexColors.empty()) {
		mesh.vertexColors.reserve(sources.size());
		for (const VIndex idxVertex : sources)
			mesh.vertexColors.push_back(vertexColors[idxVertex]);
	}
	if (!vertexNormals.empty()) {
		mesh.vertexNormals.reserve(sources.size());
		for (const VIndex idxVertex : sources)
			mesh.vertexNormals.push_back(vertexNormals[idxVertex]);
	}
	if (!faceNormals.empty() && faceNormals.size() == faces.size()) {
		mesh.faceNormals.reserve(faceIndices.size());
		for (const FIndex idxFace : faceIndices)
			mesh.faceNormals.push_back(faceNormals[idxFace]);
	}
	if (!faceIndices.empty() && !faceTexcoords.empty() && faceTexcoords.size() == faces.size() * 3) {
		mesh.faceTexcoords.reserve(numCorners);
		for (const FIndex idxFace : faceIndices)
			mesh.faceTexcoords.insert(mesh.faceTexcoords.end(), faceTexcoords.begin() + size_t(idxFace) * 3, faceTexcoords.begin() + size_t(idxFace) * 3 + 3);
		if (texturesDiffuse.empty()) {
			if (faceTexblobs.size() == faces.size())
				for (const FIndex idxFace : faceIndices)
					mesh.faceTexblobs.push_back(faceTexblobs[idxFace]);
		} else if (faceTexblobs.empty()) {
			mesh.texturesDiffuse.push_back(texturesDiffuse.front());
		} else {
			// keep only the textures the faces use, in order of first use
			std::vector<int> mapTexblobs(texturesDiffuse.size(), -1);
			mesh.faceTexblobs.reserve(faceIndices.size());
			for (const FIndex idxFace : faceIndices) {
				const TexIndex idxTexblob = faceTexblobs[idxFace];
				ASSERT(idxTexblob < texturesDiffuse.size());
				int& idxNew = mapTexblobs[idxTexblob];
				if (idxNew < 0) {
					idxNew = static_cast<int>(mesh.texturesDiffuse.size());
					mesh.texturesDiffuse.push_back(texturesDiffuse[idxTexblob]);
				}
				mesh.faceTexblobs.push_back(static_cast<TexIndex>(idxNew));
			}
			if (mesh.texturesDiffuse.size() == 1)
				mesh.faceTexblobs.clear();
		}
	}
	if (vertexMap)
		*vertexMap = std::move(sources);
	return mesh;
}

void Mesh::SamplePoints(double density, uint32_t seed, std::vector<Vertex>& points, std::vector<Pixel>* colors) const
{
	ASSERT(density >= 0);
	SyncFacesConst();
	points.clear();
	const bool textured = colors != nullptr && faceTexcoords.size() == faces.size() * 3 && !texturesDiffuse.empty();
	if (colors)
		colors->clear();
	const size_t expected = static_cast<size_t>(std::ceil(ComputeArea() * density));
	points.reserve(expected);
	if (textured)
		colors->reserve(expected);
	std::mt19937 rnd(seed);
	std::uniform_real_distribution<double> dist(0, 1);
	for (FIndex idxFace = 0; idxFace < static_cast<FIndex>(faces.size()); ++idxFace) {
		const Face& face = faces[idxFace];
		// the triangle as O + x*u + y*v
		const Vertex& O = vertices[face[0]];
		const Vertex u = vertices[face[1]] - O;
		const Vertex v = vertices[face[2]] - O;
		const Vertex n = u.cross(v);
		const double area = static_cast<double>(std::sqrt(n.x() * n.x() + n.y() * n.y() + n.z() * n.z())) * 0.5;
		// the points this face takes, the fraction left with its probability
		const double toAdd = area * density;
		unsigned numPoints = static_cast<unsigned>(toAdd);
		if (dist(rnd) <= toAdd - static_cast<double>(numPoints))
			++numPoints;
		for (unsigned i = 0; i < numPoints; ++i) {
			double x = dist(rnd), y = dist(rnd);
			// fold the half of the unit square outside the triangle back in
			if (x + y > 1.0) {
				x = 1.0 - x;
				y = 1.0 - y;
			}
			points.emplace_back(O + static_cast<Type>(x) * u + static_cast<Type>(y) * v);
			if (textured) {
				const TexCoord* tc = &faceTexcoords[static_cast<size_t>(idxFace) * 3];
				const TexCoord t = tc[0] + static_cast<TexCoord::Scalar>(x) * (tc[1] - tc[0]) + static_cast<TexCoord::Scalar>(y) * (tc[2] - tc[0]);
				const Image3u& texture = texturesDiffuse[FTexblob(idxFace)];
				const auto color = SampleImage<LinearInterp<float>>(texture, Eigen::Vector2f(t.x(), t.y()));
				colors->emplace_back(Pixel(static_cast<uint8_t>(std::clamp(std::lround(color.x()), 0l, 255l)),
				                           static_cast<uint8_t>(std::clamp(std::lround(color.y()), 0l, 255l)),
				                           static_cast<uint8_t>(std::clamp(std::lround(color.z()), 0l, 255l))));
			}
		}
	}
}

void Mesh::ListVertexFaces()
{
	SyncFaces();
	vertexFaces.clear();
	vertexFaces.resize(vertices.size());
	// First pass: count incident face-corners per vertex so each list is reserved
	// exactly, avoiding the ~4 reallocations per vertex from growing from empty
	// (a valence-6 vertex otherwise triggers 1->2->4->8 regrows).  Degenerate
	// corners over-count harmlessly (reserve is an upper bound).
	std::vector<uint32_t> valence(vertices.size(), 0);
	for (const Face& face : faces)
		for (int v = 0; v < 3; ++v)
			++valence[face[v]];
	FOREACHIDX (VIndex, iV, vertices)
		vertexFaces[iV].reserve(valence[iV]);
	// Second pass: append in ascending face order (keeps each list sorted -- an
	// invariant RemoveFaces' lower_bound relies on); back()!=iF dedups the
	// repeated corners of a degenerate face.
	FOREACHIDX (FIndex, iF, faces) {
		const Face& face = faces[iF];
		for (int v = 0; v < 3; ++v) {
			VertexFaces& vfs = vertexFaces[face[v]];
			ASSERT(std::find(vfs.begin(), vfs.end(), iF) == vfs.end() || std::find(vfs.begin(), vfs.end(), iF) == vfs.end() - 1 /*in case of degenerate faces*/);
			if (vfs.empty() || vfs.back() != iF) {
				vfs.emplace_back(iF);
			}
		}
	}
}

bool Mesh::ValidateVertexFaces()
{
	std::vector<VertexFaces> prevVertexFaces = std::move(vertexFaces);
	ListVertexFaces();
	if (prevVertexFaces.size() != vertexFaces.size())
		return false;
	RFOREACH (i, vertexFaces) {
		if (prevVertexFaces[i] != vertexFaces[i])
			return false;
	}
	return true;
}

void Mesh::ListHalfEdges()
{
	if (!halfMesh.Empty())
		return;
	ASSERT(!(faces.empty() && !vertices.empty()));
	// Fast path: Build assumes a manifold mesh but now *detects* non-manifold
	// input cheaply and returns false instead of producing a corrupt structure
	// (which used to hang the adjacency walk).  On failure, repair to manifold
	// and rebuild via the safe path, so any mesh handed to the half-edge code is
	// checked & fixed first.  A known-manifold mesh pays only the inline checks.
	if (!halfMesh.Build(*this)) {
		REPORT_WARNING("non-manifold mesh; repairing to manifold before half-edge build");
		ListHalfEdgesSafe();
	}
	ASSERT(ValidateInvariants());
}

// NOTE: ListHalfEdgesSafe() and FixNonManifold() are implemented in
// MeshRepair.cpp: ListHalfEdgesSafe depends on FixNonManifold, so both
// live in the repair TU.

std::vector<Mesh::VIndex> Mesh::VAdjacentVertices(VIndex iV) const
{
	ASSERT(vertexFaces.size() == vertices.size());
	std::unordered_set<VIndex> seenIndices;
	for (FIndex iF : vertexFaces[iV]) {
		const Face& face = faces[iF];
		for (int v = 0; v < 3; ++v) {
			const VIndex iVAdj = face[v];
			if (iV != iVAdj)
				seenIndices.emplace(iVAdj);
		}
	}
	return std::vector<VIndex>(seenIndices.begin(), seenIndices.end());
}

Mesh::VIndex Mesh::FVertexIdx(FIndex idxFace, VIndex iV) const
{
	SyncFacesConst();
	const Face& face = faces[idxFace];
	for (VIndex i = 0; i < 3; ++i)
		if (face[i] == iV)
			return i;
	return NO_ID;
}

bool Mesh::FSameVertices(FIndex idxFace0, FIndex idxFace1) const
{
	const Face& face0 = faces[idxFace0];
	for (VIndex i = 0; i < 3; ++i)
		if (FVertexIdx(idxFace1, face0[i]) == NO_ID)
			return false;
	return true;
}

bool Mesh::FEdgeOrientation(FIndex idxFace, VIndex iV0, VIndex iV1) const
{
	const VIndex i0 = FVertexIdx(idxFace, iV0);
	ASSERT(i0 != NO_ID);
	ASSERT(faces[idxFace][(i0 + 1) % 3] == iV1 || faces[idxFace][(i0 + 2) % 3] == iV1);
	return faces[idxFace][(i0 + 1) % 3] == iV1;
}

Mesh::FIndex Mesh::FEdgeAdjacentFace(FIndex idxFace, VIndex iV0, VIndex iV1) const
{
	// iterate over all associated faces at the current vertex
	ASSERT(vertexFaces.size() == vertices.size());
	FIndex idxFaceAdj = NO_ID;
	for (FIndex iF : vertexFaces[iV0]) {
		// if the face associated at this vertex is not equal to the current face
		if (iF != idxFace) {
			const Face& face = faces[iF];
			for (int i = 0; i < 3; ++i) {
				// if this face is adjacent to the vertex
				if (face[i] == iV1) {
					// check if there are more than two adjacent faces (manifold constraint)
					if (idxFaceAdj != NO_ID)
						return NO_ID;
					// check if edge vertices ordering is opposite in the two faces (manifold constraint)
					if (FEdgeOrientation(idxFace, iV0, iV1) == FEdgeOrientation(iF, iV0, iV1))
						return NO_ID;
					idxFaceAdj = iF;
				}
			}
		}
	}
	return idxFaceAdj;
}

void Mesh::ECollapse(EIndex iE)
{
	ASSERT(ValidateInvariants());
	ASSERT(!halfMesh.Empty());
	HalfMesh::RemovedData removedData;
	halfMesh.ERemove(iE, removedData);
	ASSERT(removedData.numVerts == 1);
	vertices[removedData.verts[0]] = vertices.back();
	vertices.pop_back();
	// the per-vertex attributes move in lockstep with the position swap-pop,
	// mirroring RemoveUnreferencedVertices/RemoveVertices.
	VertexAttributesSwapPop(removedData.verts[0]);
	InvalidateFaces();
	// re-harvesting the whole array for one collapse is O(F); a caller collapsing
	// many edges should scope the loop in BeginHalfEdgePipeline, which suppresses
	// this and harvests once at the end
	SyncFacesOnPublicExit();
	ASSERT(ValidateInvariants());
}

Mesh::VIndex Mesh::RemoveUnreferencedVertices()
{
	return halfMesh.Empty() ? RemoveUnreferencedVerticesArrays() : RemoveUnreferencedVerticesHalfEdge();
}

Mesh::VIndex Mesh::RemoveUnreferencedVerticesArrays()
{
	if (vertexFaces.size() != vertices.size())
		ListVertexFaces();
	VIndex numVerticesRemoved = 0;
	RFOREACHIDX (VIndex, idxVert, vertices) {
		if (vertexFaces[idxVert].empty()) {
			const VIndex idxVertMoved = static_cast<VIndex>(vertices.size() - 1);
			for (FIndex idxFace : vertexFaces.back()) {
				Face& face = faces[idxFace];
				for (int i = 0; i < 3; ++i) {
					if (face[i] == idxVertMoved)
						face[i] = idxVert;
				}
			}
			vertices[idxVert] = vertices.back();
			vertices.pop_back();
			vertexFaces[idxVert] = std::move(vertexFaces.back());
			vertexFaces.pop_back();
			VertexAttributesSwapPop(idxVert);
			++numVerticesRemoved;
		}
	}
	if (numVerticesRemoved > 0)
		halfMesh.Clear();
	return numVerticesRemoved;
}

Mesh::VIndex Mesh::RemoveUnreferencedVerticesHalfEdge()
{
	ASSERT(!halfMesh.Empty());
	ASSERT(halfMesh.VSize() == vertices.size());
	std::vector<VIndex> removedVerts;
	halfMesh.VRemoveUnreferenced(removedVerts);
	for (VIndex removed : removedVerts) {
		vertices[removed] = vertices.back();
		vertices.pop_back();
		VertexAttributesSwapPop(removed);
	}
	if (!removedVerts.empty())
		InvalidateFaces();
	SyncFacesOnPublicExit();
	// cheap contract check only; the O(F log F) ValidateHalfMesh() rebuild-and-compare
	// is run by the test suite after every native mutator (see tests/AGENTS.md)
	ASSERT(ValidateInvariants());
	return static_cast<VIndex>(removedVerts.size());
}

void Mesh::RemoveVertices(std::vector<VIndex>& vertexRemoves, bool updateLists)
{
	if (vertexFaces.size() != vertices.size())
		ListVertexFaces();
	std::sort(vertexRemoves.begin(), vertexRemoves.end());
	VIndex idxVertLast = NO_ID;
	if (!updateLists) {
		RFOREACHPTR (ptrIdxVert, vertexRemoves) {
			const VIndex idxVert = *ptrIdxVert;
			if (idxVertLast == idxVert)
				continue;
			const VIndex idxVertMove = vertices.size() - 1;
			if (idxVert < idxVertMove) {
				// update all faces of the moved vertex
				const VertexFaces& vfs = vertexFaces[idxVertMove];
				for (FIndex idxFace : vfs)
					FVertex(idxFace, idxVertMove) = idxVert;
			}
			vertexFaces[idxVert] = std::move(vertexFaces.back());
			vertexFaces.pop_back();
			vertices[idxVert] = vertices.back();
			vertices.pop_back();
			VertexAttributesSwapPop(idxVert);
			idxVertLast = idxVert;
		}
		if (!vertexRemoves.empty())
			halfMesh.Clear();
		return;
	}
	std::vector<FIndex> faceRemoves;
	RFOREACHPTR (ptrIdxVert, vertexRemoves) {
		const VIndex idxVert = *ptrIdxVert;
		if (idxVertLast == idxVert)
			continue;
		const VIndex idxVertMove = vertices.size() - 1;
		if (idxVert < idxVertMove) {
			// update all faces of the moved vertex
			const VertexFaces& vfs = vertexFaces[idxVertMove];
			for (FIndex idxFace : vfs)
				FVertex(idxFace, idxVertMove) = idxVert;
		}
		faceRemoves.insert(faceRemoves.end(), vertexFaces[idxVert].begin(), vertexFaces[idxVert].end());
		vertexFaces[idxVert] = std::move(vertexFaces.back());
		vertexFaces.pop_back();
		vertices[idxVert] = vertices.back();
		vertices.pop_back();
		VertexAttributesSwapPop(idxVert);
		idxVertLast = idxVert;
	}
	RemoveFaces(faceRemoves);
	if (!vertexRemoves.empty())
		halfMesh.Clear();
}

void Mesh::RemoveFaces(std::vector<FIndex>& faceRemoves, bool updateLists)
{
	SyncFaces();
	const auto RemoveAt = [this](FIndex idxFace) {
		ASSERT(idxFace < faces.size());
		if (!faceTexcoords.empty()) {
			const FIndex idxT(idxFace * 3);
			ASSERT(faceTexcoords.size() == faces.size() * 3);
			// swap this face's 3 per-corner UVs with the last face's, then drop
			// the tail; must index the corner slot (idxT + i), not the face id.
			for (int i = 2; i >= 0; --i) {
				faceTexcoords[idxT + i] = faceTexcoords.back();
				faceTexcoords.pop_back();
			}
		}
		// faceTexblobs is independent of faceTexcoords: a single-texture mesh
		// has per-corner UVs but an EMPTY texblob array (empty == all blob 0), so
		// guard it on its own or we index an empty vector.
		if (!faceTexblobs.empty()) {
			ASSERT(faceTexblobs.size() == faces.size());
			faceTexblobs[idxFace] = faceTexblobs.back();
			faceTexblobs.pop_back();
		}
		if (!faceNormals.empty()) {
			faceNormals[idxFace] = faceNormals.back();
			faceNormals.pop_back();
		}
		faces[idxFace] = faces.back();
		faces.pop_back();
	};
	std::sort(faceRemoves.begin(), faceRemoves.end());
	FIndex idxFaceLast = NO_ID;
	if (!updateLists || vertexFaces.empty()) {
		RFOREACHPTR (ptrIdxFace, faceRemoves) {
			const FIndex idxFace = *ptrIdxFace;
			if (idxFaceLast == idxFace)
				continue;
			RemoveAt(idxFace);
			idxFaceLast = idxFace;
		}
		vertexFaces.clear();
	} else if (faceRemoves.size() > faces.size() / 10) {
		// Bulk-update threshold: removing a large fraction of faces, one O(F)
		// ListVertexFaces() rebuild beats per-face incremental surgery (each
		// removal is up to 6 lower_bound scans + list edits).  The compacted
		// `faces` array is identical to the incremental path's, and
		// ListVertexFaces is a deterministic ascending function of it, so the
		// resulting vertexFaces is byte-identical.
		ASSERT(vertices.size() == vertexFaces.size());
		RFOREACHPTR (ptrIdxFace, faceRemoves) {
			const FIndex idxFace = *ptrIdxFace;
			if (idxFaceLast == idxFace)
				continue;
			RemoveAt(idxFace);
			idxFaceLast = idxFace;
		}
		ListVertexFaces();
	} else {
		ASSERT(vertices.size() == vertexFaces.size());
		RFOREACHPTR (ptrIdxFace, faceRemoves) {
			const FIndex idxFace = *ptrIdxFace;
			if (idxFaceLast == idxFace)
				continue;
			{
				// remove face from vertex face list
				const Face& face = faces[idxFace];
				for (int v = 0; v < 3; ++v) {
					const VIndex idxVert = face[v];
					VertexFaces& vfs = vertexFaces[idxVert];
					VertexFaces::iterator it = std::lower_bound(vfs.begin(), vfs.end(), idxFace);
					if (it != vfs.end() && *it == idxFace) {
						ASSERT(it + 1 == vfs.end() || *(it + 1) != idxFace);
						vfs.erase(it);
					}
				}
			}
			const FIndex idxFaceMove = static_cast<FIndex>(faces.size() - 1);
			if (idxFace < idxFaceMove) {
				// update all vertices of the moved face
				const Face& face = faces[idxFaceMove];
				for (int v = 0; v < 3; ++v) {
					const VIndex idxVert = face[v];
					VertexFaces& vfs = vertexFaces[idxVert];
					VertexFaces::iterator it = std::lower_bound(vfs.begin(), vfs.end(), idxFaceMove);
					if (it != vfs.end() && *it == idxFaceMove) {
						ASSERT(it + 1 == vfs.end() || *(it + 1) != idxFaceMove);
						// idxFaceMove is the global max face index, hence the last
						// entry of this sorted list: drop it and re-insert idxFace
						// at its lower_bound slot (O(d) memmove, no re-sort).
						vfs.erase(it);
						VertexFaces::iterator pos = std::lower_bound(vfs.begin(), vfs.end(), idxFace);
						vfs.insert(pos, idxFace);
					}
				}
			}
			RemoveAt(idxFace);
			idxFaceLast = idxFace;
		}
	}
	if (!faceRemoves.empty())
		halfMesh.Clear();
}

bool Mesh::RemoveFacesHalfEdgeImpl(std::vector<FIndex>& faceRemoves, std::vector<VIndex>& removedVerts, std::vector<VIndex>& splitSrcVerts)
{
	ASSERT(!halfMesh.Empty());
	ASSERT(ValidateInvariants());
	const FIndex initialFaces = halfMesh.FSize();
	halfMesh.FRemoveBulk(faceRemoves, removedVerts, splitSrcVerts);
	if (halfMesh.FSize() == initialFaces)
		return false;
	for (VIndex source : splitSrcVerts) {
		vertices.emplace_back(vertices[source]);
		VertexAttributesAppendFrom(source);
	}
	for (VIndex vertex : removedVerts) {
		vertices[vertex] = vertices.back();
		vertices.pop_back();
		VertexAttributesSwapPop(vertex);
	}
	InvalidateFaces();
	ASSERT(vertices.size() == halfMesh.VSize());
	ASSERT(ValidateInvariants());
	return true;
}

void Mesh::RemoveFacesHalfEdge(std::vector<FIndex>& faceRemoves)
{
	std::vector<VIndex> removedVerts;
	std::vector<VIndex> splitSrcVerts;
	RemoveFacesHalfEdgeImpl(faceRemoves, removedVerts, splitSrcVerts);
	// Public exit: harvest unless a BeginHalfEdgePipeline scope owns the single
	// final harvest. Unconditional, so a half-edge-only entry that removed
	// nothing still leaves the caller with a face snapshot.
	SyncFacesOnPublicExit();
}

} // namespace halfmesh
