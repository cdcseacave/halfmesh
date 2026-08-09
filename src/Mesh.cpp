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

#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <cstdint>
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
	faceNormals = std::vector<Normal>();
	faceTexcoords = std::vector<TexCoord>();
	faceTexblobs = std::vector<FIndex>();
	texturesDiffuse = std::vector<Image3u>();
	vertexFaces = std::vector<VertexFaces>();
}

void Mesh::ComputeFaceNormals()
{
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
	if (faceNormals.size() != faces.size())
		ComputeFaceNormals();
	// Angle-weighted (Thurmer & Wuthrich 1998) pseudonormals: weight each
	// incident (possibly smoothed) face normal by the triangle's corner angle at
	// the vertex.  This is tessellation-independent -- unlike uniform averaging,
	// a fan of slivers on one side cannot bias the normal -- while still
	// honoring ComputeSmoothFaceNormals output (we scale the cached face normal,
	// not a fresh cross product).
	std::vector<Normal> vertexNormals(vertices.size(), Normal::Zero());
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
			vertexNormals[face[i]] += faceNormal * std::acos(cosAngle);
		}
	}
	for (Normal& vertexNormal : vertexNormals)
		vertexNormal.normalize();
	return vertexNormals;
}

real Mesh::ComputeArea() const
{
	real area(0);
	for (const Face& face : faces)
		area += ComputeFaceDoubleArea(face);
	return area * real(0.5);
}

real Mesh::ComputeArea(const std::vector<FIndex>& indices) const
{
	real area(0);
	for (FIndex idxFace : indices)
		area += ComputeFaceDoubleArea(idxFace);
	return area * real(0.5);
}

Eigen::AlignedBox<Mesh::Type, 3> Mesh::ComputeAABBox() const
{
	Eigen::AlignedBox<Type, 3> bbox;
	for (const Vertex& vert : vertices)
		bbox.extend(vert);
	return bbox;
}

void Mesh::ListVertexFaces()
{
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
	// Freshness gate: vertex count alone misses face-only mutations (RemoveFaces
	// leaves halfMesh untouched by design — see RemoveSmallestComponents, which
	// keeps using the old structure mid-surgery and clears it itself). Compare
	// BOTH counts so any face add/remove forces a rebuild here.
	if (halfMesh.vHalfedges.size() == vertices.size() && halfMesh.FSize() == faces.size())
		return;
	// Fast path: Build assumes a manifold mesh but now *detects* non-manifold
	// input cheaply and returns false instead of producing a corrupt structure
	// (which used to hang the adjacency walk).  On failure, repair to manifold
	// and rebuild via the safe path, so any mesh handed to the half-edge code is
	// checked & fixed first.  A known-manifold mesh pays only the inline checks.
	if (!halfMesh.Build(*this)) {
		REPORT_WARNING("non-manifold mesh; repairing to manifold before half-edge build");
		ListHalfEdgesSafe();
	}
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
	ASSERT(!halfMesh.Empty());
	HalfMesh::RemovedData removedData;
	halfMesh.ERemove(iE, removedData);
	ASSERT(removedData.numVerts == 1);
	vertices[removedData.verts[0]] = vertices.back();
	vertices.pop_back();
	// vertexColors moves in lockstep with the position swap-pop, mirroring
	// RemoveUnreferencedVertices/RemoveVertices.
	if (!vertexColors.empty()) {
		vertexColors[removedData.verts[0]] = vertexColors.back();
		vertexColors.pop_back();
	}
}

Mesh::VIndex Mesh::RemoveUnreferencedVertices()
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
			if (!vertexColors.empty()) {
				vertexColors[idxVert] = vertexColors.back();
				vertexColors.pop_back();
			}
			++numVerticesRemoved;
		}
	}
	return numVerticesRemoved;
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
			if (!vertexColors.empty()) {
				vertexColors[idxVert] = vertexColors.back();
				vertexColors.pop_back();
			}
			idxVertLast = idxVert;
		}
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
		if (!vertexColors.empty()) {
			vertexColors[idxVert] = vertexColors.back();
			vertexColors.pop_back();
		}
		idxVertLast = idxVert;
	}
	RemoveFaces(faceRemoves);
}

void Mesh::RemoveFaces(std::vector<FIndex>& faceRemoves, bool updateLists)
{
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
}

} // namespace halfmesh
