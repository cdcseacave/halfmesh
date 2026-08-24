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
#include <mutex>
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

void Mesh::InvalidateFaces()
{
	const bool droppedAttributes = !faceTexcoords.empty() || !faceTexblobs.empty() || !faceNormals.empty() || !texturesDiffuse.empty();
	faces.clear();
	faceTexcoords.clear();
	faceTexblobs.clear();
	faceNormals.clear();
	texturesDiffuse.clear();
	vertexFaces.clear();
	if (droppedAttributes) {
		static std::once_flag warningFlag;
		std::call_once(warningFlag, []() {
			REPORT_WARNING("face attributes dropped: processing methods expect untextured meshes");
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
	return (faces.empty() || halfMesh.Empty() || faces.size() == halfMesh.FSize()) && (halfMesh.Empty() || vertices.size() == halfMesh.VSize());
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
	const_cast<Mesh*>(this)->SyncFaces();
	real area(0);
	for (const Face& face : faces)
		area += ComputeFaceDoubleArea(face);
	return area * real(0.5);
}

real Mesh::ComputeArea(const std::vector<FIndex>& indices) const
{
	const_cast<Mesh*>(this)->SyncFaces();
	real area(0);
	for (FIndex idxFace : indices)
		area += ComputeFaceDoubleArea(idxFace);
	return area * real(0.5);
}

Mesh::Type Mesh::ComputeMeanEdgeLength()
{
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
	const_cast<Mesh*>(this)->SyncFaces();
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
	// vertexColors moves in lockstep with the position swap-pop, mirroring
	// RemoveUnreferencedVertices/RemoveVertices.
	if (!vertexColors.empty()) {
		vertexColors[removedData.verts[0]] = vertexColors.back();
		vertexColors.pop_back();
	}
	InvalidateFaces();
	SyncFaces();
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
			if (!vertexColors.empty()) {
				vertexColors[idxVert] = vertexColors.back();
				vertexColors.pop_back();
			}
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
		if (!vertexColors.empty()) {
			vertexColors[removed] = vertexColors.back();
			vertexColors.pop_back();
		}
	}
	if (!removedVerts.empty())
		InvalidateFaces();
	SyncFacesOnPublicExit();
	ASSERT(ValidateHalfMesh());
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
			if (!vertexColors.empty()) {
				vertexColors[idxVert] = vertexColors.back();
				vertexColors.pop_back();
			}
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
		if (!vertexColors.empty()) {
			vertexColors[idxVert] = vertexColors.back();
			vertexColors.pop_back();
		}
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
		if (!vertexColors.empty())
			vertexColors.emplace_back(vertexColors[source]);
	}
	for (VIndex vertex : removedVerts) {
		vertices[vertex] = vertices.back();
		vertices.pop_back();
		if (!vertexColors.empty()) {
			vertexColors[vertex] = vertexColors.back();
			vertexColors.pop_back();
		}
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
	if (RemoveFacesHalfEdgeImpl(faceRemoves, removedVerts, splitSrcVerts))
		SyncFaces();
}

} // namespace halfmesh
