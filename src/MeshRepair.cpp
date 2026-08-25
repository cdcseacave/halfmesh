/*
* MeshRepair.cpp
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
#include <halfmesh/Util/Accumulator.h>
#include <halfmesh/Util/Hash.h> // std::hash<std::tuple<...>>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace math;

namespace halfmesh {

// ---------------------------------------------------------------------------
// ListHalfEdgesSafe (original line 635)
// Moved here from mesh_core because it calls FixNonManifold.
// ---------------------------------------------------------------------------
void Mesh::ListHalfEdgesSafe()
{
	SyncFaces();
	halfMesh.Clear();
	// Make the mesh manifold so the (manifold-only) half-edge build succeeds.
	// Order matters: weld coincident vertices first (recovers shared edges from
	// seam-split / per-texture sub-meshes), then drop duplicate and degenerate
	// faces (HalfMesh::Build rejects self-edge/duplicate-directed-edge faces),
	// then separate the remaining non-manifold fans/bow-ties.
	//
	// RemoveDegenerateFaces(0.f) — thArea=0 — is topology-only: it removes ONLY
	// faces with a repeated vertex index (the self-edge faces Build rejects) and
	// keeps every distinct-index face, including tiny slivers. Auto-repair must be
	// geometry-preserving (a photogrammetry mesh can have millions of valid
	// near-zero-area faces); the default 1e-5 area cull would silently delete them
	// and punch holes. Deliberate sliver-cleaning stays opt-in via an explicit
	// RemoveDegenerateFaces(th) call.
	RemoveDuplicateVertices();
	// keep ONE copy of each duplicated face (removeBothFaces=false): the
	// default removes every copy, deleting valid surface — auto-repair must be
	// geometry-preserving (same principle as the thArea=0 choice below), and
	// one surviving copy has unique directed edges, satisfying Build.
	RemoveDuplicateFaces(false);
	RemoveDegenerateFaces(0.f);
	RemoveUnreferencedVertices();
	FixNonManifold();
	if (!halfMesh.Build(*this))
		REPORT_WARNING("mesh still non-manifold after repair; half-edge build incomplete");
}

// ---------------------------------------------------------------------------
// IsManifold — cheap O(faces) manifoldness test (no half-edge build).
// ---------------------------------------------------------------------------
bool Mesh::IsManifold() const
{
	SyncFacesConst();
	// Edge-non-manifold iff: a self-edge face (a==b), a directed edge used by >1
	// face (edge shared by >2 faces or a duplicate face). Necessary but NOT
	// sufficient for HalfMesh::Build: Build additionally rejects non-manifold
	// VERTICES (bow-ties) via its single-fan walk — see the doc on the
	// declaration in Mesh.h and the counterexample there.
	std::unordered_map<uint64_t, uint8_t> directed;
	directed.reserve(faces.size() * 3);
	for (const Face& face : faces) {
		for (int e = 0; e < 3; ++e) {
			const VIndex a = face[e], b = face[(e + 1) % 3];
			if (a == b)
				return false; // self-edge
			const uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
			if (++directed[key] > 1)
				return false; // directed edge reused => non-manifold
		}
	}
	return true;
}

// ---------------------------------------------------------------------------
// RemoveDuplicateVertices — weld spatially-coincident vertices.
// ---------------------------------------------------------------------------
Mesh::VIndex Mesh::RemoveDuplicateVertices(Type epsilon)
{
	SyncFaces();
	if (vertices.empty())
		return 0;
	const VIndex n = static_cast<VIndex>(vertices.size());
	const bool hasColors = vertexColors.size() == vertices.size();
	std::vector<VIndex> remap(n);
	std::vector<Vertex> unique;
	std::vector<Pixel> uniqueColors;
	unique.reserve(n);
	if (hasColors)
		uniqueColors.reserve(n);

	// Key each vertex by its (optionally grid-snapped) coordinates.  glTF and
	// other seam-splitting formats duplicate positions bit-for-bit, so the
	// exact-match (epsilon == 0) path recovers the shared vertices perfectly.
	const auto addUnique = [&](VIndex i) {
		remap[i] = static_cast<VIndex>(unique.size());
		unique.emplace_back(vertices[i]);
		if (hasColors)
			uniqueColors.emplace_back(vertexColors[i]);
	};
	if (epsilon <= 0) {
		std::unordered_map<std::tuple<uint32_t, uint32_t, uint32_t>, VIndex> seen;
		seen.reserve(static_cast<std::size_t>(n) * 2);
		FOREACHIDX (VIndex, i, vertices) {
			const Vertex& v = vertices[i];
			const std::tuple<uint32_t, uint32_t, uint32_t> key{
			    std::bit_cast<uint32_t>(v.x()), std::bit_cast<uint32_t>(v.y()),
			    std::bit_cast<uint32_t>(v.z())};
			const auto [it, inserted] = seen.emplace(key, static_cast<VIndex>(unique.size()));
			if (inserted)
				addUnique(i);
			else
				remap[i] = it->second;
		}
	} else {
		// epsilon>0: snap to an epsilon-sized grid, but (1) compute the cell index in
		// DOUBLE — v.x()*inv done in float loses a whole cell once |coord/eps| > 2^23
		// (ulp exceeds the grid step), so large-coordinate models bucket distant
		// vertices together — and (2) probe the +/-1 neighbor cells with an exact
		// squared-distance test, making the tolerance a true radius: near pairs that
		// straddle a cell boundary still weld, and far pairs at large coordinates that
		// happen to share a coarse key do NOT.
		const double epsD = static_cast<double>(epsilon);
		const double epsSq = epsD * epsD;
		using Cell = std::tuple<int64_t, int64_t, int64_t>;
		const auto cellOf = [epsD](const Vertex& v) -> Cell {
			return Cell{static_cast<int64_t>(std::llround(static_cast<double>(v.x()) / epsD)),
			            static_cast<int64_t>(std::llround(static_cast<double>(v.y()) / epsD)),
			            static_cast<int64_t>(std::llround(static_cast<double>(v.z()) / epsD))};
		};
		std::unordered_map<Cell, std::vector<VIndex>> seen; // cell -> unique-vertex reps
		seen.reserve(static_cast<std::size_t>(n) * 2);
		FOREACHIDX (VIndex, i, vertices) {
			const Vertex& v = vertices[i];
			const auto [cx, cy, cz] = cellOf(v);
			VIndex match = NO_ID;
			// probe the 3x3x3 neighborhood in a fixed order (determinism); the first
			// existing representative within epsilon wins.
			for (int64_t dz = -1; dz <= 1 && match == NO_ID; ++dz)
				for (int64_t dy = -1; dy <= 1 && match == NO_ID; ++dy)
					for (int64_t dx = -1; dx <= 1 && match == NO_ID; ++dx) {
						const auto it = seen.find(Cell{cx + dx, cy + dy, cz + dz});
						if (it == seen.end())
							continue;
						for (VIndex u : it->second) {
							const Vertex& w = unique[u];
							const double ddx = static_cast<double>(v.x()) - static_cast<double>(w.x());
							const double ddy = static_cast<double>(v.y()) - static_cast<double>(w.y());
							const double ddz = static_cast<double>(v.z()) - static_cast<double>(w.z());
							if (ddx * ddx + ddy * ddy + ddz * ddz <= epsSq) {
								match = u;
								break;
							}
						}
					}
			if (match == NO_ID) {
				const VIndex u = static_cast<VIndex>(unique.size());
				addUnique(i);
				seen[Cell{cx, cy, cz}].push_back(u);
			} else {
				remap[i] = match;
			}
		}
	}

	if (unique.size() == vertices.size())
		return 0; // nothing coincident

	const VIndex removed = n - static_cast<VIndex>(unique.size());
	for (Face& face : faces)
		for (int k = 0; k < 3; ++k)
			face[k] = remap[face[k]];
	vertices.swap(unique);
	if (hasColors)
		vertexColors.swap(uniqueColors);
	// vertex indices changed: any cached adjacency / half-edge is now stale.
	vertexFaces = std::vector<VertexFaces>();
	halfMesh.Clear();
	return removed;
}

// ---------------------------------------------------------------------------
// RemoveDuplicateFaces (original line 938)
// ---------------------------------------------------------------------------
Mesh::FIndex Mesh::RemoveDuplicateFaces(bool removeBothFaces)
{
	SyncFaces();
	struct SortedFace
	{
		union {
			struct
			{
				VIndex v0, v1, v2;
			};
			VIndex v[3];
		};
		SortedFace() {}
		explicit SortedFace(VIndex _v0, VIndex _v1, VIndex _v2) :
		    v0(_v0), v1(_v1), v2(_v2)
		{
			std::sort(v, v + 3);
		}
		bool operator<(const SortedFace& f) const
		{
			return v2 != f.v2 ? v2 < f.v2 : (v1 != f.v1 ? v1 < f.v1 : v0 < f.v0);
		}
		bool operator==(const SortedFace& f) const
		{
			return v0 == f.v0 && v1 == f.v1 && v2 == f.v2;
		}
	};
	const FIndex numFaces = faces.size();
	// FIndex is unsigned: with 0 faces the `numFaces - 1` loop bound below
	// wraps to 0xFFFFFFFF and indexes empty vectors. Fewer than 2 faces can
	// have no duplicates — return before doing any work.
	if (numFaces < 2)
		return 0;
	std::vector<SortedFace> sortedFaces;
	sortedFaces.reserve(numFaces);
	for (const Face& face : faces) {
		sortedFaces.emplace_back(face.x(), face.y(), face.z());
	}
	std::vector<FIndex> indices(numFaces);
	std::iota(indices.begin(), indices.end(), FIndex(0));
	std::sort(indices.begin(), indices.end(), [&sortedFaces](FIndex i, FIndex j) {
		return sortedFaces[i] < sortedFaces[j];
	});
	std::vector<FIndex> faceRemoves;
	for (FIndex i = 0; i < numFaces - 1; ++i) {
		if (sortedFaces[indices[i]] == sortedFaces[indices[i + 1]]) {
			if (removeBothFaces) {
				// next check required if the input mesh is not manifold
				if (faceRemoves.empty() || faceRemoves.back() != indices[i])
					faceRemoves.emplace_back(indices[i]);
				faceRemoves.emplace_back(indices[i + 1]);
			} else {
				faceRemoves.emplace_back(indices[i]);
			}
		}
	}
	if (faceRemoves.empty())
		return 0;
	RemoveFaces(faceRemoves, true);
	return faceRemoves.size();
}

// ---------------------------------------------------------------------------
// RemoveDegenerateFaces(Type thArea) (original line 989)
// ---------------------------------------------------------------------------
Mesh::FIndex Mesh::RemoveDegenerateFaces(Type thArea)
{
	return halfMesh.Empty() ? RemoveDegenerateFacesArrays(thArea) : RemoveDegenerateFacesHalfEdge(thArea);
}

Mesh::FIndex Mesh::RemoveDegenerateFacesArrays(Type thArea)
{
	SyncFaces();
	if (vertexFaces.size() != vertices.size())
		ListVertexFaces();
	const Type thDoubleAreaSq = SQUARE(thArea * 2);
	std::vector<FIndex> faceRemoves;
	std::vector<std::pair<VIndex /*replace with*/, VIndex>> vertexPairs;
	RFOREACHIDX (FIndex, idxFace, faces) {
		const Face& face = faces[idxFace];
		// check first case when one or more vertices have same index
		if (face[0] == face[1] || face[0] == face[2] || face[1] == face[2]) {
			// just remove the face
			faceRemoves.emplace_back(idxFace);
			continue;
		}
		if (thDoubleAreaSq <= 0)
			continue;
		// check if the face has almost 0 area (see EdgeFunction())
		const Vertex& v0 = vertices[face[0]];
		const Vertex& v1 = vertices[face[1]];
		const Vertex& v2 = vertices[face[2]];
		const Vertex A(v2 - v0);
		const Vertex B(v1 - v0);
		const Type doubleAreaSq = B.cross(A).squaredNorm();
		if (doubleAreaSq <= thDoubleAreaSq) {
			// remove the face
			faceRemoves.emplace_back(idxFace);
			const Type lengthSqA = A.squaredNorm();
			const Type lengthSqB = B.squaredNorm();
			const Type lengthSqC = (v2 - v1).squaredNorm();
			// remove two of the vertices,
			// moving all adjacent face to the remaining vertex
			if (lengthSqA <= thArea && lengthSqB <= thArea) {
				vertexPairs.emplace_back(face[2], face[0]);
				vertexPairs.emplace_back(face[1], face[0]);
			} else if (lengthSqA <= thArea && lengthSqC <= thArea) {
				vertexPairs.emplace_back(face[0], face[2]);
				vertexPairs.emplace_back(face[1], face[2]);
			} else if (lengthSqB <= thArea && lengthSqC <= thArea) {
				vertexPairs.emplace_back(face[0], face[1]);
				vertexPairs.emplace_back(face[2], face[1]);
			} else
				// remove one of the vertices,
				// moving all adjacent face to the closest remaining vertices
				if (lengthSqA <= thArea) {
					vertexPairs.emplace_back(face[2], face[0]);
				} else if (lengthSqB <= thArea) {
					vertexPairs.emplace_back(face[1], face[0]);
				} else if (lengthSqC <= thArea) {
					vertexPairs.emplace_back(face[1], face[2]);
				} else
				// the vertices are (almost) collinear, remove the smallest edge
				{
					const Type cosLengthAB = A.dot(B);
					(void)cosLengthAB;
					if (lengthSqA < lengthSqB) {
						if (lengthSqA < lengthSqC)
							vertexPairs.emplace_back(face[2], face[0]);
						else
							vertexPairs.emplace_back(face[2], face[1]);
					} else {
						if (lengthSqB < lengthSqC)
							vertexPairs.emplace_back(face[1], face[0]);
						else
							vertexPairs.emplace_back(face[2], face[1]);
					}
				}
		}
	}
	if (faceRemoves.empty())
		return 0;
	RemoveFaces(faceRemoves, true);
	if (vertexPairs.empty()) {
		REPORT_STATUS_NOW("Removed {} degenerate faces", faceRemoves.size());
		return faceRemoves.size();
	}
	// replace first vertex with the second
	std::vector<VIndex> mapRemovedVerts(vertices.size(), NO_ID);
	const auto TraceMovedVertex = [&mapRemovedVerts](VIndex idx) {
		while (mapRemovedVerts[idx] != NO_ID)
			idx = mapRemovedVerts[idx];
		return idx;
	};
	std::sort(vertexPairs.begin(), vertexPairs.end());
	vertexPairs.erase(std::unique(vertexPairs.begin(), vertexPairs.end()), vertexPairs.end());
	RFOREACHPTR (ptrIdxPair, vertexPairs) {
		auto p = *ptrIdxPair;
		p.first = TraceMovedVertex(p.first);
		p.second = TraceMovedVertex(p.second);
		if (p.first == p.second)
			continue;
		VertexFaces& firstVfs = vertexFaces[p.first];
		for (const FIndex idxFace : firstVfs) {
			Face& face = faces[idxFace];
			for (VIndex i = 0; i < 3; ++i)
				if (face[i] == p.first)
					face[i] = p.second;
		}
		VertexFaces& secondVfs = vertexFaces[p.second];
		secondVfs.insert(secondVfs.end(), firstVfs.begin(), firstVfs.end());
		std::sort(secondVfs.begin(), secondVfs.end());
		secondVfs.erase(std::unique(secondVfs.begin(), secondVfs.end()), secondVfs.end());
		firstVfs.clear();
		mapRemovedVerts[p.first] = p.second;
	}
	const size_t numRemovedFaces = faceRemoves.size() + RemoveDegenerateFacesArrays(0.f);
	REPORT_STATUS_NOW("Removed {} zero-area faces", numRemovedFaces);
	return numRemovedFaces;
}

Mesh::FIndex Mesh::RemoveDegenerateFacesHalfEdge(Type thArea)
{
	ASSERT(!halfMesh.Empty());
	ASSERT(halfMesh.VSize() == vertices.size());
	if (thArea <= 0) {
		SyncFacesOnPublicExit();
		return 0;
	}

	const Type thDoubleAreaSq = SQUARE(thArea * 2);
	const FIndex initialFaces = halfMesh.FSize();
	FIndex numCollapses = 0;
	FIndex numFlips = 0;
	bool topologyChanged = false;
	// Terminates: a collapse strictly shrinks the face count, and a flip is only
	// applied when BOTH resulting triangles clear the area threshold, so it makes
	// two degenerate faces healthy without moving a vertex (no face outside the
	// pair changes shape) and strictly shrinks the degenerate count. A round that
	// repairs nothing ends the sweep.
	for (;;) {
		bool repaired = false;
		for (FIndex idxFace = 0; idxFace < halfMesh.FSize();) {
			const HIndex firstHalfedge = halfMesh.FHalfedge(idxFace);
			HIndex halfedges[3];
			Type lengthsSq[3];
			HIndex halfedge = firstHalfedge;
			for (int i = 0; i < 3; ++i) {
				halfedges[i] = halfedge;
				const auto edgeVertices = halfMesh.EVertices(halfMesh.HeEdge(halfedge));
				lengthsSq[i] = (vertices[edgeVertices.first] - vertices[edgeVertices.second]).squaredNorm();
				halfedge = halfMesh.HeNext(halfedge);
			}
			const Face face = halfMesh.F(idxFace);
			const Type doubleAreaSq = (vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]).squaredNorm();
			if (doubleAreaSq > thDoubleAreaSq) {
				++idxFace;
				continue;
			}

			int shortest = 0;
			int longest = 0;
			for (int i = 1; i < 3; ++i) {
				if (lengthsSq[i] < lengthsSq[shortest])
					shortest = i;
				if (lengthsSq[i] > lengthsSq[longest])
					longest = i;
			}

			// A near-zero-area triangle is either a NEEDLE (one edge collapses to
			// nothing) or a CAP (three comparable edges around a ~180 degrees
			// corner). Classify by edge-length RATIO so the split does not depend on
			// the mesh's absolute scale, and try to replace a cap's long diagonal
			// before falling back to the needle's shortest-edge collapse.
			constexpr Type capAspectSq = Type(1) / Type(64); // shortest >= longest/8
			if (lengthsSq[shortest] >= lengthsSq[longest] * capAspectSq) {
				const EIndex longEdge = halfMesh.HeEdge(halfedges[longest]);
				if (halfMesh.EIsFlipValid(longEdge, vertices)) {
					const HIndex hA0 = halfMesh.EHalfedge(longEdge);
					const HIndex hB0 = halfMesh.HeTwin(hA0);
					const VIndex v0 = halfMesh.HeVertex(hA0);
					const VIndex v2 = halfMesh.HeVertex(hB0);
					const VIndex v1 = halfMesh.HeVertex(halfMesh.HeNext(halfMesh.HeNext(hA0)));
					const VIndex v3 = halfMesh.HeVertex(halfMesh.HeNext(halfMesh.HeNext(hB0)));
					const Type newAreaSq0 = (vertices[v3] - vertices[v0]).cross(vertices[v1] - vertices[v0]).squaredNorm();
					const Type newAreaSq1 = (vertices[v1] - vertices[v2]).cross(vertices[v3] - vertices[v2]).squaredNorm();
					if (newAreaSq0 > thDoubleAreaSq && newAreaSq1 > thDoubleAreaSq) {
						// a flip keeps every face index, so the scan simply advances
						halfMesh.EFlip(longEdge);
						++numFlips;
						topologyChanged = repaired = true;
						++idxFace;
						continue;
					}
				}
			}

			const EIndex shortEdge = halfMesh.HeEdge(halfedges[shortest]);
			if (!halfMesh.EIsCollapseValidTopologically(shortEdge)) {
				++idxFace;
				continue;
			}
			const auto edgeVertices = halfMesh.EVertices(shortEdge);
			const Vertex midpoint = (vertices[edgeVertices.first] + vertices[edgeVertices.second]) * Type(0.5);
			if (!halfMesh.EIsCollapseValidGeometrically(shortEdge, midpoint, vertices)) {
				++idxFace;
				continue;
			}

			HalfMesh::RemovedData removedData;
			const VIndex vertexMoved = halfMesh.ERemove(shortEdge, removedData);
			ASSERT(removedData.numVerts == 1);
			vertices[removedData.verts[0]] = vertices.back();
			vertices.pop_back();
			if (!vertexColors.empty()) {
				vertexColors[removedData.verts[0]] = vertexColors.back();
				vertexColors.pop_back();
			}
			vertices[vertexMoved] = midpoint;
			++numCollapses;
			topologyChanged = repaired = true;
			// the collapse swap-popped faces from the end into the freed slots, so
			// idxFace may now hold a different face: re-test it instead of advancing
			// (faces relocated BEFORE the cursor are caught by the next round)
		}
		if (!repaired)
			break;
	}

	const FIndex numRemovedFaces = initialFaces - halfMesh.FSize();
	if (topologyChanged)
		InvalidateFaces();
	SyncFacesOnPublicExit();
	// cheap contract check only; the O(F log F) ValidateHalfMesh() rebuild-and-compare
	// is run by the test suite after every native mutator (see tests/AGENTS.md)
	ASSERT(ValidateInvariants());
	if (numRemovedFaces > 0 || numFlips > 0)
		REPORT_STATUS_NOW("Repaired degenerate topology ({} faces removed by {} collapses, {} cap flips)", numRemovedFaces, numCollapses, numFlips);
	return numRemovedFaces;
}

// ---------------------------------------------------------------------------
// RemoveDegenerateFaces(unsigned maxIterations, Type thArea) (original line 1114)
// ---------------------------------------------------------------------------
Mesh::FIndex Mesh::RemoveDegenerateFaces(unsigned maxIterations, Type thArea)
{
	FIndex totalNumRemovedFaces = 0;
	for (unsigned iter = 0; iter < maxIterations; ++iter) {
		const FIndex numRemovedFaces = RemoveDegenerateFaces(thArea);
		if (numRemovedFaces == 0)
			break;
		totalNumRemovedFaces += numRemovedFaces;
	}
	return totalNumRemovedFaces;
}

// ---------------------------------------------------------------------------
// RemoveFacesOutside (original line 1194)
// ---------------------------------------------------------------------------
unsigned Mesh::RemoveFacesOutside(const OBB& obb)
{
	SyncFaces();
	ASSERT(!obb.IsEmpty());
	std::vector<bool> insideVertices(vertices.size());
	FOREACH (i, vertices)
		insideVertices[i] = obb.Contains(vertices[i].cast<real>());
	std::vector<FIndex> faceRemoves;
	FOREACH (i, faces) {
		const Face& face = faces[i];
		if (!insideVertices[face.x()] || !insideVertices[face.y()] || !insideVertices[face.z()])
			faceRemoves.emplace_back(i);
	}
	RemoveFaces(faceRemoves);
	return faceRemoves.size();
}

// ---------------------------------------------------------------------------
// FixNonManifold (original line 1210)
// ---------------------------------------------------------------------------
unsigned Mesh::FixNonManifold(float thMoveDuplicate, std::vector<VIndex>* duplicatedVertices)
{
	if (!halfMesh.Empty())
		return 0;
	SyncFaces();
	// graceful no-op on empty input in every build mode (matches the smoothers'
	// early-return convention; an assert here diverged Debug from Release)
	if (vertices.empty() || faces.empty())
		return 0;
	TIMER_START("FixNonManifold");
	if (vertexFaces.size() != vertices.size())
		ListVertexFaces();
	// iterate over all vertices and
	// separates all components that are incident to the vertices by adding new vertices
	unsigned numNonManifold(0);
	std::vector<int> components(faces.size());
	FOREACHIDX (VIndex, idxVert, vertices) {
		// get reference to the vector of associated faces at the current vertex
		const VertexFaces& vfaces = vertexFaces[idxVert];
		// reset components to which each face connected to this vertex belongs
		for (FIndex iF : vfaces)
			components[iF] = -1;
		// find the components
		std::vector<FIndex> fqueue;
		fqueue.reserve(vfaces.size());
		FIndex idxFaceNext(0);
		int component(0);
		for (;; ++component) {
			// find one face not yet belonging to a component
			while (idxFaceNext < vfaces.size()) {
				const FIndex iF(vfaces[idxFaceNext++]);
				if (components[iF] == -1) {
					// add component as seed to the list
					fqueue.push_back(iF);
					// mark the current face with the right component
					components[iF] = component;
					// process component
					goto ProcessFace;
				}
			}
			// no more components found
			break;
		ProcessFace:
			// grow from the seed face until no more connected faces where found
			do {
				// get next element from the faces queue
				const FIndex idxFaceCurrent(fqueue.back());
				fqueue.pop_back();
				const Face& face = faces[idxFaceCurrent];
				// go over all vertices of the current face
				for (int i = 0; i < 3; ++i) {
					const VIndex idxVertAdj(face[i]);
					// if the vertex of the face is not equal to the current iteration vertex
					if (idxVertAdj != idxVert) {
						// if neighbor face found
						const FIndex idxFaceAdj(FEdgeAdjacentFace(idxFaceCurrent, idxVert, idxVertAdj));
						if (idxFaceAdj != NO_ID) {
							// tag it with the right component and push it to the faces queue
							if (components[idxFaceAdj] == -1) {
								components[idxFaceAdj] = component;
								fqueue.push_back(idxFaceAdj);
							}
						}
					}
				}
			} while (!fqueue.empty());
		}
		if (component <= 1)
			continue;
		// separate the components at the current vertex
		for (int c = 1; c < component; ++c) {
			// duplicate the point to achieve the separation
			const VIndex idxVertNew = static_cast<VIndex>(vertices.size());
			const Vertex v = vertices[idxVert];
			vertices.emplace_back(v);
			// the split copy carries the source vertex's colour, as in
			// RemoveFacesHalfEdgeImpl's pinch splits
			if (!vertexColors.empty())
				vertexColors.emplace_back(vertexColors[idxVert]);
			if (duplicatedVertices)
				duplicatedVertices->emplace_back(idxVert);
			// update the face indices of the current component
			VertexFaces& vfacesMut = vertexFaces[idxVert];
			VertexFaces nvfaces;
			RFOREACH (i, vfacesMut) {
				const FIndex idxFace = vfacesMut[i];
				// if the face belongs to the current component
				if (components[idxFace] == c) {
					Face& face = faces[idxFace];
					// iterate over its vertices
					for (int j = 0; j < 3; ++j) {
						// relink to the new vertex by updating the face and pushing it in as new
						if (face[j] == idxVert) {
							face[j] = idxVertNew;
							nvfaces.insert(nvfaces.begin(), idxFace);
							break;
						}
					}
					// erase the old one
					vfacesMut.erase(vfacesMut.begin() + i);
				}
			}
			vertexFaces.emplace_back(std::move(nvfaces));
			++numNonManifold;
		}
		// adjust vertex positions
		if (thMoveDuplicate > 0) {
			// list affected vertices
			std::vector<VIndex> verts(component);
			verts[0] = idxVert;
			for (int c = 1; c < component; ++c)
				verts[c] = static_cast<VIndex>(vertices.size()) - (component - c);
			// for each vertex, compute the direction to the center of the first ring,
			// and adjust its position towards it
			FOREACH (i, verts) {
				const VIndex idxVertI(verts[i]);
				std::vector<VIndex> indices = VAdjacentVertices(idxVertI);
				WeightedAccumulator<Vertex> accum;
				for (VIndex iV : indices)
					accum.Add(vertices[iV]);
				const Vertex bv(accum.Normalized());
				Vertex& vv(vertices[idxVertI]);
				const Vertex dir(bv - vv);
				vv += dir * thMoveDuplicate;
			}
		}
	}
	vertexFaces = std::vector<VertexFaces>();
	if (numNonManifold > 0)
		halfMesh.Clear();
	REPORT_STATUS_NOW("Fixed {} non-manifold issues ({})",
	                  numNonManifold, TIMER_STR());
	return numNonManifold;
}

// ---------------------------------------------------------------------------
// RemoveSmallComponents (original line 1333)
// ---------------------------------------------------------------------------
unsigned Mesh::RemoveSmallComponents(unsigned minComponentSize)
{
	TIMER_START("RemoveSmallComponents");
	ListHalfEdges();
	std::vector<FIndex> components;
	const FIndex numComponents = halfMesh.ConnectedComponents(components);
	if (numComponents <= 1) {
		SyncFacesOnPublicExit();
		return 0;
	}
	std::vector<unsigned> componentSizes(numComponents, 0);
	for (FIndex component : components)
		++componentSizes[component];
	const unsigned numSmallComponents = std::accumulate(componentSizes.begin(), componentSizes.end(), 0u,
	                                                    [minComponentSize](unsigned num, unsigned size) { return size < minComponentSize ? num + 1 : num; });
	if (numSmallComponents == 0) {
		SyncFacesOnPublicExit();
		return 0;
	}
	std::vector<FIndex> faceRemoves;
	faceRemoves.reserve(halfMesh.FSize());
	for (FIndex idxFace = 0; idxFace < halfMesh.FSize(); ++idxFace)
		if (componentSizes[components[idxFace]] < minComponentSize)
			faceRemoves.emplace_back(idxFace);
	std::vector<VIndex> removedVerts;
	std::vector<VIndex> splitSrcVerts;
	RemoveFacesHalfEdgeImpl(faceRemoves, removedVerts, splitSrcVerts);
	SyncFacesOnPublicExit();
	// cheap contract check only; the O(F log F) ValidateHalfMesh() rebuild-and-compare
	// is run by the test suite after every native mutator (see tests/AGENTS.md)
	ASSERT(ValidateInvariants());
	REPORT_STATUS_NOW("Removed {} small components from {} total ({})",
	                  numSmallComponents, numComponents, TIMER_STR());
	return numSmallComponents;
}

Mesh::FIndex Mesh::RemoveSpuriousComponents(float factor)
{
	if (factor <= 0.f) {
		SyncFacesOnPublicExit();
		return 0;
	}
	if (vertices.empty() || (faces.empty() && halfMesh.Empty()))
		return 0;
	TIMER_START("RemoveSpuriousComponents");
	const FIndex initialFaces = halfMesh.Empty() ? static_cast<FIndex>(faces.size()) : halfMesh.FSize();
	// ListHalfEdges, not ListHalfEdgesSafe: the manifold build is a fraction of the
	// cost of the weld/dedupe/repair sweep, and it falls back to the safe path by
	// itself when the input turns out to be non-manifold
	ListHalfEdges();
	if (halfMesh.Empty()) {
		SyncFacesOnPublicExit();
		return initialFaces - static_cast<FIndex>(faces.size());
	}

	std::vector<float> edgeLengths;
	edgeLengths.reserve(halfMesh.ESize());
	for (EIndex edge = 0; edge < halfMesh.ESize(); ++edge) {
		const auto verts = halfMesh.EVertices(edge);
		edgeLengths.emplace_back((vertices[verts.first] - vertices[verts.second]).norm());
	}
	if (edgeLengths.empty()) {
		SyncFacesOnPublicExit();
		return 0;
	}
	std::vector<float> percentiles(edgeLengths);
	const size_t idx95 = percentiles.size() * 95 / 100;
	const size_t idx55 = percentiles.size() * 55 / 100;
	std::nth_element(percentiles.begin(), percentiles.begin() + idx95, percentiles.end());
	const float maxEdgeLength = percentiles[idx95] * factor;
	// the prefix left by the pass above already holds the idx95 smallest lengths,
	// so the lower percentile only has to be selected within it
	std::nth_element(percentiles.begin(), percentiles.begin() + idx55, percentiles.begin() + idx95);
	const float minComponentDiameter = percentiles[idx55] * factor;

	std::vector<FIndex> removeFaces;
	for (EIndex edge = 0; edge < halfMesh.ESize(); ++edge) {
		if (edgeLengths[edge] <= maxEdgeLength)
			continue;
		for (HIndex iHe : halfMesh.EAdjacentInteriorHalfedges(edge))
			removeFaces.emplace_back(halfMesh.HeFace(iHe));
	}
	std::sort(removeFaces.begin(), removeFaces.end());
	removeFaces.erase(std::unique(removeFaces.begin(), removeFaces.end()), removeFaces.end());
	if (!removeFaces.empty()) {
		std::vector<VIndex> removedVerts;
		std::vector<VIndex> splitSrcVerts;
		RemoveFacesHalfEdgeImpl(removeFaces, removedVerts, splitSrcVerts);
	}
	if (halfMesh.Empty()) {
		SyncFacesOnPublicExit();
		return initialFaces;
	}

	std::vector<FIndex> components;
	const FIndex numComponents = halfMesh.ConnectedComponents(components);
	if (numComponents > 1) {
		std::vector<Eigen::AlignedBox<float, 3>> bounds(numComponents);
		for (FIndex idxFace = 0; idxFace < halfMesh.FSize(); ++idxFace) {
			const Face face = halfMesh.F(idxFace);
			for (int i = 0; i < 3; ++i)
				bounds[components[idxFace]].extend(vertices[face[i]]);
		}
		removeFaces.clear();
		for (FIndex idxFace = 0; idxFace < halfMesh.FSize(); ++idxFace) {
			const Eigen::AlignedBox<float, 3>& bound = bounds[components[idxFace]];
			if (!bound.isEmpty() && bound.diagonal().norm() < minComponentDiameter)
				removeFaces.emplace_back(idxFace);
		}
		if (!removeFaces.empty()) {
			std::vector<VIndex> removedVerts;
			std::vector<VIndex> splitSrcVerts;
			RemoveFacesHalfEdgeImpl(removeFaces, removedVerts, splitSrcVerts);
		}
	}

	const FIndex removed = initialFaces - halfMesh.FSize();
	SyncFacesOnPublicExit();
	// cheap contract check only; the O(F log F) ValidateHalfMesh() rebuild-and-compare
	// is run by the test suite after every native mutator (see tests/AGENTS.md)
	ASSERT(ValidateInvariants());
	if (removed > 0)
		REPORT_STATUS_NOW("Removed {} spurious faces ({})", removed, TIMER_STR());
	return removed;
}

unsigned Mesh::RemoveSpikes(unsigned maxIterations)
{
	return halfMesh.Empty() ? RemoveSpikesArrays(maxIterations) : RemoveSpikesHalfEdge(maxIterations);
}

unsigned Mesh::RemoveSpikesArrays(unsigned maxIterations)
{
	SyncFaces();
	if (vertices.empty())
		return 0;
	TIMER_START("RemoveSpikes");
	unsigned numSpikes = 0;
	for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
		if (vertexFaces.size() != vertices.size())
			ListVertexFaces();
		std::vector<VIndex> spikes;
		FOREACHIDX (VIndex, idxVert, vertices) {
			if (vertexFaces[idxVert].size() <= 1)
				spikes.emplace_back(idxVert);
		}
		if (spikes.empty())
			break;
		numSpikes += static_cast<unsigned>(spikes.size());
		// drops the spike vertices together with their incident face, which can
		// starve a neighbour down to a single face and expose it next round
		RemoveVertices(spikes, true);
	}
	if (numSpikes == 0)
		return 0;
	vertexFaces = std::vector<Mesh::VertexFaces>();
	halfMesh.Clear();
	SyncFaces();
	REPORT_STATUS_NOW("Removed {} spike vertices ({})", numSpikes, TIMER_STR());
	return numSpikes;
}

unsigned Mesh::RemoveSpikesHalfEdge(unsigned maxIterations)
{
	if (vertices.empty() || maxIterations == 0) {
		SyncFacesOnPublicExit();
		return 0;
	}
	ListHalfEdges();
	if (halfMesh.Empty()) {
		SyncFacesOnPublicExit();
		return 0;
	}
	TIMER_START("RemoveSpikesHalfEdge");
	unsigned numSpikes = 0;
	for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
		// A spike is a vertex left with a single incident face -- isolated slots
		// cannot occur in a valid half-edge, so the array arm's zero-face case has
		// no counterpart here. Dropping the face can starve a neighbour, so the
		// sweep repeats; restricting the re-scan to the neighbours of the faces
		// just removed would not pay, because the bulk removal below already
		// touches O(V+F) per round.
		std::vector<FIndex> faceRemoves;
		unsigned numRoundSpikes = 0;
		for (VIndex vertex = 0; vertex < halfMesh.VSize(); ++vertex) {
			if (halfMesh.VFaceDegree(vertex) != 1)
				continue;
			++numRoundSpikes;
			// a valence-1 vertex is on the boundary, so its representative is the
			// face-bearing side by the boundary-canonical invariant
			faceRemoves.emplace_back(halfMesh.HeFace(halfMesh.VHalfedge(vertex)));
		}
		if (faceRemoves.empty())
			break;
		std::vector<VIndex> removedVerts;
		std::vector<VIndex> splitSrcVerts;
		if (!RemoveFacesHalfEdgeImpl(faceRemoves, removedVerts, splitSrcVerts)) {
			ASSERT(false && "RemoveSpikes: a spike face was not removable");
			break;
		}
		numSpikes += numRoundSpikes;
	}

	SyncFacesOnPublicExit();
	// cheap contract check only; the O(F log F) ValidateHalfMesh() rebuild-and-compare
	// is run by the test suite after every native mutator (see tests/AGENTS.md)
	ASSERT(ValidateInvariants());
	if (numSpikes > 0)
		REPORT_STATUS_NOW("Removed {} spike vertices ({})", numSpikes, TIMER_STR());
	return numSpikes;
}

} // namespace halfmesh
