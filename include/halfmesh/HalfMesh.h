/*
* HalfMesh.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#pragma once

#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Types.h>

// enable HalfMesh data structure optimized for triangular faces
#ifndef HALFMESH_TRIS
	#define HALFMESH_TRIS 1
#endif

namespace halfmesh {

class Mesh;

// Structure used to encode a manifold mesh (triangles or polygons) using a
// half-edge representation in an original way that is both very compact and
// fast to use. All the walks over the mesh, ex. iterating over adjacent
// vertices, faces, edges, half-edges for vertices, faces, edges, have O(1)
// complexity, usually requiring only one look up per iteration.
// The connectivity is also compact as it represents many of the required
// structure in a implicit way using indices:
//  - the twin half-edges representing one edge are always stored consecutively;
//    first half-edge is from current (tail) to the next (head) vertex and
//    twin half-edge is from next (head) to the current (tail) vertex
//  - the half-edge to edge mapping and reverse is done by discarding the last bit;
//    an edge index is thus the half-edge index shifted 1 bit to the right
//  - an edge is on the border iff the face index corresponding the second half-edge
//    is not defined (math::NO_ID)
//  - a vertex is on the border iff the twin of its first half-edge is on the border
//  - adjacency loops are represented in the same way for on/off the border vertices
//    by artificially closing the loops by pointing the half-edge on the border to the
//    next face on the other side of the border vertex
// See https://geometry-central.net/surface/surface_mesh/basics/
// for a similar structure with more details.
class HalfMesh
{
	public:
	typedef float Type; // vertex precision
	typedef Eigen::Matrix<Type, 3, 1> Vertex; // vertex point
	typedef uint32_t VIndex; // vertex index
	typedef uint32_t FIndex; // face index
	typedef FIndex EIndex; // edge index
	typedef FIndex HIndex; // half-edge index
#if HALFMESH_TRIS
	typedef Eigen::Matrix<VIndex, 3, 1> Face; // vertex indices of the triangle
#else
	typedef Eigen::Matrix<VIndex, Eigen::Dynamic, 1> Face; // vertex indices of the polygon
#endif
	// return the indices of the various removed elements
	struct RemovedData
	{
		VIndex verts[3]; // indices of the one/two/three vertices removed
		EIndex edges[3]; // indices of the two/three edges removed
		FIndex faces[2]; // indices of the one/two faces removed
		uint8_t numVerts{0};
		uint8_t numEdges{0};
		uint8_t numFaces{0};
	};

	static_assert(sizeof(VIndex) == sizeof(decltype(math::NO_ID)), "NO_ID and vertex index type must match");
	static_assert(sizeof(FIndex) == sizeof(decltype(math::NO_ID)), "NO_ID and face index type must match");

	// map vertices to a representative OUTGOING half-edge (always valid).
	//
	// Build() establishes a stronger invariant: every representative is EVEN. This
	// holds because the first face that introduces a vertex v always creates the
	// edge v->next fresh with v as the tail, whose even half-edge becomes v's
	// representative. So immediately after a (re)build, vHalfedges is all-even.
	//
	// In-place mutations relax it: EFlip (and hence ESplit, cleaning up after it)
	// may leave an ODD outgoing half-edge on an INTERIOR vertex — and it can't be
	// fixed back to even in place, because the vertex may have no even outgoing
	// half-edge left and "even == EHalfedge == 2*edge" is positional (restoring it
	// requires renumbering, i.e. a fresh Build). Nothing reads the parity, so this
	// is benign; algorithms that want the all-even form back should rebuild.
	//
	// The part that IS always maintained (and load-bearing): for a BOUNDARY vertex
	// the representative is the canonical even (interior-face) half-edge of one of
	// its boundary edges, so VIsBoundary (== EHeIsBoundary(VHalfedge)) and
	// HePrevBoundary / boundary-loop iteration work. EFlip cannot corrupt this (it
	// only updates a representative equal to the flipped INTERIOR edge's
	// half-edge), and ESplit sets it canonically for the new vertex.
	std::vector<HIndex> vHalfedges;
	std::vector<HIndex> fHalfedges; // map faces to half-edge indices (all valid)
	std::vector<HIndex> heNexts; // next half-edge and twin half-edge indices (all valid)
	std::vector<VIndex> heVertices; // map half-edges to vertex indices (all valid)
	std::vector<FIndex> heFaces; // map half-edges to face indices (invalid only for border edges)

	// true iff every vHalfedges entry is an even (canonical) half-edge — see the
	// vHalfedges note above. Established by Build(), conservatively cleared by any
	// in-place mutation that stores an odd representative. Restore with
	// GuaranteeAlwaysEven(). (Conservative: false may be a false-negative; true is
	// always accurate.)
	bool alwaysEven = true;

	// single choke point for writing a vertex's representative half-edge: clears
	// alwaysEven when an odd (non-canonical) representative is stored.
	void SetVHalfedge(VIndex v, HIndex he)
	{
		vHalfedges[v] = he;
		if (he & 1u)
			alwaysEven = false;
	}

	HalfMesh() {}
	explicit HalfMesh(const Mesh& mesh)
	{
		Build(mesh);
	}
	HalfMesh(VIndex numVertices, const std::vector<Face>& faces)
	{
		Build(numVertices, faces);
	}
	bool Empty() const
	{
		return vHalfedges.empty();
	}
	void Clear();
	// Instrumentation for the "one Build per pipeline" budget: process-wide
	// counters of the Build / FFaces calls made through the public entry points
	// (internal validation rebuilds are not counted). Atomic but not scoped, so
	// Reset + read is only meaningful while one pipeline is running.
	static uint64_t BuildCount();
	static void ResetBuildCount();
	static uint64_t FFacesCount();
	static void ResetFFacesCount();

	// create half-edge data-structures for the given mesh
	bool Build(const Mesh& mesh);
	bool Build(VIndex numVertices, const std::vector<Face>& faces);

	private:
	friend class Mesh;
	bool BuildForValidation(VIndex numVertices, const std::vector<Face>& faces);
	bool BuildImpl(VIndex numVertices, const std::vector<Face>& faces, bool countBuild);
	void FFacesForValidation(std::vector<Face>& faces) const { FFacesImpl(faces, false); }
	void FFacesImpl(std::vector<Face>& faces, bool countHarvest) const;

	// Undo log for FAdd: the previous value of every PRE-EXISTING slot an
	// insertion overwrites. Slots the insertion appended need no record — the
	// rollback truncates them. Entries are appended without de-duplication and
	// replayed in reverse, so the oldest value always wins, which is what lets
	// one log span a whole FAddDisk while each FAdd can still unwind just its
	// own tail. That is the difference between an O(patch) rejected patch and
	// copying the entire connectivity up front.
	struct AddUndo
	{
		std::vector<std::pair<HIndex, HIndex>> nexts; // heNexts[iHe]
		std::vector<std::pair<HIndex, FIndex>> faces; // heFaces[iHe]
		std::vector<std::pair<VIndex, HIndex>> representatives; // vHalfedges[iV]
	};
	// where an insertion started, i.e. the state a rollback restores
	struct AddMark
	{
		std::size_t nexts{0}, faces{0}, representatives{0};
		std::size_t numHalfedges{0}, numFaces{0};
		bool alwaysEven{true};
	};
	AddMark AddUndoMark(const AddUndo&) const;
	void AddUndoRollback(AddUndo&, const AddMark&);
	FIndex FAddImpl(const Face&, AddUndo&);

	public:
	// restore the canonical all-even vHalfedges form (see alwaysEven). No-op
	// when already canonical; otherwise rebuilds in place from the current faces.
	void GuaranteeAlwaysEven();
	// rearrange vertex start half-edge such that it is always
	// a bounding half-edge if the vertex is on the border;
	// in the same time close the loops for these vertices
	// such that we can loop over adjacent vertices/faces/edges
	// for all vertices, on the border or not
	// returns false if a corrupt (non-manifold) adjacency would loop forever
	bool ConnectBorders(HIndex&);
	bool ConnectBorders();

	// find and enumerate mesh boundary loops (holes)
	void EnumerateHoles(std::vector<std::vector<VIndex>>& holes) const;
	// triangulate the hole indicated by the given half-edge
	void TriangulateHole(HIndex, const std::vector<Vertex>&);

	// find and enumerate connected components:
	//  - components: store component ID per face
	//  - validator: check the edge between two neighbor faces is also user valid
	//  - return: number of components
	template <typename EdgeValidator>
	FIndex ConnectedComponents(std::vector<FIndex>& components, const EdgeValidator& validator) const;
	FIndex ConnectedComponents(std::vector<FIndex>& components) const
	{
		return ConnectedComponents(components, [](FIndex, FIndex) { return true; });
	}

	// connectivity
	HIndex HeTwin(HIndex iHe) const { return iHe ^ 1; }
	HIndex HeBack(HIndex iHe) const { return iHe | 1; }
	HIndex HeNext(HIndex iHe) const { return heNexts[iHe]; }
#if HALFMESH_TRIS
	HIndex HePrev(HIndex iHe) const { return HeIsBoundary(iHe) ? HePrevBoundary(iHe) : HeNext(HeNext(iHe)); }
#else
	HIndex HePrev(HIndex iHe) const { return HePrevOrbitFace(iHe); }
#endif
	VIndex HeVertex(HIndex iHe) const { return heVertices[iHe]; }
	FIndex HeFace(HIndex iHe) const { return heFaces[iHe]; }
	HIndex VHalfedge(VIndex iV) const { return vHalfedges[iV]; }
	HIndex FHalfedge(FIndex iF) const { return fHalfedges[iF]; }
	HIndex HeNextIncomingHalfedge(HIndex iHe) const { return HeTwin(HeNext(iHe)); }
	HIndex HeNextOutgoingHalfedge(HIndex iHe) const { return HeNext(HeTwin(iHe)); }
	VIndex HeHeadVertex(HIndex iHe) const
	{
		ASSERT(HeVertex(HeTwin(iHe)) == HeVertex(HeNext(iHe)));
		return HeVertex(HeNext(iHe));
	}
	VIndex HeTailVertex(HIndex iHe) const { return HeVertex(iHe); }
	EIndex HeEdge(HIndex iHe) const { return iHe / 2; }
	HIndex EHalfedge(EIndex iE) const { return iE * 2; }
	bool HeIsBoundary(HIndex iHe) const
	{
		ASSERT(heFaces[iHe] != math::NO_ID || (iHe & 1));
		return heFaces[iHe] == math::NO_ID;
	}
	bool EHeIsBoundary(HIndex iHe) const { return HeIsBoundary(HeBack(iHe)); }
	bool EIsBoundary(EIndex iE) const { return EHeIsBoundary(EHalfedge(iE)); }
	bool VIsBoundary(VIndex iV) const { return EHeIsBoundary(VHalfedge(iV)); }

	// Iterator template base used for all looping iterators, implemented by
	// a while loop that stops when it arrives back to the first element.
	// The loopStart member is used to mark the start of the loop.
	// The following function members can be customize:
	//  - IsValid(): returns false if currentElem should be skipped by the iterator
	//  - Next(): advances the iterator once, updating currentElem in-place
	//  - Value(): returns the value of the current element
	template <typename N>
	class IteratorBase
	{
		public:
		IteratorBase(typename N::ElemType elem, bool _loop_start) :
		    state{{elem}}, loopStart(_loop_start)
		{
			while (!state.IsValid()) { // get first valid element
				state.Next();
				if (state.currentElem == elem) {
					loopStart = false; // no valid elements, end loop
					break;
				}
			}
		}
		IteratorBase& operator++()
		{
			state.Next();
			while (!state.IsValid()) {
				state.Next();
			}
			loopStart = false;
			return *this;
		}
		IteratorBase operator++(int)
		{
			const IteratorBase it = *this;
			this->operator++();
			return it;
		}
		bool operator==(const IteratorBase& other) const
		{
			return state.currentElem == other.state.currentElem && loopStart == other.loopStart;
		}
		bool operator!=(const IteratorBase& other) const
		{
			return !(*this == other);
		}
		const typename N::ElemType& GetElem() const
		{
			return state.currentElem;
		}
		typename N::ReturnType operator*() const
		{
			return state.Value();
		}

		typedef std::input_iterator_tag iterator_category;
		typedef std::ptrdiff_t difference_type;
		typedef typename N::ReturnType value_type;
		typedef value_type* pointer;
		typedef value_type& reference;

		private:
		N state; // iterator state
		bool loopStart; // distinguish between begin/end
	};

	template <typename N>
	class IteratorStateBase
	{
		public:
		explicit IteratorStateBase(typename N::ElemType elem) :
		    endElem{IteratorBase<N>(elem, false)} {}
		IteratorBase<N> begin() const
		{
			return IteratorBase<N>(endElem.GetElem(), true);
		}
		const IteratorBase<N>& end() const
		{
			return endElem;
		}
		IteratorBase<N>& end()
		{
			return endElem;
		}

		private:
		IteratorBase<N> endElem; // cache begin/end iterator
	};

	//////////////////////
	// Half-edge adjacency
	struct Halfedge
	{
		HIndex idx;
		const HalfMesh& mesh;
		operator HIndex() const { return idx; }
		Halfedge& operator=(HIndex i)
		{
			idx = i;
			return *this;
		}
		bool operator==(const Halfedge& rhs) const
		{
			ASSERT(&mesh == &rhs.mesh);
			return idx == rhs.idx;
		}
	};
	struct HalfedgeIterator
	{
		typedef Halfedge ElemType;
		ElemType currentElem;
		bool IsValid() const { return true; }
	};

	// number of half-edges
	EIndex HeSize() const
	{
		ASSERT(heNexts.size() == heVertices.size());
		ASSERT(heNexts.size() == heFaces.size());
		ASSERT(heNexts.size() % 2 == 0);
		return heNexts.size();
	}
	// fetch the previous half-edge (general)
	HIndex HePrevOrbitFace(HIndex iHe) const
	{
		HIndex currHe = iHe;
		while (true) {
			const HIndex nextHe = HeNext(currHe);
			if (nextHe == iHe) {
				break;
			}
			currHe = nextHe;
		}
		ASSERT(HeNext(currHe) == iHe);
		return currHe;
	}
	HIndex HePrevOrbitVertex(HIndex iHe) const
	{
		HIndex currHe = HeTwin(iHe);
		while (true) {
			const HIndex nextHe = HeNext(currHe);
			if (nextHe == iHe) {
				break;
			}
			currHe = HeTwin(nextHe);
		}
		ASSERT(HeNext(currHe) == iHe);
		return currHe;
	}
	// fetch the previous half-edge in case of a boundary edge
	HIndex HePrevBoundary(HIndex iHe) const
	{
		ASSERT(iHe == HeBack(iHe) && HeIsBoundary(iHe));
		ASSERT(HeTwin(VHalfedge(HeVertex(iHe))) == HePrevOrbitFace(iHe));
		return HeTwin(VHalfedge(HeVertex(iHe)));
	}
#if HALFMESH_TRIS
	// remove the half-edge, return the new index of the half-edge border
	// or math::NO_ID if the entire edge was removed (twin already on border)
	HIndex HeRemove(HIndex);
#endif

	///////////////////
	// Vertex adjacency
	struct VertexAdjacentVertexIterator : HalfedgeIterator
	{
		typedef VIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNextOutgoingHalfedge(currentElem); }
		ReturnType Value() const { return currentElem.mesh.HeHeadVertex(currentElem); }
	};
	struct VertexIncomingHalfedgeIterator : HalfedgeIterator
	{
		typedef HIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNextIncomingHalfedge(currentElem); }
		ReturnType Value() const { return currentElem; }
	};
	struct VertexOutgoingHalfedgeIterator : HalfedgeIterator
	{
		typedef HIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNextOutgoingHalfedge(currentElem); }
		ReturnType Value() const { return currentElem; }
	};
	struct VertexAdjacentFaceIterator : HalfedgeIterator
	{
		typedef FIndex ReturnType;
		void Next()
		{
			if (currentElem.mesh.HeIsBoundary(currentElem.mesh.HeTwin(currentElem))) {
				currentElem = currentElem.mesh.HeNextOutgoingHalfedge(currentElem);
			}
			currentElem = currentElem.mesh.HeNextOutgoingHalfedge(currentElem);
		}
		ReturnType Value() const { return currentElem.mesh.HeFace(currentElem); }
	};
	struct VertexAdjacentEdgeIterator : HalfedgeIterator
	{
		typedef EIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNextOutgoingHalfedge(currentElem); }
		ReturnType Value() const { return currentElem.mesh.HeEdge(currentElem); }
	};
	struct VertexBoundaryLoopVertexIterator : HalfedgeIterator
	{
		typedef VIndex ReturnType;
		void Next()
		{
			ASSERT(currentElem.mesh.HeIsBoundary(currentElem));
			ASSERT(currentElem == currentElem.mesh.HeTwin(currentElem.mesh.VHalfedge(currentElem.mesh.HeHeadVertex(currentElem))));
			currentElem = currentElem.mesh.HeNext(currentElem);
		}
		ReturnType Value() const { return currentElem.mesh.HeVertex(currentElem); }
	};

	IteratorStateBase<VertexIncomingHalfedgeIterator> VIncomingHalfedges(VIndex i) const
	{
		return IteratorStateBase<VertexIncomingHalfedgeIterator>(Halfedge{HePrev(VHalfedge(i)), *this});
	}
	IteratorStateBase<VertexOutgoingHalfedgeIterator> VOutgoingHalfedges(VIndex i) const
	{
		return IteratorStateBase<VertexOutgoingHalfedgeIterator>(Halfedge{VHalfedge(i), *this});
	}
	IteratorStateBase<VertexAdjacentVertexIterator> VAdjacentVertices(VIndex i) const
	{
		return IteratorStateBase<VertexAdjacentVertexIterator>(Halfedge{VHalfedge(i), *this});
	}
	IteratorStateBase<VertexAdjacentFaceIterator> VAdjacentFaces(VIndex i) const
	{
		return IteratorStateBase<VertexAdjacentFaceIterator>(Halfedge{VHalfedge(i), *this});
	}
	IteratorStateBase<VertexAdjacentEdgeIterator> VAdjacentEdges(VIndex i) const
	{
		return IteratorStateBase<VertexAdjacentEdgeIterator>(Halfedge{VHalfedge(i), *this});
	}
	IteratorStateBase<VertexBoundaryLoopVertexIterator> VBoundaryLoopVertices(VIndex i) const
	{
		ASSERT(VIsBoundary(i));
		return IteratorStateBase<VertexBoundaryLoopVertexIterator>(Halfedge{HeTwin(VHalfedge(i)), *this});
	}

	// number of vertices
	VIndex VSize() const
	{
		return vHalfedges.size();
	}
	EIndex VDegree(VIndex i) const
	{
		EIndex k = 0;
		for ([[maybe_unused]] EIndex iE : VAdjacentEdges(i)) {
			++k;
		}
		return k;
	}
	FIndex VFaceDegree(VIndex i) const
	{
		FIndex k = 0;
		for ([[maybe_unused]] FIndex iF : VAdjacentFaces(i)) {
			++k;
		}
		return k;
	}
	// replace given vertex with the last vertex, update links, and pop back vertex
	void VRemoveOnly(VIndex);
	// same as above, but for a range of vertices
	void VRemoveOnly(VIndex*, unsigned numVerts);
	// remove every unreferenced (NO_ID representative) vertex, reporting the
	// descending swap-pop order used by VRemoveOnly
	void VRemoveUnreferenced(std::vector<VIndex>& removedVerts);

	/////////////////
	// Face adjacency
	struct FaceAdjacentVertexIterator : HalfedgeIterator
	{
		typedef VIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNext(currentElem); }
		ReturnType Value() const { return currentElem.mesh.HeVertex(currentElem); }
	};
	struct FaceAdjacentHalfedgeIterator : HalfedgeIterator
	{
		typedef HIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNext(currentElem); }
		ReturnType Value() const { return currentElem; }
	};
	struct FaceAdjacentEdgeIterator : HalfedgeIterator
	{
		typedef EIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeNext(currentElem); }
		ReturnType Value() const { return currentElem.mesh.HeEdge(currentElem); }
	};
	struct FaceAdjacentFaceIterator
	{
		// current half-edge for this face and adjacent half-edge for return face
		typedef std::pair<Halfedge, HIndex> ElemType;
		ElemType currentElem;
		typedef FIndex ReturnType;
		void Next()
		{
			currentElem.second = currentElem.first.mesh.HeTwin(currentElem.second);
			if (currentElem.first == currentElem.second) {
				currentElem.second = currentElem.first = currentElem.first.mesh.HeNext(currentElem.first);
			}
		}
		bool IsValid() const { return currentElem.first != currentElem.second && !currentElem.first.mesh.HeIsBoundary(currentElem.second); }
		ReturnType Value() const { return currentElem.first.mesh.HeFace(currentElem.second); }
	};

	IteratorStateBase<FaceAdjacentHalfedgeIterator> FAdjacentHalfedges(FIndex i) const
	{
		return IteratorStateBase<FaceAdjacentHalfedgeIterator>(Halfedge{FHalfedge(i), *this});
	}
	IteratorStateBase<FaceAdjacentVertexIterator> FAdjacentVertices(FIndex i) const
	{
		return IteratorStateBase<FaceAdjacentVertexIterator>(Halfedge{FHalfedge(i), *this});
	}
	IteratorStateBase<FaceAdjacentFaceIterator> FAdjacentFaces(FIndex i) const
	{
		return IteratorStateBase<FaceAdjacentFaceIterator>(std::make_pair(Halfedge{FHalfedge(i), *this}, FHalfedge(i)));
	}
	IteratorStateBase<FaceAdjacentEdgeIterator> FAdjacentEdges(FIndex i) const
	{
		return IteratorStateBase<FaceAdjacentEdgeIterator>(Halfedge{FHalfedge(i), *this});
	}

	// number of faces
	FIndex FSize() const
	{
		return fHalfedges.size();
	}
	bool FIsTriangle(FIndex i) const
	{
#if HALFMESH_TRIS
		return true;
#else
		const HIndex iHe = FHalfedge(i);
		return iHe == HeNext(HeNext(HeNext(iHe)));
#endif
	}
	FIndex FDegree(FIndex i) const
	{
		FIndex k = 0;
		for ([[maybe_unused]] HIndex iHe : FAdjacentHalfedges(i)) {
			++k;
		}
		return k;
	}
	// find vertex index in the face polygon array
	VIndex FVertexIth(FIndex i, VIndex iV) const
	{
		VIndex k = 0;
		for (HIndex iHe : FAdjacentHalfedges(i)) {
			if (HeVertex(iHe) == iV) {
				return k;
			}
			++k;
		}
		return math::NO_ID;
	}
	VIndex FVertexIth(HIndex iHe, const Face& face) const
	{
		for (VIndex i = 0; i < face.rows(); ++i) {
			if (HeVertex(iHe) == face[i]) {
				return i;
			}
		}
		return math::NO_ID;
	}
	// fetch the n-th half-edge index in the face polygon array
	VIndex FHalfEdgeNth(FIndex i, unsigned nth) const
	{
#if HALFMESH_TRIS
		ASSERT(nth < 3);
#else
		ASSERT(nth < FDegree(i));
#endif
		HIndex iHe = FHalfedge(i);
		for (unsigned v = 0; v < nth; ++v, iHe = HeNext(iHe))
			;
		return iHe;
	}
	// fetch the requested face
	Face F(FIndex i) const
	{
		return FHe(fHalfedges[i]);
	}
	Face FHe(HIndex iHe) const
	{
#if HALFMESH_TRIS
		// assume triangular face
		Face face;
		for (int v = 0; v < 3; ++v, iHe = HeNext(iHe)) {
			face[v] = HeVertex(iHe);
		}
		ASSERT(face[0] == HeVertex(iHe));
		return face;
#else
		// polygon face
		std::vector<VIndex> verts;
		for (VIndex iV : IteratorStateBase<FaceAdjacentVertexIterator>(Halfedge{iHe, *this})) {
			verts.push_back(iV);
		}
		return Eigen::Map<const Face>(verts.data(), verts.size());
#endif
	}
	// return the index of the adjacent face to the given edge
	FIndex FAdjacent(HIndex iHe) const
	{
		return HeFace(HeTwin(iHe));
	}
	// fetch the array of faces back in the native Mesh format
	void FFaces(std::vector<Face>& faces) const
	{
		FFacesImpl(faces, true);
	}
	// create a new face between the vertices defined by the given face; a corner
	// may be an isolated vertex (NO_ID representative) the caller pre-appended.
	// Returns NO_ID if the face can not be added, as is the case if the face does
	// not form a manifold connection with the mesh; a rejected add leaves every
	// array byte-identical.
	FIndex FAdd(const Face&);
	// attach a pre-triangulated disk in dependency order; retries faces that are
	// not yet attachable and restores the exact input structure on failure;
	// caller pre-appends every interior vertex as a NO_ID vHalfedges slot
	bool FAddDisk(const std::vector<Face>& faces);
#if HALFMESH_TRIS
	// check if the face is a corner: triangle having two or three boundary edges
	bool FIsCorner(FIndex) const;
	// remove the face; notes:
	//  - does not support faces with more than one border-edge
	//  - after, call ConnectBorders to restore border connectivity
	//  - call ConnectBorders before removing a face with a border-edge
	//    on the same border as a face previously removed
	// Prefer FRemoveBulk below: it has none of these restrictions, keeps the
	// connectivity valid on its own, and is faster for more than one face.
	void FRemove(FIndex);
	// remove arbitrary faces while preserving valid manifold connectivity;
	// reports vertex swap-pops and the source of every pinch-split duplicate
	void FRemoveBulk(std::vector<FIndex>& faceRemoves,
	                 std::vector<VIndex>& removedVerts,
	                 std::vector<VIndex>& splitSrcVerts);
	// remove corner face: triangle having two or three boundary edges
	// return false if it's a stand alone face (all edges are boundary)
	bool FRemoveCorner(FIndex, RemovedData&);
#endif
	// replace given face with the last face, update links, and pop back face
	void FRemoveOnly(FIndex);
	// same as above, but for a range of faces
	void FRemoveOnly(FIndex*, unsigned numFaces);

	/////////////////
	// Edge adjacency
	struct EdgeIterator : HalfedgeIterator
	{
		typedef EIndex ReturnType;
		void Next()
		{
			if ((currentElem.idx += 2) >= currentElem.mesh.heNexts.size())
				currentElem = 0;
		}
		ReturnType Value() const { return currentElem.mesh.HeEdge(currentElem); }
	};
	struct EdgeAdjacentHalfedgeIterator : HalfedgeIterator
	{
		typedef HIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeTwin(currentElem); }
		ReturnType Value() const { return currentElem; }
	};
	struct EdgeAdjacentInteriorHalfedgeIterator : HalfedgeIterator
	{
		typedef HIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeTwin(currentElem); }
		bool IsValid() const { return !currentElem.mesh.HeIsBoundary(currentElem); }
		ReturnType Value() const { return currentElem; }
	};
	struct EdgeAdjacentFaceIterator : HalfedgeIterator
	{
		typedef FIndex ReturnType;
		void Next() { currentElem = currentElem.mesh.HeTwin(currentElem); }
		bool IsValid() const { return !currentElem.mesh.HeIsBoundary(currentElem); }
		ReturnType Value() const { return currentElem.mesh.HeFace(currentElem); }
	};

	IteratorStateBase<EdgeIterator> EEdges(EIndex i = 0) const
	{
		return IteratorStateBase<EdgeIterator>(Halfedge{EHalfedge(i), *this});
	}
	IteratorStateBase<EdgeAdjacentHalfedgeIterator> EAdjacentHalfedges(EIndex i) const
	{
		return IteratorStateBase<EdgeAdjacentHalfedgeIterator>(Halfedge{EHalfedge(i), *this});
	}
	IteratorStateBase<EdgeAdjacentInteriorHalfedgeIterator> EAdjacentInteriorHalfedges(EIndex i) const
	{
		return IteratorStateBase<EdgeAdjacentInteriorHalfedgeIterator>(Halfedge{EHalfedge(i), *this});
	}
	IteratorStateBase<EdgeAdjacentFaceIterator> EAdjacentFaces(EIndex i) const
	{
		return IteratorStateBase<EdgeAdjacentFaceIterator>(Halfedge{EHalfedge(i), *this});
	}

	// number of edges
	EIndex ESize() const
	{
		return HeEdge(HeSize());
	}
	VIndex EFirstVertex(EIndex i) const
	{
		return HeTailVertex(EHalfedge(i));
	}
	VIndex ESecondVertex(EIndex i) const
	{
		return HeHeadVertex(EHalfedge(i));
	}
	std::pair<VIndex, VIndex> EVertices(EIndex i) const
	{
		const HIndex iHe = EHalfedge(i);
		return std::make_pair(HeTailVertex(iHe), HeHeadVertex(iHe));
	}
	VIndex EOtherVertex(EIndex i, VIndex v) const
	{
		const HIndex iHe = EHalfedge(i);
		return HeTailVertex(iHe) == v ? HeHeadVertex(iHe) : HeTailVertex(iHe);
	}
	EIndex EDegree(EIndex i) const
	{
		EIndex k = 0;
		for ([[maybe_unused]] HIndex iHe : EAdjacentInteriorHalfedges(i)) {
			++k;
		}
		return k;
	}
	// find the edge index given two vertices
	// (not needed most of the time, use only if there is no other way)
	EIndex EEdge(VIndex iV0, VIndex iV1) const
	{
		ASSERT(iV0 != iV1);
		for (HIndex iHe : VOutgoingHalfedges(iV0)) {
			if (HeHeadVertex(iHe) == iV1) {
				return HeEdge(iHe);
			}
		}
		return math::NO_ID;
	}
	std::pair<FIndex, FIndex> EAdjacentFaceIndices(EIndex i) const
	{
		const HIndex iHe = EHalfedge(i);
		return std::make_pair(HeFace(iHe), HeFace(HeTwin(iHe)));
	}
#if HALFMESH_TRIS
	// check if edge can be collapsed: would not break the topological consistency of the mesh;
	// an edge v0<->v1 can be collapsed iff it satisfies:
	//  - the link condition: for every vertex v adjacent to both v0 and v1,
	//    v0,v,v1 define a face of the mesh
	//  - the boundary condition: if v0 and v1 are boundary vertices, v0<->v1
	//    is a boundary edge
	// see pr. 4.2: "Mesh Optimization", Hoppe et al, 1993
	bool EIsCollapseValidTopologically(EIndex) const;
	// check if edge can be collapsed: in the resulting local mesh no two adjacent triangles form an internal
	// dihedral angle greater than a fixed threshold (i.e. triangles do not "fold" into each other),
	// where the given point is the new position of the kept vertex
	bool EIsCollapseValidGeometrically(EIndex, const Vertex&, const std::vector<Vertex>&) const;
	// collapse the given edge, remove it, remove both adjacent faces, and remove one the head vertex;
	// return the index of the repositioned vertex
	VIndex ERemove(EIndex, RemovedData&);
	// check if edge is flippable
	bool EIsFlipValid(EIndex, const std::vector<Vertex>&) const;
	// flip the given edge between two triangles (only if EIsFlipValid())
	void EFlip(EIndex);
	// split the given edge at a new midpoint vertex, updating connectivity in
	// place (no full rebuild). Interior edge -> +1 vertex, +2 faces, +3 edges;
	// boundary edge -> +1 vertex, +1 face, +2 edges. Appends one vHalfedges
	// entry and returns the new vertex index; the caller appends its position.
	VIndex ESplit(EIndex);
#endif
	// replace given edge with the last edge, update links, and pop back edge
	void ERemoveOnly(EIndex);
	// same as above, but for a range of edges
	void ERemoveOnly(EIndex*, unsigned numEdges);
};

template <typename EdgeValidator>
HalfMesh::FIndex HalfMesh::ConnectedComponents(std::vector<FIndex>& components, const EdgeValidator& validator) const
{
	FIndex numComponents = 0;
	components.clear();
	components.resize(fHalfedges.size(), math::NO_ID);
	std::vector<HIndex> stack;
	stack.reserve(8 * 1024);
	FOREACHIDX (FIndex, iFStart, fHalfedges) {
		if (components[iFStart] != math::NO_ID)
			continue;
		ASSERT(stack.empty());
		FIndex iF = iFStart;
	// visit connected faces depth-first
	ContinueVisit:
		components[iF] = numComponents;
		stack.push_back(fHalfedges[iF]);
		do {
			// loop over face neighbors
			const HIndex iHeStart = stack.back();
			ASSERT(!HeIsBoundary(iHeStart));
			ASSERT(components[HeFace(iHeStart)] == numComponents);
			HIndex iHe = iHeStart;
			while (true) {
				iF = HeFace(HeTwin(iHe));
				if (iF != math::NO_ID && components[iF] == math::NO_ID && validator(HeFace(iHe), iF))
					goto ContinueVisit;
				if ((iHe = HeNext(iHe)) == iHeStart)
					break;
			}
			stack.pop_back();
		} while (!stack.empty());
		++numComponents;
	}
	return numComponents;
}

} // namespace halfmesh
