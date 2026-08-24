/*
* HalfMesh.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/HalfMesh.h>
#include <halfmesh/Mesh.h>
#include <halfmesh/Util/Hash.h>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Maths.h>
#include <limits>
#include <algorithm>
#include <unordered_map>

using namespace math;

namespace halfmesh {

void HalfMesh::Clear()
{
	heNexts.clear();
	heVertices.clear();
	heFaces.clear();
	vHalfedges.clear();
	fHalfedges.clear();
	alwaysEven = true; // empty mesh is vacuously canonical
}

bool HalfMesh::Build(const Mesh& mesh)
{
	const_cast<Mesh&>(mesh).SyncFaces();
	return Build(static_cast<VIndex>(mesh.vertices.size()), mesh.faces);
}
bool HalfMesh::Build(VIndex numVertices, const std::vector<Face>& faces)
{
	Clear();
	if (numVertices == 0 || faces.empty())
		return true;
	heNexts.reserve(numVertices * 6);
	heVertices.reserve(heNexts.capacity());
	heFaces.reserve(heNexts.capacity());
	vHalfedges.resize(numVertices, NO_ID);
	fHalfedges.resize(faces.size(), NO_ID);
	// walk the faces, creating half-edges if non-existing.
	// Edges are keyed by a packed 64-bit UNDIRECTED key (min(tail,head)<<32)|max:
	// VIndex is uint32, so the pack is lossless and gives an identity hash + single
	// 64-bit compare instead of a two-round HashCombine chain over a tuple. Both
	// directions of an edge share one key, so a single emplace per corner both
	// creates the pair (on the first, first-seen direction) and locates the twin
	// (on the second) — halving hash traffic and node allocations vs probing two
	// directed keys. The map stores the EVEN half-edge of each pair; the odd twin is
	// its XOR-partner. Output is bit-identical to a two-directed-key probe: the pair
	// is still created oriented from the first-seen corner (even = idxTail, so the
	// same vHalfedges / heVertices), the same corner triggers creation (first
	// undirected encounter == twin-not-yet-created), and a directed edge reused by a
	// second face is rejected the same way (its half-edge already carries a face).
	// The map is never iterated — half-edge indices are assigned in face-walk order.
	const auto edgeKey = [](VIndex a, VIndex b) -> uint64_t {
		if (a > b)
			std::swap(a, b);
		return (static_cast<uint64_t>(a) << 32) | b;
	};
	std::unordered_map<uint64_t, HIndex> createdHalfedges;
	createdHalfedges.reserve(heNexts.capacity());
	FOREACHIDX (FIndex, idxFace, faces) {
		const Face& face = faces[idxFace];
		// walk around this face
		HIndex prevIHe = NO_ID, firstIHe = NO_ID;
		for (Eigen::Index v = 0; v < face.rows(); ++v) {
			const VIndex idxTail = face[v];
			const VIndex idxHead = face[(v + 1) % face.rows()];
			// Non-manifold guard (active in Release, not just _DEBUG): a self-edge
			// (face[i]==face[i+1]) cannot be represented as a half-edge.  One cheap
			// integer compare on values already in hand; bail on the first one so a
			// known-manifold mesh pays essentially nothing.
			if (idxTail == idxHead) {
				Clear();
				return false; // self-edge => non-manifold/degenerate input
			}
			// one emplace per corner: locate (or create) this edge's half-edge pair
			HIndex& iHeEven = createdHalfedges.emplace(
			                                      edgeKey(idxTail, idxHead), NO_ID)
			                      .first->second;
			HIndex iHe;
			if (iHeEven == NO_ID) {
				// first time this edge is seen: create and initialize a new pair,
				// oriented from this corner (even half-edge has idxTail as tail)
				ASSERT(heNexts.size() == heVertices.size() && heNexts.size() == heFaces.size());
				iHe = static_cast<HIndex>(heNexts.size());
				iHeEven = iHe;
				vHalfedges[idxTail] = iHe;
				heNexts.emplace_back(NO_ID);
				heNexts.emplace_back(NO_ID);
				heVertices.emplace_back(idxTail);
				heVertices.emplace_back(idxHead);
				heFaces.emplace_back(NO_ID);
				heFaces.emplace_back(NO_ID);
			} else {
				// edge already exists: pick the half-edge for this corner's direction
				// (the stored even half-edge if it leaves idxTail, else its twin)
				iHe = (heVertices[iHeEven] == idxTail) ? iHeEven : HeTwin(iHeEven);
				// A directed edge seen twice means an edge shared by >2 faces (or a
				// duplicate face): the half-edge for this direction already carries a
				// face, so the manifold structure cannot represent it. Bail early.
				if (heFaces[iHe] != NO_ID) {
					Clear();
					return false; // duplicate directed edge => non-manifold input
				}
			}
			// update data-structures
			heFaces[iHe] = idxFace;
			if (v == 0) {
				fHalfedges[idxFace] = iHe;
				firstIHe = iHe;
			} else {
				heNexts[prevIHe] = iHe;
			}
			prevIHe = iHe;
		}
		// initialize first half-edge next index, skipped above
		heNexts[prevIHe] = firstIHe;
	}
	// restore border connectivity
	if (!ConnectBorders()) {
		// adjacency walk did not terminate => the input was non-manifold in a way
		// the inline edge checks above did not catch (e.g. a bow-tie vertex).
		Clear();
		return false;
	}
	// Reject non-manifold VERTICES (bow-ties): a vertex shared by two or more
	// otherwise-disjoint fans is edge-manifold — so the per-edge checks above and
	// ConnectBorders accept it — yet it breaks the single-fan-per-vertex assumption
	// the half-edge collapse/remove ops rely on (they would follow a NO_ID link and
	// fault; observed as a SIGBUS in QEM decimation). A vertex is single-fan iff
	// walking its outgoing half-edges reaches every half-edge that has it as tail;
	// if not, fail so ListHalfEdges falls back to the repairing ListHalfEdgesSafe.
	{
		std::vector<HIndex> tailCount(vHalfedges.size(), 0);
		for (const VIndex v : heVertices)
			++tailCount[v];
		FOREACHIDX (VIndex, v, vHalfedges) {
			if (vHalfedges[v] == NO_ID)
				continue; // unreferenced vertex: no fan to validate
			HIndex reached = 0;
			for (const HIndex iHe : VOutgoingHalfedges(v)) {
				(void)iHe;
				if (++reached > tailCount[v])
					break; // malformed cycle guard (should not happen post-ConnectBorders)
			}
			if (reached != tailCount[v]) {
				Clear();
				return false; // bow-tie / non-manifold vertex
			}
		}
	}
#ifndef NDEBUG
	// Structural post-conditions of a SUCCESSFUL build. Every non-manifold
	// rejection above has passed, so the half-edge structure is manifold and both
	// properties MUST hold — a violation here is a bug in Build/ConnectBorders,
	// not bad input. (They are deliberately not checked mid-build: there they are
	// caller states Build reports via `return false`, which is what lets
	// Mesh::ListHalfEdges fall back to the repairing ListHalfEdgesSafe.)
	// Guarded by NDEBUG — the portable spelling used throughout this codebase —
	// not the MSVC-only _DEBUG, so the validation runs on every platform.
	{
		// 1. every vertex is referenced by some half-edge: ConnectBorders() rejects
		//    a NO_ID anchor explicitly, so reaching here implies all are set.
		for (const HIndex iHe : vHalfedges) {
			ASSERT(iHe != NO_ID); // unreferenced vertex
		}
		// 2. each boundary neighborhood is a disk or a half-disk, i.e. a vertex is
		//    the tail of at most one boundary half-edge (two would be a bow-tie,
		//    rejected by ConnectBorders + the single-fan check above).
		//    Boundary is tested via heFaces (HeIsBoundary), NOT heNexts: the old
		//    heNexts == NO_ID test only identified boundary half-edges BEFORE
		//    ConnectBorders linked them, and would be vacuous here.
		std::vector<bool> boundaryVertices(vHalfedges.size(), false);
		FOREACHIDX (HIndex, iHe, heFaces) {
			if (heFaces[iHe] == NO_ID) {
				const VIndex v = heVertices[iHe];
				ASSERT(!boundaryVertices[v]); // vertex appears in more than one boundary loop
				boundaryVertices[v] = true;
			}
		}
	}
#endif
	// Build produces the canonical all-even form: each vertex's representative is
	// written directly (not via SetVHalfedge) as the freshly-created even half-edge
	// of the first edge leaving it, and ConnectBorders only stores even boundary
	// representatives. Because those direct writes bypass the alwaysEven
	// bookkeeping — and the flag may be stale from an in-place mutation before this
	// rebuild — set it true explicitly to record the invariant we just established.
	alwaysEven = true;
	return true;
}

void HalfMesh::GuaranteeAlwaysEven()
{
	if (alwaysEven)
		return; // already canonical: O(1) fast path
	// The all-even form can only be restored by a rebuild (the parity is
	// positional). Harvest the current faces and rebuild in place; Build sets
	// alwaysEven = true. Vertex positions live in the Mesh and are untouched.
	const VIndex nv = VSize(); // capture before Build()'s Clear()
	std::vector<Face> faces;
	FFaces(faces); // FFaces appends, so pass an empty vector
	Build(nv, faces);
}

namespace {

template <typename BeforeNextWrite>
bool ConnectBordersImpl(HalfMesh& mesh, HalfMesh::HIndex& iHeStart, BeforeNextWrite&& beforeNextWrite)
{
	using HIndex = HalfMesh::HIndex;
	using VIndex = HalfMesh::VIndex;
	if (iHeStart == NO_ID || iHeStart >= mesh.heNexts.size())
		return false;

	// Find the face-bearing outgoing side of a boundary edge. Representatives on
	// interior vertices may be odd after in-place mutations, so face occupancy,
	// not pair parity, decides when the boundary has been reached.
	const HIndex original = iHeStart;
	const VIndex vertex = mesh.HeVertex(original);
	const std::size_t guardLimit = mesh.heNexts.size() + 1;
	std::size_t guard = 0;
	HIndex boundaryOut = original;
	while (mesh.heFaces[boundaryOut] == NO_ID || mesh.heFaces[mesh.HeTwin(boundaryOut)] != NO_ID) {
		const HIndex twin = mesh.HeTwin(boundaryOut);
		if (mesh.heNexts[twin] == NO_ID)
			return false;
		boundaryOut = mesh.HeNextOutgoingHalfedge(boundaryOut);
		if (boundaryOut == original)
			return true; // interior vertex
		if (++guard > guardLimit)
			return false;
	}

	// Find the outgoing boundary half-edge on the other side of this vertex's
	// fan. Defer both writes until this walk succeeds so failure is non-mutating.
	guard = 0;
	HIndex boundaryNext = boundaryOut;
	do {
		boundaryNext = mesh.HeTwin(mesh.HePrev(boundaryNext));
		if (++guard > guardLimit)
			return false;
	} while (mesh.heFaces[boundaryNext] != NO_ID);

	ASSERT(mesh.HeVertex(boundaryNext) == vertex);
	const HIndex boundaryIn = mesh.HeTwin(boundaryOut);
	ASSERT(mesh.heFaces[boundaryOut] != NO_ID && mesh.heFaces[boundaryIn] == NO_ID);
	mesh.SetVHalfedge(vertex, boundaryOut);
	// All in-tree callers pass the representative itself by reference. Preserve
	// the public by-reference behavior for any caller that passes a local copy.
	if (iHeStart != boundaryOut)
		iHeStart = boundaryOut;
	beforeNextWrite(boundaryIn);
	mesh.heNexts[boundaryIn] = boundaryNext;
	return true;
}

} // anonymous namespace

bool HalfMesh::ConnectBorders(HIndex& iHeStart)
{
	return ConnectBordersImpl(*this, iHeStart, [](HIndex) {});
}

bool HalfMesh::ConnectBorders()
{
	for (HIndex& iHeStart : vHalfedges) {
		if (!ConnectBorders(iHeStart))
			return false;
		ASSERT([&]() {
			HIndex iHe = iHeStart;
			do {
				iHe = HeNextOutgoingHalfedge(iHe);
			} while (iHe != iHeStart);
			return true;
		}());
	}
	return true;
}

void HalfMesh::EnumerateHoles(std::vector<std::vector<VIndex>>& holes) const
{
	// flat visited bitmap instead of a node-based std::set: O(1) test-and-set, a
	// single allocation, sequential memory. The traversal order (vertex scan +
	// boundary heNexts walk) is unchanged, so the hole list is identical.
	std::vector<uint8_t> parsedVertices(vHalfedges.size(), 0);
	FOREACHIDX (VIndex, iV, vHalfedges) {
		if (!VIsBoundary(iV) || parsedVertices[iV])
			continue;
		std::vector<VIndex> holeVertices;
		const HIndex iHeStart = HeTwin(VHalfedge(iV));
		HIndex iHe = iHeStart;
		do {
			ASSERT(HeIsBoundary(iHe));
			const VIndex i = HeVertex(iHe);
			holeVertices.emplace_back(i);
			ASSERT(!parsedVertices[i]);
			parsedVertices[i] = 1;
		} while ((iHe = HeNext(iHe)) != iHeStart);
		holes.emplace_back(std::move(holeVertices));
	}
}

void HalfMesh::TriangulateHole(HIndex iHeStart, const std::vector<Vertex>& vertices)
{
	// enumerate hole vertices
	std::vector<VIndex> holeVertices;
	HIndex iHe = iHeStart;
	do {
		ASSERT(HeIsBoundary(iHe));
		holeVertices.emplace_back(HeVertex(iHe));
	} while ((iHe = HeNext(iHe)) != iHeStart);
	const uint32_t n = static_cast<uint32_t>(holeVertices.size());
	// compute minimal triangulation by dynamic programming
	struct Weight
	{
		Weight operator+(const Weight& _rhs) const
		{
			return Weight{std::max(angle, _rhs.angle), area + _rhs.area};
		}
		bool operator<(const Weight& _rhs) const
		{
			return angle < _rhs.angle || (angle == _rhs.angle && area < _rhs.area);
		}
		Type angle{std::numeric_limits<Type>::max()};
		Type area{std::numeric_limits<Type>::max()};
	};
	typedef Eigen::Matrix<Type, 3, 1> Vector;
	Eigen::Matrix<Weight, Eigen::Dynamic, Eigen::Dynamic> weights(n, n);
	Eigen::Matrix<VIndex, Eigen::Dynamic, Eigen::Dynamic> indices(n, n);
	const auto ComputeNormal = [&vertices](VIndex a, VIndex b, VIndex c) -> Vector {
		return (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]).normalized();
	};
	const auto ComputeAngle = [](const Vector& n1, const Vector& n2) -> Type {
		return Type(1) - n1.dot(n2);
	};
	// opposite vertex across the boundary edge at hole vertex i; hoisted out of
	// ComputeWeight (called O(n^3) times) and captured by reference so the O(n^3)
	// DP does not copy holeVertices on every invocation.
	const auto VOpposite = [this, &holeVertices](uint32_t i) -> VIndex {
		return HeHeadVertex(HeNext(VHalfedge(holeVertices[i])));
	};
	const auto ComputeWeight = [&](uint32_t i, uint32_t j, uint32_t k) -> Weight {
		const VIndex a = holeVertices[i];
		const VIndex b = holeVertices[j];
		const VIndex c = holeVertices[k];
		// if one of the potential edges already exists, this would result
		// in an invalid triangulation -> prevent by giving infinite weight
		const auto EIsInterior = [this](VIndex iV0, VIndex iV1) -> bool {
			const EIndex iE = EEdge(iV0, iV1);
			return iE != NO_ID ? !EIsBoundary(iE) : false;
		};
		if (EIsInterior(a, b) || EIsInterior(b, c) || EIsInterior(c, a))
			return Weight{};
		// compute dihedral angles with
		const Vector nomalScaled = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
		// A degenerate (collinear) candidate has an exactly/near-zero cross
		// product: normalized() yields NaN, ComputeAngle propagates it, and
		// std::max(0, NaN) silently DISCARDS it -- the zero-area triangle would
		// score as perfect (angle=0, area=0) and win the DP on both keys. Forbid
		// it outright (same sentinel as the interior-edge case above); the
		// dimensionless area/maxEdgeSq^2 <= sin^2(corner angle) test rejects by
		// SHAPE at any mesh scale (same class and fix as HoleFilling::
		// ComputeWeight in MeshHoles.cpp, HOLES-3). Coincident points and NaN
		// inputs fail the comparison and are refused too.
		const Type area2 = nomalScaled.squaredNorm();
		const Type maxEdgeSq = std::max({(vertices[b] - vertices[a]).squaredNorm(),
		                                 (vertices[c] - vertices[b]).squaredNorm(),
		                                 (vertices[a] - vertices[c]).squaredNorm()});
		if (!(area2 > Type(1e-12) * maxEdgeSq * maxEdgeSq))
			return Weight{};
		const Vector nomal = nomalScaled.normalized();
		// neighbor to (i,j)
		VIndex d = (i + 1 == j) ? VOpposite(j) : holeVertices[indices(i, j)];
		Type angle = std::max(Type(0), ComputeAngle(nomal, ComputeNormal(a, d, b)));
		// neighbor to (j,k)
		d = (j + 1 == k) ? VOpposite(k) : holeVertices[indices(j, k)];
		angle = std::max(angle, ComputeAngle(nomal, ComputeNormal(b, d, c)));
		// and neighbor to (k,i) if (k,i)==(n-1, 0)
		if (i == 0 && k + 1 == holeVertices.size()) {
			d = VOpposite(0);
			angle = std::max(angle, ComputeAngle(nomal, ComputeNormal(c, d, a)));
		}
		// area term for the lexicographic tie-break after the max-dihedral term.
		// Liepa 2003 minimizes SUMMED triangle area, so use |cross| (= 2*area);
		// nomalScaled.squaredNorm() would minimize the sum of SQUARED areas, which
		// changes the argmin on non-planar holes (losing the minimal-surface fill)
		// and squares the dynamic range. The constant 1/2 cancels in the argmin.
		const Type area = nomalScaled.norm();
		return Weight{angle, area};
	};
	// initialize 2-clicks
	for (uint32_t i = 0; i < n - 1; ++i) {
		weights(i, i + 1) = Weight{0, 0};
		indices(i, i + 1) = NO_ID;
	}
	// n-clicks with n>2
	for (uint32_t j = 2; j < n; ++j) {
		// for all n-clicks [i,i+j]
		for (uint32_t i = 0; i < n - j; ++i) {
			const uint32_t k = i + j;
			// find best split i < m < i+j
			Weight minW;
			VIndex minIV = NO_ID;
			for (uint32_t m = i + 1; m < k; ++m) {
				const Weight w = weights(i, m) + ComputeWeight(i, m, k) + weights(m, k);
				if (w < minW) {
					minW = w;
					minIV = static_cast<VIndex>(m);
				}
			}
			weights(i, k) = minW;
			indices(i, k) = minIV;
		}
	}
	// add the new triangles to the mesh
	typedef std::pair<uint32_t, uint32_t> IRange;
	std::vector<IRange> stack;
	std::vector<Face> queue;
	stack.reserve(n * 2);
	stack.push_back(IRange(0, n - 1));
	while (!stack.empty()) {
		const IRange r = stack.back();
		stack.pop_back();
		ASSERT(r.first <= r.second);
		if (r.second - r.first < 2)
			continue;
		const uint32_t split = indices(r.first, r.second);
		if (split == NO_ID) {
			// every candidate triangle for this span was forbidden (degenerate
			// or edge-duplicating): leave the span untriangulated rather than
			// indexing holeVertices[NO_ID].
			continue;
		}
		const Face face{holeVertices[r.first], holeVertices[split], holeVertices[r.second]};
		if (FAdd(face) == NO_ID) {
			// can not create this face right now as it breaks topology constraints;
			// store face to retry later
			queue.emplace_back(std::move(face));
		}
		stack.push_back(IRange(r.first, split));
		stack.push_back(IRange(split, r.second));
	}
	while (!queue.empty()) {
		RFOREACH (i, queue) {
			if (FAdd(queue[i]) != NO_ID) {
				queue[i] = queue.back();
				queue.pop_back();
			}
		}
	}
}

#if HALFMESH_TRIS
HalfMesh::HIndex HalfMesh::HeRemove(HIndex iHe)
{
	if (vHalfedges[heVertices[iHe]] == iHe) {
		HIndex iHeNext = iHe;
		do {
			iHeNext = HeNextOutgoingHalfedge(iHeNext);
		} while (EHeIsBoundary(iHeNext) && iHeNext != iHe);
		ASSERT(heVertices[iHe] == heVertices[iHeNext]);
		SetVHalfedge(heVertices[iHe], iHeNext);
	}
	if (iHe & 1) {
		// full edge, half-edge already positioned correctly;
		// just mark it as border
		ASSERT(!EHeIsBoundary(iHe));
		heNexts[iHe] = NO_ID;
		heFaces[iHe] = NO_ID;
		return iHe;
	}
	const HIndex iHeTwin = HeTwin(iHe);
	if (HeIsBoundary(iHeTwin)) {
		// edge already on the border, so removing this half-edge
		// removes the entire edge;
		// move last edge in its place and update links
		ASSERT(vHalfedges[heVertices[iHeTwin]] != iHeTwin);
		const HIndex iHeTwinPrev = HePrev(iHeTwin);
		ASSERT(heNexts[iHeTwinPrev] == iHeTwin);
		ERemoveOnly(HeEdge(iHe));
		heNexts[iHeTwinPrev] = NO_ID;
		return NO_ID;
	}
	// edge becomes now a border edge, swap its twin half-edges
	// in order to maintain the convention, and update links
	ASSERT(vHalfedges[heVertices[iHeTwin]] != iHeTwin);
	if (fHalfedges[heFaces[iHeTwin]] == iHeTwin)
		fHalfedges[heFaces[iHeTwin]] = iHe;
	ASSERT(heNexts[HePrev(iHeTwin)] == iHeTwin);
	heNexts[HePrev(iHeTwin)] = iHe;
	heNexts[iHe] = heNexts[iHeTwin];
	heNexts[iHeTwin] = NO_ID;
	std::swap(heVertices[iHe], heVertices[iHeTwin]);
	heFaces[iHe] = heFaces[iHeTwin];
	heFaces[iHeTwin] = NO_ID;
	return iHeTwin;
}
#endif

void HalfMesh::VRemoveOnly(VIndex iV)
{
	if (iV + 1 < vHalfedges.size()) {
		SetVHalfedge(iV, vHalfedges.back());
		const HIndex iHeStart = vHalfedges[iV];
		HIndex iHe = iHeStart;
		while (true) {
			ASSERT(heVertices[iHe] == vHalfedges.size() - 1);
			heVertices[iHe] = iV;
			if ((iHe = HeNextOutgoingHalfedge(iHe)) == iHeStart)
				break;
		}
	}
	vHalfedges.pop_back();
}

void HalfMesh::VRemoveOnly(VIndex* verts, unsigned numVerts)
{
	ASSERT(verts != NULL);
	std::sort(verts, verts + numVerts, [](EIndex i, EIndex j) { return i > j; });
	for (unsigned i = 0; i < numVerts; ++i)
		VRemoveOnly(verts[i]);
}

void HalfMesh::VRemoveUnreferenced(std::vector<VIndex>& removedVerts)
{
	const std::size_t firstRemoved = removedVerts.size();
	FOREACHIDX (VIndex, vertex, vHalfedges)
		if (vHalfedges[vertex] == NO_ID)
			removedVerts.emplace_back(vertex);
	if (removedVerts.size() > firstRemoved)
		VRemoveOnly(removedVerts.data() + firstRemoved, static_cast<unsigned>(removedVerts.size() - firstRemoved));
}

HalfMesh::FIndex HalfMesh::FAdd(const Face& face)
{
// check if adding the face maintains a manifold mesh
#if HALFMESH_TRIS
	const unsigned numVertices = 3;
	HIndex hedges[3];
	unsigned numNewEdges[3] = {0, 0, 0};
	bool isolated[3];
#else
	const unsigned numVertices = face.rows();
	std::vector<HIndex> hedges(numVertices);
	std::vector<unsigned> numNewEdges(numVertices);
	std::vector<bool> isolated(numVertices);
#endif
	for (unsigned v = 0; v < numVertices; ++v) {
		if (face[v] >= VSize())
			return NO_ID;
		isolated[v] = VHalfedge(face[v]) == NO_ID;
		if (!isolated[v] && !VIsBoundary(face[v]))
			return NO_ID;
		const unsigned v1 = (v + 1) % numVertices;
		if (face[v] == face[v1])
			return NO_ID;
		const EIndex iE = isolated[v] || VHalfedge(face[v1]) == NO_ID ? NO_ID : EEdge(face[v], face[v1]);
		if (iE == NO_ID) {
			hedges[v] = NO_ID;
			++numNewEdges[v];
			++numNewEdges[v1];
		} else {
			if (!EIsBoundary(iE))
				return NO_ID; // duplicate/interior edge
			hedges[v] = HeBack(EHalfedge(iE));
			if (HeVertex(hedges[v]) != face[v])
				return NO_ID; // non-manifold edge (opposite orientation)
		}
	}
	for (unsigned v = 0; v < numVertices; ++v) {
		if (numNewEdges[v] > (isolated[v] ? 2u : 1u))
			return NO_ID; // non-manifold vertex
	}
	const std::size_t oldHeSize = heNexts.size();
	const std::size_t oldFaceSize = fHalfedges.size();
	const bool oldAlwaysEven = alwaysEven;
	std::vector<std::pair<VIndex, HIndex>> oldRepresentatives;
	oldRepresentatives.reserve(numVertices);
	for (unsigned v = 0; v < numVertices; ++v) {
		const VIndex vertex = face[v];
		if (std::find_if(oldRepresentatives.begin(), oldRepresentatives.end(), [vertex](const auto& entry) { return entry.first == vertex; }) == oldRepresentatives.end())
			oldRepresentatives.emplace_back(vertex, vHalfedges[vertex]);
	}
	std::vector<std::pair<HIndex, HIndex>> oldNexts;
	std::vector<std::pair<HIndex, FIndex>> oldHeFaces;
	oldNexts.reserve(numVertices * 3);
	oldHeFaces.reserve(numVertices);
	const auto RememberNext = [&](HIndex iHe) {
		if (iHe >= oldHeSize)
			return;
		if (std::find_if(oldNexts.begin(), oldNexts.end(), [iHe](const auto& entry) { return entry.first == iHe; }) == oldNexts.end())
			oldNexts.emplace_back(iHe, heNexts[iHe]);
	};
	const auto RememberFace = [&](HIndex iHe) {
		if (iHe >= oldHeSize)
			return;
		if (std::find_if(oldHeFaces.begin(), oldHeFaces.end(), [iHe](const auto& entry) { return entry.first == iHe; }) == oldHeFaces.end())
			oldHeFaces.emplace_back(iHe, heFaces[iHe]);
	};
	const auto Rollback = [&]() {
		for (const auto& [iHe, next] : oldNexts)
			heNexts[iHe] = next;
		for (const auto& [iHe, iF] : oldHeFaces)
			heFaces[iHe] = iF;
		heNexts.resize(oldHeSize);
		heVertices.resize(oldHeSize);
		heFaces.resize(oldHeSize);
		fHalfedges.resize(oldFaceSize);
		for (const auto& [vertex, representative] : oldRepresentatives)
			SetVHalfedge(vertex, representative);
		alwaysEven = oldAlwaysEven;
	};
	// add the new face
	const FIndex iF = static_cast<FIndex>(fHalfedges.size());
	fHalfedges.emplace_back(NO_ID);
	// walk around this face
	HIndex prevIHe = NO_ID, firstIHe = NO_ID;
	for (unsigned v = 0; v < numVertices; ++v) {
		// get an index for this half-edge
		HIndex iHe = hedges[v];
		if (iHe == NO_ID) {
			// create and initialize a new edge (half-edge pair)
			ASSERT(heNexts.size() == heVertices.size() && heNexts.size() == heFaces.size());
			const VIndex idxTail = face[v];
			const VIndex idxHead = face[(v + 1) % numVertices];
			iHe = static_cast<HIndex>(heNexts.size());
			SetVHalfedge(idxTail, iHe);
			heNexts.emplace_back(NO_ID);
			heNexts.emplace_back(NO_ID);
			heVertices.emplace_back(idxTail);
			heVertices.emplace_back(idxHead);
			heFaces.emplace_back(NO_ID);
			heFaces.emplace_back(NO_ID);
		}
		// update data-structures
		RememberFace(iHe);
		heFaces[iHe] = iF;
		if (v == 0) {
			fHalfedges[iF] = iHe;
			firstIHe = iHe;
		} else {
			RememberNext(prevIHe);
			heNexts[prevIHe] = iHe;
		}
		prevIHe = iHe;
	}
	// initialize first half-edge next index, skipped above
	RememberNext(prevIHe);
	heNexts[prevIHe] = firstIHe;
	// define the border edges
	for (unsigned v = 0; v < numVertices; ++v) {
		HIndex& iHe = vHalfedges[face[v]];
		if (!ConnectBordersImpl(*this, iHe, RememberNext)) {
			Rollback();
			return NO_ID;
		}
	}
	return iF;
}

bool HalfMesh::FAddDisk(const std::vector<Face>& faces)
{
	if (faces.empty())
		return true;
	HalfMesh original(*this);
	std::vector<Face> pending(faces);
	std::vector<FIndex> addedFaces;
	addedFaces.reserve(faces.size());
	while (!pending.empty()) {
		std::vector<Face> retry;
		retry.reserve(pending.size());
		for (const Face& face : pending) {
			const FIndex added = FAdd(face);
			if (added == NO_ID)
				retry.emplace_back(face);
			else
				addedFaces.emplace_back(added);
		}
		if (retry.empty())
			return true;
		if (retry.size() == pending.size()) {
			if (!addedFaces.empty()) {
				std::vector<VIndex> removedVerts;
				std::vector<VIndex> splitSrcVerts;
				FRemoveBulk(addedFaces, removedVerts, splitSrcVerts);
			}
			*this = std::move(original);
			return false;
		}
		pending.swap(retry);
	}
	return true;
}

#if HALFMESH_TRIS

bool HalfMesh::FIsCorner(FIndex iF) const
{
	unsigned numBorderEdges = 0;
	const HIndex iHeStart = fHalfedges[iF];
	ASSERT(!HeIsBoundary(iHeStart));
	HIndex iHe = iHeStart;
	while (true) {
		if (EHeIsBoundary(iHe))
			++numBorderEdges;
		if ((iHe = HeNext(iHe)) == iHeStart)
			break;
	}
	return numBorderEdges > 1;
}

void HalfMesh::FRemove(FIndex iF)
{
	if (fHalfedges.size() == 1) {
		*this = HalfMesh();
		return;
	}
	// remove face half-edges
	const HIndex iHeStart = fHalfedges[iF];
	HIndex iHe = HeNext(iHeStart);
	while (true) {
		const HIndex iHeNext = HeNext(iHe);
		HeRemove(iHe);
		if (iHe == iHeStart)
			break;
		iHe = iHeNext;
	}
	// remove face
	FRemoveOnly(iF);
}

void HalfMesh::FRemoveBulk(std::vector<FIndex>& faceRemoves,
                           std::vector<VIndex>& removedVerts,
                           std::vector<VIndex>& splitSrcVerts)
{
	if (faceRemoves.empty() || fHalfedges.empty())
		return;

	std::sort(faceRemoves.begin(), faceRemoves.end());
	faceRemoves.erase(std::unique(faceRemoves.begin(), faceRemoves.end()), faceRemoves.end());
	faceRemoves.erase(std::remove_if(faceRemoves.begin(), faceRemoves.end(), [this](FIndex face) { return face >= FSize(); }), faceRemoves.end());
	if (faceRemoves.empty())
		return;
	const std::size_t firstRemoved = removedVerts.size();

	const VIndex originalVertexCount = VSize();
	const EIndex originalEdgeCount = ESize();
	const FIndex originalFaceCount = FSize();
	std::vector<bool> removedFace(originalFaceCount, false);
	for (FIndex face : faceRemoves)
		removedFace[face] = true;
	const auto FaceSurvives = [&](FIndex face) {
		return face != NO_ID && !removedFace[face];
	};

	std::vector<VIndex> affectedVertices;
	std::vector<EIndex> affectedEdges;
	affectedVertices.reserve(faceRemoves.size() * 3);
	affectedEdges.reserve(faceRemoves.size() * 3);
	for (FIndex face : faceRemoves) {
		for (HIndex iHe : FAdjacentHalfedges(face)) {
			affectedVertices.emplace_back(HeTailVertex(iHe));
			affectedVertices.emplace_back(HeHeadVertex(iHe));
			affectedEdges.emplace_back(HeEdge(iHe));
		}
	}
	std::sort(affectedVertices.begin(), affectedVertices.end());
	affectedVertices.erase(std::unique(affectedVertices.begin(), affectedVertices.end()), affectedVertices.end());
	std::sort(affectedEdges.begin(), affectedEdges.end());
	affectedEdges.erase(std::unique(affectedEdges.begin(), affectedEdges.end()), affectedEdges.end());

	std::vector<bool> touchedVertex(originalVertexCount, false);
	std::vector<VIndex> touchedVertices;
	touchedVertices.reserve(affectedVertices.size() + faceRemoves.size());
	const auto TouchVertex = [&](VIndex vertex) {
		if (vertex >= touchedVertex.size())
			touchedVertex.resize(static_cast<std::size_t>(vertex) + 1, false);
		if (!touchedVertex[vertex]) {
			touchedVertex[vertex] = true;
			touchedVertices.emplace_back(vertex);
		}
	};
	std::vector<bool> removedVertex(originalVertexCount, false);
	// Split every surviving disconnected fan before changing face/edge links.
	// One face-bearing outgoing half-edge represents each incident face in the
	// cyclic vertex orbit; removed faces and an existing boundary separate runs.
	for (VIndex vertex : affectedVertices) {
		std::vector<HIndex> ring;
		for (HIndex iHe : VOutgoingHalfedges(vertex))
			ring.emplace_back(iHe);
		if (ring.empty()) {
			removedVerts.emplace_back(vertex);
			removedVertex[vertex] = true;
			continue;
		}

		std::vector<std::vector<HIndex>> fans;
		std::size_t separator = ring.size();
		for (std::size_t i = 0; i < ring.size(); ++i)
			if (!FaceSurvives(heFaces[ring[i]])) {
				separator = i;
				break;
			}
		if (separator == ring.size()) {
			fans.emplace_back(ring);
		} else {
			std::vector<HIndex> fan;
			for (std::size_t step = 1; step <= ring.size(); ++step) {
				const HIndex iHe = ring[(separator + step) % ring.size()];
				if (FaceSurvives(heFaces[iHe])) {
					fan.emplace_back(iHe);
				} else if (!fan.empty()) {
					fans.emplace_back(std::move(fan));
					fan.clear();
				}
			}
			if (!fan.empty())
				fans.emplace_back(std::move(fan));
		}

		if (fans.empty()) {
			removedVerts.emplace_back(vertex);
			removedVertex[vertex] = true;
			continue;
		}
		const HIndex oldRepresentative = vHalfedges[vertex];
		const auto originalFan = std::find_if(fans.begin(), fans.end(), [oldRepresentative](const std::vector<HIndex>& fan) {
			return std::find(fan.begin(), fan.end(), oldRepresentative) != fan.end();
		});
		if (originalFan != fans.end() && originalFan != fans.begin())
			std::iter_swap(fans.begin(), originalFan);
		SetVHalfedge(vertex, fans.front().front());
		if (fans.size() > 1)
			TouchVertex(vertex);

		for (std::size_t fanIndex = 1; fanIndex < fans.size(); ++fanIndex) {
			const VIndex duplicate = VSize();
			vHalfedges.emplace_back(NO_ID);
			TouchVertex(duplicate);
			splitSrcVerts.emplace_back(vertex);
			for (HIndex faceOut : fans[fanIndex]) {
				const HIndex previous = HePrev(faceOut);
				heVertices[faceOut] = duplicate;
				heVertices[HeTwin(previous)] = duplicate;
			}
			SetVHalfedge(duplicate, fans[fanIndex].front());
		}
	}

	std::vector<EIndex> edgeRemoves;
	edgeRemoves.reserve(faceRemoves.size() * 2);
	for (EIndex edge : affectedEdges) {
		ASSERT(edge < originalEdgeCount);
		const HIndex even = EHalfedge(edge);
		const HIndex odd = HeTwin(even);
		const bool evenSurvives = FaceSurvives(heFaces[even]);
		const bool oddSurvives = FaceSurvives(heFaces[odd]);
		if (evenSurvives && oddSurvives)
			continue;

		const VIndex endpoint0 = heVertices[even];
		const VIndex endpoint1 = heVertices[odd];
		TouchVertex(endpoint0);
		TouchVertex(endpoint1);
		if (!evenSurvives && !oddSurvives) {
			edgeRemoves.emplace_back(edge);
			continue;
		}

		if (evenSurvives) {
			heFaces[odd] = NO_ID;
			heNexts[odd] = NO_ID;
		} else {
			const FIndex survivingFace = heFaces[odd];
			const VIndex survivingTail = heVertices[odd];
			const HIndex previous = HePrev(odd);
			heNexts[previous] = even;
			heNexts[even] = heNexts[odd];
			heNexts[odd] = NO_ID;
			if (vHalfedges[survivingTail] == odd)
				SetVHalfedge(survivingTail, even);
			std::swap(heVertices[even], heVertices[odd]);
			heFaces[even] = survivingFace;
			heFaces[odd] = NO_ID;
			if (fHalfedges[survivingFace] == odd)
				fHalfedges[survivingFace] = even;
		}
	}

	// Rebuild only the boundary links at vertices whose local fan changed.
	std::sort(touchedVertices.begin(), touchedVertices.end());
	for (VIndex vertex : touchedVertices) {
		if (vertex < removedVertex.size() && removedVertex[vertex])
			continue;
		if (!ConnectBorders(vHalfedges[vertex])) {
			ASSERT(false && "FRemoveBulk produced an invalid surviving fan");
			return;
		}
	}

	if (!edgeRemoves.empty())
		ERemoveOnly(edgeRemoves.data(), static_cast<unsigned>(edgeRemoves.size()));
	FRemoveOnly(faceRemoves.data(), static_cast<unsigned>(faceRemoves.size()));
	if (removedVerts.size() > firstRemoved)
		VRemoveOnly(removedVerts.data() + firstRemoved, static_cast<unsigned>(removedVerts.size() - firstRemoved));
}

bool HalfMesh::FRemoveCorner(FIndex iF, RemovedData& removedData)
{
	ASSERT(FIsCorner(iF));
	// remove corner face
	const auto RemoveCorner = [&](HIndex iHeInterior) {
		HIndex iHe = iHeInterior;
		while (true) {
			const HIndex iHeNext = HeNext(iHe);
			if (EHeIsBoundary(iHe)) {
				// just remove the border edge
				removedData.edges[removedData.numEdges++] = HeEdge(iHe);
				if (iHeNext == iHeInterior) {
					VRemoveOnly(removedData.verts[removedData.numVerts++] = HeVertex(iHe));
				}
			} else {
				if (iHe != HeBack(iHe)) {
					// swap edge half-edges such that the second corresponds to the boundary link
					ASSERT(heNexts[HeNext(HeNext(HeBack(iHe)))] == HeBack(iHe));
					heNexts[HeNext(HeNext(HeBack(iHe)))] = iHe;
					heNexts[iHe] = heNexts[HeBack(iHe)];
					heFaces[iHe] = heFaces[HeBack(iHe)];
					std::swap(heVertices[iHe], heVertices[HeBack(iHe)]);
					if (fHalfedges[heFaces[HeBack(iHe)]] == HeBack(iHe))
						fHalfedges[heFaces[HeBack(iHe)]] = iHe;
					iHe = HeBack(iHe);
				}
				// update boundary links
				heNexts[HePrevBoundary(HeTwin(HeNext(iHeNext)))] = iHe;
				heNexts[iHe] = heNexts[HeTwin(iHeNext)];
				heFaces[iHe] = NO_ID;
				SetVHalfedge(heVertices[HeTwin(iHe)], HeTwin(iHe));
			}
			if (iHeNext == iHeInterior)
				break;
			iHe = iHeNext;
		}
		ERemoveOnly(removedData.edges, removedData.numEdges);
		FRemoveOnly(removedData.faces[removedData.numFaces++] = iF);
	};
	// find non-border edge
	const HIndex iHeStart = fHalfedges[iF];
	HIndex iHe = iHeStart;
	while (true) {
		if (!EHeIsBoundary(iHe)) {
			RemoveCorner(iHe);
			return true;
		}
		if ((iHe = HeNext(iHe)) == iHeStart)
			break;
	}
	// remove stand alone face
	iHe = iHeStart;
	while (true) {
		removedData.edges[removedData.numEdges++] = HeEdge(iHe);
		removedData.verts[removedData.numVerts++] = HeVertex(iHe);
		if ((iHe = HeNext(iHe)) == iHeStart)
			break;
	}
	VRemoveOnly(removedData.verts, removedData.numVerts);
	ERemoveOnly(removedData.edges, removedData.numEdges);
	FRemoveOnly(iF);
	return false;
}
#endif

void HalfMesh::FRemoveOnly(FIndex iF)
{
	if (iF + 1 < fHalfedges.size()) {
		const HIndex iHeStart = fHalfedges[iF] = fHalfedges.back();
		HIndex iHe = iHeStart;
		while (true) {
			ASSERT(heFaces[iHe] == fHalfedges.size() - 1);
			heFaces[iHe] = iF;
			if ((iHe = HeNext(iHe)) == iHeStart)
				break;
		}
	}
	fHalfedges.pop_back();
}

void HalfMesh::FRemoveOnly(FIndex* faces, unsigned numFaces)
{
	ASSERT(faces != NULL);
	std::sort(faces, faces + numFaces, [](EIndex i, EIndex j) { return i > j; });
	for (unsigned i = 0; i < numFaces; ++i)
		FRemoveOnly(faces[i]);
}

#if HALFMESH_TRIS

//        v1
//        /|\
//       / | \
//      /  |  \
//  vL / L | R \ vR
//     \   |   /
//      \  |  /
//     x \ | / x
//        \|/
//        v0
bool HalfMesh::EIsCollapseValidTopologically(EIndex iE) const
{
	// test that neither left nor right face are corner faces:
	// both other face edges are on the border
	const HIndex v0V1 = EHalfedge(iE);
	const HIndex v1VL = HeNext(v0V1);
	const HIndex vLV0 = HeNext(v1VL);
	ASSERT(!HeIsBoundary(v0V1));
	if (HeIsBoundary(HeTwin(v1VL)) && HeIsBoundary(HeTwin(vLV0)))
		return false;
	const HIndex v1V0 = HeTwin(v0V1);
	const HIndex v0VR = HeNext(v1V0);
	const HIndex vRV1 = HeNext(v0VR);
	const VIndex v0 = HeVertex(v0V1);
	const VIndex v1 = HeVertex(v1V0);
	if (!HeIsBoundary(v1V0)) {
		if (HeIsBoundary(HeTwin(v0VR)) && HeIsBoundary(HeTwin(vRV1)))
			return false;
		// edge is not on the border, check that at least one vertex is not on the border either
		if (VIsBoundary(v0) && VIsBoundary(v1))
			return false;
	}
	// check the link condition for v0<->v1 edge
	const VIndex vL = HeVertex(vLV0);
	const VIndex vR = HeVertex(vRV1);
	// One-ring marking (OpenMesh/PMP style): circulate v1 once into a small stack
	// buffer (thread-local heap fallback for pathological degrees), then test each
	// neighbor of v0 for membership by a contiguous scan instead of re-circulating
	// every neighbor's full incoming ring. Same accept/reject verdict: a common
	// neighbor of v0 and v1 must be a wing (vL, or vR when the edge is interior).
	// v1 is never in its own one-ring, so the neighbor v==v1 along the collapsed
	// edge is naturally excluded (no need for the old outgoing/incoming skip).
	constexpr uint32_t linkStackCap = 64;
	VIndex n1Stack[linkStackCap];
	thread_local std::vector<VIndex> n1Heap;
	uint32_t n1Size = 0;
	bool n1UseHeap = false;
	for (const HIndex v1V : VIncomingHalfedges(v1)) {
		const VIndex v = HeVertex(v1V);
		if (!n1UseHeap) {
			if (n1Size < linkStackCap) {
				n1Stack[n1Size++] = v;
				continue;
			}
			// overflow: migrate the stack buffer into the thread-local heap buffer
			n1Heap.assign(n1Stack, n1Stack + linkStackCap);
			n1UseHeap = true;
		}
		n1Heap.emplace_back(v);
	}
	const VIndex* const n1 = n1UseHeap ? n1Heap.data() : n1Stack;
	const uint32_t n1Count = n1UseHeap ? static_cast<uint32_t>(n1Heap.size()) : n1Size;
	for (const HIndex v0V : VIncomingHalfedges(v0)) {
		const VIndex v = HeVertex(v0V);
		bool common = false;
		for (uint32_t i = 0; i < n1Count; ++i) {
			if (n1[i] == v) {
				common = true;
				break;
			}
		}
		if (!common)
			continue;
		// v0-v-v1 are connected, determine if this triangle is a face of the mesh
		const bool isFace = (v == vL || (v == vR && !HeIsBoundary(v1V0)));
		if (!isFace)
			return false;
	}
	return true;
}

bool HalfMesh::EIsCollapseValidGeometrically(EIndex iE, const Vertex& p0, const std::vector<Vertex>& vertices) const
{
	// list vertices incident to v0 and v1 (excluding themselves)
	// stored as outgoing half-edges
	const HIndex v0V1 = EHalfedge(iE);
	const HIndex v1V0 = HeTwin(v0V1);
	// thread-local scratch reused across calls: steady-state zero-allocation on the
	// decimation hot path. thread_local (never static) — MeshRemesh calls this
	// predicate concurrently. Not reentrant on a single thread, so clear-and-refill
	// per call is safe.
	thread_local std::vector<HIndex> linkHeVertices;
	linkHeVertices.clear();
	{
		const auto vector = IteratorStateBase<VertexOutgoingHalfedgeIterator>(Halfedge{v0V1, *this});
		auto it = HeIsBoundary(v1V0) ? ++vector.begin() : ++(++vector.begin());
		if (it == vector.end())
			return false;
		do {
			linkHeVertices.emplace_back(HeNext(*it));
		} while (++it != vector.end());
	}
	{
		const auto vector = IteratorStateBase<VertexOutgoingHalfedgeIterator>(Halfedge{v1V0, *this});
		auto it = ++(++vector.begin());
		if (it == vector.end())
			return false;
		do {
			linkHeVertices.emplace_back(HeNext(*it));
		} while (++it != vector.end());
	}
	ASSERT(!linkHeVertices.empty());
	// given triangles p0,p1,p2 and p0,p2,p3, sharing edge v0-v2, determine if they are geometrically valid:
	//  - the ratio of their respective areas is no greater than a given threshold
	//  - the internal dihedral angle formed by their supporting planes is no greater than a given threshold
	const auto ValidateSharedTriangles = [](const Vertex& p0, const Vertex& p1, const Vertex& p2, const Vertex& p3) {
		// Evaluate the area-ratio and dihedral guards in double. Both comparands
		// scale as edge-length^4..^8: in float they flush to 0 on fine detail /
		// small coordinates ("0 <= 0" then accepts a 180deg fold-over) and to inf
		// on large coordinates ("inf <= inf" also accepts), making the verdict
		// scale-dependent. Float subtraction then exact widening keeps vertex
		// storage in float while giving the products ~1e+-300 of headroom; the
		// verdict is unchanged at ordinary scales.
		typedef Eigen::Matrix<double, 3, 1> Vector;
		const Vector e01 = (p1 - p0).cast<double>();
		const Vector e02 = (p2 - p0).cast<double>();
		const Vector e03 = (p3 - p0).cast<double>();
		const Vector n012 = e01.cross(e02);
		const Vector n023 = e02.cross(e03);
		const double l012 = n012.squaredNorm();
		const double l023 = n023.squaredNorm();
		const double larger = std::max(l012, l023);
		const double smaller = std::min(l012, l023);
		// l012/l023 are |cross|^2 = (2*area)^2, so this threshold bounds the true
		// area ratio of the two post-collapse triangles at sqrt(maxAreaRatio):
		// 1e4 here = 100:1. Retuned 2026-08 (was 1e8 = 1e4:1, measured inert — 3
		// of 9,937 geometric rejects on the pipes CAD assembly). Swept {1e2, 1e4,
		// 1e5, 1e6, 1e8} on the challenge fixture, pipes and a UV sphere at 1e4:
		// clean meshes are byte-insensitive; on the fixture the worst output
		// sliver improves 5.6x (min quality 0.0023 -> 0.0129) at unchanged
		// target reachability and fidelity; on pipes the worst adjacent ratio
		// drops 8,810 -> 834 and max surface deviation halves, for +1.3% on an
		// already link-condition-dominated stall floor. 1e2 (10:1) is too tight:
		// it degrades p1 triangle quality by forcing worse collapses elsewhere.
		const double maxAreaRatio = 1e4;
		const double maxDihedralAngleSquaredCos = SQUARE(std::cos(D2R(1.0)));
		if (larger < maxAreaRatio * smaller) {
			const double l0123 = n012.dot(n023);
			if (l0123 > 0)
				return true;
			if (SQUARE(l0123) <= maxDihedralAngleSquaredCos * (l012 * l023))
				return true;
		}
		return false;
	};
	// iterate over all 3 consecutive vertices in the link
	FOREACH (l, linkHeVertices) {
		// get the three consecutive half-edges along the link
		// and their corresponding v1,v2,v3 consecutive vertices along the link
		const HIndex h12 = linkHeVertices[l + 1 == lSize ? 0 : l + 1];
		const HIndex h23 = linkHeVertices[l];
		const HIndex h3n = linkHeVertices[l == 0 ? lSize - 1 : l - 1];
		// if edge h21 exists (not boundary edge)
		if (!EHeIsBoundary(h12)) {
			const VIndex v1 = HeVertex(h12);
			const VIndex v2 = HeVertex(h23);
			ASSERT(HeHeadVertex(h12) == v2);
			// if edge h32 exists (not boundary edge)
			if (!EHeIsBoundary(h23)) {
				// there will be two adjacent triangles v0,v1,v2 and v0,v2,v3 after the collapse
				const VIndex v3 = HeVertex(h3n);
				ASSERT(HeHeadVertex(h23) == v3);
				if (!ValidateSharedTriangles(p0, vertices[v1], vertices[v2], vertices[v3]))
					return false;
			}
			// the triangle exterior to the link for edge h21 exists,
			// check also triangles v0,v1,v2 and v4,v2,v1
			const VIndex v4 = HeHeadVertex(HeNextOutgoingHalfedge(h12));
			if (!ValidateSharedTriangles(vertices[v1], vertices[v4], vertices[v2], p0))
				return false;
		}
	}
	return true;
}

HalfMesh::VIndex HalfMesh::ERemove(EIndex iE, RemovedData& removedData)
{
	HIndex v0V1 = EHalfedge(iE);
	HIndex v1V0 = HeTwin(v0V1);
	if (!HeIsBoundary(v1V0) && (EHeIsBoundary(HeNext(v1V0)) || EHeIsBoundary(HeNext(HeNext(v0V1))))) {
		// swap half-edges
		std::swap(v0V1, v1V0);
	}
	const HIndex v1VL = HeNext(v0V1);
	const HIndex vLV0 = HeNext(v1VL);
	const VIndex v0 = HeVertex(v0V1);
	const VIndex v1 = HeVertex(v1V0);
	// unite vV0 into v1V twin half-edges
	// note: at least one edge is not on the border (excluded by the topology check)
	const auto UniteFaceHalfedges = [this, &removedData](HIndex v1V, HIndex vV0) {
		ASSERT(!EHeIsBoundary(vV0));
		const HIndex vV0Twin = HeTwin(vV0);
		heNexts[HePrev(vV0Twin)] = v1V;
		heNexts[v1V] = HeNext(vV0Twin);
		heFaces[v1V] = HeFace(vV0Twin);
		if (fHalfedges[heFaces[v1V]] == vV0Twin)
			fHalfedges[heFaces[v1V]] = v1V;
		removedData.faces[removedData.numFaces++] = heFaces[vV0];
	};
	if (!HeIsBoundary(v1V0)) {
		// remove right face
		const HIndex v0VR = HeNext(v1V0);
		const HIndex vRV1 = HeNext(v0VR);
		UniteFaceHalfedges(vRV1, v0VR);
		if (vHalfedges[heVertices[vRV1]] == HeTwin(v0VR))
			SetVHalfedge(heVertices[vRV1], vRV1);
		if (vHalfedges[v1] == v1V0)
			SetVHalfedge(v1, v1VL);
		removedData.edges[removedData.numEdges++] = HeEdge(v0VR);
		// set v0 to v1
		if (VIsBoundary(v0)) {
			ASSERT(!EHeIsBoundary(v0VR) && v0VR != VHalfedge(v0));
			SetVHalfedge(v1, VHalfedge(v0));
		}
		for (HIndex iHe = HeNext(vRV1); iHe != v0V1; iHe = HeNextOutgoingHalfedge(iHe)) {
			ASSERT(heVertices[iHe] == v0);
			heVertices[iHe] = v1;
		}
	} else {
		if (EHeIsBoundary(v1VL) || EHeIsBoundary(vLV0)) {
			[[maybe_unused]] const bool cornerFace = FRemoveCorner(HeFace(v0V1), removedData);
			ASSERT(cornerFace);
			return (v0 == removedData.verts[0] && v1 < vHalfedges.size()) ? v1 : v0;
		}
		// update border half-edge connections
		ASSERT(HeNext(HeTwin(vHalfedges[v1])) == v1V0);
		ASSERT(HeTwin(vHalfedges[v0]) == v1V0);
		heNexts[HePrev(v1V0)] = HeNext(v1V0);
		// set v0 to v1
		ASSERT(HeIsBoundary(HeNextOutgoingHalfedge(v0V1)));
		for (HIndex iHe = HeNextOutgoingHalfedge(v0V1); iHe != v0V1; iHe = HeNextOutgoingHalfedge(iHe)) {
			ASSERT(heVertices[iHe] == v0);
			heVertices[iHe] = v1;
		}
	}
	// remove left face
	ASSERT(!HeIsBoundary(v0V1));
	UniteFaceHalfedges(v1VL, vLV0);
	if (vHalfedges[heVertices[vLV0]] == vLV0)
		SetVHalfedge(heVertices[vLV0], HeTwin(v1VL));
	removedData.edges[removedData.numEdges++] = HeEdge(vLV0);
	// remove vertex
	VRemoveOnly(removedData.verts[removedData.numVerts++] = v0);
	// remove edges
	removedData.edges[removedData.numEdges++] = iE;
	ERemoveOnly(removedData.edges, removedData.numEdges);
	// remove faces
	FRemoveOnly(removedData.faces, removedData.numFaces);
	return v1 < vHalfedges.size() ? v1 : v0;
}

//                                     +1
//           v1                     v1
//          /  \                   /|\
//         /  a \               d / | \
//        /      \               /  |  \
//       /   iE   \          -1 /   |   \ -1
//      v0--------v2 ========> v0 b | a v2
//       \        /             \   |   /
//        \   b  /               \  |  / d
//         \    /                 \ | /
//          \  /                   \|/ +1
//           v3                     v3
//       Before Flip            After Flip

bool HalfMesh::EIsFlipValid(EIndex iE, const std::vector<Vertex>& vertices) const
{
	if (EIsBoundary(iE))
		return false;
	// check if topologically flippable
	const HIndex iHa0 = EHalfedge(iE);
	const HIndex iHa1 = HeNext(iHa0);
	const HIndex iHa2 = HeNext(iHa1);
	const HIndex iHb0 = HeTwin(iHa0);
	const HIndex iHb1 = HeNext(iHb0);
	const HIndex iHb2 = HeNext(iHb1);
	// incident on degree 1 vertex
	if (iHa1 == iHb0 || iHb1 == iHa0)
		return false;
	// reject only exactly-degenerate (zero-area) new triangles: .norm() is an
	// UNSIGNED cross-product magnitude, so these two tests can never fire on a
	// fold-over (detecting that would need a signed/oriented area). This is by
	// design — a flip's correctness here is purely topological (interior edge with
	// the opposite edge absent, checked below) and any caller that cares about
	// fold-over geometry gates it itself; this guard only avoids emitting a null
	// triangle from a degenerate input.
	const Type A1 = (vertices[HeVertex(iHb2)] - vertices[HeVertex(iHa0)]).cross(vertices[HeVertex(iHa2)] - vertices[HeVertex(iHa0)]).norm();
	if (A1 <= 0)
		return false;
	const Type A2 = (vertices[HeVertex(iHa2)] - vertices[HeVertex(iHb0)]).cross(vertices[HeVertex(iHb2)] - vertices[HeVertex(iHb0)]).norm();
	if (A2 <= 0)
		return false;
	// check the new edge doesn't exist already
	return EEdge(HeVertex(iHa2), HeVertex(iHb2)) == NO_ID;
}

void HalfMesh::EFlip(EIndex iE)
{
	// get vertices and faces
	const HIndex iHa0 = EHalfedge(iE);
	const HIndex iHa1 = HeNext(iHa0);
	const HIndex iHa2 = HeNext(iHa1);
	const HIndex iHb0 = HeTwin(iHa0);
	const HIndex iHb1 = HeNext(iHb0);
	const HIndex iHb2 = HeNext(iHb1);
	const VIndex v0 = HeVertex(iHa0);
	const VIndex v2 = HeVertex(iHb0);
	const VIndex v1 = HeVertex(iHa2);
	const VIndex v3 = HeVertex(iHb2);
	const FIndex fa = HeFace(iHa0);
	const FIndex fb = HeFace(iHb0);
	// update vertex pointers
	if (vHalfedges[v0] == iHa0)
		SetVHalfedge(v0, iHb1);
	if (vHalfedges[v2] == iHb0)
		SetVHalfedge(v2, iHa1);
	// update face pointers
	if (fHalfedges[fa] != iHa1)
		fHalfedges[fa] = iHb2;
	if (fHalfedges[fb] != iHb1)
		fHalfedges[fb] = iHa2;
	// update half-edge pointers
	heNexts[iHa0] = iHb2;
	heNexts[iHb2] = iHa1;
	heNexts[iHa1] = iHa0;
	heNexts[iHb0] = iHa2;
	heNexts[iHa2] = iHb1;
	heNexts[iHb1] = iHb0;
	heVertices[iHa0] = v1;
	heVertices[iHb0] = v3;
	heFaces[iHa2] = fb;
	heFaces[iHb2] = fa;
}

HalfMesh::VIndex HalfMesh::ESplit(EIndex iE)
{
	// iHa0 (even) is always the interior/face side; iHb0 (odd) is the boundary
	// side iff the edge is on the boundary.
	const HIndex iHa0 = EHalfedge(iE); // a -> b, face fa
	const HIndex iHb0 = HeTwin(iHa0); // b -> a, face fb OR boundary
	const HIndex iHa1 = HeNext(iHa0); // b -> c
	const HIndex iHa2 = HeNext(iHa1); // c -> a
	const VIndex b = heVertices[iHb0];
	const VIndex c = heVertices[iHa2];
	const FIndex fa = heFaces[iHa0];
	ASSERT(fa != NO_ID); // even side always carries a face

	const VIndex m = static_cast<VIndex>(vHalfedges.size());
	const HIndex H = static_cast<HIndex>(heNexts.size()); // even (pairs)

	if (!HeIsBoundary(iHb0)) {
		// ---- interior edge: fa=(a,b,c), fb=(b,a,d) -> 4 triangles ----
		const HIndex iHb1 = HeNext(iHb0); // a -> d
		const HIndex iHb2 = HeNext(iHb1); // d -> b
		const VIndex d = heVertices[iHb2];
		const FIndex fb = heFaces[iHb0];

		const HIndex heMb = H, heMbT = H + 1; // m->b , b->m
		const HIndex heMc = H + 2, heMcT = H + 3; // m->c , c->m
		const HIndex heMd = H + 4, heMdT = H + 5; // m->d , d->m
		heNexts.resize(H + 6);
		heVertices.resize(H + 6);
		heFaces.resize(H + 6);
		const FIndex fc = static_cast<FIndex>(fHalfedges.size()); // (m,b,c)
		const FIndex fd = fc + 1; // (m,a,d)
		fHalfedges.resize(fc + 2);

		heVertices[iHb0] = m; // b->a becomes m->a
		// clang-format off: tabular layout shows the parallel per-fan writes at a glance
		heVertices[heMb] = m;   heVertices[heMbT] = b;
		heVertices[heMc] = m;   heVertices[heMcT] = c;
		heVertices[heMd] = m;   heVertices[heMdT] = d;

		heFaces[iHa0] = fa;   heFaces[heMc] = fa;   heFaces[iHa2] = fa;
		heFaces[heMb] = fc;  heFaces[iHa1] = fc;    heFaces[heMcT] = fc;
		heFaces[heMbT] = fb; heFaces[heMd] = fb;  heFaces[iHb2] = fb;
		heFaces[iHb0] = fd;   heFaces[iHb1] = fd;    heFaces[heMdT] = fd;

		// fa=(a,m,c): iHa0 -> heMc -> iHa2
		heNexts[iHa0] = heMc;   heNexts[heMc] = iHa2;   heNexts[iHa2] = iHa0;
		// fc=(m,b,c): heMb -> iHa1 -> heMcT
		heNexts[heMb] = iHa1;   heNexts[iHa1] = heMcT; heNexts[heMcT] = heMb;
		// fb=(b,m,d): heMbT -> heMd -> iHb2
		heNexts[heMbT] = heMd; heNexts[heMd] = iHb2;  heNexts[iHb2] = heMbT;
		// fd=(m,a,d): iHb0 -> iHb1 -> heMdT
		heNexts[iHb0] = iHb1;    heNexts[iHb1] = heMdT; heNexts[heMdT] = iHb0;
		// clang-format on

		fHalfedges[fa] = iHa0;
		fHalfedges[fb] = heMbT;
		fHalfedges[fc] = heMb;
		fHalfedges[fd] = iHb0;

		vHalfedges.emplace_back(heMb); // m's even outgoing
		// iHb0 is the only existing half-edge whose tail changed (b -> m). If it
		// was b's representative (EFlip can leave an odd one there), replace it
		// with a still-outgoing-from-b half-edge. a, c, d reps stay valid.
		if (vHalfedges[b] == iHb0)
			SetVHalfedge(b, heMbT); // b -> m
	} else {
		// ---- boundary edge: only fa=(a,b,c); iHb0 is the boundary half-edge ----
		const HIndex iHbNext = heNexts[iHb0]; // boundary he outgoing from a
		const HIndex iHbPrev = HePrevBoundary(iHb0); // boundary he into b (before surgery)

		const HIndex heMb = H, heMbT = H + 1; // m->b (face fc) , b->m (boundary)
		const HIndex heMc = H + 2, heMcT = H + 3; // m->c (face fa) , c->m (face fc)
		heNexts.resize(H + 4);
		heVertices.resize(H + 4);
		heFaces.resize(H + 4);
		const FIndex fc = static_cast<FIndex>(fHalfedges.size()); // (m,b,c)
		fHalfedges.resize(fc + 1);

		heVertices[iHb0] = m; // boundary b->a becomes m->a
		// clang-format off: tabular layout shows the parallel per-fan writes at a glance
		heVertices[heMb] = m;  heVertices[heMbT] = b;
		heVertices[heMc] = m;  heVertices[heMcT] = c;

		heFaces[iHa0] = fa;   heFaces[heMc] = fa;   heFaces[iHa2] = fa;
		heFaces[heMb] = fc;  heFaces[iHa1] = fc;    heFaces[heMcT] = fc;
		heFaces[heMbT] = NO_ID; // boundary
		// iHb0 stays a boundary half-edge (heFaces[iHb0] already NO_ID)

		// fa=(a,m,c): iHa0 -> heMc -> iHa2
		heNexts[iHa0] = heMc;   heNexts[heMc] = iHa2;   heNexts[iHa2] = iHa0;
		// fc=(m,b,c): heMb -> iHa1 -> heMcT
		heNexts[heMb] = iHa1;   heNexts[iHa1] = heMcT; heNexts[heMcT] = heMb;
		// boundary loop: iHbPrev -> heMbT(b->m) -> iHb0(m->a) -> iHbNext
		heNexts[iHbPrev] = heMbT; heNexts[heMbT] = iHb0; heNexts[iHb0] = iHbNext;
		// clang-format on

		fHalfedges[fa] = iHa0;
		fHalfedges[fc] = heMb;

		vHalfedges.emplace_back(heMb); // m's even outgoing; edge (m,b) is boundary
		// iHb0's tail changed b -> m; replace b's representative if needed
		// (heMbT is b's outgoing on the new boundary edge, so VIsBoundary(b) holds).
		if (vHalfedges[b] == iHb0)
			SetVHalfedge(b, heMbT); // b -> m (boundary)
	}
	return m;
}
#endif

void HalfMesh::ERemoveOnly(EIndex iE)
{
	ASSERT(heNexts.size() == heVertices.size() && heNexts.size() == heFaces.size());
	const auto MoveBackHalfedgeTo = [this](HIndex iHe) {
		if (iHe + 1 < heNexts.size()) {
			const HIndex iHeLast = static_cast<HIndex>(heNexts.size()) - 1;
			const HIndex iHePrev = HePrev(iHeLast);
			ASSERT(heNexts[iHePrev] == iHeLast);
			heNexts[iHe] = heNexts.back();
			heNexts[iHePrev] = iHe;
			heVertices[iHe] = heVertices.back();
			if (vHalfedges[heVertices[iHe]] == iHeLast)
				SetVHalfedge(heVertices[iHe], iHe);
			heFaces[iHe] = heFaces.back();
			if (!HeIsBoundary(iHe) && fHalfedges[heFaces[iHe]] == iHeLast)
				fHalfedges[heFaces[iHe]] = iHe;
		}
		heNexts.pop_back();
		heVertices.pop_back();
		heFaces.pop_back();
	};
	const HIndex iHe = EHalfedge(iE);
	const HIndex iHeTwin = HeTwin(iHe);
	MoveBackHalfedgeTo(iHeTwin);
	MoveBackHalfedgeTo(iHe);
}

void HalfMesh::ERemoveOnly(EIndex* edges, unsigned numEdges)
{
	ASSERT(edges != NULL);
	std::sort(edges, edges + numEdges, [](EIndex i, EIndex j) { return i > j; });
	for (unsigned i = 0; i < numEdges; ++i)
		ERemoveOnly(edges[i]);
}

} // namespace halfmesh
