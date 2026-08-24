/*
* HalfMeshInvariantsTest.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Durable structural-invariant + edge-collapse tests for the HalfMesh half-edge
// structure. These are self-contained (no external reference): they assert the
// half-edge representation is internally consistent and matches the topology of a
// corpus of meshes (including a genus-1 torus and a two-boundary cylinder), and
// that the edge-collapse primitive ERemove and its collapse-validity predicates
// behave correctly. Together they pin the structural contract future work can
// rely on.

#include <gtest/gtest.h>

#include <halfmesh/HalfMesh.h>
#include <halfmesh/Mesh.h>

#include "Corpus.h"
#include "Metrics.h"

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <numbers>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace halfmesh {
namespace {

namespace corpus = hmtest::corpus;
namespace metrics = hmtest::metrics;

// ---------------------------------------------------------------------------
// Structural invariants that must hold for ANY correctly built half-edge mesh,
// independent of the specific topology (uses only the HalfMesh itself).
// ---------------------------------------------------------------------------
void CheckStructuralInvariants(const HalfMesh& hm)
{
	const HalfMesh::HIndex H = static_cast<HalfMesh::HIndex>(hm.heNexts.size());

	// Array-size consistency: the three half-edge arrays are parallel, and
	// half-edges come in twin pairs (even count).
	ASSERT_EQ(hm.heVertices.size(), hm.heNexts.size());
	ASSERT_EQ(hm.heFaces.size(), hm.heNexts.size());
	ASSERT_EQ(H % 2u, 0u);

	const HalfMesh::VIndex V = hm.VSize();
	const HalfMesh::FIndex F = hm.FSize();

	for (HalfMesh::HIndex h = 0; h < H; ++h) {
		// Indices stay in range; heFaces may be NO_ID only on the boundary.
		EXPECT_LT(hm.heNexts[h], H) << "he_nexts out of range at " << h;
		EXPECT_LT(hm.heVertices[h], V) << "he_vertices out of range at " << h;
		EXPECT_TRUE(hm.heFaces[h] == math::NO_ID || hm.heFaces[h] < F)
		    << "he_faces out of range at " << h;

		// "Next starts where current ends": the tail of HeNext(h) and the tail
		// of HeTwin(h) are both the head vertex of h. This is the core half-edge
		// consistency invariant (the assert inside HeHeadVertex).
		EXPECT_EQ(hm.heVertices[hm.HeTwin(h)], hm.heVertices[hm.HeNext(h)])
		    << "next/twin head-vertex mismatch at he " << h;

		// Boundary half-edges must be the odd (back) half-edge of their pair.
		if (hm.heFaces[h] == math::NO_ID)
			EXPECT_TRUE(h & 1u) << "boundary he must be odd: " << h;
	}

	// Orbit closure: following HeNext from any half-edge returns to it, and the
	// face is constant along the orbit (a triangle for interior faces).
	for (HalfMesh::HIndex h = 0; h < H; ++h) {
		const HalfMesh::FIndex f = hm.heFaces[h];
		HalfMesh::HIndex cur = h;
		int steps = 0;
		do {
			EXPECT_EQ(hm.heFaces[cur], f) << "orbit face not constant from he " << h;
			cur = hm.HeNext(cur);
			++steps;
			ASSERT_LE(steps, static_cast<int>(H)) << "HeNext orbit did not close from he " << h;
		} while (cur != h);
		if (f != math::NO_ID)
			EXPECT_EQ(steps, 3) << "interior face orbit must be a triangle (he " << h << ")";
	}

	// Vertex/face start half-edges point where they should.
	for (HalfMesh::VIndex v = 0; v < V; ++v)
		EXPECT_EQ(hm.heVertices[hm.VHalfedge(v)], v) << "v_halfedge not outgoing for vertex " << v;
	for (HalfMesh::FIndex f = 0; f < F; ++f)
		EXPECT_EQ(hm.heFaces[hm.FHalfedge(f)], f) << "f_halfedge not in face " << f;
}

// ---------------------------------------------------------------------------
// Brute-force adjacency built directly from the face list (ground truth).
// ---------------------------------------------------------------------------
struct BruteForceAdjacency
{
	std::vector<std::set<uint32_t>> vertVerts; // 1-ring neighbor vertices
	std::vector<std::set<uint32_t>> vertFaces; // incident faces
};

BruteForceAdjacency BruteForce(const Mesh& m)
{
	BruteForceAdjacency bf;
	bf.vertVerts.resize(m.vertices.size());
	bf.vertFaces.resize(m.vertices.size());
	for (size_t f = 0; f < m.faces.size(); ++f) {
		const Mesh::Face& face = m.faces[f];
		for (int i = 0; i < 3; ++i) {
			const uint32_t v = face[i];
			bf.vertVerts[v].insert(face[(i + 1) % 3]);
			bf.vertVerts[v].insert(face[(i + 2) % 3]);
			bf.vertFaces[v].insert(static_cast<uint32_t>(f));
		}
	}
	return bf;
}

template <typename Range>
std::set<uint32_t> ToSet(Range&& range)
{
	std::set<uint32_t> s;
	for (uint32_t x : range)
		s.insert(x);
	return s;
}

// ---------------------------------------------------------------------------
// Full topology check against a mesh with known topology.
// ---------------------------------------------------------------------------
void CheckAgainstKnown(const std::string& name, const Mesh& m, const corpus::KnownTopology& known)
{
	SCOPED_TRACE(name);
	const HalfMesh hm(m);

	CheckStructuralInvariants(hm);

	EXPECT_EQ(hm.VSize(), known.numVertices);
	EXPECT_EQ(hm.FSize(), known.numFaces);
	EXPECT_EQ(hm.ESize(), known.numEdges);
	// Euler characteristic V - E + F.
	EXPECT_EQ(static_cast<int32_t>(hm.VSize()) - static_cast<int32_t>(hm.ESize()) + static_cast<int32_t>(hm.FSize()),
	          known.euler);

	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	EXPECT_EQ(holes.size(), known.numBoundaryLoops) << "boundary-loop count";

	// Adjacency iterators must match the brute-force 1-ring / incidence.
	const BruteForceAdjacency bf = BruteForce(m);
	for (HalfMesh::VIndex v = 0; v < hm.VSize(); ++v) {
		EXPECT_EQ(ToSet(hm.VAdjacentVertices(v)), bf.vertVerts[v]) << "VAdjacentVertices mismatch at v " << v;
		EXPECT_EQ(ToSet(hm.VAdjacentFaces(v)), bf.vertFaces[v]) << "VAdjacentFaces mismatch at v " << v;
	}
	for (HalfMesh::FIndex f = 0; f < hm.FSize(); ++f) {
		std::set<uint32_t> expectVerts(m.faces[f].data(), m.faces[f].data() + 3);
		EXPECT_EQ(ToSet(hm.FAdjacentVertices(f)), expectVerts) << "FAdjacentVertices mismatch at f " << f;
	}

	// Round-trip: faces reconstructed from the half-edge structure are the same
	// triangulation as the input (up to relabeling / cyclic rotation).
	Mesh rebuilt;
	rebuilt.vertices = m.vertices;
	hm.FFaces(rebuilt.faces);
	EXPECT_TRUE(metrics::CanonicallyEqual(m, rebuilt)) << "FFaces round-trip not canonically equal";
}

struct CorpusCase
{
	std::string name;
	std::function<Mesh()> make;
	corpus::KnownTopology known;
};

std::vector<CorpusCase> Cases()
{
	return {
	    {"Triangle", [] { return corpus::Triangle(); }, corpus::Triangle_Known()},
	    {"Quad", [] { return corpus::Quad(); }, corpus::Quad_Known()},
	    {"Tetrahedron", [] { return corpus::TetrahedronMesh(); }, corpus::TetrahedronMesh_Known()},
	    {"Cube", [] { return corpus::CubeMesh(); }, corpus::CubeMesh_Known()},
	    {"Icosahedron", [] { return corpus::IcosahedronMesh(); }, corpus::IcosahedronMesh_Known()},
	    {"GridPlane4", [] { return corpus::GridPlane(4); }, corpus::GridPlane_Known(4)},
	    {"OpenCylinder", [] { return corpus::OpenCylinder(8, 4); }, corpus::OpenCylinder_Known(8, 4)},
	    {"Cone", [] { return corpus::Cone(8); }, corpus::Cone_Known(8)},
	    {"UVSphere", [] { return corpus::UVSphere(8, 12); }, corpus::UVSphere_Known(8, 12)},
	    {"Torus", [] { return corpus::TorusMesh(12, 8); }, corpus::TorusMesh_Known(12, 8)},
	};
}

bool HasOddRepresentative(const HalfMesh& hm)
{
	for (HalfMesh::HIndex iHe : hm.vHalfedges)
		if (iHe & 1u)
			return true;
	return false;
}

void CheckBoundaryRepresentatives(const HalfMesh& hm)
{
	std::vector<bool> boundary(hm.VSize(), false);
	for (HalfMesh::HIndex iHe = 0; iHe < hm.HeSize(); ++iHe) {
		if (hm.heFaces[iHe] != math::NO_ID)
			continue;
		boundary[hm.HeVertex(iHe)] = true;
		boundary[hm.HeVertex(hm.HeTwin(iHe))] = true;
	}
	for (HalfMesh::VIndex vertex = 0; vertex < hm.VSize(); ++vertex) {
		if (!boundary[vertex])
			continue;
		const HalfMesh::HIndex representative = hm.VHalfedge(vertex);
		EXPECT_EQ(representative & 1u, 0u) << "boundary representative parity at vertex " << vertex;
		EXPECT_NE(hm.heFaces[representative], math::NO_ID) << "boundary representative must carry a face";
		EXPECT_EQ(hm.heFaces[hm.HeTwin(representative)], math::NO_ID) << "boundary representative twin must be boundary";
	}
}

void CheckReaddedFace(Mesh& mesh)
{
	HalfMesh& hm = mesh.halfMesh;
	const HalfMesh::FIndex removedFace = hm.FSize() - 1;
	const HalfMesh::Face face = hm.F(removedFace);
	hm.FRemove(removedFace);
	ASSERT_TRUE(hm.ConnectBorders());
	CheckStructuralInvariants(hm);
	CheckBoundaryRepresentatives(hm);

	std::vector<std::vector<HalfMesh::VIndex>> holes;
	hm.EnumerateHoles(holes);
	ASSERT_EQ(holes.size(), 1u);
	EXPECT_EQ(holes.front().size(), 3u);

	EXPECT_NE(hm.FAdd(face), math::NO_ID);
	holes.clear();
	hm.EnumerateHoles(holes);
	EXPECT_TRUE(holes.empty());
	CheckStructuralInvariants(hm);
	CheckBoundaryRepresentatives(hm);

	mesh.InvalidateFaces();
	mesh.SyncFaces();
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	HalfMesh rebuilt;
	ASSERT_TRUE(rebuilt.Build(mesh));
	EXPECT_EQ(rebuilt.VSize(), hm.VSize());
	EXPECT_EQ(rebuilt.ESize(), hm.ESize());
	EXPECT_EQ(rebuilt.FSize(), hm.FSize());
	for (HalfMesh::VIndex vertex = 0; vertex < hm.VSize(); ++vertex) {
		EXPECT_EQ(ToSet(rebuilt.VAdjacentVertices(vertex)), ToSet(hm.VAdjacentVertices(vertex)));
		EXPECT_EQ(ToSet(rebuilt.VAdjacentFaces(vertex)), ToSet(hm.VAdjacentFaces(vertex)));
	}
}

void ApplyBulkRemoval(Mesh& mesh, std::vector<Mesh::FIndex> removes,
                      std::vector<Mesh::VIndex>* removedOut = nullptr,
                      std::vector<Mesh::VIndex>* splitOut = nullptr)
{
	mesh.ListHalfEdges();
	std::vector<Mesh::VIndex> removed;
	std::vector<Mesh::VIndex> split;
	mesh.halfMesh.FRemoveBulk(removes, removed, split);
	for (Mesh::VIndex source : split)
		mesh.vertices.emplace_back(mesh.vertices[source]);
	for (Mesh::VIndex vertex : removed) {
		mesh.vertices[vertex] = mesh.vertices.back();
		mesh.vertices.pop_back();
	}
	mesh.InvalidateFaces();
	mesh.SyncFaces();
	if (removedOut != nullptr)
		*removedOut = removed;
	if (splitOut != nullptr)
		*splitOut = split;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
TEST(HalfMeshInvariants, CorpusTopologyAndAdjacency)
{
	for (const CorpusCase& c : Cases())
		CheckAgainstKnown(c.name, c.make(), c.known);
}

// A vertex shared by two otherwise-disjoint triangle fans (a "bow-tie") is
// edge-manifold (every edge has <=2 faces, no repeated directed edge) but
// vertex-non-manifold. Build must REJECT it: the half-edge collapse/remove ops
// assume a single fan per vertex and will dereference a NO_ID link on a bow-tie
// (observed: SIGBUS in QEM decimation). Rejecting it makes ListHalfEdges fall
// back to the repairing ListHalfEdgesSafe (FixNonManifold). Regression test.
TEST(HalfMeshInvariants, BuildRejectsBowtieVertex)
{
	Mesh m;
	m.vertices.assign(7, Mesh::Vertex::Zero());
	m.vertices[0] = Mesh::Vertex(0, 0, 1); // apex A, shared by both cones
	m.vertices[1] = Mesh::Vertex(1, 0, 0);
	m.vertices[2] = Mesh::Vertex(0, 1, 0);
	m.vertices[3] = Mesh::Vertex(-1, 0, 0);
	m.vertices[4] = Mesh::Vertex(3, 0, 0);
	m.vertices[5] = Mesh::Vertex(3, 1, 0);
	m.vertices[6] = Mesh::Vertex(4, 0, 0);
	const auto add = [&](uint32_t a, uint32_t b, uint32_t c) {
		Mesh::Face f;
		f[0] = a;
		f[1] = b;
		f[2] = c;
		m.faces.push_back(f);
	};
	add(0, 1, 2);
	add(0, 2, 3);
	add(0, 3, 1); // cone 1 around apex A
	add(0, 4, 5);
	add(0, 5, 6);
	add(0, 6, 4); // cone 2 around apex A (disjoint ring)

	HalfMesh hm;
	EXPECT_FALSE(hm.Build(m)) << "Build must reject a bow-tie (vertex-non-manifold) mesh";

	// And the high-level path must auto-repair rather than crash: ListHalfEdges
	// sees Build fail and routes to ListHalfEdgesSafe (FixNonManifold).
	Mesh repaired = m;
	repaired.ListHalfEdges();
	EXPECT_EQ(repaired.halfMesh.vHalfedges.size(), repaired.vertices.size());
}

TEST(HalfMeshInvariants, MultipleComponents)
{
	// Two disconnected cubes: the half-edge structure must stay consistent and
	// report two components.
	const Mesh m = corpus::DirtyManyComponents(2);
	const HalfMesh hm(m);
	CheckStructuralInvariants(hm);
	std::vector<HalfMesh::FIndex> components;
	EXPECT_EQ(hm.ConnectedComponents(components), 2u);
}

// ---------------------------------------------------------------------------
// Edge collapse (ERemove) — the primitive QEM decimation is built on.
// ---------------------------------------------------------------------------
TEST(HalfMeshInvariants, EdgeCollapseInteriorIsConsistent)
{
	Mesh m = corpus::GridPlane(8);
	HalfMesh hm(m);

	// Find an interior edge whose collapse is topologically valid.
	HalfMesh::EIndex edge = math::NO_ID;
	for (HalfMesh::EIndex e = 0; e < hm.ESize(); ++e) {
		if (!hm.EIsBoundary(e) && hm.EIsCollapseValidTopologically(e)) {
			edge = e;
			break;
		}
	}
	ASSERT_NE(edge, math::NO_ID) << "no collapsible interior edge found on GridPlane(8)";

	const HalfMesh::VIndex v0 = hm.EFirstVertex(edge);
	const HalfMesh::VIndex v1 = hm.ESecondVertex(edge);
	const HalfMesh::Vertex midpoint = 0.5f * (m.vertices[v0] + m.vertices[v1]);
	EXPECT_TRUE(hm.EIsCollapseValidGeometrically(edge, midpoint, m.vertices))
	    << "midpoint collapse of a flat interior edge should be geometrically valid";

	const HalfMesh::VIndex vBefore = hm.VSize();
	const HalfMesh::EIndex eBefore = hm.ESize();
	const HalfMesh::FIndex fBefore = hm.FSize();

	HalfMesh::RemovedData removed;
	hm.ERemove(edge, removed);

	// An interior triangle-mesh edge collapse removes exactly 1 vertex, 2 faces,
	// and 3 edges (the collapsed edge plus the two pairs that merge).
	EXPECT_EQ(removed.numVerts, 1u);
	EXPECT_EQ(removed.numFaces, 2u);
	EXPECT_EQ(removed.numEdges, 3u);

	EXPECT_EQ(hm.VSize(), vBefore - 1u);
	EXPECT_EQ(hm.FSize(), fBefore - 2u);
	EXPECT_EQ(hm.ESize(), eBefore - 3u);

	// The structure must still be a valid half-edge mesh after the collapse.
	CheckStructuralInvariants(hm);
}

TEST(HalfMeshInvariants, CollapseGeometricallyRejectsDegenerate)
{
	// The geometric guard accepts the in-plane midpoint but rejects a collapse
	// point placed on top of an existing neighbor of v0: that produces a
	// zero-area triangle, so the area-ratio test fails (this is the guard that
	// keeps decimation from creating slivers / folding the surface).
	Mesh m = corpus::GridPlane(8);
	HalfMesh hm(m);

	HalfMesh::EIndex edge = math::NO_ID;
	for (HalfMesh::EIndex e = 0; e < hm.ESize(); ++e) {
		if (!hm.EIsBoundary(e) && hm.EIsCollapseValidTopologically(e)) {
			edge = e;
			break;
		}
	}
	ASSERT_NE(edge, math::NO_ID);

	const HalfMesh::VIndex v0 = hm.EFirstVertex(edge);
	const HalfMesh::VIndex v1 = hm.ESecondVertex(edge);
	const HalfMesh::Vertex midpoint = 0.5f * (m.vertices[v0] + m.vertices[v1]);

	// A link neighbor of v0 other than v1.
	HalfMesh::VIndex w = math::NO_ID;
	for (HalfMesh::VIndex u : hm.VAdjacentVertices(v0)) {
		if (u != v1) {
			w = u;
			break;
		}
	}
	ASSERT_NE(w, math::NO_ID);

	EXPECT_TRUE(hm.EIsCollapseValidGeometrically(edge, midpoint, m.vertices));
	EXPECT_FALSE(hm.EIsCollapseValidGeometrically(edge, m.vertices[w], m.vertices))
	    << "collapsing onto an existing neighbor (zero-area triangle) must be rejected";
}

// The collapse geometric guard must be SCALE-INVARIANT: a folded link
// configuration rejected at unit scale must stay rejected when all coordinates
// are uniformly shrunk (x1e-6) or grown (x1e5). The area-ratio and dihedral
// tests multiply edge-length^4..^8 products; in float these flush to 0 at small
// scale ("0 <= 0" falsely accepts a 180deg fold-over) and to inf at large scale
// ("inf <= inf" also falsely accepts). Evaluating the guard in double gives the
// dynamic range to keep the verdict stable. Regression test for that L^8
// underflow/overflow accept-of-fold-over.
TEST(HalfMeshInvariants, CollapseGeometricGuardIsScaleInvariant)
{
	const Mesh base = corpus::IcosahedronMesh();
	HalfMesh hm(base);
	// Interior edge with the collapse target placed well outside the unit
	// icosahedron, so the post-collapse link folds over -> rejected at unit scale.
	const HalfMesh::EIndex edge = 0;
	ASSERT_FALSE(hm.EIsBoundary(edge));
	const HalfMesh::Vertex p0(-2.f, -2.f, 0.25f);
	ASSERT_FALSE(hm.EIsCollapseValidGeometrically(edge, p0, base.vertices))
	    << "a folded collapse must be rejected at unit scale";

	auto scaled = [&](float s) {
		std::vector<HalfMesh::Vertex> v(base.vertices.size());
		for (size_t i = 0; i < v.size(); ++i)
			v[i] = base.vertices[i] * s;
		return v;
	};
	// Connectivity (hm) is scale-free; only the geometry passed in scales.
	for (const float s : {1e-6f, 1e5f}) {
		const std::vector<HalfMesh::Vertex> v = scaled(s);
		EXPECT_FALSE(hm.EIsCollapseValidGeometrically(edge, p0 * s, v))
		    << "the same fold-over must stay rejected at scale " << s;
	}
}

// ---------------------------------------------------------------------------
// Incremental edge split (ESplit) — splits in place without a full rebuild.
// Splits a deterministic subset of the original edges (interior AND boundary on
// the open meshes), then checks that the in-place structure is still consistent,
// the V/E/F counts match the Euler bookkeeping (interior +1/+3/+2, boundary
// +1/+2/+1), and the result rebuilds cleanly from its own face list — i.e. the
// incremental surgery is equivalent to building from scratch.
// ---------------------------------------------------------------------------
TEST(HalfMeshInvariants, EdgeSplitIsConsistent)
{
	auto allRepsEven = [](const HalfMesh& hm) {
		for (HalfMesh::VIndex v = 0; v < hm.VSize(); ++v)
			if (hm.VHalfedge(v) & 1u)
				return false;
		return true;
	};
	bool anyBecameDirty = false;
	for (const CorpusCase& c : Cases()) {
		SCOPED_TRACE(c.name);
		Mesh m = c.make();
		m.ListHalfEdges();
		HalfMesh& hm = m.halfMesh;

		// Fresh Build => canonical all-even form, and the flag agrees.
		EXPECT_TRUE(hm.alwaysEven) << "always_even must be true after Build";
		EXPECT_TRUE(allRepsEven(hm)) << "fresh Build must be all-even";

		const HalfMesh::VIndex v0 = hm.VSize();
		const HalfMesh::FIndex f0 = hm.FSize();
		const HalfMesh::EIndex e0 = hm.ESize();

		// Perturb vHalfedges away from the canonical even representatives: EFlip
		// can leave an ODD half-edge as a vertex's representative, and the split
		// must not assume even reps (a real failure had vHalfedges[b] pointing at
		// the very half-edge the split repoints from b to the midpoint). Flips
		// don't change V/E/F, so the counts below are unaffected.
		for (HalfMesh::EIndex e = 1; e < e0; e += 5) {
			if (hm.EIsFlipValid(e, m.vertices))
				hm.EFlip(e);
		}

		// Split EVERY original edge (so a vertex gets multiple incident edges split
		// in one pass — the case that exposed a broken orbit on real meshes). New
		// edges (index >= e0) are not revisited.
		unsigned nInt = 0, nBnd = 0;
		for (HalfMesh::EIndex e = 0; e < e0; e += 1) {
			const HalfMesh::HIndex he = hm.EHalfedge(e);
			const HalfMesh::VIndex a = hm.HeTailVertex(he);
			const HalfMesh::VIndex b = hm.HeHeadVertex(he);
			const bool boundary = hm.EIsBoundary(e);
			const HalfMesh::VIndex mv = hm.ESplit(e);
			ASSERT_EQ(mv, m.vertices.size()) << "new vertex index must be the appended slot";
			m.vertices.push_back(0.5f * (m.vertices[a] + m.vertices[b]));
			if (boundary)
				++nBnd;
			else
				++nInt;
		}
		ASSERT_GT(nInt + nBnd, 0u);

		// The flag is a sound over-approximation: alwaysEven==true must imply the
		// representatives are genuinely all-even (it may be conservatively false).
		EXPECT_TRUE(!hm.alwaysEven || allRepsEven(hm)) << "always_even=true but an odd rep exists";
		anyBecameDirty |= !hm.alwaysEven;

		// Structural invariants hold on the in-place (un-rebuilt) structure.
		CheckStructuralInvariants(hm);

		// Vertex orbits must close on the IN-PLACE structure (the remesher walks
		// these via HeNextOutgoingHalfedge; a broken umbrella that the face-orbit
		// checks miss would hang it).
		for (HalfMesh::VIndex v = 0; v < hm.VSize(); ++v) {
			const HalfMesh::HIndex start = hm.VHalfedge(v);
			HalfMesh::HIndex cur = start;
			int steps = 0;
			do {
				ASSERT_EQ(hm.HeTailVertex(cur), v) << "outgoing he not from v " << v;
				cur = hm.HeNextOutgoingHalfedge(cur);
				ASSERT_LE(++steps, static_cast<int>(hm.HeSize()))
				    << "vertex orbit did not close at v " << v;
			} while (cur != start);
		}

		// Euler bookkeeping.
		EXPECT_EQ(hm.VSize(), v0 + nInt + nBnd) << "vertex count";
		EXPECT_EQ(hm.FSize(), f0 + 2 * nInt + nBnd) << "face count";
		EXPECT_EQ(hm.ESize(), e0 + 3 * nInt + 2 * nBnd) << "edge count";

		// Harvest faces from the in-place structure and rebuild from scratch: the
		// incremental result must be a valid manifold with identical counts.
		// (FFaces appends, so clear the stale face list first.)
		m.faces.clear();
		hm.FFaces(m.faces);

		// VIsBoundary must stay correct on the IN-PLACE structure: boundary-vertex
		// representatives must remain canonical (the only consumers that care about
		// VHalfedge's exact value — VIsBoundary / HePrevBoundary / boundary-loop
		// iteration — depend on it). Compare against brute force from the faces.
		{
			std::map<std::pair<HalfMesh::VIndex, HalfMesh::VIndex>, int> edgeFaces;
			for (const auto& f : m.faces)
				for (int i = 0; i < 3; ++i) {
					HalfMesh::VIndex u = f[i], w = f[(i + 1) % 3];
					++edgeFaces[{std::min(u, w), std::max(u, w)}];
				}
			std::vector<bool> bruteBoundary(m.vertices.size(), false);
			for (const auto& kv : edgeFaces)
				if (kv.second == 1) {
					bruteBoundary[kv.first.first] = true;
					bruteBoundary[kv.first.second] = true;
				}
			for (HalfMesh::VIndex v = 0; v < hm.VSize(); ++v)
				EXPECT_EQ(hm.VIsBoundary(v), bruteBoundary[v]) << "VIsBoundary wrong at v " << v;
		}
		EXPECT_EQ(m.faces.size(), f0 + 2 * nInt + nBnd);
		HalfMesh rebuilt;
		ASSERT_TRUE(rebuilt.Build(m)) << "incremental split produced a non-manifold mesh";
		EXPECT_EQ(rebuilt.VSize(), hm.VSize());
		EXPECT_EQ(rebuilt.FSize(), hm.FSize());
		EXPECT_EQ(rebuilt.ESize(), hm.ESize());
		CheckStructuralInvariants(rebuilt);
		EXPECT_TRUE(rebuilt.alwaysEven) << "rebuilt mesh must be all-even";

		// GuaranteeAlwaysEven restores the canonical all-even form in place.
		hm.GuaranteeAlwaysEven();
		EXPECT_TRUE(hm.alwaysEven);
		EXPECT_TRUE(allRepsEven(hm)) << "GuaranteeAlwaysEven must restore all-even";
		CheckStructuralInvariants(hm);
	}
	EXPECT_TRUE(anyBecameDirty)
	    << "expected the flip/split perturbation to clear always_even on some mesh";
}

TEST(HalfMeshInvariants, FAddAfterSplitAndFlipOnDirtyHalfMesh)
{
	Mesh mesh = corpus::UVSphere(8, 12);
	mesh.ListHalfEdges();
	HalfMesh& hm = mesh.halfMesh;

	bool split = false;
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
		if (hm.EIsBoundary(edge))
			continue;
		const auto [a, b] = hm.EVertices(edge);
		const Mesh::Vertex midpoint = (mesh.vertices[a] + mesh.vertices[b]) * 0.5f;
		ASSERT_EQ(hm.ESplit(edge), mesh.vertices.size());
		mesh.vertices.emplace_back(midpoint);
		split = true;
		break;
	}
	ASSERT_TRUE(split);

	bool flipped = false;
	for (unsigned pass = 0; pass < 4 && !HasOddRepresentative(hm); ++pass) {
		for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
			if (!hm.EIsFlipValid(edge, mesh.vertices))
				continue;
			hm.EFlip(edge);
			flipped = true;
			if (HasOddRepresentative(hm))
				break;
		}
	}
	ASSERT_TRUE(flipped);
	ASSERT_FALSE(hm.alwaysEven);
	ASSERT_TRUE(HasOddRepresentative(hm));
	CheckReaddedFace(mesh);
}

TEST(HalfMeshInvariants, FAddAfterEdgeCollapse)
{
	Mesh mesh = corpus::UVSphere(8, 12);
	mesh.ListHalfEdges();
	HalfMesh& hm = mesh.halfMesh;

	bool collapsed = false;
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
		const auto [a, b] = hm.EVertices(edge);
		const Mesh::Vertex midpoint = (mesh.vertices[a] + mesh.vertices[b]) * 0.5f;
		if (!hm.EIsCollapseValidTopologically(edge) || !hm.EIsCollapseValidGeometrically(edge, midpoint, mesh.vertices))
			continue;
		HalfMesh::RemovedData removed;
		const HalfMesh::VIndex moved = hm.ERemove(edge, removed);
		ASSERT_EQ(removed.numVerts, 1u);
		mesh.vertices[removed.verts[0]] = mesh.vertices.back();
		mesh.vertices.pop_back();
		mesh.vertices[moved] = midpoint;
		collapsed = true;
		break;
	}
	ASSERT_TRUE(collapsed);
	CheckReaddedFace(mesh);
}

TEST(HalfMeshInvariants, RejectedFAddIsBitIdentical)
{
	Mesh mesh = corpus::Triangle();
	mesh.ListHalfEdges();
	HalfMesh& hm = mesh.halfMesh;
	const HalfMesh before = hm;

	EXPECT_EQ(hm.FAdd(mesh.faces.front()), math::NO_ID);
	EXPECT_EQ(hm.vHalfedges, before.vHalfedges);
	EXPECT_EQ(hm.fHalfedges, before.fHalfedges);
	EXPECT_EQ(hm.heNexts, before.heNexts);
	EXPECT_EQ(hm.heVertices, before.heVertices);
	EXPECT_EQ(hm.heFaces, before.heFaces);
	EXPECT_EQ(hm.alwaysEven, before.alwaysEven);
}

TEST(HalfMeshInvariants, BoundaryRepresentativesStayCanonicalNearMutations)
{
	Mesh mesh = corpus::GridPlane(6);
	mesh.ListHalfEdges();
	HalfMesh& hm = mesh.halfMesh;
	CheckBoundaryRepresentatives(hm);

	bool split = false;
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
		if (!hm.EIsBoundary(edge))
			continue;
		const auto [a, b] = hm.EVertices(edge);
		const Mesh::Vertex midpoint = (mesh.vertices[a] + mesh.vertices[b]) * 0.5f;
		ASSERT_EQ(hm.ESplit(edge), mesh.vertices.size());
		mesh.vertices.emplace_back(midpoint);
		split = true;
		CheckBoundaryRepresentatives(hm);
		break;
	}
	ASSERT_TRUE(split);

	bool flipped = false;
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
		const auto [a, b] = hm.EVertices(edge);
		if ((!hm.VIsBoundary(a) && !hm.VIsBoundary(b)) || !hm.EIsFlipValid(edge, mesh.vertices))
			continue;
		hm.EFlip(edge);
		flipped = true;
		CheckBoundaryRepresentatives(hm);
		break;
	}
	ASSERT_TRUE(flipped);

	bool collapsed = false;
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
		const auto [a, b] = hm.EVertices(edge);
		if (!hm.VIsBoundary(a) && !hm.VIsBoundary(b))
			continue;
		const Mesh::Vertex midpoint = (mesh.vertices[a] + mesh.vertices[b]) * 0.5f;
		if (!hm.EIsCollapseValidTopologically(edge) || !hm.EIsCollapseValidGeometrically(edge, midpoint, mesh.vertices))
			continue;
		HalfMesh::RemovedData removed;
		const HalfMesh::VIndex moved = hm.ERemove(edge, removed);
		ASSERT_EQ(removed.numVerts, 1u);
		mesh.vertices[removed.verts[0]] = mesh.vertices.back();
		mesh.vertices.pop_back();
		mesh.vertices[moved] = midpoint;
		collapsed = true;
		CheckBoundaryRepresentatives(hm);
		break;
	}
	ASSERT_TRUE(collapsed);

	mesh.InvalidateFaces();
	mesh.SyncFaces();
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(HalfMeshInvariants, BulkRemoveHandlesTwoAndThreeBoundaryEdges)
{
	Mesh quad = corpus::Quad();
	std::vector<Mesh::FIndex> removes{0};
	quad.ListHalfEdges();
	quad.RemoveFacesHalfEdge(removes);
	EXPECT_EQ(quad.faces.size(), 1u);
	EXPECT_EQ(quad.vertices.size(), 3u);
	EXPECT_TRUE(quad.ValidateHalfMesh());
	CheckBoundaryRepresentatives(quad.halfMesh);

	Mesh triangle = corpus::Triangle();
	removes = {0};
	triangle.ListHalfEdges();
	triangle.RemoveFacesHalfEdge(removes);
	EXPECT_TRUE(triangle.vertices.empty());
	EXPECT_TRUE(triangle.faces.empty());
	EXPECT_TRUE(triangle.halfMesh.Empty());
	EXPECT_TRUE(triangle.ValidateHalfMesh());
}

TEST(HalfMeshInvariants, BulkRemoveSplitsInteriorPinch)
{
	Mesh mesh;
	mesh.vertices.emplace_back(0.f, 0.f, 0.f);
	for (int i = 0; i < 6; ++i) {
		const float angle = static_cast<float>(i) * static_cast<float>(2.0 * std::numbers::pi / 6.0);
		mesh.vertices.emplace_back(std::cos(angle), std::sin(angle), 0.f);
	}
	for (Mesh::VIndex i = 1; i <= 6; ++i)
		mesh.faces.emplace_back(0, i, i == 6 ? 1 : i + 1);

	std::vector<Mesh::VIndex> removed;
	std::vector<Mesh::VIndex> split;
	ApplyBulkRemoval(mesh, {0, 3}, &removed, &split);
	EXPECT_TRUE(removed.empty());
	ASSERT_EQ(split.size(), 1u);
	EXPECT_EQ(split.front(), 0u);
	EXPECT_EQ(mesh.vertices.size(), 8u);
	EXPECT_EQ(mesh.faces.size(), 4u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	HalfMesh rebuilt;
	EXPECT_TRUE(rebuilt.Build(mesh)) << "pinch split must leave rebuildable topology";
}

TEST(HalfMeshInvariants, BulkRemoveSplitsBoundaryPinch)
{
	Mesh mesh;
	mesh.vertices = {
	    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}};
	mesh.faces = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}};

	std::vector<Mesh::VIndex> removed;
	std::vector<Mesh::VIndex> split;
	ApplyBulkRemoval(mesh, {1}, &removed, &split);
	EXPECT_TRUE(removed.empty());
	ASSERT_EQ(split.size(), 1u);
	EXPECT_EQ(split.front(), 0u);
	EXPECT_EQ(mesh.vertices.size(), 6u);
	EXPECT_EQ(mesh.faces.size(), 2u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	CheckBoundaryRepresentatives(mesh.halfMesh);
}

TEST(HalfMeshInvariants, BulkRemoveScatteredFacesAndWholeComponent)
{
	Mesh grid = corpus::GridPlane(8);
	ApplyBulkRemoval(grid, {3, 19, 47, 82});
	EXPECT_EQ(grid.faces.size(), 124u);
	EXPECT_TRUE(grid.ValidateHalfMesh());
	HalfMesh rebuiltGrid;
	EXPECT_TRUE(rebuiltGrid.Build(grid));

	Mesh components = corpus::TetrahedronMesh();
	const Mesh second = corpus::TetrahedronMesh();
	const Mesh::VIndex offset = static_cast<Mesh::VIndex>(components.vertices.size());
	for (const Mesh::Vertex& vertex : second.vertices)
		components.vertices.emplace_back(vertex + Mesh::Vertex(3.f, 0.f, 0.f));
	for (const Mesh::Face& face : second.faces)
		components.faces.emplace_back(face[0] + offset, face[1] + offset, face[2] + offset);
	ApplyBulkRemoval(components, {4, 5, 6, 7});
	EXPECT_EQ(components.vertices.size(), 4u);
	EXPECT_EQ(components.faces.size(), 4u);
	EXPECT_TRUE(components.ValidateHalfMesh());
}

TEST(HalfMeshInvariants, BulkRemoveReportsIsolatedVerticesDescending)
{
	Mesh mesh;
	for (int component = 0; component < 3; ++component) {
		const float x = static_cast<float>(component * 3);
		const Mesh::VIndex base = static_cast<Mesh::VIndex>(mesh.vertices.size());
		mesh.vertices.emplace_back(x, 0.f, 0.f);
		mesh.vertices.emplace_back(x + 1.f, 0.f, 0.f);
		mesh.vertices.emplace_back(x, 1.f, 0.f);
		mesh.faces.emplace_back(base, base + 1, base + 2);
	}
	std::vector<Mesh::VIndex> removed;
	ApplyBulkRemoval(mesh, {0, 2}, &removed);
	EXPECT_EQ(removed, (std::vector<Mesh::VIndex>{8, 7, 6, 2, 1, 0}));
	EXPECT_EQ(mesh.vertices.size(), 3u);
	EXPECT_EQ(mesh.faces.size(), 1u);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(HalfMeshInvariants, BulkRemoveWorksOnDirtyHalfMesh)
{
	Mesh mesh = corpus::UVSphere(8, 12);
	mesh.ListHalfEdges();
	HalfMesh& hm = mesh.halfMesh;
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize(); ++edge) {
		if (hm.EIsBoundary(edge))
			continue;
		const auto [a, b] = hm.EVertices(edge);
		const Mesh::Vertex midpoint = (mesh.vertices[a] + mesh.vertices[b]) * 0.5f;
		ASSERT_EQ(hm.ESplit(edge), mesh.vertices.size());
		mesh.vertices.emplace_back(midpoint);
		break;
	}
	for (HalfMesh::EIndex edge = 0; edge < hm.ESize() && hm.alwaysEven; ++edge)
		if (hm.EIsFlipValid(edge, mesh.vertices))
			hm.EFlip(edge);
	ASSERT_FALSE(hm.alwaysEven);
	mesh.InvalidateFaces();

	std::vector<Mesh::FIndex> removes{2, 17, 43};
	mesh.RemoveFacesHalfEdge(removes);
	EXPECT_TRUE(mesh.ValidateHalfMesh());
	HalfMesh rebuilt;
	EXPECT_TRUE(rebuilt.Build(mesh));
}

TEST(HalfMeshInvariants, VRemoveUnreferencedReportsDescending)
{
	Mesh mesh = corpus::Triangle();
	mesh.ListHalfEdges();
	mesh.vertices.resize(6, Mesh::Vertex::Zero());
	mesh.halfMesh.vHalfedges.resize(6, math::NO_ID);
	std::vector<Mesh::VIndex> removed{42};
	mesh.halfMesh.VRemoveUnreferenced(removed);
	EXPECT_EQ(removed, (std::vector<Mesh::VIndex>{42, 5, 4, 3}));
	for (auto it = removed.begin() + 1; it != removed.end(); ++it) {
		const Mesh::VIndex vertex = *it;
		mesh.vertices[vertex] = mesh.vertices.back();
		mesh.vertices.pop_back();
	}
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(HalfMeshInvariants, BulkRemoveAppendsReportsAndEmptyInputIsNoOp)
{
	Mesh mesh = corpus::Triangle();
	mesh.faceTexcoords.resize(3, Mesh::TexCoord::Zero());
	mesh.texturesDiffuse.emplace_back(1, 1);
	mesh.ListHalfEdges();
	const HalfMesh before = mesh.halfMesh;
	std::vector<Mesh::FIndex> noRemoves;
	mesh.RemoveFacesHalfEdge(noRemoves);
	EXPECT_EQ(mesh.halfMesh.vHalfedges, before.vHalfedges);
	EXPECT_EQ(mesh.halfMesh.fHalfedges, before.fHalfedges);
	EXPECT_EQ(mesh.halfMesh.heNexts, before.heNexts);
	EXPECT_EQ(mesh.halfMesh.heVertices, before.heVertices);
	EXPECT_EQ(mesh.halfMesh.heFaces, before.heFaces);
	EXPECT_EQ(mesh.faceTexcoords.size(), 3u);
	EXPECT_EQ(mesh.texturesDiffuse.size(), 1u);

	std::vector<Mesh::VIndex> removed{99};
	std::vector<Mesh::VIndex> split{98};
	std::vector<Mesh::FIndex> removes{0};
	mesh.halfMesh.FRemoveBulk(removes, removed, split);
	EXPECT_EQ(removed, (std::vector<Mesh::VIndex>{99, 2, 1, 0}));
	EXPECT_EQ(split, (std::vector<Mesh::VIndex>{98}));
}

TEST(HalfMeshInvariants, BulkRemoveWrapperKeepsColorsInLockstep)
{
	Mesh mesh;
	mesh.vertices = {
	    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}};
	mesh.faces = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}};
	for (uint8_t i = 0; i < mesh.vertices.size(); ++i)
		mesh.vertexColors.emplace_back(i, i, i);
	mesh.ListHalfEdges();
	std::vector<Mesh::FIndex> removes{1};
	mesh.RemoveFacesHalfEdge(removes);
	ASSERT_EQ(mesh.vertexColors.size(), mesh.vertices.size());
	ASSERT_EQ(mesh.vertices.size(), 6u);
	EXPECT_EQ(mesh.vertexColors.back(), mesh.vertexColors.front());
	EXPECT_TRUE(mesh.ValidateHalfMesh());
}

TEST(HalfMeshInvariants, BulkRemoveEveryCubeFaceSubsetStaysValid)
{
	const Mesh source = corpus::CubeMesh();
	ASSERT_EQ(source.faces.size(), 12u);
	for (uint32_t mask = 1; mask < (1u << source.faces.size()); ++mask) {
		SCOPED_TRACE(mask);
		Mesh mesh = source;
		mesh.ListHalfEdges();
		std::vector<Mesh::FIndex> removes;
		for (Mesh::FIndex face = 0; face < source.faces.size(); ++face)
			if (mask & (1u << face))
				removes.emplace_back(face);
		mesh.RemoveFacesHalfEdge(removes);
		EXPECT_TRUE(mesh.ValidateHalfMesh());
		if (!mesh.halfMesh.Empty()) {
			HalfMesh rebuilt;
			EXPECT_TRUE(rebuilt.Build(mesh));
		}
	}
}

} // namespace
} // namespace halfmesh
