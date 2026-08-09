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
#include <functional>
#include <map>
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

} // namespace
} // namespace halfmesh
