/*
* MeshHoles.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Hole filling, following the algorithm and structure of the Polygon Mesh
// Processing Library (pmp-library), file
//   src/pmp/algorithms/hole_filling.cpp  (class HoleFilling)
// which in turn implements:
//   P. Liepa, "Filling Holes in Meshes", Eurographics Symposium on Geometry
//   Processing, 2003.
//
// The pipeline per hole is:
//   1. triangulate_hole  — minimum-weight triangulation of the boundary loop by
//                          dynamic programming, minimising a (max-dihedral-angle,
//                          area) Weight lexicographically (Liepa / PMP).
//   2. refine            — isotropic remeshing of the patch (split long edges /
//                          collapse short edges / flip for valence / Laplacian
//                          relaxation / remove caps) to match the surrounding
//                          edge-length density. May add interior vertices.
//   3. fairing           — curvature-/area-minimising Laplacian smoothing of the
//                          patch-interior vertices via an Eigen SPD solve
//                          (SimplicialLDLT), following PMP's relaxation system.
//
// This is a native adaptation, not a transcription: PMP operates on an
// incremental pmp::SurfaceMesh, whereas halfmesh stores faces/vertices in flat
// arrays with a HalfMesh half-edge view rebuilt on demand. We therefore carry
// out the per-hole patch construction and refinement on a small, self-contained
// "patch" sub-mesh (positions + triangles + boundary/interior tags), then append
// the new faces (and any interior vertices) to the parent Mesh. The Weight
// struct, its lexicographic ordering, the DP, and the refine thresholds are kept
// faithful to PMP; the data-structure plumbing is halfmesh-native.

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h>
#include <halfmesh/Types.h>

#include "MeshRemeshShared.h"

#include <BS_thread_pool.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace halfmesh {
namespace {

// ---------------------------------------------------------------------------
// HoleFilling
//
// Self-contained patch-based hole filler. Constructed from the parent Mesh and
// a boundary loop (ordered list of parent vertex indices). It builds the
// minimum-weight triangulation, optionally refines and fairs the patch, then
// the caller harvests the resulting triangles / interior vertices.
//
// Patch representation (local indices 0..P-1):
//   - points:   local vertex positions (double precision for stability)
//   - boundary_: the first B entries (0..B-1) are the hole-boundary vertices, in
//                loop order; they map back to parent vertices via parentVidx
//                and are "locked" (never moved, never collapsed away).
//   - vlocked:  true for the original boundary vertices (locked), false for any
//                interior vertex added during refine.
//   - tris:     patch triangles as triples of local indices.
//   - parentVidx: parent Mesh vertex index for boundary vertices, NO_ID for
//                interior vertices created during refine.
// ---------------------------------------------------------------------------
class HoleFilling
{
	public:
	using Type = Mesh::Type;
	using VIndex = Mesh::VIndex;
	using FIndex = Mesh::FIndex;
	using Point = Eigen::Matrix<double, 3, 1>;

	// mesh:      parent mesh (read for boundary positions / opposite normals)
	// loop:      ordered parent vertex indices forming the boundary loop
	// oppNorms: outward normal of the existing face opposite boundary edge
	//            (loop[i] -> loop[i+1]); used as the dihedral neighbour for the
	//            corresponding boundary edge (PMP opposite_normal()).
	HoleFilling(const Mesh& mesh,
	            const std::vector<VIndex>& loop,
	            const std::vector<Point>& oppNorms) :
	    nb(static_cast<int>(loop.size())), hm(&mesh.halfMesh), pverts(&mesh.vertices),
	    oppNorms(oppNorms)
	{
		points.reserve(loop.size());
		parentVidx.reserve(loop.size());
		vlocked.reserve(loop.size());
		for (VIndex v : loop) {
			const Mesh::Vertex& p = mesh.vertices[v];
			points.emplace_back(p.x(), p.y(), p.z());
			parentVidx.emplace_back(v);
			vlocked.emplace_back(true);
		}
		BuildInteriorEdgeTable();
	}

	// Run stage 1 (always), plus refine/fair when requested.
	// Returns false only if the minimum-weight triangulation failed to produce
	// any triangle (degenerate / non-simple loop).
	bool Fill(bool doRefine, bool doFair)
	{
		if (!TriangulateHole())
			return false;
		if (doRefine)
			Refine(doFair);
		return true;
	}

	// Number of boundary vertices (locked, already in the parent mesh).
	int NumBoundary() const { return nb; }
	// Parent vertex index for local boundary vertex i (0..nb-1).
	VIndex ParentVertex(int i) const { return parentVidx[i]; }
	// Interior vertices added during refine (local indices nb..P-1).
	int NumInterior() const { return static_cast<int>(points.size()) - nb; }
	Mesh::Vertex InteriorPoint(int k) const
	{
		const Point& p = points[nb + k];
		return Mesh::Vertex(static_cast<Type>(p.x()),
		                    static_cast<Type>(p.y()),
		                    static_cast<Type>(p.z()));
	}
	const std::vector<Eigen::Matrix<int, 3, 1>>& Triangles() const { return tris; }

	private:
	// ---- Weight (faithful to PMP HoleFilling::Weight) ----
	struct Weight
	{
		Weight(double angle = std::numeric_limits<double>::max(),
		       double area = std::numeric_limits<double>::max()) :
		    angle(angle), area(area) {}

		Weight operator+(const Weight& rhs) const
		{
			return Weight(std::max(angle, rhs.angle), area + rhs.area);
		}
		bool operator<(const Weight& rhs) const
		{
			return (angle < rhs.angle || (angle == rhs.angle && area < rhs.area));
		}
		double angle;
		double area;
	};

	// ---- geometry helpers (faithful to PMP) ----
	static double ComputeArea(const Point& a, const Point& b, const Point& c)
	{
		// PMP uses squared norm of the cross product as the area term.
		return (b - a).cross(c - a).squaredNorm();
	}
	static Point ComputeNormal(const Point& a, const Point& b, const Point& c)
	{
		return (b - a).cross(c - a).normalized();
	}
	static double ComputeAngle(const Point& n1, const Point& n2)
	{
		return 1.0 - n1.dot(n2);
	}

	// Does an *interior* (non-boundary) edge already exist between the parent
	// vertices of boundary indices i,j?  Faithful to PMP HoleFilling
	// is_interior_edge() and to halfmesh's HalfMesh::TriangulateHole: an existing
	// edge that is itself a mesh boundary edge (the hole's own boundary edges) is
	// *not* an interior edge and so does not forbid the diagonal; an existing
	// interior edge would make the diagonal a duplicate -> forbidden (infinite
	// weight). This correctly handles pinched / adjacent holes where two
	// boundary vertices are already joined through the surrounding mesh.
	//
	// Precomputed once (BuildInteriorEdgeTable) into an nb x nb bitset so the
	// O(n^3) DP does three O(1) table lookups per candidate instead of three
	// parent-mesh half-edge fan walks (hm->EEdge). The predicate is identical, so
	// the triangulation is bit-for-bit unchanged.
	bool IsInteriorEdge(int i, int j) const
	{
		return interiorEdge[i][j] != 0;
	}

	// Fill interiorEdge[i][j] = 1 iff an existing *interior* (non-boundary)
	// parent edge joins the parent vertices of loop indices i and j. Loops that
	// reach HoleFilling are pre-validated simple, so a single-valued parent-vidx
	// hash suffices. Each boundary vertex's outgoing half-edges are enumerated
	// once: O(sum deg) total, replacing O(n^3) EEdge walks.
	void BuildInteriorEdgeTable()
	{
		const int n = nb;
		interiorEdge.assign(n, std::vector<char>(n, 0));
		std::unordered_map<VIndex, int> local;
		local.reserve(static_cast<std::size_t>(n) * 2);
		for (int i = 0; i < n; ++i)
			if (parentVidx[i] != math::NO_ID)
				local.emplace(parentVidx[i], i);
		for (int i = 0; i < n; ++i) {
			const VIndex a = parentVidx[i];
			if (a == math::NO_ID)
				continue;
			for (const HalfMesh::HIndex he : hm->VOutgoingHalfedges(a)) {
				const auto it = local.find(hm->HeHeadVertex(he));
				if (it == local.end() || it->second == i)
					continue;
				if (!hm->EIsBoundary(hm->HeEdge(he)))
					interiorEdge[i][it->second] = 1; // interior edge -> forbidden
			}
		}
		// A loop vertex becomes a full interior vertex once THIS hole is filled iff
		// its only boundary wedge is this hole (exactly one outgoing boundary
		// half-edge). Such vertices are scored against interior target valence 6;
		// vertices that stay on another boundary keep target 4 (PMP full-mesh rule).
		becomesInterior.assign(n, false);
		for (int i = 0; i < n; ++i) {
			if (parentVidx[i] == math::NO_ID)
				continue;
			int nbHe = 0;
			for (const HalfMesh::HIndex he : hm->VOutgoingHalfedges(parentVidx[i]))
				if (hm->HeIsBoundary(he))
					++nbHe;
			becomesInterior[i] = (nbHe == 1);
		}
	}

	// opposite normal for boundary edge (i-1 -> i): the existing face across it.
	// PMP indexes opposite_normal(i) = normal of face across halfedge hole_[i],
	// where hole_[i] is the boundary halfedge whose head is vertex i. Our
	// oppNorms[e] is the normal across boundary edge loop[e]->loop[e+1].
	// The halfedge with head i is edge (i-1 -> i) == oppNorms[(i-1+n)%n].
	Point OppositeNormal(int i) const
	{
		const int n = nb;
		return oppNorms[(i - 1 + n) % n];
	}

	Weight ComputeWeight(int i, int j, int k) const
	{
		// forbid diagonals that coincide with an existing edge
		if (IsInteriorEdge(i, j) || IsInteriorEdge(j, k) || IsInteriorEdge(k, i))
			return {};

		const Point& a = points[i];
		const Point& b = points[j];
		const Point& c = points[k];

		const double area = ComputeArea(a, b, c);

		// A collinear triple has an exactly-zero cross product: ComputeNormal's
		// normalized() then yields NaN, ComputeAngle propagates it, and
		// std::max(0.0, NaN) silently DISCARDS it -- the degenerate triangle
		// would score as perfect (angle=0, area=0) and win the DP on both keys.
		// Forbid it outright (same sentinel as the interior-edge case above).
		// `area` is the squared cross norm (length^4 units); dividing by the
		// squared max squared-edge-length makes the guard dimensionless --
		// area / maxEdgeSq^2 <= sin^2(corner angle) -- so it rejects by SHAPE
		// (near-collinearity, angle <~ 1e-6 rad) at any mesh scale, where an
		// absolute threshold wrongly refused every triangle on meshes smaller
		// than ~1e-3 units. Coincident points (maxEdgeSq == 0) and NaN inputs
		// still fail the comparison and are refused.
		const double maxEdgeSq = std::max({(b - a).squaredNorm(),
		                                   (c - b).squaredNorm(),
		                                   (a - c).squaredNorm()});
		if (!(area > 1e-12 * maxEdgeSq * maxEdgeSq))
			return {};

		double angle = 0.0;
		const Point n = ComputeNormal(a, b, c);
		Point n2;

		// ...neighbour to (i,j): the (i,j) sub-polygon's chosen triangle normal,
		// cached when weight[i][j] was finalized (sentinel for an all-forbidden
		// span, which lives only inside an already-infinite branch, so the selected
		// weights are unchanged and never read points[-1]).
		n2 = (i + 1 == j) ? OppositeNormal(j) : normal[i][j];
		angle = std::max(angle, ComputeAngle(n, n2));

		// ...neighbour to (j,k)
		n2 = (j + 1 == k) ? OppositeNormal(k) : normal[j][k];
		angle = std::max(angle, ComputeAngle(n, n2));

		// ...neighbour to (k,i) only for the closing edge (0, n-1)
		if (i == 0 && k + 1 == nb) {
			n2 = OppositeNormal(0);
			angle = std::max(angle, ComputeAngle(n, n2));
		}

		return {angle, area};
	}

	// Liepa / PMP minimum-weight triangulation by dynamic programming.
	bool TriangulateHole()
	{
		const int n = nb;
		if (n < 3)
			return false;

		weight.assign(n, std::vector<Weight>(n, Weight()));
		index.assign(n, std::vector<int>(n, 0));
		// Cache the chosen sub-triangle normal per interval (i,j), filled as
		// weight[i][j] is finalized so ComputeWeight reads it instead of
		// re-normalizing a cross product it already computed. Sentinel (zero) for
		// spans with no valid split (index==-1): those only feed infinite-weight
		// branches, so the selected triangulation is unchanged.
		normal.assign(n, std::vector<Point>(n, Point::Zero()));

		// initialize 2-gons
		for (int i = 0; i < n - 1; ++i) {
			weight[i][i + 1] = Weight(0, 0);
			index[i][i + 1] = -1;
		}

		// n-gons with n>2
		for (int j = 2; j < n; ++j) {
			for (int i = 0; i < n - j; ++i) {
				const int k = i + j;
				Weight wmin = Weight();
				int imin = -1;
				for (int m = i + 1; m < k; ++m) {
					const Weight w =
					    weight[i][m] + ComputeWeight(i, m, k) + weight[m][k];
					if (w < wmin) {
						wmin = w;
						imin = m;
					}
				}
				weight[i][k] = wmin;
				index[i][k] = imin;
				if (imin >= 0)
					normal[i][k] = ComputeNormal(points[i], points[imin], points[k]);
			}
		}

		// emit triangles (iterative split, as PMP)
		std::vector<std::pair<int, int>> todo;
		todo.reserve(n);
		todo.emplace_back(0, n - 1);
		while (!todo.empty()) {
			const std::pair<int, int> tri = todo.back();
			todo.pop_back();
			const int start = tri.first;
			const int end = tri.second;
			if (end - start < 2)
				continue;
			const int split = index[start][end];
			if (split < 0)
				return false; // no valid triangulation
			tris.emplace_back(start, split, end);
			todo.emplace_back(start, split);
			todo.emplace_back(split, end);
		}

		weight.clear();
		index.clear();
		normal.clear();
		return !tris.empty();
	}

	// -----------------------------------------------------------------------
	// Patch adjacency (rebuilt from tris on demand). Edges are undirected pairs
	// (min,max) of local vertex indices. We track, per directed corner, enough
	// to perform flips and to identify interior vs boundary edges.
	// -----------------------------------------------------------------------
	struct EdgeKey
	{
		int a, b;
		EdgeKey(int x, int y) :
		    a(std::min(x, y)), b(std::max(x, y)) {}
		bool operator<(const EdgeKey& o) const
		{
			return a < o.a || (a == o.a && b < o.b);
		}
	};

	// One entry per undirected patch edge: the first two incident triangles (in
	// triangle-scan order) plus the incidence count. count>2 (non-manifold) edges
	// are kept only as a count so the size()!=2 guards still fire; patch edges have
	// at most two incident triangles in practice.
	struct EdgeEntry
	{
		EdgeKey key;
		int t0, t1;
		int count;
	};

	// Build the undirected-edge table as a flat vector sorted by EdgeKey — same
	// iteration order as the old std::map (so refine output is byte-identical) but
	// without per-edge red-black-tree node / std::vector<int> heap allocations. A
	// stable sort preserves the triangle-scan order of the two incident triangles,
	// which the flip/cap winding (t0 then t1) depends on.
	void BuildEdgeTable(std::vector<EdgeEntry>& tbl) const
	{
		std::vector<std::pair<EdgeKey, int>> flat;
		flat.reserve(tris.size() * 3);
		for (int t = 0; t < static_cast<int>(tris.size()); ++t) {
			const auto& f = tris[t];
			for (int e = 0; e < 3; ++e)
				flat.emplace_back(EdgeKey(f[e], f[(e + 1) % 3]), t);
		}
		std::stable_sort(flat.begin(), flat.end(),
		                 [](const std::pair<EdgeKey, int>& x, const std::pair<EdgeKey, int>& y) {
			                 return x.first < y.first;
		                 });
		tbl.clear();
		tbl.reserve(flat.size());
		for (std::size_t i = 0; i < flat.size();) {
			std::size_t j = i;
			EdgeEntry e{flat[i].first, flat[i].second, -1, 0};
			while (j < flat.size() && !(flat[i].first < flat[j].first) && !(flat[j].first < flat[i].first)) {
				if (e.count == 0)
					e.t0 = flat[j].second;
				else if (e.count == 1)
					e.t1 = flat[j].second;
				++e.count;
				++j;
			}
			tbl.push_back(e);
			i = j;
		}
	}

	// Does the (sorted) edge table contain edge (x,y)?  Binary search by EdgeKey.
	bool EdgeExists(const std::vector<EdgeEntry>& tbl, int x, int y) const
	{
		const EdgeKey k(x, y);
		int lo = 0, hi = static_cast<int>(tbl.size());
		while (lo < hi) {
			const int mid = (lo + hi) / 2;
			if (tbl[mid].key < k)
				lo = mid + 1;
			else
				hi = mid;
		}
		return lo < static_cast<int>(tbl.size()) && !(k < tbl[lo].key) && !(tbl[lo].key < k);
	}

	double EdgeLen(int a, int b) const { return (points[a] - points[b]).norm(); }

	// -----------------------------------------------------------------------
	// Refine — isotropic remeshing of the patch (PMP refine()).
	// -----------------------------------------------------------------------
	void Refine(bool doFair)
	{
		const int n = nb;
		double meanLength = 0.0;
		for (int i = 0; i < n; ++i)
			meanLength += EdgeLen(i, (i + 1) % n);
		meanLength /= static_cast<double>(n);
		if (meanLength <= 0.0)
			return;

		// Bound the refined patch's triangle count: the density target above
		// comes from the hole BOUNDARY's mean edge length, which says nothing
		// about the patch's area, so a large-area hole rimmed by fine edges
		// requests an unbounded refinement (measured on the challenge fixture's
		// 2,739-edge outer scan boundary: 2,737 Liepa triangles split past
		// 701,000 and every subsequent O(T log T) pass ground for >15 minutes).
		// Floor the target edge length so ~budget triangles cover the patch
		// area. Refined edges settle near ~1.1x the target (between the 0.7x
		// collapse and 1.5x split thresholds), giving a mean triangle area of
		// ~0.54 * target^2 (measured on the hemisphere regression test).
		// Disk-like holes density-match to ~nb^2/5 triangles and are
		// unaffected up to nb ~ 280.
		{
			double area = 0.0;
			for (const auto& t : tris)
				area += 0.5 * (points[t[1]] - points[t[0]]).cross(points[t[2]] - points[t[0]]).norm();
			const double budget = std::max(16384.0, 8.0 * n);
			const double lBudget = std::sqrt(area / (0.54 * budget));
			if (lBudget > meanLength) {
				REPORT_STATUS("CloseHoles: coarsening the patch of a {}-edge hole "
				              "(area {:g}) to ~{} triangles",
				              n, area, static_cast<unsigned>(budget));
				meanLength = lBudget;
			}
		}

		const double minLength = 0.7 * meanLength;
		const double maxLength = 1.5 * meanLength;

		for (int iter = 0; iter < 10; ++iter) {
			SplitLongEdges(maxLength);
			CollapseShortEdges(minLength);
			FlipEdges();
			Relaxation();
		}
		RemoveCaps();
		if (doFair)
			Fairing();
	}

	// Split patch edges longer than lmax by inserting their midpoint (PMP
	// split_long_edges). Boundary edges of the hole are left intact (PMP elocked_
	// guard). Each pass builds the edge map once and processes every currently
	// long edge whose triangles have not yet been touched this pass, deferring
	// newly-created edges to the next pass (mirrors PMP's sweep-and-repeat).
	void SplitLongEdges(double lmax)
	{
		const double lmaxSq = lmax * lmax;
		std::vector<EdgeEntry> tbl;
		for (int pass = 0; pass < 10; ++pass) {
			BuildEdgeTable(tbl);
			std::vector<char> triDirty(tris.size(), 0);
			bool ok = true;

			for (const auto& e : tbl) {
				const int a = e.key.a;
				const int b = e.key.b;
				if (IsOriginalBoundaryEdge(a, b))
					continue;
				if ((points[a] - points[b]).squaredNorm() <= lmaxSq)
					continue;
				if (e.count > 2)
					continue; // non-manifold patch edge
				const int inc[2] = {e.t0, e.t1};
				bool touched = false;
				for (int i = 0; i < e.count; ++i)
					if (inc[i] >= static_cast<int>(triDirty.size()) || triDirty[inc[i]]) {
						touched = true;
						break;
					}
				if (touched)
					continue;

				const int mid = static_cast<int>(points.size());
				points.emplace_back(0.5 * (points[a] + points[b]));
				vlocked.emplace_back(false);
				parentVidx.emplace_back(math::NO_ID);
				for (int i = 0; i < e.count; ++i) {
					triDirty[inc[i]] = 1;
					SplitTriangleAtEdge(inc[i], a, b, mid);
				}
				ok = false;
			}
			if (ok)
				break;
		}
	}

	// Replace triangle t (which contains edge a-b) by two triangles using mid.
	void SplitTriangleAtEdge(int t, int a, int b, int mid)
	{
		auto& f = tris[t];
		// find local positions of a and b
		int ia = -1, ib = -1;
		for (int e = 0; e < 3; ++e) {
			if (f[e] == a)
				ia = e;
			if (f[e] == b)
				ib = e;
		}
		ASSERT(ia >= 0 && ib >= 0);
		const int ic = 3 - ia - ib; // the third corner
		const int c = f[ic];
		// preserve orientation: original winding f[0],f[1],f[2].
		// Triangle (a, b, c) in original order -> (a, mid, c) and (mid, b, c).
		// Determine the directed order a->b in the face winding.
		bool aBeforeB = ((ia + 1) % 3 == ib);
		if (aBeforeB) {
			// ... a -> b ... with c the remaining vertex
			tris[t] = Eigen::Matrix<int, 3, 1>(a, mid, c);
			tris.emplace_back(mid, b, c);
		} else {
			// b -> a in winding
			tris[t] = Eigen::Matrix<int, 3, 1>(b, mid, c);
			tris.emplace_back(mid, a, c);
		}
	}

	// Collapse patch edges shorter than lmin (PMP collapse_short_edges). Only an
	// unlocked endpoint may be removed; never collapse onto/away a locked
	// (boundary) vertex pair, and never if it would invert the patch.
	//
	// Each pass applies EVERY non-conflicting collapse in the snapshot, mirroring
	// FlipEdges' batching, instead of one-collapse-then-rebuild: marking the whole
	// one-ring of each removed vertex dirty keeps the batched collapses vertex- (and
	// hence triangle-) disjoint, so they commute and CollapseOk stays valid for each
	// against the snapshot. The old code applied one collapse per O(T log T) rebuild
	// (capped at 10 per refine iteration), starving collapses on large patches.
	void CollapseShortEdges(double lmin)
	{
		const double lminSq = lmin * lmin;
		std::vector<EdgeEntry> tbl;
		for (int pass = 0; pass < 10; ++pass) {
			BuildEdgeTable(tbl);
			// Per-wave vertex->incident-triangle adjacency (indices into tris).
			// Valid for the whole harvest loop below: collapses are only *collected*
			// here; tris/points are mutated exclusively by ApplyCollapses, AFTER the
			// loop. Rebuilt each pass so CollapseOk never reads it stale. Patch
			// triangles are non-degenerate, so each triangle contributes its index
			// once to each of its three (distinct) corners; v2t[v] is thus ascending.
			std::vector<std::vector<int>> v2t(points.size());
			for (int t = 0; t < static_cast<int>(tris.size()); ++t) {
				const auto& f = tris[t];
				v2t[f[0]].push_back(t);
				v2t[f[1]].push_back(t);
				v2t[f[2]].push_back(t);
			}
			std::vector<char> vDirty(points.size(), 0);
			std::vector<std::pair<int, int>> collapses; // (remove, keep), disjoint

			for (const auto& e : tbl) {
				if (e.count != 2)
					continue; // only interior edges
				const int a = e.key.a;
				const int b = e.key.b;
				if ((points[a] - points[b]).squaredNorm() >= lminSq)
					continue;
				int remove = -1, keep = -1;
				if (!vlocked[a]) {
					remove = a;
					keep = b;
				} else if (!vlocked[b]) {
					remove = b;
					keep = a;
				}
				if (remove < 0)
					continue;
				if (vDirty[a] || vDirty[b])
					continue;
				if (!CollapseOk(remove, keep, v2t))
					continue;
				// mark the one-ring of `remove` dirty so no conflicting collapse
				// joins this batch (keeps the batch vertex/triangle-disjoint). Same
				// vertex set as scanning all of tris, via the per-wave adjacency.
				for (const int t : v2t[remove]) {
					const auto& f = tris[t];
					vDirty[f[0]] = vDirty[f[1]] = vDirty[f[2]] = 1;
				}
				collapses.emplace_back(remove, keep);
			}
			if (collapses.empty())
				break; // converged — a further pass would redo the full O(V+T)
				    // ApplyCollapses compaction as pure identity work
			ApplyCollapses(collapses);
		}
	}

	// Apply a batch of vertex-disjoint collapses at once: relabel each removed
	// vertex to its kept vertex in every triangle, drop the (now-degenerate)
	// collapsed triangles, then compact the removed vertex slots in a single sweep
	// (highest index first, so swap-compaction never moves a slot still to remove).
	// One edge-map rebuild thus handles k collapses instead of k rebuilds.
	void ApplyCollapses(const std::vector<std::pair<int, int>>& collapses)
	{
		std::vector<int> vmap(points.size());
		for (int i = 0; i < static_cast<int>(vmap.size()); ++i)
			vmap[i] = i;
		for (const auto& c : collapses)
			vmap[c.first] = c.second; // disjoint -> no chaining

		std::vector<Eigen::Matrix<int, 3, 1>> kept;
		kept.reserve(tris.size());
		for (auto f : tris) {
			for (int e = 0; e < 3; ++e)
				f[e] = vmap[f[e]];
			if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0])
				continue; // collapsed pair -> degenerate, dropped
			kept.push_back(f);
		}
		tris.swap(kept);

		std::vector<int> removed;
		removed.reserve(collapses.size());
		for (const auto& c : collapses)
			removed.push_back(c.first);
		std::sort(removed.begin(), removed.end());
		// Batched swap-with-last compaction: each removed slot (highest first)
		// is refilled by the current last vertex, but the triangle relabel
		// happens ONCE at the end. The per-removal relabel was
		// O(collapses * T) — measured 9.6 s for a single 23k-collapse wave on a
		// 266k-triangle patch (the split cascade overshoots far above the final
		// patch size, so T is large exactly when collapses are many). Chained
		// moves (a moved-in survivor becoming `last` later) are resolved through
		// the occupant map.
		std::vector<int> finalIdx(points.size());
		std::vector<int> occupant(points.size());
		for (int i = 0; i < static_cast<int>(points.size()); ++i)
			finalIdx[i] = occupant[i] = i;
		int last = static_cast<int>(points.size()) - 1;
		for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
			const int idx = *it;
			if (idx != last) {
				points[idx] = points[last];
				vlocked[idx] = vlocked[last];
				parentVidx[idx] = parentVidx[last];
				const int v = occupant[last];
				finalIdx[v] = idx;
				occupant[idx] = v;
			}
			--last;
		}
		points.resize(last + 1);
		vlocked.resize(last + 1);
		parentVidx.resize(last + 1);
		for (auto& f : tris)
			for (int e = 0; e < 3; ++e)
				f[e] = finalIdx[f[e]];
	}

	// Would collapsing `remove` onto `keep` keep the patch valid (no inverted /
	// degenerate triangle, no non-manifold fold, no duplicate face)? Geometric
	// guard analogous to PMP relying on SurfaceMesh::is_collapse_ok.
	// `v2t` is the per-wave vertex->incident-triangle adjacency built by the caller
	// (CollapseShortEdges) against the current tris/points snapshot; both scans
	// below need only the triangles around `remove`/`keep`, so this replaces the old
	// O(V*T) / O(T) full-tris sweeps with O(deg) work. It is a pure function of the
	// same inputs and produces the same booleans, so the predicate is bit-identical.
	bool CollapseOk(int remove, int keep,
	                const std::vector<std::vector<int>>& v2t) const
	{
		ASSERT(v2t.size() == points.size());
		ASSERT(remove >= 0 && keep >= 0 && remove != keep);
		// Freshness: every triangle v2t claims for `remove` must still contain
		// `remove`. v2t is snapshot-derived — sound only while tris is untouched
		// during the harvest loop (ApplyCollapses runs after it) and batched
		// collapses stay disjoint (vDirty marking); this catches an edit that
		// breaks either.
		ASSERT([&]() {
			for (const int t : v2t[remove]) {
				const auto& f = tris[t];
				if (f[0] != remove && f[1] != remove && f[2] != remove)
					return false;
			}
			return true;
		}());
		// Link condition: `remove` and `keep` may share at most the two common
		// neighbours forming the two triangles on edge remove-keep. More would
		// collapse two surviving triangles onto the same vertex triple -> duplicate
		// patch faces (the recent flip gate has an equivalent already-exists guard).
		// Compute the one-ring neighbour set of each endpoint (the other two corners
		// of every incident triangle, deduplicated) and count common neighbours. Each
		// common vertex is counted ONCE regardless of how many triangles witness it,
		// exactly as the old per-vertex outer loop did.
		const auto Ring = [&](int v, std::vector<int>& out) {
			out.clear();
			for (const int t : v2t[v]) {
				const auto& f = tris[t];
				for (int e = 0; e < 3; ++e)
					if (f[e] != v)
						out.push_back(f[e]);
			}
			std::sort(out.begin(), out.end());
			out.erase(std::unique(out.begin(), out.end()), out.end());
		};
		std::vector<int> nr, nk;
		Ring(remove, nr);
		Ring(keep, nk);
		int common = 0;
		for (std::size_t i = 0, j = 0; i < nr.size() && j < nk.size();) {
			if (nr[i] < nk[j]) {
				++i;
			} else if (nk[j] < nr[i]) {
				++j;
			} else {
				// common neighbour; remove/keep can never land here (each Ring
				// excludes its own vertex), but keep the w!=remove,keep guard to
				// mirror the old skip exactly.
				if (nr[i] != remove && nr[i] != keep && ++common > 2)
					return false;
				++i;
				++j;
			}
		}
		// Fold-over / degeneracy: only triangles incident to `remove` can change.
		// Iterating v2t[remove] (ascending triangle index) visits exactly the same
		// triangles in the same order as the old full-tris scan filtered by
		// has_remove, so the per-triangle FP test below is byte-for-byte identical.
		for (const int t : v2t[remove]) {
			const auto& f = tris[t];
			bool hasKeep = false;
			for (int e = 0; e < 3; ++e)
				if (f[e] == keep)
					hasKeep = true;
			if (hasKeep)
				continue; // this triangle is removed by the collapse
			// simulate substitution remove->keep; the surviving triangle must be
			// non-degenerate AND keep its orientation (newNormal.dot(oldNormal) > 0),
			// mirroring FlipGeometryOk. A pure degeneracy test let fold-overs pass.
			Point pOld[3], pNew[3];
			for (int e = 0; e < 3; ++e) {
				pOld[e] = points[f[e]];
				pNew[e] = (f[e] == remove) ? points[keep] : points[f[e]];
			}
			const Point oldN = (pOld[1] - pOld[0]).cross(pOld[2] - pOld[0]);
			const Point newN = (pNew[1] - pNew[0]).cross(pNew[2] - pNew[0]);
			if (!(newN.squaredNorm() > 0.0))
				return false; // degenerate
			if (newN.dot(oldN) <= 0.0)
				return false; // fold-over
		}
		return true;
	}

	// Remove an interior vertex slot by swapping the last vertex into it.

	// Flip interior edges to improve vertex valence (PMP flip_edges). Each pass
	// sweeps all edges once over a snapshot map and applies every improving flip
	// whose two triangles are still untouched, then repeats (no mid-sweep
	// rebuild -> no iterator invalidation, no quadratic blow-up).
	void FlipEdges()
	{
		// Full-mesh valence seed for locked boundary vertices: their parent one-ring
		// degree minus the two loop edges (re-counted from the patch emap below), so
		// a boundary vertex is scored against its true post-fill valence instead of
		// appearing under-connected (2-3 vs target) and attracting patch diagonals
		// onto an already-crowded surrounding fan. Matches PMP's full-mesh valence.
		std::vector<int> bseed(nb);
		for (int i = 0; i < nb; ++i)
			bseed[i] = static_cast<int>(hm->VDegree(parentVidx[i])) - 2;
		std::vector<EdgeEntry> tbl;
		for (int pass = 0; pass < 10; ++pass) {
			BuildEdgeTable(tbl);
			std::vector<int> valence(points.size(), 0);
			for (int i = 0; i < nb; ++i)
				valence[i] = bseed[i];
			for (const auto& e : tbl) {
				++valence[e.key.a];
				++valence[e.key.b];
			}
			std::vector<char> triDirty(tris.size(), 0);
			bool flippedAny = false;

			for (const auto& e : tbl) {
				if (e.count != 2)
					continue;
				const int a = e.key.a;
				const int b = e.key.b;
				if (IsOriginalBoundaryEdge(a, b))
					continue;
				const int t0 = e.t0;
				const int t1 = e.t1;
				if (triDirty[t0] || triDirty[t1])
					continue;
				const int c = OppositeCorner(t0, a, b);
				const int d = OppositeCorner(t1, a, b);
				if (c < 0 || d < 0 || c == d)
					continue;
				if (EdgeExists(tbl, c, d))
					continue; // new edge already present
				if (!FlipImprovesValence(a, b, c, d, valence))
					continue;
				if (!FlipGeometryOk(t0, t1, a, b, c, d))
					continue;
				DoFlip(t0, t1, a, b, c, d);
				triDirty[t0] = triDirty[t1] = 1;
				--valence[a];
				--valence[b];
				++valence[c];
				++valence[d];
				flippedAny = true;
			}
			if (!flippedAny)
				break;
		}
	}

	// optimal valence: 6 interior, 6 for a boundary vertex that becomes interior
	// once this hole is filled (its only boundary was this hole), else 4 (stays on
	// another boundary) — PMP scores hole-boundary vertices at their post-fill
	// valence, not the locked-boundary default.
	int OptValence(int v, const std::vector<int>&) const
	{
		if (v < nb)
			return detail::IdealValence(!becomesInterior[v]);
		return detail::IdealValence(vlocked[v]);
	}

	bool FlipImprovesValence(int a, int b, int c, int d,
	                         const std::vector<int>& valence) const
	{
		return detail::FlipImprovesValence(
		    valence[a], valence[b], valence[c], valence[d],
		    OptValence(a, valence), OptValence(b, valence),
		    OptValence(c, valence), OptValence(d, valence));
	}

	bool FlipGeometryOk(int t0, int t1, int a, int b, int c, int d) const
	{
		// The two new triangles (with the exact winding DoFlip will assign) must
		// be non-degenerate AND keep the local orientation: flipping a-b -> c-d
		// across a non-planar quad can fold a triangle over, which the old
		// degeneracy-only test let through (the remesher's TestEdgeFlip applies
		// the same normal-consistency gate).
		auto triNormal = [&](int i, int j, int k) {
			return Eigen::Vector3d(
			    (points[j] - points[i]).cross(points[k] - points[i]));
		};
		Eigen::Vector3d new0, new1;
		if (DirectedAB(t0, a, b)) {
			new0 = triNormal(a, d, c);
			new1 = triNormal(b, c, d);
		} else {
			new0 = triNormal(b, d, c);
			new1 = triNormal(a, c, d);
		}
		if (!(new0.squaredNorm() > 0.0) || !(new1.squaredNorm() > 0.0))
			return false;
		const auto& f0 = tris[t0];
		const auto& f1 = tris[t1];
		const Eigen::Vector3d oldN = triNormal(f0[0], f0[1], f0[2]) + triNormal(f1[0], f1[1], f1[2]);
		return new0.dot(oldN) > 0.0 && new1.dot(oldN) > 0.0;
	}

	int OppositeCorner(int t, int a, int b) const
	{
		const auto& f = tris[t];
		for (int e = 0; e < 3; ++e)
			if (f[e] != a && f[e] != b)
				return f[e];
		return -1;
	}

	void DoFlip(int t0, int t1, int a, int b, int c, int d)
	{
		// Replace edge a-b by c-d. Preserve orientation by inheriting winding
		// from the two original triangles: t0 = (a,b,c)-ish, t1 = (b,a,d)-ish.
		// Build new triangles (a,d,c) and (b,c,d) consistent with the originals.
		// Determine directed order of a->b in t0.
		bool aBeforeBT0 = DirectedAB(t0, a, b);
		if (aBeforeBT0) {
			tris[t0] = Eigen::Matrix<int, 3, 1>(a, d, c);
			tris[t1] = Eigen::Matrix<int, 3, 1>(b, c, d);
		} else {
			tris[t0] = Eigen::Matrix<int, 3, 1>(b, d, c);
			tris[t1] = Eigen::Matrix<int, 3, 1>(a, c, d);
		}
	}

	bool DirectedAB(int t, int a, int b) const
	{
		const auto& f = tris[t];
		for (int e = 0; e < 3; ++e)
			if (f[e] == a)
				return f[(e + 1) % 3] == b;
		return false;
	}

	// Membrane (uniform-umbrella) relaxation of the free interior vertices with
	// the boundary locked.  Shared with the isotropic remesher through
	// detail::FairMesh(k=1) (one-ring graph Laplacian, locked verts on the rhs,
	// Eigen SimplicialLDLT).  Leaves positions unchanged on solver failure.
	void Relaxation()
	{
		detail::FairMesh(points, tris, vlocked, /*k=*/1);
	}

	// Remove caps: flip any edge whose opposite triangle has an angle > 170deg
	// (PMP remove_caps). Single sweep over a snapshot map; apply every cap-fixing
	// flip whose two triangles are still untouched (no mid-sweep rebuild).
	void RemoveCaps()
	{
		const double aa = std::cos(170.0 * std::numbers::pi / 180.0);
		std::vector<EdgeEntry> tbl;
		BuildEdgeTable(tbl);
		std::vector<char> triDirty(tris.size(), 0);
		for (const auto& e : tbl) {
			if (e.count != 2)
				continue;
			const int a = e.key.a;
			const int b = e.key.b;
			if (IsOriginalBoundaryEdge(a, b))
				continue;
			const int t0 = e.t0;
			const int t1 = e.t1;
			if (triDirty[t0] || triDirty[t1])
				continue;
			const int vb = OppositeCorner(t0, a, b);
			const int vd = OppositeCorner(t1, a, b);
			if (vb < 0 || vd < 0 || vb == vd)
				continue;
			const double a0 = (points[a] - points[vb]).normalized().dot((points[b] - points[vb]).normalized());
			const double a1 = (points[a] - points[vd]).normalized().dot((points[b] - points[vd]).normalized());
			const double amin = std::min(a0, a1);
			if (amin < aa) {
				if (EdgeExists(tbl, vb, vd))
					continue;
				if (!FlipGeometryOk(t0, t1, a, b, vb, vd))
					continue;
				DoFlip(t0, t1, a, b, vb, vd);
				triDirty[t0] = triDirty[t1] = 1;
			}
		}
	}

	// Fairing of the patch-interior vertices. Following PMP, fairing is only
	// triggered when refine inserted interior vertices.
	//
	// Thin-plate (bi-Laplacian) fairing via detail::FairMesh(k=2): minimises
	// curvature (PMP minimize_curvature), the dependency-light uniform variant of
	// PMP's cotangent bi-Laplacian. This is the SOTA upgrade over the previous
	// membrane (area-minimising) fairing — a single direct SPD solve replaces the
	// old repeated umbrella relaxation (which, being a direct solve of an
	// unchanged system, only ever produced the membrane result). Falls back to
	// membrane (k=1) if the higher-order solve fails on a degenerate patch.
	//
	// The bi-Laplacian row of a free vertex next to the boundary needs the boundary
	// vertex's FULL one-ring; with patch-only edges L(boundary,.) is truncated, so
	// the solve is blind to the surrounding surface tangent and folds interior
	// triangles on non-planar patches. We therefore augment the system with the
	// parent one-ring of faces around the boundary — appended as LOCKED vertices at
	// their parent positions — so the boundary Laplacian rows are complete and the
	// patch inherits the surrounding slope (PMP runs minimize_curvature on the whole
	// mesh, reaching the second ring outside the hole). The augmentation is local to
	// the solve: only the free interior patch positions are copied back, so
	// Triangles()/NumInterior()/InteriorPoint() never see the appended ring.
	void Fairing()
	{
		bool hasInterior = false;
		for (size_t v = 0; v < vlocked.size(); ++v)
			if (!vlocked[v]) {
				hasInterior = true;
				break;
			}
		if (!hasInterior)
			return;

		// Augmented copies: patch + surrounding parent one-ring (all locked).
		std::vector<Point> augPts = points;
		std::vector<Eigen::Matrix<int, 3, 1>> augTris = tris;
		std::vector<bool> augLocked = vlocked;

		// parent vertex -> augmented local index; boundary loop verts keep 0..nb-1.
		std::unordered_map<VIndex, int> pmap;
		pmap.reserve(static_cast<std::size_t>(nb) * 4);
		for (int i = 0; i < nb; ++i)
			if (parentVidx[i] != math::NO_ID)
				pmap.emplace(parentVidx[i], i);
		const auto ringLocal = [&](VIndex pv) -> int {
			const auto it = pmap.find(pv);
			if (it != pmap.end())
				return it->second;
			const int idx = static_cast<int>(augPts.size());
			const Mesh::Vertex& p = (*pverts)[pv];
			augPts.emplace_back(p.x(), p.y(), p.z());
			augLocked.push_back(true);
			pmap.emplace(pv, idx);
			return idx;
		};

		// Surrounding parent faces incident to the boundary loop (deduped, ordered
		// for determinism); their edges complete the boundary vertices' one-rings.
		std::set<FIndex> ringFaces;
		for (int i = 0; i < nb; ++i)
			for (const FIndex f : hm->VAdjacentFaces(parentVidx[i]))
				ringFaces.insert(f);
		for (const FIndex f : ringFaces) {
			const HalfMesh::Face face = hm->F(f);
			augTris.emplace_back(ringLocal(face[0]), ringLocal(face[1]),
			                     ringLocal(face[2]));
		}

		// Cotangent weights for the thin-plate solve (PMP/Liepa minimize_curvature):
		// they remove the tessellation-density bias of the uniform Laplacian so the
		// faired patch follows geometry, not vertex distribution. Fall back to the
		// uniform membrane (k=1) on solver failure.
		if (detail::FairMesh(augPts, augTris, augLocked, /*k=*/2, /*cotangent=*/true)) {
			// copy back only the free interior patch positions
			for (int v = nb; v < static_cast<int>(points.size()); ++v)
				points[v] = augPts[v];
		} else {
			detail::FairMesh(points, tris, vlocked, /*k=*/1);
		}
	}

	bool IsOriginalBoundaryEdge(int a, int b) const
	{
		// an original boundary edge connects two locked vertices that are
		// consecutive in the loop (indices < nb).
		if (a >= nb || b >= nb)
			return false;
		const int n = nb;
		const int d = std::abs(a - b);
		return d == 1 || d == n - 1;
	}

	// members
	int nb; // number of boundary vertices
	const HalfMesh* hm; // parent half-edge mesh (for interior-edge queries)
	const std::vector<Mesh::Vertex>* pverts; // parent positions (ring-support fairing)
	std::vector<Point> points;
	std::vector<bool> vlocked;
	std::vector<VIndex> parentVidx;
	std::vector<Point> oppNorms;
	std::vector<Eigen::Matrix<int, 3, 1>> tris;

	// interior-edge (forbidden-diagonal) table, precomputed in the constructor
	std::vector<std::vector<char>> interiorEdge;
	// per loop vertex: does it become a full interior vertex once this hole fills?
	std::vector<char> becomesInterior;

	// DP scratch
	std::vector<std::vector<Weight>> weight;
	std::vector<std::vector<int>> index;
	std::vector<std::vector<Point>> normal; // cached sub-triangle normal per (i,j)
};

// ---------------------------------------------------------------------------
// BoundaryLoop — a hole as discovered by walking boundary half-edges.
//   verts:     ordered parent vertex indices (loop[k] -> loop[k+1] is an edge)
//   oppNorms: normalized interior-face normal across boundary edge k
// ---------------------------------------------------------------------------
struct BoundaryLoop
{
	std::vector<Mesh::VIndex> verts;
	std::vector<HoleFilling::Point> oppNorms;
};

// Trace every boundary loop of `hm` (one per connected boundary), recording for
// each boundary edge the normal of the interior face across it (the dihedral
// neighbour used by HoleFilling, mirroring PMP opposite_normal()).
static void EnumerateBoundaryLoops(const HalfMesh& hm,
                                   const std::vector<Mesh::Vertex>& vertices,
                                   std::vector<BoundaryLoop>& loops)
{
	loops.clear();
	const HalfMesh::HIndex nHe = hm.HeSize();
	std::vector<bool> visited(nHe, false);
	for (HalfMesh::HIndex h = 0; h < nHe; ++h) {
		if (visited[h] || !hm.HeIsBoundary(h))
			continue;
		BoundaryLoop loop;
		HalfMesh::HIndex it = h;
		do {
			visited[it] = true;
			const Mesh::VIndex tail = hm.HeVertex(it); // loop[k]
			loop.verts.push_back(tail);
			// interior face across this boundary edge (tail -> head)
			const HalfMesh::HIndex twin = hm.HeTwin(it);
			const Mesh::FIndex f = hm.HeFace(twin);
			HoleFilling::Point n(0, 0, 1);
			if (f != math::NO_ID) {
				const HalfMesh::Face face = hm.F(f);
				const Mesh::Vertex& a = vertices[face[0]];
				const Mesh::Vertex& b = vertices[face[1]];
				const Mesh::Vertex& c = vertices[face[2]];
				const Mesh::Normal nn = (b - a).cross(c - a);
				const double len = nn.norm();
				if (len > 0)
					n = HoleFilling::Point(nn.x() / len, nn.y() / len, nn.z() / len);
			}
			loop.oppNorms.push_back(n);
			it = hm.HeNext(it);
		} while (it != h);
		loops.push_back(std::move(loop));
	}
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Mesh::CloseHoles
//
// Enumerate the mesh's boundary loops (holes) and fill the smallest `nCloseHoles`
// of them by Liepa minimum-weight triangulation, optionally refining and fairing
// the patch (PMP pipeline). New patch triangles are appended to `faces` and any
// interior vertices added by refine/fairing are appended to `vertices`. If
// `holesFaces` is non-null it receives, per filled hole, the indices of the new
// faces. Returns the number of holes actually closed.
// ---------------------------------------------------------------------------
unsigned Mesh::CloseHoles(unsigned nCloseHoles,
                          std::vector<std::vector<FIndex>>* holesFaces)
{
	if (faces.empty() || nCloseHoles == 0)
		return 0;

	// Stage 2/3 (refine + fairing) are enabled by default: they remesh the patch
	// to the surrounding density and smooth the interior. They never remove
	// boundary vertices and degrade gracefully if the SPD solve fails.
	const bool doRefine = true;
	const bool doFair = true;

	// Force a fresh half-edge build: ListHalfEdges() caches by vertex count, so a
	// prior build with the same vertex count would otherwise be reused even if
	// faces changed since.
	halfMesh.Clear();
	ListHalfEdges();
	if (halfMesh.Empty())
		return 0;

	std::vector<BoundaryLoop> loops;
	EnumerateBoundaryLoops(halfMesh, vertices, loops);
	if (loops.empty())
		return 0;

	// Fill the smallest holes first (by boundary-vertex count), as the brief
	// requires. A hole needs at least 3 boundary vertices to be triangulable.
	std::vector<unsigned> order(loops.size());
	for (unsigned i = 0; i < order.size(); ++i)
		order[i] = i;
	std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
		return loops[a].verts.size() < loops[b].verts.size();
	});

	// Collect the simple, triangulable candidate loops (>= 3 vertices, no repeated
	// vertex) in the fixed smallest-first order.
	std::vector<unsigned> candidates;
	candidates.reserve(order.size());
	for (unsigned oi = 0; oi < order.size(); ++oi) {
		const BoundaryLoop& loop = loops[order[oi]];
		if (loop.verts.size() < 3)
			continue;
		std::unordered_map<VIndex, int> seen;
		bool simple = true;
		for (VIndex v : loop.verts)
			if (++seen[v] > 1) {
				simple = false;
				break;
			}
		if (simple)
			candidates.push_back(order[oi]);
	}

	// Fill holes in parallel and harvest sequentially (phased launch-then-harvest).
	// Each HoleFilling only reads immutable parent state (halfMesh is never rebuilt
	// during the loop; boundary positions are copied at construction; the ring
	// support reads original parent faces/positions), so concurrent Fills are
	// race-free. Determinism is preserved by a FIXED harvest order (the smallest
	// -first candidate order): a hole's Fill result is independent of whether an
	// earlier hole was already harvested, so the appended vertices/faces are
	// byte-identical to a serial fill. Batches of `need` candidates are launched
	// and topped up when some fills fail.
	unsigned closed = 0;
	unsigned next = 0;
	BS::light_thread_pool pool;
	while (closed < nCloseHoles && next < candidates.size()) {
		const unsigned need = nCloseHoles - closed;
		const unsigned begin = next;
		const unsigned end =
		    std::min<unsigned>(begin + need, static_cast<unsigned>(candidates.size()));
		next = end;

		std::vector<HoleFilling> hfs;
		hfs.reserve(end - begin);
		for (unsigned i = begin; i < end; ++i) {
			const BoundaryLoop& loop = loops[candidates[i]];
			hfs.emplace_back(*this, loop.verts, loop.oppNorms);
		}
		std::vector<char> ok(hfs.size(), 0);
		pool.detach_blocks(std::size_t(0), hfs.size(),
		                   [&](std::size_t b, std::size_t e) {
			                   for (std::size_t j = b; j < e; ++j)
				                   ok[j] = hfs[j].Fill(doRefine, doFair);
		                   });
		pool.wait();

		for (std::size_t j = 0; j < hfs.size() && closed < nCloseHoles; ++j) {
			if (!ok[j])
				continue;
			HoleFilling& hf = hfs[j];

			// Append interior vertices first; remember their parent indices.
			std::vector<VIndex> interiorParent(hf.NumInterior());
			for (int k = 0; k < hf.NumInterior(); ++k) {
				interiorParent[k] = static_cast<VIndex>(vertices.size());
				vertices.emplace_back(hf.InteriorPoint(k));
			}

			// Map a local patch index to a parent vertex index.
			const int nb = hf.NumBoundary();
			auto LocalToParent = [&](int local) -> VIndex {
				return (local < nb) ? hf.ParentVertex(local)
				                    : interiorParent[local - nb];
			};

			// Append patch triangles.
			std::vector<FIndex> newFaces;
			newFaces.reserve(hf.Triangles().size());
			for (const auto& t : hf.Triangles()) {
				const VIndex a = LocalToParent(t[0]);
				const VIndex b = LocalToParent(t[1]);
				const VIndex c = LocalToParent(t[2]);
				if (a == b || b == c || c == a)
					continue; // skip any degenerate triangle defensively
				newFaces.push_back(static_cast<FIndex>(faces.size()));
				faces.emplace_back(Face(a, b, c));
			}

			if (newFaces.empty()) {
				// nothing usable was produced; drop any interior verts we added
				vertices.resize(vertices.size() - interiorParent.size());
				continue;
			}

			if (holesFaces != nullptr)
				holesFaces->emplace_back(std::move(newFaces));
			++closed;
		}
	}

	if (closed > 0) {
		// structural change: invalidate/rebuild derived data
		faceNormals.clear();
		vertexFaces.clear();
		halfMesh.Clear();
		ListHalfEdges();
	}
	return closed;
}

} // namespace halfmesh
