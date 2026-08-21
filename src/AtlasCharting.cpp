/*
* AtlasCharting.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// src/AtlasCharting.cpp — the atlas pipeline's segmentation (Module A), density
// normalisation (Module C) and the GenerateAtlas orchestrator. The developable
// (D-Charts) segmentation method — cone-Lloyd → developable merge → flip repair —
// and the approaches that did NOT work are documented in full atop
// halfmesh/AtlasCharting.h. Per-chart UV flattening (Module B) is in
// src/Parametrize.cpp; rectangle packing (Module D) is in src/AtlasPacking.cpp.
//
// Density (Module C) overview — rescale each chart's UVs to a uniform
// texels-per-world-unit D:
//   scale_c = D * sqrt(world_area_c / uv_area_c);  auto D = resolution / sqrt(Σ world_area_c).
//
// Segmentation depends on the flattener (Module B) only through one bridge,
// detail::ChartFacesFold ("does this chart fold when flattened?", the flip-repair
// test); it is defined in src/Parametrize.cpp and declared in ChartFlattenCache.h
// together with the flatten-artifact cache the flip-repair fills for
// ParametrizeCharts (so shipping charts are not flattened twice).

#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>

#include "ChartFlattenCache.h" // ChartFacesFold bridge + flatten-artifact reuse

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <halfmesh/HalfMesh.h>
#include <Eigen/Eigenvalues>
#include <BS_thread_pool.hpp> // parallel per-chart flatten in RepairDevelopableFlips
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#ifdef HM_ATLAS_DEBUG
	#include <iostream> // verbose cone-Lloyd / flip-repair diagnostics (HALFMESH_ATLAS_DEBUG only)
#endif

namespace halfmesh {

namespace {

using Vec3 = Eigen::Vector3d; // BisectFaces works in real 3-D geometry

using FIndex = Mesh::FIndex;
using EIndex = Mesh::EIndex;
using HIndex = Mesh::HIndex;
using Normal = Eigen::Matrix<double, 3, 1>;

constexpr unsigned NONE = std::numeric_limits<unsigned>::max();

// D-Charts developability error of a chart of weighted face normals, given the
// accumulators W=Σw, s1=Σw·n, s2=Σw·n·nᵀ. A cone proxy ⟨axis N, half-angle θ⟩
// captures the whole developable family (plane θ=0, cylinder θ=90°, cone
// between); the best-fit cone error is the SMALLEST eigenvalue of the weighted
// normal covariance C = s2 − s1·s1ᵀ/W (optimal cosθ = N·n̄ ⇒ error = λ_min(C)).
// This bounds FLATTENABILITY (zero for any developable surface), unlike the
// planar L2,1 metric which a half-cylinder/saddle can pass while folding. Used
// to decide which adjacent charts may merge into one large flattenable chart.
inline double ConeFitError(double W, const Normal& s1, const Eigen::Matrix3d& s2)
{
	if (W <= 0.0)
		return 0.0;
	const Eigen::Matrix3d C = s2 - (s1 * s1.transpose()) / W;
	// Closed-form (Cardano) 3x3 eigenvalues — ~5x faster than the iterative QR path
	// and accurate to ~1e-14 relative, which is irrelevant for a cost that only RANKS
	// merges.
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
	es.computeDirect(C, Eigen::EigenvaluesOnly);
	// Fail-closed on solver failure: a failed fit must never look converged nor
	// authorize a merge (tryPush filters inf ≤ budget, ConeMoments::error marks the
	// chart over-budget → re-seeded). computeDirect is effectively always Success, so
	// this branch is dead-but-safe.
	if (es.info() != Eigen::Success)
		return std::numeric_limits<double>::infinity();
	return std::max(0.0, es.eigenvalues()(0)); // smallest eigenvalue = cone error
}

// Best-fit cone ⟨axis, cosθ⟩ + total error for the moments W, s1=Σw·n, s2=Σw·n·nᵀ.
// axis = smallest-eigenvector of the normal covariance; cosθ = axis·n̄; error =
// λ_min(C). For a near-plane (no clear cone axis) we fall back to the mean
// normal with cosθ=1 so off-plane faces are still penalized.
struct Cone
{
	Normal axis = Normal::UnitZ();
	double cosTheta = 1.0;
	double error = 0.0; // per-area developability error (λ_min / W)
	// cone-fit cost of a single unit normal against this cone: (axis·n − cosθ)².
	double cost(const Normal& n) const
	{
		const double d = axis.dot(n) - cosTheta;
		return d * d;
	}
};
inline Cone FitConeProxy(double W, const Normal& s1, const Eigen::Matrix3d& s2)
{
	Cone c;
	if (W <= 0.0)
		return c;
	const Normal nbar = s1 / W;
	const Eigen::Matrix3d C = s2 - W * nbar * nbar.transpose();
	// Closed-form (Cardano) 3x3 eigen-decomposition — ~5x faster than iterative QR;
	// eigenvalues stay ascending so (0)/(2) and eigenvectors().col(0) keep their meaning.
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
	es.computeDirect(C);
	// Scale-invariant plane-vs-cone gate: C's entries scale linearly with the total
	// chart area W (unit normals ⇒ trace(C) ≤ W), so an ABSOLUTE 1e-12 floor makes
	// every chart on a tiny-unit model take the plane fallback (degrading the flood to
	// planar VSA, which shatters curved regions). Compare λ_max/W — the mean-squared
	// normal spread, dimensionless.
	if (es.info() == Eigen::Success && es.eigenvalues()(2) > 1e-12 * W) {
		c.axis = es.eigenvectors().col(0).normalized();
		c.error = std::max(0.0, es.eigenvalues()(0)) / W;
	} else {
		const double l = nbar.norm();
		c.axis = (l > 1e-12) ? Normal(nbar / l) : Normal(Normal::UnitZ());
		c.error = 0.0;
	}
	if (c.axis.dot(nbar) < 0.0)
		c.axis = -c.axis;
	c.cosTheta = std::clamp(c.axis.dot(nbar), -1.0, 1.0);
	return c;
}

// -------------------------------------------------------------------------
// SegmentState — holds all precomputed per-face / per-edge data shared by the
// segmentation phases.
// -------------------------------------------------------------------------
struct SegmentState
{
	const Mesh& mesh;
	const HalfMesh& hm;
	const ParametrizeParams& params;

	FIndex numFaces = 0;
	std::vector<Normal> normals; // unit face normals
	std::vector<double> weights; // per-face importance weight (default: area)
	bool hasTexblobs = false; // mesh carries per-face material/texture blobs
	// Virtual (segmentation-only) denoised vertex positions; empty ⇒ use the mesh
	// positions verbatim. Feeds normals + angle defect on the developable path.
	std::vector<Eigen::Vector3d> spos;
	// Per-face centroids in segmentation space (denoised when smoothing was
	// applied) — feeds the distance term's centroid-path accumulation. Filled
	// only when developableDistanceExponent > 0.
	std::vector<Eigen::Vector3d> centroids;

	// Segmentation-space position of a vertex (denoised if smoothing was applied).
	Eigen::Vector3d Pos(Mesh::VIndex v) const
	{
		return spos.empty() ? mesh.vertices[v].template cast<double>() : spos[v];
	}

	explicit SegmentState(const Mesh& m, const ParametrizeParams& p) :
	    mesh(m), hm(m.halfMesh), params(p) {}

	// The neighbor face a chart may grow into across half-edge iHe, or NONE. Charts
	// span everything EXCEPT the topological/material seams that must always bound a
	// chart: mesh-border, non-manifold, and texblob (material) edges. Creases are
	// deliberately NOT a boundary — developability (the cone error + angle-defect
	// cap) alone decides chart edges, so noisy MVS normals cannot spawn charts.
	FIndex TopoNeighbor(HIndex iHe) const
	{
		const HIndex iTwin = hm.HeTwin(iHe);
		if (hm.HeIsBoundary(iHe) || hm.HeIsBoundary(iTwin))
			return NONE; // mesh border
		// No non-manifold-edge check is needed here: HalfMesh::Build rejects any edge
		// carrying a third incident face (duplicate directed edge) and any bow-tie
		// vertex, so once BOTH half-edges are interior (checked above) the edge is
		// manifold and EDegree()==2 unconditionally. The old `EDegree(...)!=2` test was
		// therefore dead code — plus an edge-ring walk — on every flood visit.
		const FIndex nb = hm.HeFace(iTwin);
		if (nb == math::NO_ID)
			return NONE;
		if (hasTexblobs && mesh.faceTexblobs[hm.HeFace(iHe)] != mesh.faceTexblobs[nb])
			return NONE; // material border
		return nb;
	}
};

// -------------------------------------------------------------------------
// Precompute normals, weights and the hard-edge mask.
// -------------------------------------------------------------------------
void Precompute(SegmentState& s)
{
	const Mesh& mesh = s.mesh;
	const HalfMesh& hm = s.hm;
	s.numFaces = static_cast<FIndex>(mesh.faces.size());

	s.normals.resize(s.numFaces);
	s.weights.resize(s.numFaces);
	const bool hasTexblobs = mesh.faceTexblobs.size() == mesh.faces.size();
	s.hasTexblobs = hasTexblobs;

	// Adaptive virtual geometry denoise: Taubin lambda|mu smoothing over a TEMPORARY copy
	// of the vertex positions. Both the face normals and the angle defect derive from
	// triangle geometry, so denoising once makes the developable analysis robust to
	// MVS noise (the negative mu pass cancels the lambda pass's Laplacian shrinkage, so
	// genuine features keep their normal spread; boundary vertices are pinned). It is
	// ADAPTIVE: skipped on small meshes — noise only matters at scale, and smoothing a
	// tiny synthetic/clean mesh (e.g. a cube) would deform its sharp features.
	// This is deliberately distinct from Mesh::SmoothTaubin (the general-purpose
	// surface smoother): that one smooths border vertices along the boundary curve,
	// whereas segmentation here PINS them; do not consolidate the two.
	constexpr FIndex minFacesToSmooth = 2000;
	const unsigned geoIters =
	    (s.numFaces >= minFacesToSmooth) ? s.params.developableSmoothIters : 0u;
	if (geoIters > 0) {
		const std::size_t nv = mesh.vertices.size();
		std::vector<Eigen::Vector3d> P(nv);
		for (std::size_t v = 0; v < nv; ++v)
			P[v] = mesh.vertices[v].template cast<double>();
		std::vector<char> boundary(nv, 0);
		for (std::size_t v = 0; v < nv; ++v)
			boundary[v] = hm.VIsBoundary(static_cast<Mesh::VIndex>(v)) ? 1 : 0;
		constexpr double lambda = 0.330, mu = -0.331; // |mu|>lambda => no shrinkage
		std::vector<Eigen::Vector3d> Q(nv);
		for (unsigned pass = 0; pass < geoIters * 2u; ++pass) {
			const double step = (pass % 2u == 0u) ? lambda : mu;
			for (std::size_t v = 0; v < nv; ++v) {
				if (boundary[v]) {
					Q[v] = P[v];
					continue;
				}
				Eigen::Vector3d acc = Eigen::Vector3d::Zero();
				unsigned cnt = 0;
				for (Mesh::VIndex nb : hm.VAdjacentVertices(static_cast<Mesh::VIndex>(v))) {
					acc += P[nb];
					++cnt;
				}
				Q[v] = (cnt > 0) ? Eigen::Vector3d(P[v] + step * (acc / cnt - P[v])) : P[v];
			}
			P.swap(Q);
		}
		s.spos.swap(P);
	}

	// Unit face normals (from the denoised positions when smoothing was applied, so
	// we don't depend on prior mesh state). Weights stay on the TRUE geometry (area)
	// so texel density / packing are measured on the real surface.
	for (FIndex f = 0; f < s.numFaces; ++f) {
		Normal n;
		if (s.spos.empty()) {
			const Mesh::Normal mn = mesh.ComputeFaceNormal(f);
			n = mn.cast<double>();
		} else {
			const Mesh::Face& face = mesh.faces[f];
			n = (s.spos[face[1]] - s.spos[face[0]]).cross(s.spos[face[2]] - s.spos[face[0]]);
		}
		const double len = n.norm();
		s.normals[f] = (len > 0.0) ? Normal(n / len) : Normal::Zero();
		// default weight = face area (= half the double-area) on the REAL mesh.
		double w = static_cast<double>(mesh.ComputeFaceDoubleArea(f)) * 0.5;
		if (s.params.faceWeight)
			w = static_cast<double>(s.params.faceWeight(f));
		// guard against zero / degenerate weights so seeds always anchor.
		s.weights[f] = (w > 0.0) ? w : std::numeric_limits<double>::min();
	}

	if (s.params.developableDistanceExponent > 0.f) {
		s.centroids.resize(s.numFaces);
		for (FIndex f = 0; f < s.numFaces; ++f) {
			const Mesh::Face& face = mesh.faces[f];
			s.centroids[f] = (s.Pos(face[0]) + s.Pos(face[1]) + s.Pos(face[2])) / 3.0;
		}
	}
}

// -------------------------------------------------------------------------
// Min-heap candidate for best-first flood growth: the cheapest pending
// (cost, face, chart) wins. Used by the cone flood-assign.
// -------------------------------------------------------------------------
struct Candidate
{
	double cost;
	FIndex face;
	unsigned chart;
	unsigned depth; // flood hops from the chart's seed
	double dist; // accumulated centroid-path distance from the chart's seed
	    // (data for the distance term — NOT a tie-break)
	// Min-heap via std::priority_queue (a max-heap on operator<): the CHEAPEST cost
	// pops first. On developable regions every candidate cost ties at ~0, so the
	// boundary between competing charts would otherwise be decided by the STL heap's
	// implementation-defined pop order — snaky, jagged seams AND non-determinism across
	// STL builds. Break exact-cost ties by the NEAREST face (smallest flood depth) for
	// compact, straight Voronoi seams, then by smallest face id, then smallest chart id
	// (the same face is pushed by several charts) for cross-platform determinism.
	bool operator<(const Candidate& o) const
	{
		if (cost != o.cost)
			return cost > o.cost;
		if (depth != o.depth)
			return depth > o.depth;
		if (face != o.face)
			return face > o.face;
		return chart > o.chart;
	}
};

// -------------------------------------------------------------------------
// Connected "regions": maximal face sets connected through TopoNeighbor edges
// (i.e. across everything but mesh-border / non-manifold / texblob seams). A
// chart can never span two regions. Returns the region count, fills regionId.
// -------------------------------------------------------------------------
unsigned ComputeRegions(const SegmentState& s, std::vector<unsigned>& regionId)
{
	regionId.assign(s.numFaces, NONE);
	const HalfMesh& hm = s.hm;
	unsigned num = 0;
	std::vector<FIndex> stack;
	for (FIndex start = 0; start < s.numFaces; ++start) {
		if (regionId[start] != NONE)
			continue;
		regionId[start] = num;
		stack.push_back(start);
		while (!stack.empty()) {
			const FIndex f = stack.back();
			stack.pop_back();
			for (HIndex iHe : hm.FAdjacentHalfedges(f)) {
				const FIndex nb = s.TopoNeighbor(iHe);
				if (nb != NONE && regionId[nb] == NONE) {
					regionId[nb] = num;
					stack.push_back(nb);
				}
			}
		}
		++num;
	}
	return num;
}

// -------------------------------------------------------------------------
// Farthest-point sampling of additional seeds, constrained per region.
// `seeds` already contains one seed per region (mandatory for coverage). This
// adds up to `extra` more seeds, each at the face farthest (hop distance over
// topo edges) from all current seeds.
// -------------------------------------------------------------------------
void AddFarthestSeeds(const SegmentState& s, std::vector<FIndex>& seeds, unsigned extra)
{
	if (extra == 0)
		return;
	const HalfMesh& hm = s.hm;
	// Multi-source BFS distance from all current seeds.
	std::vector<unsigned> dist(s.numFaces, NONE);
	std::queue<FIndex> bfs;
	for (FIndex sf : seeds) {
		dist[sf] = 0;
		bfs.push(sf);
	}
	// Standard incremental multi-source BFS with distance RELAXATION. The old
	// `dist[nb] == NONE` guard only ever expanded into never-visited faces, so after
	// the first multi-source pass filled every reachable face no later runBfs (from a
	// freshly added seed) could lower a distance — every subsequent "farthest" pick
	// read stale distances and landed adjacent to the previous extra seed, clustering
	// the extra seeds at one spot. Relaxing (dist[f]+1 < dist[nb]) lets dist[best]=0
	// propagate lower distances around each new seed so successive picks are true
	// farthest points. dist[f] is always set here (f
	// was popped after being assigned), and dist[nb]==NONE (UINT_MAX) relaxes correctly.
	auto runBfs = [&]() {
		while (!bfs.empty()) {
			const FIndex f = bfs.front();
			bfs.pop();
			for (HIndex iHe : hm.FAdjacentHalfedges(f)) {
				const FIndex nb = s.TopoNeighbor(iHe);
				if (nb != NONE && dist[f] + 1 < dist[nb]) {
					dist[nb] = dist[f] + 1;
					bfs.push(nb);
				}
			}
		}
	};
	runBfs();
	for (unsigned i = 0; i < extra; ++i) {
		// pick the reachable face with the largest distance.
		FIndex best = NONE;
		unsigned bestD = 0;
		for (FIndex f = 0; f < s.numFaces; ++f) {
			if (dist[f] != NONE && dist[f] != 0 && dist[f] > bestD) {
				bestD = dist[f];
				best = f;
			}
		}
		if (best == NONE || bestD == 0)
			break; // no farther face to seed
		seeds.push_back(best);
		// incrementally update distances from the new seed.
		dist[best] = 0;
		bfs.push(best);
		runBfs();
	}
}

// -------------------------------------------------------------------------
// Compact relabel: remap arbitrary chart ids to 0..N-1 in first-appearance
// order. Returns the number of distinct charts.
// -------------------------------------------------------------------------
unsigned Compact(std::vector<unsigned>& chart)
{
	unsigned maxId = 0;
	for (unsigned c : chart)
		if (c != NONE && c > maxId)
			maxId = c;
	std::vector<unsigned> table(maxId + 1, NONE);
	unsigned next = 0;
	for (unsigned& c : chart) {
		if (c == NONE)
			continue;
		if (table[c] == NONE)
			table[c] = next++;
		c = table[c];
	}
	return next;
}

// -------------------------------------------------------------------------
// Split disconnected charts so each chart is a single connected face set.
//
// A chart may become disconnected after merges/bisections, so we re-flood every
// chart over TopoNeighbor edges and give each connected piece of the same chart a
// distinct new id. This guarantees per-chart connectivity AND that no chart spans
// a mesh-border / non-manifold / texblob seam. Relabels compactly; returns count.
// -------------------------------------------------------------------------
unsigned EnforceConnectivity(const SegmentState& s, std::vector<unsigned>& chart)
{
	const HalfMesh& hm = s.hm;
	std::vector<unsigned> newc(s.numFaces, NONE);
	unsigned next = 0;
	std::vector<FIndex> stack;
	for (FIndex start = 0; start < s.numFaces; ++start) {
		if (newc[start] != NONE)
			continue;
		const unsigned home = chart[start];
		newc[start] = next;
		stack.push_back(start);
		while (!stack.empty()) {
			const FIndex f = stack.back();
			stack.pop_back();
			for (HIndex iHe : hm.FAdjacentHalfedges(f)) {
				const FIndex nb = s.TopoNeighbor(iHe);
				if (nb != NONE && newc[nb] == NONE && chart[nb] == home) {
					newc[nb] = next;
					stack.push_back(nb);
				}
			}
		}
		++next;
	}
	chart.swap(newc);
	return next;
}

// -------------------------------------------------------------------------
// DevelopableMerge — D-Charts-style consolidation into the fewest large
// FLATTENABLE charts. Starting from the fine planar charts, greedily merge
// adjacent charts — crossing creases (only mesh-border / non-manifold / texblob
// edges block) — whenever the merged chart's best-fit-cone error per area stays
// below the budget AND the merge would not enclose a high-curvature vertex. The
// cone error bounds developability and the angle-defect cap is the hard anti-fold
// guarantee, so charts grow large over cylinders/cones yet stay flip-free.
// Greedy lowest-error pair via a lazy heap on a union-find chart graph
// (~O(E·log)). Relabels `chart`, returns the new count.
// -------------------------------------------------------------------------
void ComputeVertexDefect(const SegmentState& s, std::vector<double>& dabs,
                         std::vector<unsigned>& vtotal);

unsigned DevelopableMerge(const SegmentState& s, const ParametrizeParams& params,
                          std::vector<unsigned>& chart, unsigned numCharts, double budget)
{
	if (numCharts <= 1 || budget <= 0.0)
		return numCharts;
	const HalfMesh& hm = s.hm;
	const Mesh& mesh = s.mesh;

	// Per-chart cone moments: W=Σw, s1=Σw·n, s2=Σw·n·nᵀ, plus size.
	std::vector<double> W(numCharts, 0.0), sz(numCharts, 0.0);
	std::vector<Normal> s1(numCharts, Normal::Zero());
	std::vector<Eigen::Matrix3d> s2(numCharts, Eigen::Matrix3d::Zero());
	for (FIndex f = 0; f < s.numFaces; ++f) {
		const unsigned c = chart[f];
		const double w = s.weights[f];
		const Normal& n = s.normals[f];
		W[c] += w;
		s1[c] += w * n;
		s2[c].noalias() += w * (n * n.transpose());
		sz[c] += 1.0;
	}

	// --- Anti-fold cap: a high-curvature (cone/saddle) vertex must NEVER become
	// interior to a chart, or the chart folds when flattened. Such vertices are
	// few; for each we record the fine charts it touches and forbid any merge
	// that would unify all of them (making the vertex interior). ----------------
	std::vector<double> dabs;
	std::vector<unsigned> vtotal;
	ComputeVertexDefect(s, dabs, vtotal);
	const double K_max = static_cast<double>(params.developableMaxVertexDefect);
	std::vector<std::vector<unsigned>> hdFine; // per high-defect vertex: incident fine charts
	std::vector<std::unordered_set<unsigned>> hdAtRoot(numCharts); // root → hd-vertex indices
	{
		std::unordered_map<Mesh::VIndex, unsigned> hdIndex;
		for (FIndex f = 0; f < s.numFaces; ++f) {
			const unsigned c = chart[f];
			const Mesh::Face& face = mesh.faces[f];
			for (int i = 0; i < 3; ++i) {
				const Mesh::VIndex v = face[i];
				if (dabs[v] <= K_max)
					continue;
				auto [it, fresh] = hdIndex.try_emplace(v, static_cast<unsigned>(hdFine.size()));
				if (fresh)
					hdFine.emplace_back();
				hdFine[it->second].push_back(c);
				hdAtRoot[c].insert(it->second);
			}
		}
		for (std::vector<unsigned>& vc : hdFine) {
			std::sort(vc.begin(), vc.end());
			vc.erase(std::unique(vc.begin(), vc.end()), vc.end());
		}
	}
	// Chart adjacency across creases (block only true topological/material seams).
	// TopoNeighbor already encodes exactly that rule (mesh-border / non-manifold /
	// texblob), so reuse it instead of re-inlining the same checks.
	std::vector<std::unordered_set<unsigned>> adj(numCharts);
	for (FIndex f = 0; f < s.numFaces; ++f) {
		const unsigned c = chart[f];
		for (HIndex iHe : hm.FAdjacentHalfedges(f)) {
			const FIndex nb = s.TopoNeighbor(iHe);
			if (nb == NONE)
				continue;
			const unsigned d = chart[nb];
			if (d != c)
				adj[c].insert(d);
		}
	}

	std::vector<unsigned> parent(numCharts);
	std::iota(parent.begin(), parent.end(), 0u);
	std::function<unsigned(unsigned)> find = [&](unsigned x) {
		while (parent[x] != x) {
			parent[x] = parent[parent[x]];
			x = parent[x];
		}
		return x;
	};
	// per-area cone error of the chart formed by merging roots a and b.
	auto combinedError = [&](unsigned a, unsigned b) {
		const double Wab = W[a] + W[b];
		return ConeFitError(Wab, s1[a] + s1[b], s2[a] + s2[b]) / std::max(Wab, 1e-300);
	};
	auto doMerge = [&](unsigned a, unsigned b) -> unsigned {
		if (sz[a] > sz[b])
			std::swap(a, b);
		parent[a] = b;
		W[b] += W[a];
		s1[b] += s1[a];
		s2[b] += s2[a];
		sz[b] += sz[a];
		if (adj[a].size() > adj[b].size())
			adj[a].swap(adj[b]);
		for (unsigned n : adj[a]) {
			const unsigned nr = find(n);
			if (nr != b) {
				adj[b].insert(nr);
				adj[nr].insert(b);
			}
		}
		adj[a].clear();
		if (hdAtRoot[a].size() > hdAtRoot[b].size())
			hdAtRoot[a].swap(hdAtRoot[b]);
		for (unsigned idx : hdAtRoot[a])
			hdAtRoot[b].insert(idx);
		hdAtRoot[a].clear();
		return b;
	};

	// Would merging roots a,b make a high-curvature vertex interior (all its
	// incident charts ∈ {a,b}, both present)? Such a merge folds → forbid it.
	auto wouldEnclose = [&](unsigned a, unsigned b) {
		const unsigned small = hdAtRoot[a].size() <= hdAtRoot[b].size() ? a : b;
		for (unsigned idx : hdAtRoot[small]) {
			bool hasA = false, hasB = false, other = false;
			for (unsigned fc : hdFine[idx]) {
				const unsigned r = find(fc);
				if (r == a)
					hasA = true;
				else if (r == b)
					hasB = true;
				else {
					other = true;
					break;
				}
			}
			if (!other && hasA && hasB)
				return true;
		}
		return false;
	};

	using Entry = std::tuple<double, unsigned, unsigned>; // (cost, rootA, rootB)
	std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
	auto tryPush = [&](unsigned a, unsigned b) {
		a = find(a);
		b = find(b);
		if (a == b)
			return;
		const double e = combinedError(a, b);
		if (e <= budget && !wouldEnclose(a, b))
			pq.emplace(e, std::min(a, b), std::max(a, b));
	};
	for (unsigned c = 0; c < numCharts; ++c) {
		if (find(c) != c)
			continue;
		for (unsigned n : adj[c])
			tryPush(c, n);
	}
	while (!pq.empty()) {
		const auto [e, a, b] = pq.top();
		pq.pop();
		const unsigned ar = find(a), br = find(b);
		if (ar == br)
			continue;
		const double cur = combinedError(ar, br);
		if (cur > budget)
			continue;
		if (cur > e + 1e-12) { // stale (moments changed) → re-queue at true cost
			pq.emplace(cur, std::min(ar, br), std::max(ar, br));
			continue;
		}
		if (wouldEnclose(ar, br))
			continue; // would enclose a cone/saddle vertex → fold; skip
		const unsigned r = doMerge(ar, br);
		for (unsigned n : adj[r])
			tryPush(r, n);
	}

	for (FIndex f = 0; f < s.numFaces; ++f)
		chart[f] = find(chart[f]);
	return Compact(chart);
}

// -------------------------------------------------------------------------
// Per-vertex absolute Gaussian curvature (angle defect) and incident-face count.
// defect(v) = (2π or π on the mesh boundary) − Σ incident corner angles; a cone/
// saddle vertex has large |defect|, a flat or straight-fold vertex ~0.
// -------------------------------------------------------------------------
void ComputeVertexDefect(const SegmentState& s, std::vector<double>& dabs,
                         std::vector<unsigned>& vtotal)
{
	const Mesh& mesh = s.mesh;
	const std::size_t nv = mesh.vertices.size();
	std::vector<double> angsum(nv, 0.0);
	vtotal.assign(nv, 0u);
	const auto corner = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
		const double na = a.norm(), nb = b.norm();
		if (na < 1e-20 || nb < 1e-20)
			return 0.0;
		return std::acos(std::clamp(a.dot(b) / (na * nb), -1.0, 1.0));
	};
	for (FIndex f = 0; f < s.numFaces; ++f) {
		const Mesh::Face& face = mesh.faces[f];
		const Eigen::Vector3d p0 = s.Pos(face[0]);
		const Eigen::Vector3d p1 = s.Pos(face[1]);
		const Eigen::Vector3d p2 = s.Pos(face[2]);
		angsum[face[0]] += corner(p1 - p0, p2 - p0);
		angsum[face[1]] += corner(p0 - p1, p2 - p1);
		angsum[face[2]] += corner(p0 - p2, p1 - p2);
		for (int i = 0; i < 3; ++i)
			++vtotal[face[i]];
	}
	// Only a REFLEX boundary corner (incident interior angles summing toward / past
	// 2π, i.e. a local overlap) can fail injectivity when the chart is flattened; a
	// convex or straight border corner (angsum ≤ π on a flat patch, e.g. a 90° square
	// corner) always stays on the chart's UV boundary and unfolds isometrically, so it
	// must NOT be scored as a fold-risk (cone/saddle) vertex. The old |π − angsum| made
	// every convex border corner "high-defect", forcing the flood to strand its last
	// face and the merge to forbid rejoining — a permanent, unmergeable seam at each
	// corner. Flag boundary vertices ONLY at the injectivity limit itself (angsum >
	// 2π − small margin); interior vertices keep the standard |2π − angsum| defect, so
	// closed meshes (no boundary vertices) are byte-for-byte unchanged. RepairDevelopableFlips
	// still backstops any residual fold.
	constexpr double boundaryReflexMargin = 0.1; // rad below 2π: catch corners at the limit
	constexpr double boundaryFlag = 2.0 * M_PI; // sentinel magnitude: always > any sane K_max
	dabs.assign(nv, 0.0);
	for (std::size_t v = 0; v < nv; ++v) {
		if (s.hm.VIsBoundary(static_cast<Mesh::VIndex>(v)))
			dabs[v] = (angsum[v] > 2.0 * M_PI - boundaryReflexMargin) ? boundaryFlag : 0.0;
		else
			dabs[v] = std::abs(2.0 * M_PI - angsum[v]);
	}
}

// ===========================================================================
// Cone-Lloyd developable segmentation (the scalable D-Charts core).
//
// A global, SIMULTANEOUS Lloyd relaxation on CONE proxies (D-Charts,
// Julius/Kraevoy/Sheffer 2005): a cone ⟨axis, half-angle⟩ fits the whole
// developable family (its fit error is λ_min of the chart's weighted normal
// covariance — zero for any developable surface), with an angle-defect anti-fold
// cap. The steps:
//
//   1. seed a few charts (one per topo-region + farthest-point extras),
//   2. CONE flood-assign every face simultaneously to the cheapest-fitting cone,
//   3. Lloyd-iterate: recompute each chart's best-fit cone, relocate its seed to
//      the most representative face, re-flood — until the labelling is stable,
//   4. BATCHED seeding: add, in ONE batch, a seed at the worst-fitting face of
//      every chart still above the cone-error budget (plus any cap-blocked face),
//      then re-Lloyd; repeat until growth stalls (the developable merge that
//      follows consolidates the fine pieces, so a coarse-but-developable partition
//      suffices).
//
// Batching is the key to scalability: a one-seed-at-a-time grow-and-re-Lloyd is
// O(F²); adding all over-budget seeds per round converges in O(log) rounds. Charts
// grow over TopoNeighbor edges (across creases), so noisy MVS normals cannot spawn
// charts — only true developability breaks (cone error) and cone/saddle vertices
// (angle defect) cut the surface. Returns the chart count.
// ===========================================================================

// Per-chart cone accumulators: moments W=Σw, s1=Σw·n, s2=Σw·n·nᵀ and the fitted
// cone, plus the per-area developability error helper.
struct ConeMoments
{
	std::vector<double> W;
	std::vector<Normal> s1;
	std::vector<Eigen::Matrix3d> s2;
	std::vector<Cone> cones;
	double error(unsigned c) const
	{
		return (W[c] > 0.0) ? ConeFitError(W[c], s1[c], s2[c]) / W[c] : 0.0;
	}
};

void RecomputeCones(const SegmentState& s, const std::vector<unsigned>& chart,
                    unsigned k, ConeMoments& m)
{
	m.W.assign(k, 0.0);
	m.s1.assign(k, Normal::Zero());
	m.s2.assign(k, Eigen::Matrix3d::Zero());
	for (FIndex f = 0; f < s.numFaces; ++f) {
		const unsigned c = chart[f];
		if (c == NONE || c >= k)
			continue;
		const double w = s.weights[f];
		const Normal& n = s.normals[f];
		m.W[c] += w;
		m.s1[c] += w * n;
		m.s2[c].noalias() += w * (n * n.transpose());
	}
	m.cones.assign(k, Cone{});
	for (unsigned c = 0; c < k; ++c)
		if (m.W[c] > 0.0)
			m.cones[c] = FitConeProxy(m.W[c], m.s1[c], m.s2[c]);
}

// Best-first flood by CONE fit cost over topo edges (crosses creases). A face is
// refused if assigning it would make a high-defect (cone/saddle) vertex interior
// to the chart — the hard anti-fold cap; such faces are left NONE for the caller
// to re-seed. Pass K_max < 0 to disable the cap.
void ConeFloodAssign(const SegmentState& s, const std::vector<FIndex>& seeds,
                     const std::vector<Cone>& cones, const std::vector<double>& dabs,
                     const std::vector<unsigned>& vtotal, double K_max,
                     std::vector<unsigned>& chart)
{
	const HalfMesh& hm = s.hm;
	const Mesh& mesh = s.mesh;
	const FIndex nf = s.numFaces;
	chart.assign(nf, NONE);
	const std::size_t nv = mesh.vertices.size();
	std::vector<unsigned> numAssigned(nv, 0u);
	std::vector<unsigned> vchart(nv, NONE);
	constexpr unsigned mixed = NONE - 1u;
	std::priority_queue<Candidate> pq; // min-heap (Candidate::operator< inverts)

	// D-Charts distance term: rank growth by (fit + fitFloor)·dist^β when β>0.
	// fitFloor keeps the distance active on EXACT-developable ties (fit == 0),
	// turning hop-Voronoi boundaries into geometric-Voronoi; it is far below the
	// cone-error budget (0.05 default) so it never outweighs a real fit signal.
	// β == 0 reproduces the fit-only ranking exactly (same value order, same
	// tie-breaks) — the off-switch guarantee.
	const double beta = static_cast<double>(s.params.developableDistanceExponent);
	constexpr double fitFloor = 1e-8;
	const auto growCost = [&](unsigned c, FIndex nb, double dist) {
		const double fit = cones[c].cost(s.normals[nb]);
		return beta > 0.0 ? (fit + fitFloor) * std::pow(dist, beta) : fit;
	};

	const auto enclosesHighDefect = [&](FIndex g, unsigned c) {
		if (K_max < 0.0)
			return false;
		const Mesh::Face& face = mesh.faces[g];
		for (int i = 0; i < 3; ++i) {
			const Mesh::VIndex v = face[i];
			if (numAssigned[v] + 1 == vtotal[v]) { // g is v's last incident face
				const bool consistent = (vchart[v] == NONE) || (vchart[v] == c);
				if (consistent && dabs[v] > K_max)
					return true; // would enclose a cone/saddle vertex
			}
		}
		return false;
	};
	const auto commit = [&](FIndex g, unsigned c, unsigned depth, double dist) {
		chart[g] = c;
		const Mesh::Face& face = mesh.faces[g];
		for (int i = 0; i < 3; ++i) {
			const Mesh::VIndex v = face[i];
			vchart[v] = (vchart[v] == NONE) ? c : (vchart[v] == c ? c : mixed);
			++numAssigned[v];
		}
		for (HIndex iHe : hm.FAdjacentHalfedges(g)) {
			const FIndex nb = s.TopoNeighbor(iHe);
			if (nb != NONE && chart[nb] == NONE) {
				const double d = beta > 0.0
				                     ? dist + (s.centroids[nb] - s.centroids[g]).norm()
				                     : 0.0;
				pq.push({growCost(c, nb, d), nb, c, depth + 1u, d});
			}
		}
	};

	// Seed each chart (a single face can never enclose a vertex → bypass the cap).
	for (unsigned c = 0; c < seeds.size(); ++c) {
		const FIndex sf = seeds[c];
		if (sf != NONE && chart[sf] == NONE)
			commit(sf, c, 0u, 0.0);
	}
	while (!pq.empty()) {
		const Candidate top = pq.top();
		pq.pop();
		if (chart[top.face] != NONE)
			continue; // already claimed (cheaper) — skip
		if (enclosesHighDefect(top.face, top.chart))
			continue; // would fold → leave NONE for the caller to re-seed
		commit(top.face, top.chart, top.depth, top.dist);
	}
}

// Final coverage guarantee: flood every still-NONE face into an adjacent assigned
// chart over topo edges (every topo-region has its seed assigned, so this covers
// everything). Keeps charts topo-connected.
void AssignLeftovers(const SegmentState& s, std::vector<unsigned>& chart)
{
	const HalfMesh& hm = s.hm;
	std::queue<FIndex> q;
	for (FIndex f = 0; f < s.numFaces; ++f)
		if (chart[f] != NONE)
			q.push(f);
	while (!q.empty()) {
		const FIndex f = q.front();
		q.pop();
		for (HIndex iHe : hm.FAdjacentHalfedges(f)) {
			const FIndex nb = s.TopoNeighbor(iHe);
			if (nb != NONE && chart[nb] == NONE) {
				chart[nb] = chart[f];
				q.push(nb);
			}
		}
	}
}

// One cone-Lloyd partition for a fixed seed set: cone flood + iterate (recompute
// cones, relocate seeds, re-flood) until the labelling is stable. Leaves the
// final cone moments in `m` for the caller's batched-seeding error queries.
void ConeLloyd(const SegmentState& s, std::vector<FIndex>& seeds,
               const std::vector<double>& dabs, const std::vector<unsigned>& vtotal,
               double K_max, unsigned maxIters, std::vector<unsigned>& chart,
               ConeMoments& m)
{
	const unsigned k = static_cast<unsigned>(seeds.size());
	// Warm-start: seeds are only ever APPENDED across batched-seeding rounds, so the
	// converged cones the previous round left in m.cones (via its final RecomputeCones)
	// stay index-aligned for charts [0, oldK). Reuse them — an established cylinder/
	// cone chart floods on its FITTED proxy from the first iteration instead of
	// cold-restarting to a degenerate plane and re-discovering its half-angle within
	// the fixed lloydIters, so the labelling stabilises (and the chart==prev early
	// exit fires) sooner. Only the newly appended seeds [oldK, k) start from their
	// seed-face normal (degenerate cone cosθ=1). On the first call m.cones is empty
	// (oldK=0), so every chart cold-starts.
	const unsigned oldK = std::min(k, static_cast<unsigned>(m.cones.size()));
	m.cones.resize(k);
	for (unsigned c = oldK; c < k; ++c) {
		m.cones[c] = Cone{};
		m.cones[c].axis = s.normals[seeds[c]];
		m.cones[c].cosTheta = 1.0;
	}
	ConeFloodAssign(s, seeds, m.cones, dabs, vtotal, K_max, chart);
	for (unsigned it = 0; it < maxIters; ++it) {
		RecomputeCones(s, chart, k, m);
		// relocate each seed to its chart's most cone-representative face.
		std::vector<double> best(k, std::numeric_limits<double>::max());
		std::vector<FIndex> bestf(k, NONE);
		for (FIndex f = 0; f < s.numFaces; ++f) {
			const unsigned c = chart[f];
			if (c == NONE || c >= k)
				continue;
			const double cost = m.cones[c].cost(s.normals[f]);
			if (cost < best[c]) {
				best[c] = cost;
				bestf[c] = f;
			}
		}
		std::vector<unsigned> prev = chart;
		for (unsigned c = 0; c < k; ++c)
			if (bestf[c] != NONE)
				seeds[c] = bestf[c];
		ConeFloodAssign(s, seeds, m.cones, dabs, vtotal, K_max, chart);
		if (chart == prev)
			break; // converged
	}
	RecomputeCones(s, chart, static_cast<unsigned>(seeds.size()), m);
}

unsigned ConeLloydSegment(const SegmentState& s, const ParametrizeParams& params,
                          std::vector<unsigned>& chart)
{
	const FIndex nf = s.numFaces;
	chart.assign(nf, NONE);
	if (nf == 0)
		return 0;

	std::vector<double> dabs;
	std::vector<unsigned> vtotal;
	ComputeVertexDefect(s, dabs, vtotal);
	const double F_max = static_cast<double>(params.developableMaxConeError);
	const double K_max = static_cast<double>(params.developableMaxVertexDefect);
	const unsigned lloydIters = std::max(1u, params.maxIterations);
	// The anti-fold (angle-defect) cap during the cone-Lloyd flood does double duty:
	// it keeps every fine chart flattenable (cone/saddle vertices stay on seams, so
	// the per-area cone budget alone — which a large folding chart can satisfy on
	// average — cannot pass a fold), AND faces it refuses are left NONE and re-seeded,
	// which seeds a fine partition near the curvature immediately (granularity the
	// developable merge then consolidates). Without it, a loose budget yields a few
	// huge charts that fold (flips). So the flood stays capped.

	// --- Initial seeds: one per topo-region (mandatory) + farthest-point extras.
	std::vector<unsigned> regionId;
	const unsigned numRegions = ComputeRegions(s, regionId);
	std::vector<FIndex> seeds(numRegions, NONE);
	std::vector<double> regionBest(numRegions, -1.0);
	for (FIndex f = 0; f < nf; ++f) {
		const unsigned r = regionId[f];
		if (s.weights[f] > regionBest[r]) {
			regionBest[r] = s.weights[f];
			seeds[r] = f;
		}
	}
	const unsigned extra = static_cast<unsigned>(
	    std::lround(static_cast<float>(numRegions) * params.seedExtraMult));
	AddFarthestSeeds(s, seeds, extra);

	ConeMoments m;
	ConeLloyd(s, seeds, dabs, vtotal, K_max, lloydIters, chart, m);

	// --- Batched developability seeding: split every chart still above the cone
	// budget at its worst-fitting face (all at once), re-Lloyd. Repeat until few
	// charts remain over budget (diminishing returns) — the developable merge that
	// follows consolidates the fine pieces back into the fewest large charts, so we
	// only need a fine partition whose boundaries follow developable structure, not
	// a fully-converged one. -----------------------------------------------------
	constexpr unsigned maxRounds = 64;
	for (unsigned round = 0; round < maxRounds; ++round) {
		const unsigned k = static_cast<unsigned>(seeds.size());
		std::vector<char> isSeed(nf, 0);
		for (FIndex sf : seeds)
			if (sf != NONE)
				isSeed[sf] = 1;
		// worst-fitting (highest cone cost) non-seed face per over-budget chart.
		std::vector<double> worst(k, -1.0);
		std::vector<FIndex> worstf(k, NONE);
		for (FIndex f = 0; f < nf; ++f) {
			const unsigned c = chart[f];
			if (c == NONE || c >= k || isSeed[f])
				continue;
			const double cost = m.cones[c].cost(s.normals[f]);
			if (cost > worst[c]) {
				worst[c] = cost;
				worstf[c] = f;
			}
		}
		std::vector<FIndex> add;
		for (unsigned c = 0; c < k; ++c)
			if (m.error(c) > F_max && worstf[c] != NONE)
				add.push_back(worstf[c]);
		// cap-blocked / unreached faces must be re-seeded (they cover real geometry).
		unsigned blocked = 0;
		for (FIndex f = 0; f < nf; ++f)
			if (chart[f] == NONE && !isSeed[f]) {
				add.push_back(f);
				++blocked;
			}
		// Diminishing returns: once each round adds few seeds relative to the chart
		// count, the fine partition has settled — more splitting won't lower the
		// final count (the developable merge dominates), it only churns the dynamic
		// split/reflood equilibrium the cap induces near curvature. Stop; the few
		// faces still stranded are absorbed by AssignLeftovers (the final merge's cap
		// keeps the consolidated charts flip-free regardless). -----------------------
		const unsigned stopThr = std::max(2u, k / 25u);
		(void)blocked;
		if (add.size() <= stopThr)
			break;
		// dedup within the batch (all are distinct non-seed faces already).
		std::sort(add.begin(), add.end());
		add.erase(std::unique(add.begin(), add.end()), add.end());
		unsigned added = 0;
		for (FIndex f : add)
			if (!isSeed[f]) {
				seeds.push_back(f);
				isSeed[f] = 1; // guard intra-batch dups (a face can't seed twice)
				++added;
			}
		if (added == 0)
			break;
#ifdef HM_ATLAS_DEBUG
		std::cerr << "[cone-lloyd] round " << round << " charts=" << k
		          << " +seeds=" << added << " (blocked=" << blocked << ")\n";
#endif
		ConeLloyd(s, seeds, dabs, vtotal, K_max, lloydIters, chart, m);
	}

	AssignLeftovers(s, chart);
	const unsigned nc = Compact(chart);
#ifdef HM_ATLAS_DEBUG
	std::cerr << "[cone-lloyd] final charts=" << nc << "\n";
#endif
	return nc;
}

// --- Flip/topology repair (the hard flip-free guarantee) -----------------
// Topo-connected components of a face set (so every repaired chart is connected —
// the partition EnforceConnectivity would produce, which the fold verdict assumes).
// `mark` is a scratch buffer sized to the face count, all-zero on entry and exit.
// Each returned component is sorted by global face id (for the ExtractOneChart match).
std::vector<std::vector<Mesh::FIndex>> ConnectedComponents(
    const SegmentState& s, const std::vector<Mesh::FIndex>& faces, std::vector<char>& mark)
{
	const HalfMesh& hm = s.hm;
	for (Mesh::FIndex f : faces)
		mark[f] = 1;
	std::vector<std::vector<Mesh::FIndex>> comps;
	std::vector<Mesh::FIndex> stack;
	for (Mesh::FIndex start : faces) {
		if (mark[start] != 1)
			continue;
		comps.emplace_back();
		mark[start] = 2;
		stack.push_back(start);
		while (!stack.empty()) {
			const Mesh::FIndex f = stack.back();
			stack.pop_back();
			comps.back().push_back(f);
			for (HIndex he : hm.FAdjacentHalfedges(f)) {
				const Mesh::FIndex nb = s.TopoNeighbor(he);
				if (nb != NONE && mark[nb] == 1) {
					mark[nb] = 2;
					stack.push_back(nb);
				}
			}
		}
	}
	for (Mesh::FIndex f : faces)
		mark[f] = 0;
	for (auto& comp : comps)
		std::sort(comp.begin(), comp.end());
	return comps;
}

// Spatially bisect a face list along its longest PCA axis (median split on face
// centroids in REAL geometry). Cutting by space — not normals — reliably turns an
// annulus/handle into disks and halves a curved chart's enclosed curvature, so a
// few rounds drive any folding chart to flip-free disk pieces.
void BisectFaces(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                 std::vector<Mesh::FIndex>& A, std::vector<Mesh::FIndex>& B)
{
	const std::size_t n = faces.size();
	std::vector<Vec3> cen(n);
	Vec3 mean = Vec3::Zero();
	for (std::size_t i = 0; i < n; ++i) {
		const Mesh::Face& F = mesh.faces[faces[i]];
		cen[i] = (mesh.vertices[F[0]].template cast<double>() + mesh.vertices[F[1]].template cast<double>() + mesh.vertices[F[2]].template cast<double>()) / 3.0;
		mean += cen[i];
	}
	mean /= static_cast<double>(n);
	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	for (const Vec3& c : cen) {
		const Vec3 d = c - mean;
		cov.noalias() += d * d.transpose();
	}
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
	const Vec3 axis = es.eigenvectors().col(2); // largest spread
	std::vector<double> proj(n);
	for (std::size_t i = 0; i < n; ++i)
		proj[i] = axis.dot(cen[i] - mean);
	std::vector<double> sorted = proj;
	std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
	const double med = sorted[n / 2];
	for (std::size_t i = 0; i < n; ++i)
		(proj[i] < med ? A : B).push_back(faces[i]);
	if (A.empty() || B.empty()) { // all-equal projection → split by index
		A.clear();
		B.clear();
		for (std::size_t i = 0; i < n; ++i)
			(i < n / 2 ? A : B).push_back(faces[i]);
	}
}

unsigned RepairDevelopableFlips(SegmentState& s, const ParametrizeParams& params,
                                std::vector<unsigned>& chart, unsigned numCharts,
                                detail::ChartFlattenCache* cache,
                                const std::vector<unsigned>* frontierIn = nullptr)
{
	const Mesh& mesh = s.mesh;

	// Incremental, scalable repair: start from the connected charts that will ship,
	// then process a work queue — flatten a chart, and if the predicate flags it (it
	// FOLDS, or — when params.developableMaxUvDistortion > 0 — is flip-free yet
	// OVER-STRETCHED past the symmetric-Dirichlet budget), spatially bisect it and
	// split each half into TOPO-connected components, re-queueing every piece.
	// A piece is strictly smaller than its parent (≥2 components per split), so a
	// chart shrinks until it neither folds nor over-distorts (a single non-degenerate
	// triangle flattens isometrically, sym-Dir = 4 < any budget ≥ 4; ≤2-face charts
	// never fold) — convergence is guaranteed (sliver-dominated charts are excluded by
	// the predicate's guard) without re-flattening the unchanged charts, so cost is
	// ~O(F·log) rather than rounds×all-charts. Components are kept sorted by global id
	// so the verdict matches what ParametrizeCharts (ExtractCharts, global order) ships.
	//
	// frontierIn restricts repair to the listed chart ids (precondition: the
	// caller guarantees `chart` is compact and per-chart connected — true after
	// DevelopableMerge, which only unions TopoNeighbor-adjacent charts and ends
	// with Compact). Skipping EnforceConnectivity keeps those ids stable so the
	// restriction is meaningful.
	if (frontierIn == nullptr)
		numCharts = EnforceConnectivity(s, chart);
	std::vector<std::vector<Mesh::FIndex>> fl(numCharts);
	for (Mesh::FIndex f = 0; f < s.numFaces; ++f)
		fl[chart[f]].push_back(f);

	// Process the split work as WAVES over a frontier of chart ids. Each frontier
	// chart's fold verdict is a pure read of chart-local state — detail::ChartFacesFold
	// takes a const Mesh& and builds its own local ChartMesh (the very call
	// ParametrizeCharts already runs concurrently across its own pool) — so evaluate the
	// WHOLE frontier in PARALLEL, then apply the bisections SERIALLY in fixed frontier
	// order. The old serial work queue was already strict level-order BFS (comps[0]
	// keeps id c and is re-queued to the back; new pieces appended after it), so walking
	// each frontier in index order reproduces the serial run's processing order AND id
	// assignment EXACTLY: faceChart is bitwise identical, only the per-wave flattening
	// (the dominant cost) is parallelised.
	std::vector<char> mark(s.numFaces, 0);
	const unsigned cap = 4u * static_cast<unsigned>(s.numFaces); // runaway backstop
	std::size_t splits = 0;
	std::vector<unsigned> frontier;
	if (frontierIn != nullptr)
		frontier = *frontierIn;
	else {
		frontier.resize(numCharts);
		std::iota(frontier.begin(), frontier.end(), 0u);
	}
	BS::light_thread_pool pool;
	bool capped = false;
	while (!frontier.empty() && !capped) {
		const std::size_t fn = frontier.size();
		// Parallel wave: flatten each frontier chart and record whether it folds. Output
		// slots folds[i]/slots[i] are disjoint and all reads are const, so there is no
		// data race; verdicts are a pure function of the face set, hence order-independent.
		// When a cache is supplied, a passing verdict also deposits its flatten artifacts
		// (cut ChartMesh + UVs) in this wave's slot for ParametrizeCharts to reuse.
		std::vector<char> folds(fn, 0);
		std::vector<detail::ChartFlattenSlot> slots(cache != nullptr ? fn : std::size_t{0});
		pool.detach_blocks(std::size_t{0}, fn, [&](std::size_t begin, std::size_t end) {
			for (std::size_t i = begin; i < end; ++i) {
				const unsigned c = frontier[i];
				if (fl[c].size() > 2 && detail::ChartFacesFold(mesh, fl[c], params, cache != nullptr ? &slots[i] : nullptr))
					folds[i] = 1; // ≤2 faces cannot fold → skip the flatten
			}
		});
		pool.wait();
		// Serial harvest in fixed frontier order: bisect every folder, split each half
		// into topo-connected components, re-queue the pieces for the next wave.
		// Accepted charts (their face set is now FINAL — they never re-enter a
		// frontier) hand their artifacts to the cache; folding charts left their slot
		// empty, and a capped wave simply drops unharvested slots (a cache miss
		// recomputes, never corrupts).
		std::vector<unsigned> next;
		for (std::size_t i = 0; i < fn; ++i) {
			if (!folds[i]) {
				if (cache != nullptr)
					cache->Store(std::move(slots[i]));
				continue;
			}
			const unsigned c = frontier[i];
			std::vector<Mesh::FIndex> A, B;
			BisectFaces(mesh, fl[c], A, B);
			if (A.empty() || B.empty())
				continue;
			std::vector<std::vector<Mesh::FIndex>> comps = ConnectedComponents(s, A, mark);
			for (auto& cb : ConnectedComponents(s, B, mark))
				comps.push_back(std::move(cb));
			if (comps.size() <= 1)
				continue; // no real split (degenerate) — avoid an infinite loop
			fl[c] = std::move(comps[0]); // first piece keeps id c (faces already labelled c)
			next.push_back(c);
			for (std::size_t j = 1; j < comps.size(); ++j) {
				const unsigned nid = numCharts++;
				for (Mesh::FIndex f : comps[j])
					chart[f] = nid;
				fl.push_back(std::move(comps[j]));
				next.push_back(nid);
			}
			if (++splits, numCharts > cap) {
				capped = true;
				break;
			}
		}
		frontier.swap(next);
	}
#ifdef HM_ATLAS_DEBUG
	std::cerr << "[flip-repair] splits=" << splits << " charts=" << numCharts << "\n";
#else
	(void)splits;
#endif
	return Compact(chart);
}

} // namespace

// ===========================================================================
// Public entry point.
// ===========================================================================
unsigned SegmentCharts(Mesh& mesh, const ParametrizeParams& params,
                       std::vector<unsigned>& faceChart)
{
	return detail::SegmentCharts(mesh, params, faceChart, nullptr);
}

namespace detail {

// Cache-aware SegmentCharts (ChartFlattenCache.h): identical output to the public
// overload; when `cache` is non-null the flip-repair additionally deposits each
// accepted (shipping) chart's flatten artifacts for ParametrizeCharts to reuse.
unsigned SegmentCharts(Mesh& mesh, const ParametrizeParams& params,
                       std::vector<unsigned>& faceChart, ChartFlattenCache* cache)
{
	faceChart.clear();
	if (mesh.faces.empty())
		return 0;

	// Ensure adjacency + normals are available.
	if (mesh.halfMesh.Empty() || mesh.halfMesh.FSize() != mesh.faces.size())
		mesh.ListHalfEdges();
	if (mesh.faceNormals.size() != mesh.faces.size())
		mesh.ComputeFaceNormals();

	SegmentState s(mesh, params);
	Precompute(s);

	// Three stages (full method documented atop halfmesh/AtlasCharting.h):
	//   1. cone-Lloyd        — the fewest developable charts directly (global
	//                          simultaneous growth + seed relocation + batched
	//                          developability seeding),
	//   2. developable merge — stitch residual seam pieces while the combined cone
	//                          error stays in budget (angle-defect cap protects),
	//   3. flip repair        — flatten every chart and bisect any that folds, the
	//                          hard flip-free guarantee.
	// Connectivity is enforced over TOPO edges (charts span creases when the surface
	// is developable across them), so the result is robust to noisy MVS normals.
	std::vector<unsigned> chart;
	unsigned numCharts = ConeLloydSegment(s, params, chart);
	numCharts = EnforceConnectivity(s, chart);
	numCharts = DevelopableMerge(s, params, chart, numCharts,
	                             static_cast<double>(params.developableMaxConeError));
	numCharts = EnforceConnectivity(s, chart);
	if (params.developableFlipRepairRounds > 0) {
		numCharts = RepairDevelopableFlips(s, params, chart, numCharts, cache);
		// Post-repair re-merge: recombine the bisection fragments the repair
		// left behind (nothing else ever merges again), then repair ONLY the
		// merged charts — a merge that re-folds is split right back, so the
		// fold-free guarantee is preserved and a round can never regress.
		for (unsigned round = 0; round < params.postRepairMergeRounds; ++round) {
			const unsigned before = numCharts;
			std::vector<unsigned> pre(chart);
			numCharts = DevelopableMerge(s, params, chart, numCharts,
			                             static_cast<double>(params.developableMaxConeError));
			// Charts containing faces from ≥2 pre-merge charts are the merged
			// ("dirty") ones — the only ones whose fold verdict changed.
			std::vector<unsigned> firstPre(numCharts, NONE);
			std::vector<char> dirtyFlag(numCharts, 0);
			for (FIndex f = 0; f < s.numFaces; ++f) {
				const unsigned c = chart[f];
				if (firstPre[c] == NONE)
					firstPre[c] = pre[f];
				else if (firstPre[c] != pre[f])
					dirtyFlag[c] = 1;
			}
			std::vector<unsigned> dirty;
			for (unsigned c = 0; c < numCharts; ++c)
				if (dirtyFlag[c])
					dirty.push_back(c);
			if (dirty.empty())
				break; // nothing merged — converged
			numCharts = RepairDevelopableFlips(s, params, chart, numCharts, cache, &dirty);
#ifdef HM_ATLAS_DEBUG
			std::cerr << "[re-merge] round " << round << ": " << before << " -> " << numCharts << " charts\n";
#endif
			if (before - numCharts < before / 100)
				break; // <1% net change — not worth another round
		}
	}

	faceChart.assign(chart.begin(), chart.end());
	return numCharts;
}

// Test-only seam (not in any public header — declared by AtlasTest with an extern
// forward declaration): return the initial cone-Lloyd seed set (one seed per topo
// region + the farthest-point extras) so tests can verify farthest-point seeding
// spreads the extra seeds across the surface instead of clustering them. Mirrors the
// seed setup in ConeLloydSegment exactly.
std::vector<Mesh::FIndex> ComputeSegmentationSeeds(Mesh& mesh, const ParametrizeParams& params)
{
	if (mesh.halfMesh.Empty() || mesh.halfMesh.FSize() != mesh.faces.size())
		mesh.ListHalfEdges();
	if (mesh.faceNormals.size() != mesh.faces.size())
		mesh.ComputeFaceNormals();
	SegmentState s(mesh, params);
	Precompute(s);
	std::vector<unsigned> regionId;
	const unsigned numRegions = ComputeRegions(s, regionId);
	std::vector<Mesh::FIndex> seeds(numRegions, NONE);
	std::vector<double> regionBest(numRegions, -1.0);
	for (Mesh::FIndex f = 0; f < s.numFaces; ++f) {
		const unsigned r = regionId[f];
		if (s.weights[f] > regionBest[r]) {
			regionBest[r] = s.weights[f];
			seeds[r] = f;
		}
	}
	const unsigned extra = static_cast<unsigned>(
	    std::lround(static_cast<float>(numRegions) * params.seedExtraMult));
	AddFarthestSeeds(s, seeds, extra);
	return seeds;
}
} // namespace detail

namespace {

using TexCoord = Mesh::TexCoord;
using FIndex = Mesh::FIndex;

// Signed 2-D area of a triangle × 2 (can be negative; caller takes abs).
inline float SignedDoubleArea2D(const TexCoord& a,
                                const TexCoord& b,
                                const TexCoord& c)
{
	return (b.x() - a.x()) * (c.y() - a.y())
	       - (c.x() - a.x()) * (b.y() - a.y());
}

// Per-chart aggregate: 3-D world area and 2-D UV area.
struct ChartStats
{
	double worldArea = 0.0;
	double uvArea = 0.0;
	// UV bounding box (to translate bbox-min to origin).
	float uvMinX = std::numeric_limits<float>::max();
	float uvMinY = std::numeric_limits<float>::max();
};

} // anonymous namespace

// ---------------------------------------------------------------------------
float NormalizeChartDensity(Mesh& mesh,
                            const std::vector<unsigned>& faceChart,
                            unsigned numCharts,
                            const AtlasParams& params)
{
	const size_t nf = mesh.faces.size();
	if (nf == 0 || numCharts == 0)
		return 0.f;

	assert(faceChart.size() == nf && "face_chart must have one entry per face");
	assert(mesh.faceTexcoords.size() == nf * 3 && "faceTexcoords must be populated (run ParametrizeCharts first)");

	// -----------------------------------------------------------------------
	// Pass 1: accumulate per-chart world area, UV area, and UV bbox min.
	// -----------------------------------------------------------------------
	std::vector<ChartStats> stats(numCharts);

	for (size_t fi = 0; fi < nf; ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;

		ChartStats& cs = stats[cid];

		const Mesh::Face& face = mesh.faces[fi];
		const Mesh::Vertex& v0 = mesh.vertices[face[0]];
		const Mesh::Vertex& v1 = mesh.vertices[face[1]];
		const Mesh::Vertex& v2 = mesh.vertices[face[2]];

		// 3-D double-area (cross product norm).
		const double da3d = (v1 - v0).cross(v2 - v0).norm();
		cs.worldArea += 0.5 * da3d;

		// 2-D UV double-area (absolute value of signed area).
		const TexCoord& t0 = mesh.faceTexcoords[fi * 3 + 0];
		const TexCoord& t1 = mesh.faceTexcoords[fi * 3 + 1];
		const TexCoord& t2 = mesh.faceTexcoords[fi * 3 + 2];
		const float da2d = std::abs(SignedDoubleArea2D(t0, t1, t2));
		cs.uvArea += 0.5 * static_cast<double>(da2d);

		// Accumulate UV bbox min.
		cs.uvMinX = std::min({cs.uvMinX, t0.x(), t1.x(), t2.x()});
		cs.uvMinY = std::min({cs.uvMinY, t0.y(), t1.y(), t2.y()});
	}

	// -----------------------------------------------------------------------
	// Derive global density D (texels per world unit).
	// -----------------------------------------------------------------------
	float density = params.texelsPerUnit;

	if (density <= 0.f) {
		// Auto: D = resolution / sqrt(Σ world_area_c).
		double totalWorld = 0.0;
		for (unsigned c = 0; c < numCharts; ++c)
			totalWorld += stats[c].worldArea;

		if (totalWorld <= 0.0)
			return 0.f;

		density = static_cast<float>(params.resolution) / static_cast<float>(std::sqrt(totalWorld));
	}

	if (density <= 0.f)
		return 0.f;

	// -----------------------------------------------------------------------
	// Pass 2: per-chart scale + translate UV bbox-min to origin.
	// -----------------------------------------------------------------------
	// Pre-compute per-chart scale factors.
	std::vector<float> scale(numCharts, 0.f);
	// A flip-free but severely area-compressed chart (uvArea tiny yet positive)
	// would get an unbounded scale here, blowing up its own UV bbox; PackAtlas's
	// fitToResolution global k (solved from sum w*h) is then dominated by that
	// one chart and every OTHER chart collapses to sub-texel size with no
	// error. Beyond maxScaleMagnitude treat
	// the chart like the exactly-degenerate case: leave scale 0 so the s==0 skip
	// below keeps its raw flip-free UVs (PackAtlas derives each chart's local
	// bbox independently, so an unnormalized chart cannot corrupt its siblings).
	constexpr double maxScaleMagnitude = 1e4;
	for (unsigned c = 0; c < numCharts; ++c) {
		const ChartStats& cs = stats[c];
		if (cs.uvArea <= 0.0 || cs.worldArea <= 0.0)
			continue;

		// scale_c = D * sqrt(world_area_c / uv_area_c)
		const double mag = std::sqrt(cs.worldArea / cs.uvArea);
		if (mag > maxScaleMagnitude)
			continue;
		scale[c] = density * static_cast<float>(mag);
	}

	// Apply scale and translation (bbox-min to origin) in one pass.
	for (size_t fi = 0; fi < nf; ++fi) {
		const unsigned cid = faceChart[fi];
		if (cid >= numCharts)
			continue;

		const float s = scale[cid];
		const float tx = stats[cid].uvMinX;
		const float ty = stats[cid].uvMinY;

		if (s == 0.f)
			continue; // degenerate chart — leave UVs as-is.

		for (int k = 0; k < 3; ++k) {
			TexCoord& uv = mesh.faceTexcoords[fi * 3 + k];
			// Scale then translate so bbox-min maps to origin.
			uv.x() = (uv.x() - tx) * s;
			uv.y() = (uv.y() - ty) * s;
		}
	}

	return density;
}

// ---------------------------------------------------------------------------
// Full pipeline: segment + flatten + normalize + pack.
// ---------------------------------------------------------------------------
AtlasResult GenerateAtlas(Mesh& mesh,
                          const ParametrizeParams& pparams,
                          const AtlasParams& aparams)
{
	// Segment + flatten share one flatten cache: the flip-repair's final accepting
	// verdict already flattened every shipping chart (fully on the
	// developableMaxUvDistortion > 0 path, init-only otherwise), so
	// ParametrizeCharts resumes from those artifacts instead of recomputing.
	// Output is byte-identical for any cache state (a miss recomputes).
	std::vector<unsigned> faceChart;
	detail::ChartFlattenCache flattenCache;
	const unsigned numCharts = detail::SegmentCharts(mesh, pparams, faceChart, &flattenCache);
	if (numCharts == 0)
		return AtlasResult{};

	detail::ParametrizeCharts(mesh, faceChart, numCharts, pparams, &flattenCache);
	NormalizeChartDensity(mesh, faceChart, numCharts, aparams);
	// When no explicit density is requested (texelsPerUnit == 0), target a
	// single atlas of the requested resolution (like xatlas) by enabling
	// fit-to-resolution packing. When the caller DID request a density, honor
	// their fitToResolution flag so they can preserve that density and overflow
	// into multiple pages (texelsPerUnit > 0, fitToResolution = false).
	AtlasParams packParams = aparams;
	if (aparams.texelsPerUnit == 0.f)
		packParams.fitToResolution = true;
	return PackAtlas(mesh, faceChart, numCharts, packParams);
}

} // namespace halfmesh
