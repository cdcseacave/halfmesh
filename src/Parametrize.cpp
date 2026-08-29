/*
* Parametrize.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// src/Parametrize.cpp — per-chart UV flattening (Module B of the atlas pipeline).
//
// Chart SEGMENTATION (Module A) + density (C) + GenerateAtlas live in
// src/AtlasCharting.cpp, packing (D) in src/AtlasPacking.cpp; this file flattens
// each chart the segmenter produces into a 2-D UV layout and writes them into
// mesh.faceTexcoords (each chart in its own local frame). Per chart:
//   1. Extract a local submesh (unique chart vertices + reindexed faces) and its
//      boundary loop (the longest loop of chart-border edges).
//   2. Initialize: LSCM (free-boundary least-squares conformal) when flip-free,
//      else a convex-boundary Tutte map (guaranteed injective on a disk), else a
//      PCA planar projection for a closed/degenerate chart.
//   3. Local/global refinement (ARAP / SLIM) on a CONSTANT cotangent Laplacian
//      factored once with SimplicialLDLT; SLIM adds a flip-free line search.
//
// One bridge for Module A: halfmesh::detail::ChartFacesFold flattens a chart and
// reports whether it folds — the predicate the segmentation flip-repair drives on.
// Correctness is verified by properties in tests/ParametrizeTest.cpp and tests/.

#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Log.h> // REPORT_WARNING (kept-folded-chart report)
#include <halfmesh/Util/Raster.h> // ChartUVSelfOverlaps coarse-grid injectivity probe

#include "ChartFlattenCache.h" // flip-repair -> ParametrizeCharts flatten reuse

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <BS_thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint> // int32_t (ChartUVSelfOverlaps' collector first-owner buffer)
#include <functional> // std::function (cut-to-disk union-find)
#include <limits> // std::numeric_limits (cut-to-disk Dijkstra)
#include <numeric> // std::iota (cut-to-disk union-find)
#include <queue> // std::queue (cut-to-disk BFS)
#include <unordered_map>
#include <unordered_set> // cut edge / boundary-vertex sets
#include <vector>
#ifdef HM_ATLAS_DEBUG
	#include <cstdio> // std::fprintf for the flatten diagnostic summary (HALFMESH_ATLAS_DEBUG only)
#endif

namespace halfmesh {

// ===========================================================================
// Module B — per-chart flattening.
//
// Per chart:
//   1. Extract a local submesh (unique chart vertices + reindexed faces).
//   2. Build the chart boundary loop (the longest cycle of border edges) and
//      pin it onto a convex circle (Tutte boundary).
//   3. Tutte init: solve the cotangent Laplacian for the interior UVs →
//      an injective starting map on the convex boundary.
//   4. Local/global refinement (ARAP / SLIM) on a CONSTANT cotangent
//      Laplacian factored once with SimplicialLDLT.
//
// All UVs are written back to mesh.faceTexcoords in each chart's own local
// UV frame (no cross-chart packing/scale — that is Tasks 15-16).
// ===========================================================================

namespace {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;

// -------------------------------------------------------------------------
// A single chart extracted into a self-contained local triangle mesh.
//   - verts:   local vertex positions (3D), one per local vertex id.
//   - faces:   local triangles (indices into verts).
//   - globalVid: local→global mesh vertex id (for back-reference).
//   - globalFid: local→global mesh face id (for writing texcoords).
//   - boundary: ordered local vertex ids of the chart boundary loop.
// -------------------------------------------------------------------------
struct ChartMesh
{
	std::vector<Vec3> verts;
	std::vector<Eigen::Vector3i> faces;
	std::vector<Mesh::VIndex> globalVid;
	std::vector<Mesh::FIndex> globalFid;
	std::vector<int> boundary; // ordered local vertex ids (loop)
};

// Signed area of a 2D triangle (positive = CCW). Doubled.
inline double SignedArea2x(const Vec2& a, const Vec2& b, const Vec2& c)
{
	return (b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y());
}

// Cotangent of the angle at vertex `o` in triangle (o, a, b), clamped for
// numerical safety. cot = cos/sin = dot / |cross|.
inline double Cotangent(const Vec3& o, const Vec3& a, const Vec3& b)
{
	const Vec3 u = a - o, v = b - o;
	const double dot = u.dot(v);
	const double crs = u.cross(v).norm();
	if (crs < 1e-12)
		return 0.0; // degenerate corner → no contribution
	double c = dot / crs;
	// clamp to avoid blow-ups on near-degenerate triangles.
	return std::max(-1e4, std::min(1e4, c));
}

// -------------------------------------------------------------------------
// Closest 2x2 rotation to matrix M via SVD with reflection fix (Liu 2008).
// Returns R = U * diag(1, det(UV^T)) * V^T so det(R) = +1.
// -------------------------------------------------------------------------
inline Mat2 ClosestRotation(const Mat2& M)
{
	Eigen::JacobiSVD<Mat2> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Mat2 U = svd.matrixU();
	Mat2 V = svd.matrixV();
	Mat2 R = U * V.transpose();
	if (R.determinant() < 0.0) {
		// flip the smallest singular value's column of U.
		U.col(1) *= -1.0;
		R = U * V.transpose();
	}
	return R;
}

// -------------------------------------------------------------------------
// Extract every chart into its own local ChartMesh. Returns one ChartMesh per
// chart id in [0, numCharts). Charts with < 1 face are returned empty.
// -------------------------------------------------------------------------
std::vector<ChartMesh> ExtractCharts(const Mesh& mesh,
                                     const std::vector<unsigned>& faceChart,
                                     unsigned numCharts)
{
	std::vector<ChartMesh> charts(numCharts);
	// per-chart map: global vertex id -> local vertex id.
	std::vector<std::unordered_map<Mesh::VIndex, int>> g2l(numCharts);

	const Mesh::FIndex nf = static_cast<Mesh::FIndex>(mesh.faces.size());
	for (Mesh::FIndex f = 0; f < nf; ++f) {
		const unsigned c = faceChart[f];
		if (c >= numCharts)
			continue;
		ChartMesh& cm = charts[c];
		auto& map = g2l[c];
		Eigen::Vector3i lf;
		for (int k = 0; k < 3; ++k) {
			const Mesh::VIndex gv = mesh.faces[f][k];
			auto it = map.find(gv);
			int lv;
			if (it == map.end()) {
				lv = static_cast<int>(cm.verts.size());
				map.emplace(gv, lv);
				cm.verts.push_back(mesh.vertices[gv].template cast<double>());
				cm.globalVid.push_back(gv);
			} else {
				lv = it->second;
			}
			lf[k] = lv;
		}
		cm.faces.push_back(lf);
		cm.globalFid.push_back(f);
	}
	return charts;
}

// -------------------------------------------------------------------------
// Build the boundary loop of a chart submesh from its local faces. An edge is
// a boundary edge iff it is used by exactly one (chart-local) triangle. We then
// walk the directed boundary half-edges into loops and keep the LONGEST loop
// (the outer boundary). Fills cm.boundary with ordered local vertex ids and
// reports the total number of boundary loops via `numLoops` (a clean disk has
// exactly one — charts with more are multiply-connected and handled by the
// caller via a planar fallback, since Tutte/ARAP/SLIM assume a disk).
//
// Returns false if no boundary loop could be formed (e.g. a closed chart with
// no border — handled by the caller as a degenerate/skip case).
// -------------------------------------------------------------------------
bool BuildBoundaryLoop(ChartMesh& cm, int& numLoops)
{
	cm.boundary.clear();
	numLoops = 0;
	const int n = static_cast<int>(cm.verts.size());
	auto key = [](int a, int b) {
		return static_cast<long long>(a) * 2147483647LL + b;
	};
	// Collect all 3F directed edge keys and sort once. A directed half-edge a->b is
	// on the boundary iff its reverse b->a is absent (binary search) — the edge
	// belongs to a single triangle. This replaces ~6 hash-map operations per face
	// (and the per-flip-repair-probe rehash churn) with one sort + O(log F) lookups;
	// the successor / degree / visited maps become dense vectors (local ids are
	// compact). Output (cm.boundary, numLoops) is unchanged for disk / multi-loop
	// charts: every cycle is still found and the kept loop is canonicalized.
	std::vector<long long> dirkeys;
	dirkeys.reserve(cm.faces.size() * 3);
	for (const auto& f : cm.faces)
		for (int k = 0; k < 3; ++k)
			dirkeys.push_back(key(f[k], f[(k + 1) % 3]));
	std::sort(dirkeys.begin(), dirkeys.end());
	auto present = [&](long long kk) {
		return std::binary_search(dirkeys.begin(), dirkeys.end(), kk);
	};

	// Boundary successor map (dense) plus per-vertex boundary out/in degree for
	// PINCH detection (a vertex where the boundary touches itself → non-disk). At a
	// pinch (out-degree > 1) keep the SMALLEST successor so the result is a pure
	// function of geometry (the old hash map overwrote by unspecified iteration
	// order; pinched charts route to the fallback regardless — AllBoundaryLoops is
	// hardened the same way). Sorted keys are deduped so each directed edge counts once.
	std::vector<int> nxt(n, -1);
	std::vector<int> outdeg(n, 0), indeg(n, 0);
	for (size_t i = 0; i < dirkeys.size(); ++i) {
		if (i > 0 && dirkeys[i] == dirkeys[i - 1])
			continue; // dedup duplicate directed edges
		const long long kk = dirkeys[i];
		const int a = static_cast<int>(kk / 2147483647LL);
		const int b = static_cast<int>(kk % 2147483647LL);
		if (present(key(b, a)))
			continue; // interior edge (both directions present)
		if (nxt[a] == -1 || b < nxt[a])
			nxt[a] = b; // smallest-successor tie-break at a pinch
		++outdeg[a];
		++indeg[b];
	}
	int boundaryCount = 0;
	for (int a = 0; a < n; ++a)
		if (nxt[a] != -1)
			++boundaryCount;
	if (boundaryCount == 0)
		return false; // closed chart, no boundary
	bool pinched = false;
	for (int a = 0; a < n && !pinched; ++a)
		if (outdeg[a] > 1 || indeg[a] > 1)
			pinched = true;

	// Walk loops from SORTED start vertices (dense visited); count them and keep the
	// longest, canonicalized to start at its smallest vertex id (boundary[0] feeds
	// the LSCM pin / Tutte rotation gauge, so it must be geometry-determined).
	std::vector<char> visited(n, 0);
	std::vector<int> best;
	for (int s = 0; s < n; ++s) {
		if (nxt[s] == -1 || visited[s])
			continue;
		std::vector<int> loop;
		int cur = s;
		for (int guard = 0; guard <= boundaryCount; ++guard) {
			if (visited[cur])
				break;
			visited[cur] = 1;
			loop.push_back(cur);
			if (nxt[cur] == -1) {
				loop.clear();
				break;
			} // open chain (non-manifold) — skip
			cur = nxt[cur];
			if (cur == s)
				break; // closed loop
		}
		if (loop.empty())
			continue;
		++numLoops;
		std::rotate(loop.begin(),
		            std::min_element(loop.begin(), loop.end()), loop.end());
		if (loop.size() > best.size() || (loop.size() == best.size() && loop[0] < best[0]))
			best = loop;
	}
	if (best.size() < 3)
		return false;
	// A pinch makes the chart a non-disk; report >1 "loops" so the caller routes it
	// to the planar fallback (Tutte would otherwise fold at the pinch).
	if (pinched && numLoops < 2)
		numLoops = 2;
	cm.boundary = std::move(best);
	return true;
}

// -------------------------------------------------------------------------
// Tutte initialization. Pins the boundary loop onto the unit circle (arc-length
// parameterized) and solves the weighted Laplacian for the interior UVs. Uses
// STRICTLY POSITIVE mean-value weights (Floater 2003), so every interior vertex
// is a convex combination of its neighbors and the convex boundary → guaranteed
// injective (unlike cotangent weights, which can go negative on obtuse triangles
// and break the theorem). Positivity — not the exact weight — is what the theorem
// needs, so the shape-aware mean-value weights are a drop-in that starts elongated
// charts far closer to isometry than uniform weights.
//
// Returns false if the linear solve fails.
// -------------------------------------------------------------------------
bool TutteInit(const ChartMesh& cm, std::vector<Vec2>& uv, double aspect = 1.0,
               int startK = 0)
{
	const int n = static_cast<int>(cm.verts.size());
	uv.assign(n, Vec2::Zero());
	if (cm.boundary.size() < 3)
		return false;

	// Mark boundary vertices and pin them on the target convex boundary by arc
	// length: the unit circle by default, or an axis-aligned ellipse of the
	// requested aspect (semi-axes sqrt(aspect) x 1/sqrt(aspect), equal area).
	// The ellipse is strictly convex, so Tutte's injectivity theorem applies
	// unchanged, while an elongated chart maps to an elongated target at a
	// fraction of the circle's distortion (the circle forces a strip into a
	// disk). `startK` rotates the boundary so that vertex lands at parameter 0
	// (the +long-axis end) — callers pass the boundary vertex extreme along the
	// chart's principal axis so the strip's ends land on the ellipse's ends.
	std::vector<char> isBnd(n, 0);
	// Compute boundary perimeter (3D arc length) for proportional spacing.
	const int bn = static_cast<int>(cm.boundary.size());
	std::vector<double> seg(bn, 0.0);
	double perim = 0.0;
	for (int k = 0; k < bn; ++k) {
		const int a = cm.boundary[k];
		const int b = cm.boundary[(k + 1) % bn];
		seg[k] = (cm.verts[a] - cm.verts[b]).norm();
		perim += seg[k];
	}
	if (perim <= 0.0) {
		// fall back to uniform angular spacing.
		for (int k = 0; k < bn; ++k)
			seg[k] = 1.0;
		perim = static_cast<double>(bn);
	}
	if (aspect <= 1.0) {
		// exact legacy path: unit circle, angle proportional to 3D arc length
		double acc = 0.0;
		for (int k = 0; k < bn; ++k) {
			const int v = cm.boundary[k];
			const double t = 2.0 * M_PI * (acc / perim);
			uv[v] = Vec2(std::cos(t), std::sin(t));
			isBnd[v] = 1;
			acc += seg[k];
		}
	} else {
		// Ellipse of equal area: distribute boundary vertices so ELLIPSE arc
		// length (not parameter angle) is proportional to 3D arc length, via a
		// cumulative-arc lookup table.
		const double A = std::sqrt(aspect), B = 1.0 / std::sqrt(aspect);
		constexpr int LUT = 4096;
		std::vector<double> cum(LUT + 1, 0.0);
		Vec2 prev(A, 0.0);
		for (int i = 1; i <= LUT; ++i) {
			const double t = 2.0 * M_PI * i / LUT;
			const Vec2 p(A * std::cos(t), B * std::sin(t));
			cum[i] = cum[i - 1] + (p - prev).norm();
			prev = p;
		}
		const double eperim = cum[LUT];
		double acc = 0.0;
		int lo = 0;
		for (int k = 0; k < bn; ++k) {
			const int v = cm.boundary[(startK + k) % bn];
			const double s = eperim * (acc / perim);
			// cum is monotone and s is non-decreasing across k. Bound the scan at
			// LUT-1: acc sums seg[] in ROTATED order (from startK) while perim
			// summed the original order, so rounding can push the last s
			// marginally past cum[LUT] — without the bound the s1 read below
			// would be one past the end of cum.
			while (lo < LUT - 1 && cum[lo + 1] < s)
				++lo;
			const double s0 = cum[lo], s1 = cum[lo + 1];
			const double f = s1 > s0 ? (s - s0) / (s1 - s0) : 0.0;
			const double t = 2.0 * M_PI * (lo + f) / LUT;
			uv[v] = Vec2(A * std::cos(t), B * std::sin(t));
			isBnd[v] = 1;
			acc += seg[(startK + k) % bn];
		}
	}

	// Positive-weight (Tutte) barycentric map, interior rows only. Tutte's theorem
	// guarantees an injective (flip-free) embedding when every interior vertex is a
	// convex combination of its neighbors and the boundary is convex — which holds
	// for any STRICTLY POSITIVE weights. Cotangent weights can go negative on obtuse
	// triangles and break injectivity, so the INIT uses positive mean-value weights
	// (computed below; the subsequent ARAP/SLIM solve uses cotangent weights and
	// restores geometric fidelity). This makes the start flip-free even on difficult
	// charts (e.g. obtuse/thin triangles).
	auto wkey = [&](int a, int b) {
		if (a > b)
			std::swap(a, b);
		return static_cast<long long>(a) * 2147483647LL + b;
	};

	// adjacency (each undirected edge once) for assembling rows.
	std::vector<std::vector<int>> adj(n);
	{
		std::unordered_map<long long, char> seen;
		for (const auto& f : cm.faces) {
			for (int k = 0; k < 3; ++k) {
				int a = f[k], b = f[(k + 1) % 3];
				const long long kk = wkey(a, b);
				if (seen[kk])
					continue;
				seen[kk] = 1;
				adj[a].push_back(b);
				adj[b].push_back(a);
			}
		}
	}

	// Mean-value weights (Floater 2003): strictly positive, so Tutte's convex-
	// boundary injectivity theorem still applies, but shape-aware — a well-known
	// drop-in over uniform weights that starts elongated charts far closer to
	// isometry (uniform ignores geometry entirely). Per triangle, the corner angle
	// theta at i contributes tan(theta/2) to both edges of that corner; the directed
	// weight is w_{i,j} = (sum of tan(half-angle)) / |v_i - v_j|. We use the
	// SYMMETRIC edge average 0.5*(w_ij + w_ji) so the assembled matrix stays
	// symmetric SPD (SimplicialLDLT reads its lower triangle). Charts reach Tutte
	// only after LSCM rejected them (degenerate/sliver geometry), where a raw
	// mean-value weight can be non-finite (angle -> pi) or zero (coincident
	// vertices); such edges fall back to the uniform weight 1 (PARTIAL fix), keeping
	// the weights positive and the system finite rather than dropping to the
	// fold-prone PCA fallback.
	auto dirkey = [](int a, int b) { return static_cast<long long>(a) * 2147483647LL + b; };
	std::unordered_map<long long, double> tanacc; // directed sum of tan(half-angle)
	for (const auto& f : cm.faces) {
		for (int c = 0; c < 3; ++c) {
			const int i = f[c], j = f[(c + 1) % 3], k = f[(c + 2) % 3];
			const Vec3 eij = cm.verts[j] - cm.verts[i];
			const Vec3 eik = cm.verts[k] - cm.verts[i];
			const double lij = eij.norm(), lik = eik.norm();
			if (lij < 1e-20 || lik < 1e-20)
				continue;
			// tan(theta/2) = |eij x eik| / (|eij||eik| + eij.eik); grows without
			// bound as theta -> pi (denominator -> 0), so clamp for robustness.
			const double denom = lij * lik + eij.dot(eik);
			const double crs = eij.cross(eik).norm();
			const double t = (denom > 1e-20) ? std::min(crs / denom, 1e6) : 1e6;
			tanacc[dirkey(i, j)] += t;
			tanacc[dirkey(i, k)] += t;
		}
	}
	auto mvWeight = [&](int i, int j) -> double {
		const double len = (cm.verts[i] - cm.verts[j]).norm();
		if (len < 1e-20)
			return 1.0;
		auto a = tanacc.find(dirkey(i, j));
		auto b = tanacc.find(dirkey(j, i));
		const double wij = (a != tanacc.end()) ? a->second / len : 0.0;
		const double wji = (b != tanacc.end()) ? b->second / len : 0.0;
		const double w = 0.5 * (wij + wji);
		return (std::isfinite(w) && w > 0.0) ? std::min(w, 1e8) : 1.0;
	};

	std::vector<Eigen::Triplet<double>> trips;
	trips.reserve(n * 7);
	Eigen::VectorXd bx = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd by = Eigen::VectorXd::Zero(n);
	for (int i = 0; i < n; ++i) {
		if (isBnd[i]) {
			trips.emplace_back(i, i, 1.0);
			bx[i] = uv[i].x();
			by[i] = uv[i].y();
			continue;
		}
		double diag = 0.0;
		for (int j : adj[i]) {
			const double wij = mvWeight(i, j); // symmetric mean-value weight (> 0)
			diag += wij;
			if (isBnd[j]) {
				bx[i] += wij * uv[j].x();
				by[i] += wij * uv[j].y();
			} else {
				trips.emplace_back(i, j, -wij);
			}
		}
		if (diag <= 0.0)
			diag = 1.0;
		trips.emplace_back(i, i, diag);
	}

	Eigen::SparseMatrix<double> A(n, n);
	A.setFromTriplets(trips.begin(), trips.end());
	A.makeCompressed();
	// The assembled matrix is symmetric (interior-interior couplings are mirrored;
	// boundary rows are identity with their couplings moved to the RHS) and SPD on
	// a connected disk, so factor it with sparse Cholesky (SimplicialLDLT reads the
	// lower triangle of the symmetric matrix) — matching every other solve in this
	// file — rather than the general SparseLU (~2x flops/memory, ignores symmetry).
	// Tutte's flip-free guarantee is theorem-based (convex boundary + positive
	// weights), independent of the solver; the info()/finite checks below preserve
	// the return-false fallback. Both RHS solve together as one n x 2 system.
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
		return false;
	Eigen::Matrix<double, Eigen::Dynamic, 2> b(n, 2);
	b.col(0) = bx;
	b.col(1) = by;
	const Eigen::Matrix<double, Eigen::Dynamic, 2> s = solver.solve(b);
	if (solver.info() != Eigen::Success)
		return false;
	for (int i = 0; i < n; ++i) {
		if (isBnd[i])
			continue;
		uv[i] = Vec2(s(i, 0), s(i, 1));
	}
	for (int i = 0; i < n; ++i)
		if (!std::isfinite(uv[i].x()) || !std::isfinite(uv[i].y()))
			return false;
	return true;
}

// -------------------------------------------------------------------------
// LSCM (Least-Squares Conformal Map) initialization (Lévy 2002 / Mullen 2008
// "Spectral Conformal" energy form E = E_Dirichlet - Area). A free-boundary
// conformal map: only two vertices are pinned (to fix the 4-DOF similarity
// gauge), so the boundary is free to take its natural near-isometric shape —
// unlike Tutte, which forces it onto a circle. The converged SLIM/ARAP energy
// is dramatically lower from this start on elongated / non-round charts.
//
// System (variables z = [u_0..u_{n-1}, v_0..v_{n-1}]):
//   minimize  (1/2) z^T ( diag(L,L) - 2*Area ) z      (L = cotan Laplacian)
// pinned at two boundary vertices → solve the reduced free system.
// Returns false (caller falls back to Tutte) on solve failure / non-finite.
// -------------------------------------------------------------------------
bool LscmInit(const ChartMesh& cm, std::vector<Vec2>& uv)
{
	const int n = static_cast<int>(cm.verts.size());
	const int nt = static_cast<int>(cm.faces.size());
	if (n < 3 || nt < 1 || cm.boundary.size() < 3)
		return false;
	const int N = 2 * n; // unknowns: [u_0..u_{n-1}, v_0..v_{n-1}]

	// Lévy least-squares conformal energy: per triangle, the discrete
	// Cauchy-Riemann residual is a linear form in the 3 complex vertex coords.
	// Stacking all triangles gives M (2*nt × 2n); we minimize ||M z||², whose
	// normal equations M^T M are GUARANTEED SPD (a Gram matrix) — so the solve
	// always finds the true conformal minimum, even on obtuse/sliver triangles
	// where the equivalent E_Dirichlet−Area form goes indefinite.
	// Pre-scan the largest doubled triangle area for a RELATIVE degeneracy
	// threshold: an isolated zero-area sliver (routine on MVS/photogrammetry input)
	// must NOT abort the whole chart's conformal map (which then falls to a badly
	// distorting Tutte-circle init), and an absolute 1e-20 cutoff is scale-
	// dependent. Below, a degenerate triangle is SKIPPED rather than aborting: its
	// two rows are simply omitted — zero rows are harmless in the M^T M normal
	// equations, and a genuinely rank-deficient result still fails the LDLT and
	// falls back to Tutte.
	double maxDT = 0.0;
	for (const auto& f : cm.faces) {
		const Vec3 e1 = cm.verts[f[1]] - cm.verts[f[0]];
		const double l1 = e1.norm();
		if (l1 < 1e-30)
			continue;
		const Vec3 e2 = cm.verts[f[2]] - cm.verts[f[0]];
		const double x2 = e2.dot(e1 / l1);
		const double y2 = (e2 - (e1 / l1) * x2).norm();
		maxDT = std::max(maxDT, l1 * y2);
	}
	const double degenDT = maxDT * 1e-14;

	std::vector<Eigen::Triplet<double>> mtrips;
	mtrips.reserve(static_cast<size_t>(nt) * 12);
	int row = 0;
	for (const auto& f : cm.faces) {
		const Vec3& p0 = cm.verts[f[0]];
		const Vec3& p1 = cm.verts[f[1]];
		const Vec3& p2 = cm.verts[f[2]];
		// Isometric local 2D coords: corner 0 at origin, corner 1 on +x.
		const Vec3 e1 = p1 - p0;
		const double l1 = e1.norm();
		if (l1 < 1e-20)
			continue; // skip this degenerate triangle, not the whole chart
		const Vec3 ax = e1 / l1;
		const Vec3 e2 = p2 - p0;
		const double x2 = e2.dot(ax);
		const double y2 = (e2 - ax * x2).norm();
		const double dT = l1 * y2; // 2 * area
		if (dT <= degenDT)
			continue; // skip degenerate triangle (relative threshold)
		const double rw = 1.0 / std::sqrt(dT); // area weighting
		// W_j = local edge vectors (as complex Wx + i Wy):
		//   W0 = (x2 - l1, y2), W1 = (-x2, -y2), W2 = (l1, 0)
		const double Wx[3] = {x2 - l1, -x2, l1};
		const double Wy[3] = {y2, -y2, 0.0};
		const int rRe = row++;
		const int rIm = row++;
		for (int j = 0; j < 3; ++j) {
			const int vj = f[j];
			// Re(Σ W_j z_j) = Σ (Wx u - Wy v)
			mtrips.emplace_back(rRe, vj, rw * Wx[j]);
			mtrips.emplace_back(rRe, n + vj, -rw * Wy[j]);
			// Im(Σ W_j z_j) = Σ (Wy u + Wx v)
			mtrips.emplace_back(rIm, vj, rw * Wy[j]);
			mtrips.emplace_back(rIm, n + vj, rw * Wx[j]);
		}
	}
	Eigen::SparseMatrix<double> M(2 * nt, N);
	M.setFromTriplets(mtrips.begin(), mtrips.end());
	const Eigen::SparseMatrix<double> A = (M.transpose() * M).pruned();

	// Pin two boundary vertices far apart to (0,0) and (1,0) (fixes the 4-DOF
	// similarity gauge); solve the reduced SPD system A_ff z_f = -A_fc z_c.
	// Use an approximate geometric diameter pair (2-sweep farthest-point):
	// index-based pins (boundary[0], boundary[bn/2]) can land spatially close
	// on serpentine or unevenly sampled boundaries, which imposes an artificial
	// scale/shear on the conformal map and ill-conditions the gauge.
	auto farthestFrom = [&](int v) {
		int best = v;
		double bestD = -1.0;
		for (int b : cm.boundary) {
			const double d = (cm.verts[b] - cm.verts[v]).squaredNorm();
			if (d > bestD) {
				bestD = d;
				best = b;
			}
		}
		return best;
	};
	const int p0 = farthestFrom(cm.boundary[0]);
	const int p1 = farthestFrom(p0);
	if (p0 == p1)
		return false; // all boundary points coincide — degenerate chart
	std::vector<int> fixed = {p0, n + p0, p1, n + p1};
	std::vector<double> fixedVal = {0.0, 0.0, 1.0, 0.0};
	std::vector<char> isFixed(N, 0);
	for (int f : fixed)
		isFixed[f] = 1;
	std::vector<int> full2free(N, -1);
	int nfree = 0;
	for (int i = 0; i < N; ++i)
		if (!isFixed[i])
			full2free[i] = nfree++;
	if (nfree == 0)
		return false;

	Eigen::VectorXd xc = Eigen::VectorXd::Zero(N);
	for (size_t i = 0; i < fixed.size(); ++i)
		xc[fixed[i]] = fixedVal[i];
	Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nfree);
	std::vector<Eigen::Triplet<double>> ftrips;
	ftrips.reserve(static_cast<size_t>(A.nonZeros()));
	for (int col = 0; col < A.outerSize(); ++col) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
			const int r = static_cast<int>(it.row());
			const int c = static_cast<int>(it.col());
			const double v = it.value();
			if (isFixed[r])
				continue;
			if (isFixed[c])
				rhs[full2free[r]] -= v * xc[c];
			else
				ftrips.emplace_back(full2free[r], full2free[c], v);
		}
	}
	Eigen::SparseMatrix<double> Aff(nfree, nfree);
	Aff.setFromTriplets(ftrips.begin(), ftrips.end());
	Aff.makeCompressed();
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(Aff);
	if (solver.info() != Eigen::Success)
		return false;
	const Eigen::VectorXd zf = solver.solve(rhs);
	if (solver.info() != Eigen::Success)
		return false;

	uv.assign(n, Vec2::Zero());
	for (int i = 0; i < n; ++i) {
		const double u = isFixed[i] ? xc[i] : zf[full2free[i]];
		const double v = isFixed[n + i] ? xc[n + i] : zf[full2free[n + i]];
		if (!std::isfinite(u) || !std::isfinite(v))
			return false;
		uv[i] = Vec2(u, v);
	}
	// Canonicalize orientation: the conformal solve is invariant to a global
	// mirror, which can come out as an all-CW (negative-area) layout. If the net
	// signed area is negative, mirror in u so the map is CCW (matches the 3D
	// winding) — otherwise the caller's flip check would reject a perfectly good
	// conformal map.
	double signedArea = 0.0;
	for (const auto& f : cm.faces)
		signedArea += SignedArea2x(uv[f[0]], uv[f[1]], uv[f[2]]);
	if (signedArea < 0.0)
		for (Vec2& p : uv)
			p.x() = -p.x();
	return true;
}

// -------------------------------------------------------------------------
// Per-triangle data precomputed once for the local/global solver: the source
// (3D) triangle isometrically embedded in 2D and its (constant) cotangent edge
// weights. The deformation gradient of the current UVs vs. this reference
// drives the local rotation/SVD step.
// -------------------------------------------------------------------------
struct TriRef
{
	Eigen::Matrix<double, 2, 3> x; // source 2D coords of the 3 corners (isometric)
	Mat2 invX; // inverse of the reference edge basis [x1-x0, x2-x0] (constant per tri)
	double cot[3]; // cotangent weight opposite each edge (i,i+1)
	double area; // source triangle area (for energy weighting)
	bool degenerate; // reference edge basis is singular (|det| < 1e-20)
};

// Embed a 3D triangle isometrically into 2D: corner 0 at origin, corner 1 on +x.
inline TriRef MakeTriRef(const Vec3& p0, const Vec3& p1, const Vec3& p2)
{
	TriRef t;
	const Vec3 e1 = p1 - p0;
	const double l1 = e1.norm();
	Vec3 xAxis = (l1 > 1e-20) ? Vec3(e1 / l1) : Vec3(1, 0, 0);
	const Vec3 e2 = p2 - p0;
	const Vec3 nrm = e1.cross(e2);
	Vec3 yAxis = nrm.cross(xAxis);
	const double ly = yAxis.norm();
	yAxis = (ly > 1e-20) ? Vec3(yAxis / ly) : Vec3(0, 1, 0);
	t.x.col(0) = Vec2(0.0, 0.0);
	t.x.col(1) = Vec2(l1, 0.0);
	t.x.col(2) = Vec2(e2.dot(xAxis), e2.dot(yAxis));
	// Reference edge basis and its inverse — CONSTANT per triangle (t.x never
	// changes), so precompute the inverse the deformation-gradient step reuses
	// dozens of times instead of rebuilding+inverting it on every TriJacobian call.
	// Computed identically once, so downstream Jacobians are bit-for-bit unchanged.
	Mat2 Xr;
	Xr.col(0) = t.x.col(1) - t.x.col(0);
	Xr.col(1) = t.x.col(2) - t.x.col(0);
	t.degenerate = (std::abs(Xr.determinant()) < 1e-20);
	t.invX = t.degenerate ? Mat2(Mat2::Identity()) : Mat2(Xr.inverse());
	// cotangent of angle at each corner (opposite the following edge), halved
	// later when assembling L.
	t.cot[0] = Cotangent(p0, p1, p2); // at corner 0
	t.cot[1] = Cotangent(p1, p2, p0); // at corner 1
	t.cot[2] = Cotangent(p2, p0, p1); // at corner 2
	t.area = 0.5 * nrm.norm();
	return t;
}

// Symmetric-Dirichlet energy of a 2x2 Jacobian J (singular values s0,s1):
//   E = s0^2 + s1^2 + 1/s0^2 + 1/s1^2  (== |J|_F^2 + |J^-1|_F^2).
// Evaluated WITHOUT an SVD via the identities s0^2+s1^2 = |J|_F^2 and
// s0^2*s1^2 = det(J)^2, so E = f + f/det(J)^2 with f = |J|_F^2 (sign-safe: det is
// squared). The det^2 floor stands in for a per-singular-value clamp on a
// collapsed map; accepted (flip-free) candidates always have det(J) > 0, so the
// floor only guards genuinely degenerate triangles (E huge). The degenerate
// reference case (TriJacobian returns Identity -> f=2, det=1 -> E=4) matches the
// singular-value form exactly.
// NOTE: a FULLY-collapsed J (all-zero) yields f=0 and det=0, so E = 0 + 0/floor = 0
// (energy 0, NOT +inf). Callers rely on the upstream flip / zero-area checks to
// reject such degenerate candidates BEFORE consuming this energy — this function
// does not itself flag an all-zero map as invalid.
inline double SymDirEnergyJ(const Mat2& J)
{
	const double f = J.squaredNorm();
	const double d = J.determinant();
	return f + f / std::max(d * d, 1e-24);
}

// -------------------------------------------------------------------------
// Local/global flattener (ARAP or SLIM) on a chart whose UVs are already
// Tutte-initialized. The cotangent Laplacian (the global matrix) is CONSTANT
// across iterations and factored ONCE with SimplicialLDLT.
//
// Gauge: we pin a SINGLE vertex to fix the translation nullspace. The
// cotangent-Laplacian system L u = b(R) is then rank-1 deficient only in the
// (already-removed) translation direction; the global rotation/scale is fixed
// by the per-triangle rotation targets R_t, so the solver is free to recover
// the true isometric layout (a two-pin gauge would nail two vertices to fixed
// relative positions and PREVENT isometry — that is intentionally avoided).
//
// ARAP: minimize  sum_t sum_edges cot_ij * |(u_i-u_j) - R_t (x_i-x_j)|^2
//   - Local:  R_t = closest rotation to the current deformation gradient.
//   - Global: solve L u = b(R) (cotangent Laplacian; b from rotated source).
//
// SLIM: same scaffold but each triangle's cotangent weights are scaled by a
// proximal symmetric-Dirichlet reweighting, the global solve gives a search
// direction, and a flip-preventing line search picks the step → flip-free.
// -------------------------------------------------------------------------
void LocalGlobal(const ChartMesh& cm, std::vector<Vec2>& uv,
                 ParametrizeParams::FlattenMethod method, unsigned iterations)
{
	const int n = static_cast<int>(cm.verts.size());
	const int nt = static_cast<int>(cm.faces.size());
	if (n < 3 || nt < 1)
		return;

	// Precompute per-triangle reference frames + cotangent weights.
	std::vector<TriRef> tri(nt);
	for (int t = 0; t < nt; ++t) {
		const auto& f = cm.faces[t];
		tri[t] = MakeTriRef(cm.verts[f[0]], cm.verts[f[1]], cm.verts[f[2]]);
	}

	// Single translation-gauge pin (a boundary vertex if available).
	const int pin0 = (!cm.boundary.empty()) ? cm.boundary[0] : 0;
	std::vector<char> pinned(n, 0);
	pinned[pin0] = 1;

	// Emit every contribution of the (weighted) cotangent Laplacian in a FIXED
	// order: off-diagonal pairs in triangle-corner order, then one pre-accumulated
	// diagonal per vertex (pinned rows are identity). Factoring the emission out
	// lets the triplet build, the SLIM slot-recording pass, and the SLIM in-place
	// re-value pass all walk contributions in the SAME order — the prerequisite for
	// the incremental re-value to reproduce setFromTriplets bit-for-bit (each slot
	// takes <=2 contributions on a manifold chart, and IEEE addition is commutative).
	auto EmitSystem = [&](const std::vector<double>& triScale, auto&& off, auto&& dia) {
		std::vector<double> diag(n, 0.0);
		for (int t = 0; t < nt; ++t) {
			const auto& f = cm.faces[t];
			const double s = triScale.empty() ? 1.0 : triScale[t];
			for (int k = 0; k < 3; ++k) {
				const int i = f[k], j = f[(k + 1) % 3];
				// edge (i,j) opposite corner (k+2): weight = cot at that corner.
				const double wij = 0.5 * tri[t].cot[(k + 2) % 3] * s;
				if (!pinned[i] && !pinned[j]) {
					off(i, j, -wij);
					off(j, i, -wij);
				}
				if (!pinned[i])
					diag[i] += wij;
				if (!pinned[j])
					diag[j] += wij;
			}
		}
		for (int i = 0; i < n; ++i) {
			if (pinned[i]) {
				dia(i, i, 1.0);
				continue;
			}
			double d = diag[i];
			if (!(d > 1e-12))
				d = 1e-12;
			dia(i, i, d);
		}
	};

	// Assemble the CONSTANT ARAP cotangent Laplacian from scratch via triplets
	// (one scalar system reused for both u and v components). Pinned rows identity.
	auto BuildSystem = [&](const std::vector<double>& triScale,
	                       Eigen::SparseMatrix<double>& A) {
		std::vector<Eigen::Triplet<double>> trips;
		trips.reserve(nt * 12 + n);
		auto push = [&](int i, int j, double v) { trips.emplace_back(i, j, v); };
		EmitSystem(triScale, push, push);
		A.resize(n, n);
		A.setFromTriplets(trips.begin(), trips.end());
		A.makeCompressed();
	};

	// valuePtr slot of entry (r,c) in a compressed column-major matrix (binary
	// search over the sorted inner indices of column c).
	auto SlotOf = [](const Eigen::SparseMatrix<double>& A, int r, int c) -> int {
		const int* outer = A.outerIndexPtr();
		const int* inner = A.innerIndexPtr();
		int lo = outer[c], hi = outer[c + 1];
		while (lo < hi) {
			const int mid = lo + ((hi - lo) >> 1);
			if (inner[mid] < r)
				lo = mid + 1;
			else
				hi = mid;
		}
		return lo; // inner[lo] == r
	};

	// Build RHS for the rotation-fit global step given per-triangle 2x2 targets
	// R_t and per-triangle weight scale.
	auto BuildRHS = [&](const std::vector<Mat2>& R,
	                    const std::vector<double>& triScale,
	                    Eigen::VectorXd& bx, Eigen::VectorXd& by) {
		bx = Eigen::VectorXd::Zero(n);
		by = Eigen::VectorXd::Zero(n);
		for (int t = 0; t < nt; ++t) {
			const auto& f = cm.faces[t];
			const double s = triScale.empty() ? 1.0 : triScale[t];
			for (int k = 0; k < 3; ++k) {
				const int i = f[k], j = f[(k + 1) % 3];
				const double wij = 0.5 * tri[t].cot[(k + 2) % 3] * s;
				const Vec2 xe = tri[t].x.col(k) - tri[t].x.col((k + 1) % 3);
				const Vec2 re = R[t] * xe; // target edge vector
				if (!pinned[i]) {
					bx[i] += wij * re.x();
					by[i] += wij * re.y();
				}
				if (!pinned[j]) {
					bx[j] -= wij * re.x();
					by[j] -= wij * re.y();
				}
			}
		}
		// move pinned contributions to RHS for free rows.
		for (int t = 0; t < nt; ++t) {
			const auto& f = cm.faces[t];
			const double s = triScale.empty() ? 1.0 : triScale[t];
			for (int k = 0; k < 3; ++k) {
				const int i = f[k], j = f[(k + 1) % 3];
				const double wij = 0.5 * tri[t].cot[(k + 2) % 3] * s;
				if (!pinned[i] && pinned[j]) {
					bx[i] += wij * uv[j].x();
					by[i] += wij * uv[j].y();
				}
				if (!pinned[j] && pinned[i]) {
					bx[j] += wij * uv[i].x();
					by[j] += wij * uv[i].y();
				}
			}
		}
		for (int i = 0; i < n; ++i) {
			if (pinned[i]) {
				bx[i] = uv[i].x();
				by[i] = uv[i].y();
			}
		}
	};

	// Current per-triangle deformation gradient J = U_uv * X_ref^{-1}, using the
	// reference inverse precomputed once in MakeTriRef (constant per triangle).
	auto TriJacobian = [&](int t, const std::vector<Vec2>& U) -> Mat2 {
		const auto& f = cm.faces[t];
		if (tri[t].degenerate)
			return Mat2::Identity();
		Mat2 Uu;
		Uu.col(0) = U[f[1]] - U[f[0]];
		Uu.col(1) = U[f[2]] - U[f[0]];
		return Uu * tri[t].invX;
	};

	// Relative sliver threshold (mirrors ChartOverDistorted / CountRealFlips): a
	// triangle whose SOURCE area is below meanArea*1e-6 is input-degenerate — its
	// UV orientation is unconstrained numerical noise (LscmInit skips its rows
	// entirely), so no flip guard can meaningfully constrain it.
	double totArea = 0.0;
	for (int t = 0; t < nt; ++t)
		totArea += tri[t].area;
	const double sliverArea = (totArea / std::max(1, nt)) * 1e-6;

	// Count flipped triangles (signed area sign mismatch with reference).
	// `exemptSlivers` skips input-degenerate triangles: the SLIM path passes true
	// (same sliver policy as LscmInit / CountRealFlips / the shipped-precision
	// guard below) — counting a noise-oriented zero-area sliver as a real flip
	// would veto EVERY line-search candidate and stall SLIM at the init map.
	// ARAP passes false: it holds every triangle to the stricter contract.
	auto CountFlips = [&](const std::vector<Vec2>& U, bool exemptSlivers) -> int {
		int flips = 0;
		for (int t = 0; t < nt; ++t) {
			if (exemptSlivers && tri[t].area < sliverArea)
				continue;
			const auto& f = cm.faces[t];
			if (SignedArea2x(U[f[0]], U[f[1]], U[f[2]]) <= 0.0)
				++flips;
		}
		return flips;
	};

	// Largest t in (0,1] keeping all triangles with positive signed area along
	// uv + t*d (smallest positive root of the per-triangle area quadratic).
	// Shared by both methods: SLIM combines it with an energy line search; ARAP
	// needs it because its unconstrained global solve (the cotangent system is
	// indefinite on obtuse triangulations) can fold triangles, which would break
	// the flip-free contract segmentation flip-repair relies on.
	// `exemptSlivers` (SLIM true / ARAP false) as in CountFlips: a sliver's UV
	// area hovers at rounding noise, so its zero crossings along d are noise too
	// and would clamp the step to ~0, stalling the line search the same way.
	auto MaxStepNoFlip = [&](const std::vector<Vec2>& U,
	                         const std::vector<Vec2>& d, bool exemptSlivers) -> double {
		double tmax = 1.0;
		for (int t = 0; t < nt; ++t) {
			if (exemptSlivers && tri[t].area < sliverArea)
				continue;
			const auto& f = cm.faces[t];
			const Vec2 &a = U[f[0]], &b = U[f[1]], &c = U[f[2]];
			const Vec2 &da = d[f[0]], &db = d[f[1]], &dc = d[f[2]];
			// signed_area2x(t) = A2 t^2 + A1 t + A0 ; find smallest positive root.
			auto cross = [](const Vec2& p, const Vec2& q) {
				return p.x() * q.y() - p.y() * q.x();
			};
			const Vec2 u1 = b - a, u2 = c - a;
			const Vec2 v1 = db - da, v2 = dc - da;
			const double A0 = cross(u1, u2);
			const double A1 = cross(u1, v2) + cross(v1, u2);
			const double A2 = cross(v1, v2);
			// Solve A2 t^2 + A1 t + A0 = 0 (area crosses zero -> flip). Two epsilons
			// with different units: the coefficients A0/A1/A2 carry UV-area units
			// (world-scale after refinement), so classify small terms RELATIVE to the
			// coefficient magnitude; a root r is a dimensionless step fraction in
			// (0,1], so filter it with a fixed dimensionless floor (reusing the area-
			// scaled eps here would reject genuine small roots on large charts).
			const double cmax = std::max(std::abs(A0), std::max(std::abs(A1), std::abs(A2)));
			const double ceps = 1e-12 * cmax; // relative coefficient-smallness eps
			const double reps = 1e-12; // dimensionless step-fraction floor
			if (std::abs(A2) <= ceps) {
				if (std::abs(A1) > ceps) {
					const double r = -A0 / A1;
					if (r > reps && r < tmax)
						tmax = r;
				}
				continue;
			}
			const double disc = A1 * A1 - 4.0 * A2 * A0;
			if (disc < 0.0)
				continue; // never crosses zero
			// Cancellation-safe roots (Numerical Recipes): q pairs A1 with the
			// same-signed root of the discriminant, so neither root subtracts nearly
			// equal quantities. The two roots are q/A2 and A0/q.
			const double sq = std::sqrt(disc);
			const double q = -0.5 * (A1 + std::copysign(sq, A1));
			const double r0 = q / A2;
			const double r1 = (q != 0.0) ? A0 / q : r0;
			for (double r : {r0, r1})
				if (r > reps && r < tmax)
					tmax = r;
		}
		return tmax;
	};

	// A step can be flip-free in double yet quantize to a zero/negative area once
	// UVs are stored as float in faceTexcoords — check candidates at shipped
	// precision too. `exemptSlivers` skips input-degenerate triangles (SLIM path);
	// ARAP passes false, holding every triangle to the unexempted contract.
	auto FlipsAtShippedPrecision = [&](const std::vector<Vec2>& U, bool exemptSlivers) {
		for (int t = 0; t < nt; ++t) {
			if (exemptSlivers && tri[t].area < sliverArea)
				continue;
			const auto& f = cm.faces[t];
			const Eigen::Vector2f a = U[f[0]].cast<float>();
			const Eigen::Vector2f b = U[f[1]].cast<float>();
			const Eigen::Vector2f c = U[f[2]].cast<float>();
			if (SignedArea2x(Vec2(a.x(), a.y()), Vec2(b.x(), b.y()),
			                 Vec2(c.x(), c.y()))
			    <= 0.0)
				return true;
		}
		return false;
	};

	const bool slim = (method == ParametrizeParams::FlattenMethod::SLIM);

	// --- ARAP path: constant L, factor once, iterate local/global. ---
	if (!slim) {
		Eigen::SparseMatrix<double> A;
		BuildSystem({}, A);
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		solver.compute(A);
		if (solver.info() != Eigen::Success)
			return; // keep Tutte UVs
		std::vector<Mat2> R(nt, Mat2::Identity());
		// Fixed-size scratch hoisted out of the refinement loop — each pass fully
		// overwrites it (same reuse discipline as the SLIM path's re-valued A).
		Eigen::VectorXd bx, by;
		std::vector<Vec2> d(n), cand(n);
		for (unsigned it = 0; it < iterations; ++it) {
			// Local: closest rotation per triangle.
			for (int t = 0; t < nt; ++t)
				R[t] = ClosestRotation(TriJacobian(t, uv));
			// Global: solve L u = b(R).
			BuildRHS(R, {}, bx, by);
			Eigen::VectorXd sx = solver.solve(bx);
			Eigen::VectorXd sy = solver.solve(by);
			if (solver.info() != Eigen::Success)
				break;
			bool ok = true;
			for (int i = 0; i < n; ++i)
				if (!std::isfinite(sx[i]) || !std::isfinite(sy[i])) {
					ok = false;
					break;
				}
			if (!ok)
				break;
			// Flip-free step guard: if the current map is fold-free, keep it so.
			for (int i = 0; i < n; ++i)
				d[i] = Vec2(sx[i], sy[i]) - uv[i];
			if (CountFlips(uv, false) > 0) {
				// init already folded (PCA fallback) — nothing to protect.
				for (int i = 0; i < n; ++i)
					uv[i] += d[i];
				continue;
			}
			// A step can be flip-free in double yet quantize to a zero/negative
			// area once UVs are stored as float in faceTexcoords — reject the
			// candidate at shipped precision too (unexempted: ARAP's contract).
			const double tmax = MaxStepNoFlip(uv, d, false);
			double step = (tmax >= 1.0) ? 1.0 : 0.8 * tmax; // margin off the flip
			bool moved = false;
			for (int ls = 0; ls < 30 && step > 1e-12; ++ls, step *= 0.5) {
				for (int i = 0; i < n; ++i)
					cand[i] = uv[i] + step * d[i];
				if (CountFlips(cand, false) == 0 && !FlipsAtShippedPrecision(cand, false)) {
					uv = cand;
					moved = true;
					break;
				}
			}
			if (!moved)
				break; // cannot advance without folding
		}
		return;
	}

	// --- SLIM path: symmetric-Dirichlet proximal reweighting + line search. ---
	// Each iteration: (1) SVD per triangle → SLIM weights + rotation target;
	// (2) weighted global solve gives a candidate p; (3) line search along
	// (p - uv) with a max step that keeps every triangle un-flipped; accept the
	// step that reduces the symmetric-Dirichlet energy.
	auto TotalEnergy = [&](const std::vector<Vec2>& U) -> double {
		double e = 0.0;
		for (int t = 0; t < nt; ++t)
			e += tri[t].area * SymDirEnergyJ(TriJacobian(t, U));
		return e;
	};
	// One solver + one sparse matrix reused across iterations. The pattern is
	// CONSTANT (only the weight VALUES change with triScale), so we build A and
	// analyze its pattern ONCE, then RE-VALUE A in place each iteration — no
	// per-iteration triplet allocation, sort, or symbolic analysis. offSlot /
	// diagSlot map each EmitSystem contribution to its valuePtr slot in the fixed
	// emission order recorded on the first iteration.
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	Eigen::SparseMatrix<double> A;
	std::vector<int> offSlot, diagSlot;
	// Current map energy, carried across iterations: after an accepted line-search
	// step uv equals the candidate, so its energy is exactly the value we just
	// computed — cache it instead of re-summing TotalEnergy(uv) each iteration.
	double eCur = TotalEnergy(uv);
	// Per-triangle scratch hoisted out of the iteration loop: every element is
	// unconditionally overwritten each pass (triScale gets an explicit 1.0
	// fallback), so no per-iteration reinitialization is needed.
	std::vector<double> triScale(nt);
	std::vector<Mat2> R(nt);
	for (unsigned it = 0; it < iterations; ++it) {
		// Local: per-triangle SVD → SLIM weight + rotation target.
		for (int t = 0; t < nt; ++t) {
			const Mat2 J = TriJacobian(t, uv);
			Eigen::JacobiSVD<Mat2> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
			double s0 = svd.singularValues()[0];
			double s1 = svd.singularValues()[1];
			s0 = std::max(s0, 1e-8);
			s1 = std::max(s1, 1e-8);
			Mat2 U = svd.matrixU(), V = svd.matrixV();
			if ((U * V.transpose()).determinant() < 0.0) {
				U.col(1) *= -1.0;
			}
			R[t] = U * V.transpose();
			// SLIM proximal weight (Rabinovich 2017): the W^2 entry
			// (dPsi/ds)/(s-1) per singular value, averaged for an isotropic
			// per-triangle scale (a robust variant that keeps the global matrix
			// SPD). Psi = s^2 + 1/s^2 → dPsi/ds = 2s - 2/s^3. triScale multiplies
			// the cotangent quadratic form LINEARLY (it occupies SLIM's W^2 slot),
			// so return the ratio itself — the previous std::sqrt halved the
			// reweighting exponent, under-penalizing compressed triangles (~s^-3/2
			// instead of the prescribed ~s^-3) and converging to higher distortion.
			// Near s==1 the ratio is 0/0; its analytic limit is g'(1) = 8
			// (L'Hopital), so branch on closeness rather than biasing the
			// denominator (which sent near-isometric triangles to weight ~0,
			// stalling the strict-decrease line search on near-converged charts).
			auto wsv = [](double s) {
				const double g = 2.0 * s - 2.0 / (s * s * s); // dPsi/ds
				double w = (std::abs(s - 1.0) < 1e-6) ? 8.0 : g / (s - 1.0);
				if (!(w > 0.0) || !std::isfinite(w))
					w = 1.0;
				return std::min(w, 1e8);
			};
			triScale[t] = 0.5 * (wsv(s0) + wsv(s1));
			if (!(triScale[t] > 1e-8) || !std::isfinite(triScale[t]))
				triScale[t] = 1.0;
		}

		// Global: weighted solve for a target/candidate layout p. First iteration
		// builds A from triplets and records each contribution's valuePtr slot;
		// later iterations zero valuePtr and re-accumulate in the SAME emission
		// order — the summed entries are bit-identical to a fresh setFromTriplets.
		if (it == 0) {
			BuildSystem(triScale, A);
			solver.analyzePattern(A);
			offSlot.clear();
			offSlot.reserve(static_cast<size_t>(nt) * 6);
			diagSlot.assign(n, 0);
			EmitSystem(triScale, [&](int i, int j, double) { offSlot.push_back(SlotOf(A, i, j)); }, [&](int i, int j, double) { diagSlot[i] = SlotOf(A, i, j); });
		} else {
			double* vp = A.valuePtr();
			std::fill(vp, vp + A.nonZeros(), 0.0);
			size_t oi = 0;
			EmitSystem(triScale, [&](int, int, double v) { vp[offSlot[oi++]] += v; }, [&](int i, int, double v) { vp[diagSlot[i]] += v; });
#ifndef NDEBUG
			// Invariant (assert/sanitizer builds only, compiled out in Release):
			// the in-place re-value reproduces a from-scratch setFromTriplets
			// assembly bit-for-bit. Verified across every flatten fixture.
			if (it == 1) {
				Eigen::SparseMatrix<double> Aref;
				BuildSystem(triScale, Aref);
				ASSERT(Aref.nonZeros() == A.nonZeros());
				bool bitIdentical = true;
				for (Eigen::Index q = 0; q < A.nonZeros(); ++q)
					if (A.valuePtr()[q] != Aref.valuePtr()[q]) {
						bitIdentical = false;
						break;
					}
				ASSERT(bitIdentical);
			}
#endif
		}
		solver.factorize(A);
		if (solver.info() != Eigen::Success)
			break;
		Eigen::VectorXd bx, by;
		BuildRHS(R, triScale, bx, by);
		Eigen::VectorXd sx = solver.solve(bx);
		Eigen::VectorXd sy = solver.solve(by);
		if (solver.info() != Eigen::Success)
			break;
		bool ok = true;
		for (int i = 0; i < n; ++i)
			if (!std::isfinite(sx[i]) || !std::isfinite(sy[i])) {
				ok = false;
				break;
			}
		if (!ok)
			break;

		// Descent direction d = p - uv.
		std::vector<Vec2> p(n), d(n);
		for (int i = 0; i < n; ++i) {
			p[i] = Vec2(sx[i], sy[i]);
			d[i] = p[i] - uv[i];
		}

		// Flip-free max step, then backtracking line search on the energy.
		double step = 0.8 * MaxStepNoFlip(uv, d, true); // 0.8 safety margin off the flip
		if (!(step > 0.0))
			break;
		std::vector<Vec2> cand(n);
		bool improved = false;
		for (int ls = 0; ls < 30; ++ls) {
			for (int i = 0; i < n; ++i)
				cand[i] = uv[i] + step * d[i];
			// Accept only a flip-free (in double AND at shipped float precision,
			// sliver triangles exempted in BOTH) candidate that strictly lowers
			// the energy.
			if (CountFlips(cand, true) == 0 && !FlipsAtShippedPrecision(cand, true)) {
				const double eCand = TotalEnergy(cand);
				if (eCand < eCur) {
					uv = cand;
					eCur = eCand; // becomes next iteration's baseline energy
					improved = true;
					break;
				}
			}
			step *= 0.5;
			if (step < 1e-12)
				break;
		}
		if (!improved)
			break; // converged / no further descent
	}
}

// =========================================================================
// Seamster cut-to-disk (opt-in, params.cutToDisk) — Sheffer & Hart 2002.
// Slit a closed / multiply-connected chart open into a SINGLE-boundary disk so
// it flattens flip-free as ONE chart, instead of being bisected into many by the
// segmentation's disk guarantee. The chart-count reducer on hole-riddled MVS.
//
// DETERMINISM: the natural implementation iterates unordered_map/set, whose
// order is unspecified. Every observable is instead a pure function of
// (verts, faces, globalVid) in index order:
//   - AllBoundaryLoops: sort loop start vertices; each cycle is walked from its
//     smallest start id, and the returned loops are ordered by that min id.
//   - ChartVertexAdjacency: pre-sort each adj[v] (lowest-id BFS tie-break).
//   - ShortestCutEdges: seed BFS from a SORTED src list.
// (CutAlongEdges is already deterministic: duplicated-vertex ids are assigned in
// incident-face scan order, and union-find connectivity is union-order-invariant.)
// CutAlongEdges rewrites cm.verts/faces/globalVid in place but LEAVES cm.globalFid
// and the chart id untouched — it only duplicates vertices, never faces — so the
// opened chart still ships as one island through ParametrizeCharts' writeback.
// -------------------------------------------------------------------------

// Undirected edge key (a,b order-independent); matches the 2147483647 radix used
// throughout this file (assumes < 2^31 local vertices per chart).
inline long long EdgeKeyLL(int a, int b)
{
	if (a > b)
		std::swap(a, b);
	return static_cast<long long>(a) * 2147483647LL + b;
}

// All boundary loops (each an ordered vertex sequence) of a chart submesh.
// DETERMINISTIC: a vertex with two outgoing boundary half-edges (a pinch) keeps
// the successor with the smallest id; loop start vertices are sorted, so each cycle
// is emitted once, walked from its smallest-id vertex, and the loops are returned in
// ascending min-vertex-id order (the tie-break CutChartToDisk relies on).
std::vector<std::vector<int>> AllBoundaryLoops(const ChartMesh& cm)
{
	auto dkey = [](int a, int b) { return static_cast<long long>(a) * 2147483647LL + b; };
	std::unordered_map<long long, int> dir; // directed edge a->b count
	for (const auto& f : cm.faces)
		for (int k = 0; k < 3; ++k)
			dir[dkey(f[k], f[(k + 1) % 3])]++;
	std::unordered_map<int, int> nxt; // boundary half-edge successor a->b
	std::vector<int> starts; // boundary start vertices (then sorted)
	for (const auto& kv : dir) {
		const int a = static_cast<int>(kv.first / 2147483647LL);
		const int b = static_cast<int>(kv.first % 2147483647LL);
		if (dir.find(dkey(b, a)) == dir.end()) {
			auto it = nxt.find(a);
			if (it == nxt.end()) {
				nxt.emplace(a, b);
				starts.push_back(a);
			} else if (b < it->second) {
				it->second = b; // deterministic pinch tie-break: smallest successor
			}
		}
	}
	std::sort(starts.begin(), starts.end());
	std::vector<std::vector<int>> loops;
	std::unordered_set<int> seen;
	for (int s : starts) {
		if (seen.count(s))
			continue;
		std::vector<int> L;
		int c = s;
		size_t guard = 0;
		while (guard++ <= nxt.size()) {
			if (seen.count(c))
				break;
			seen.insert(c);
			L.push_back(c);
			auto it = nxt.find(c);
			if (it == nxt.end()) {
				L.clear();
				break;
			}
			c = it->second;
			if (c == s)
				break;
		}
		if (L.size() >= 3)
			loops.push_back(std::move(L));
	}
	return loops;
}

// Undirected unique vertex adjacency of a chart submesh, each list SORTED so BFS
// (ShortestCutEdges / FarthestVertex) explores neighbors lowest-id first.
void ChartVertexAdjacency(const ChartMesh& cm, std::vector<std::vector<int>>& adj)
{
	adj.assign(cm.verts.size(), {});
	std::unordered_set<long long> seen;
	for (const auto& f : cm.faces)
		for (int k = 0; k < 3; ++k) {
			const int a = f[k], b = f[(k + 1) % 3];
			if (seen.insert(EdgeKeyLL(a, b)).second) {
				adj[a].push_back(b);
				adj[b].push_back(a);
			}
		}
	for (auto& nbrs : adj)
		std::sort(nbrs.begin(), nbrs.end());
}

// Integer-quantized Euclidean edge metric for the cut Dijkstra. Seams follow the
// geometrically shortest interior path, not the fewest hops, so on irregular MVS
// triangulations they run straight instead of zig-zagging. DETERMINISM: raw double
// path lengths can be near-equal and flip under different compilers / -ffp-contract
// flags, which would change mesh CONNECTIVITY across builds and break this module's
// "pure function of chart geometry" contract. Quantizing each edge length to a
// scaled integer (relative to the chart's bounding-box diagonal) collapses that
// sub-quantum jitter to identical integers, and the exact int64 path-cost sum plus
// the (cost, vertex-id) heap order make the chosen path a pure, stable function of
// the geometry.
struct CutMetric
{
	double scale = 1.0;
	explicit CutMetric(const ChartMesh& cm)
	{
		if (cm.verts.empty())
			return;
		Vec3 lo = cm.verts[0], hi = cm.verts[0];
		for (const Vec3& v : cm.verts) {
			lo = lo.cwiseMin(v);
			hi = hi.cwiseMax(v);
		}
		const double diag = (hi - lo).norm();
		scale = (diag > 1e-20) ? (1e9 / diag) : 1e9;
	}
	// Strictly positive integer weight (the +1 avoids zero-cost edges from
	// coincident/duplicated vertices, keeping Dijkstra progress well-defined).
	long long operator()(const Vec3& a, const Vec3& b) const
	{
		return std::llround((a - b).norm() * scale) + 1;
	}
};

// Shortest (Euclidean-weighted) interior path from any src vertex to any dst vertex;
// returns its undirected edges (the cut). Empty if unreachable. DETERMINISTIC:
// Dijkstra over integer-quantized edge lengths (CutMetric) seeded from a SORTED src
// list, exploring sorted adjacency, popping in (cost, vertex-id) order — the path
// (hence the cut) is a pure, build-stable function of the chart geometry.
std::unordered_set<long long> ShortestCutEdges(const ChartMesh& cm,
                                               const std::unordered_set<int>& src,
                                               const std::unordered_set<int>& dst,
                                               const std::vector<std::vector<int>>& adj,
                                               const std::unordered_set<int>& blocked = {})
{
	const int n = static_cast<int>(cm.verts.size());
	const CutMetric metric(cm);
	constexpr long long infDist = std::numeric_limits<long long>::max();
	std::vector<long long> dist(n, infDist);
	std::vector<int> prev(n, -2);
	using Node = std::pair<long long, int>; // (accumulated cost, vertex)
	std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
	std::vector<int> srcSorted(src.begin(), src.end());
	std::sort(srcSorted.begin(), srcSorted.end());
	for (int s : srcSorted) {
		dist[s] = 0;
		prev[s] = -1;
		pq.push({0, s});
	}
	int hit = -1;
	while (!pq.empty()) {
		const Node top = pq.top();
		pq.pop();
		const long long du = top.first;
		const int u = top.second;
		if (du > dist[u])
			continue; // stale heap entry
		if (dst.count(u) && !src.count(u)) {
			hit = u;
			break;
		}
		for (int w : adj[u]) {
			// never route the cut THROUGH another boundary vertex (only the dst
			// endpoint may be a boundary) — the main source of self-touching seams.
			if (blocked.count(w) && !dst.count(w))
				continue;
			const long long nd = du + metric(cm.verts[u], cm.verts[w]);
			if (nd < dist[w]) {
				dist[w] = nd;
				prev[w] = u;
				pq.push({nd, w});
			}
		}
	}
	std::unordered_set<long long> cut;
	if (hit < 0)
		return cut;
	for (int c = hit; prev[c] != -1; c = prev[c])
		cut.insert(EdgeKeyLL(c, prev[c]));
	return cut;
}

// Slit the chart along the given undirected interior edges: each vertex is
// duplicated into one copy per one-ring "arc" (a fan of incident faces connected
// through non-cut interior edges); faces are rewired to their arc's copy. The
// vertex-duplication "cut along edges" surgery. DETERMINISTIC: incident faces are
// scanned in index order, so the arc that keeps the original vertex and the ids of
// the new copies are fixed regardless of the union-find internal root choice.
void CutAlongEdges(ChartMesh& cm, const std::unordered_set<long long>& cuts)
{
	const int nf = static_cast<int>(cm.faces.size());
	std::unordered_map<long long, int> ecount; // undirected edge -> face count
	for (const auto& f : cm.faces)
		for (int k = 0; k < 3; ++k)
			ecount[EdgeKeyLL(f[k], f[(k + 1) % 3])]++;
	std::vector<std::vector<int>> vf(cm.verts.size()); // vertex -> incident faces
	for (int f = 0; f < nf; ++f)
		for (int k = 0; k < 3; ++k)
			vf[cm.faces[f][k]].push_back(f);

	std::vector<Vec3> verts = cm.verts;
	std::vector<Mesh::VIndex> gvid = cm.globalVid;
	std::vector<Eigen::Vector3i> faces = cm.faces;
	const int n0 = static_cast<int>(cm.verts.size());
	for (int v = 0; v < n0; ++v) {
		const auto& inc = vf[v];
		if (inc.size() < 2)
			continue;
		// union-find over incident faces: connect across a shared edge (v,w) that
		// is interior (2 faces) and NOT cut → same arc.
		std::vector<int> parent(inc.size());
		std::iota(parent.begin(), parent.end(), 0);
		std::function<int(int)> find = [&](int x) {
			while (parent[x] != x) {
				parent[x] = parent[parent[x]];
				x = parent[x];
			}
			return x;
		};
		std::unordered_map<int, std::vector<int>> byW; // w -> local incident-face idxs
		for (int i = 0; i < (int)inc.size(); ++i) {
			const auto& f = cm.faces[inc[i]];
			for (int k = 0; k < 3; ++k) {
				if (f[k] == v)
					byW[f[(k + 1) % 3]].push_back(i);
				else if (f[(k + 1) % 3] == v)
					byW[f[k]].push_back(i);
			}
		}
		// Union in SORTED neighbor order. The duplicated-vertex ids are already a pure
		// function of (partition, scan order) — the union-find root id is only a compCopy
		// key, never an output — but unioning by ascending neighbor id also pins the root
		// itself, so determinism needs no reasoning about union-order root invariance.
		std::vector<int> ws;
		ws.reserve(byW.size());
		for (const auto& kv : byW)
			ws.push_back(kv.first);
		std::sort(ws.begin(), ws.end());
		for (int w : ws) {
			const std::vector<int>& fs = byW[w];
			const long long e = EdgeKeyLL(v, w);
			if (ecount[e] == 2 && cuts.find(e) == cuts.end() && fs.size() == 2)
				parent[find(fs[0])] = find(fs[1]);
		}
		std::unordered_map<int, int> compCopy; // arc root -> vertex id
		for (int i = 0; i < (int)inc.size(); ++i) {
			const int r = find(i);
			int vid;
			auto it = compCopy.find(r);
			if (it == compCopy.end()) {
				if (compCopy.empty())
					vid = v; // first arc keeps the original vertex
				else {
					vid = static_cast<int>(verts.size());
					verts.push_back(cm.verts[v]);
					gvid.push_back(cm.globalVid[v]);
				}
				compCopy[r] = vid;
			} else
				vid = it->second;
			const int f = inc[i];
			for (int k = 0; k < 3; ++k)
				if (cm.faces[f][k] == v)
					faces[f][k] = vid;
		}
	}
	cm.verts.swap(verts);
	cm.globalVid.swap(gvid);
	cm.faces.swap(faces);
}

// Farthest vertex (by Euclidean geodesic distance) from src — seeds a slit on a
// closed chart. Dijkstra over the same integer-quantized edge metric as
// ShortestCutEdges, so the endpoint is a pure, build-stable function of geometry:
// the maximum accumulated cost wins, ties broken by smallest vertex id.
int FarthestVertex(const ChartMesh& cm, int src, const std::vector<std::vector<int>>& adj)
{
	const int n = static_cast<int>(cm.verts.size());
	const CutMetric metric(cm);
	constexpr long long infDist = std::numeric_limits<long long>::max();
	std::vector<long long> dist(n, infDist);
	using Node = std::pair<long long, int>;
	std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
	dist[src] = 0;
	pq.push({0, src});
	long long best = -1;
	int far = src;
	while (!pq.empty()) {
		const Node top = pq.top();
		pq.pop();
		const long long du = top.first;
		const int u = top.second;
		if (du > dist[u])
			continue; // stale
		if (du > best || (du == best && u < far)) {
			best = du;
			far = u;
		}
		for (int w : adj[u]) {
			const long long nd = du + metric(cm.verts[u], cm.verts[w]);
			if (nd < dist[w]) {
				dist[w] = nd;
				pq.push({nd, w});
			}
		}
	}
	return far;
}

// Make a chart a topological disk by slitting: open a closed chart, then merge
// every extra boundary loop into the outer loop via a shortest interior path.
// Returns true iff it ends with exactly one boundary loop. Genus (handles) is not
// removed here — those rare charts fall through to the flip safety net. The ≤256
// merge guard bounds wall-time on a giant hole-riddled chart (15.3M-mesh risk).
bool CutChartToDisk(ChartMesh& cm)
{
	std::vector<std::vector<int>> adj;
	ChartVertexAdjacency(cm, adj);
	std::vector<std::vector<int>> loops = AllBoundaryLoops(cm);
	if (loops.empty()) {
		// closed chart: slit between the two farthest vertices to open a boundary.
		if (cm.verts.size() < 3)
			return false;
		const int a = FarthestVertex(cm, 0, adj);
		const int b = FarthestVertex(cm, a, adj);
		const auto cut = ShortestCutEdges(cm, {a}, {b}, adj);
		if (cut.empty())
			return false;
		CutAlongEdges(cm, cut);
		ChartVertexAdjacency(cm, adj);
		loops = AllBoundaryLoops(cm);
	}
	int guard = 0;
	while (loops.size() > 1 && guard++ < 256) {
		// outer = largest loop, inner = smallest of the rest; ties break by min
		// vertex id (loops are ordered by ascending min id, strict comparisons keep
		// the earlier = smaller-min-id loop).
		size_t outer = 0;
		for (size_t i = 1; i < loops.size(); ++i)
			if (loops[i].size() > loops[outer].size())
				outer = i;
		size_t inner = (outer == 0) ? 1 : 0;
		for (size_t i = 0; i < loops.size(); ++i)
			if (i != outer && loops[i].size() < loops[inner].size())
				inner = i;
		const std::unordered_set<int> src(loops[inner].begin(), loops[inner].end());
		const std::unordered_set<int> dst(loops[outer].begin(), loops[outer].end());
		// block all OTHER boundary vertices so the cut path doesn't pass through a
		// third loop (which would create a pinch).
		std::unordered_set<int> blocked;
		for (size_t i = 0; i < loops.size(); ++i)
			if (i != inner && i != outer)
				for (int v : loops[i])
					blocked.insert(v);
		const auto cut = ShortestCutEdges(cm, src, dst, adj, blocked);
		if (cut.empty())
			break; // disconnected (should not happen) — bail to the safety net
		CutAlongEdges(cm, cut);
		ChartVertexAdjacency(cm, adj);
		loops = AllBoundaryLoops(cm);
	}
	return loops.size() == 1;
}

// -------------------------------------------------------------------------
// Count folded triangles in the map `uv`, EXEMPTING input-degenerate slivers
// (source area < meanArea*1e-6, the same relative threshold ChartOverDistorted
// uses). LscmInit skips such slivers, so their UV orientation is unconstrained
// numerical noise; counting them as flips would reject an otherwise flip-free
// conformal map and force the distorting Tutte fallback. Used by both the init
// acceptance in FlattenChart and the fold verdict in ChartFolds so the repair
// decision stays consistent with the map that actually ships.
// -------------------------------------------------------------------------
// `flipped`, when non-null, receives the LOCAL triangle index of every counted
// flip (appended, not cleared) — used to build detail::FoldDiagnosis. Passing
// a non-null collector never changes the returned count: the same triangles
// are counted, in the same order, under the same sliver exemption.
int CountRealFlips(const ChartMesh& cm, const std::vector<Vec2>& uv, std::vector<int>* flipped = nullptr)
{
	const size_t nf = cm.faces.size();
	std::vector<double> src(nf, 0.0);
	double totA = 0.0;
	for (size_t t = 0; t < nf; ++t) {
		const auto& f = cm.faces[t];
		src[t] = 0.5 * (cm.verts[f[1]] - cm.verts[f[0]]).cross(cm.verts[f[2]] - cm.verts[f[0]]).norm();
		totA += src[t];
	}
	const double epsA = (totA / static_cast<double>(std::max<size_t>(1, nf))) * 1e-6;
	int flips = 0;
	for (size_t t = 0; t < nf; ++t) {
		if (src[t] < epsA)
			continue; // input-degenerate sliver: orientation is noise
		const auto& f = cm.faces[t];
		if (SignedArea2x(uv[f[0]], uv[f[1]], uv[f[2]]) <= 0.0) {
			++flips;
			if (flipped != nullptr)
				flipped->push_back(static_cast<int>(t));
		}
	}
	return flips;
}

// -------------------------------------------------------------------------
// ChartUVSelfOverlaps — global-injectivity probe for a flattened chart.
// A flip-free map can still fold back over itself GLOBALLY (zero flipped
// triangles yet doubly-covered regions — measured on the truck fixture:
// 1.34% of atlas texels fold-covered while every triangle kept consistent
// orientation), and the bake's per-texel write then paints one region with
// the other's colors. Rasterize the non-sliver triangles (same exemption as
// CountRealFlips, so the verdict matches the shipped map) onto a coarse grid
// scaled to the chart's UV bbox and count texels claimed by 2+ triangles:
// interiors of an injective map are disjoint, and a shared-edge texel centre
// resolves to exactly one owner under the rasterizer's fill rule. The small
// relative threshold absorbs stray boundary ties and sub-texel numerical
// grazing; genuine folds cover contiguous regions far above it.
// -------------------------------------------------------------------------
// gridLongSide == 0 chooses an adaptive coarse grid (cheap enough for the
// per-round repair probes); the shipping guard passes a fixed fine grid to
// also catch small folds below the coarse grid's sensitivity.
//
// `colliding`, when non-null, receives LOCAL triangle indices (appended, not
// cleared): both the first owner and the collider of every texel that turns
// an overlap. This switches the per-texel buffer from uint8 occupancy to an
// int32 first-owner-triangle id, but the covered/overlaps counting semantics
// — and hence the returned verdict and both thresholds below — are IDENTICAL
// to the uint8 path: a texel counts as an overlap exactly once, on its first
// collision; further triangles landing on an already-overlapping texel are a
// no-op either way. The two buffer strategies are kept separate (rather than
// always paying for int32) so the null-collector fast path is untouched.
bool ChartUVSelfOverlaps(const ChartMesh& cm, const std::vector<Vec2>& uv, int gridLongSide = 0, std::vector<int>* colliding = nullptr)
{
	const size_t nf = cm.faces.size();
	if (nf <= 1 || uv.size() != cm.verts.size())
		return false;
	std::vector<double> src(nf, 0.0);
	double totA = 0.0;
	for (size_t t = 0; t < nf; ++t) {
		const auto& f = cm.faces[t];
		src[t] = 0.5 * (cm.verts[f[1]] - cm.verts[f[0]]).cross(cm.verts[f[2]] - cm.verts[f[0]]).norm();
		totA += src[t];
	}
	// Wider sliver exemption than CountRealFlips' 1e-6: a near-degenerate
	// triangle's UV PLACEMENT (not just orientation) is numerical noise, and
	// treating its wander as a fold would discard/bisect maps whose real
	// triangles are perfectly injective (the bake-side sliver containment is
	// covered separately by TextureBakeTest.SliverSourceUVStaysInChart).
	const double epsA = (totA / static_cast<double>(nf)) * 1e-3;
	double minX = std::numeric_limits<double>::max(), minY = minX;
	double maxX = std::numeric_limits<double>::lowest(), maxY = maxX;
	for (const Vec2& p : uv) {
		minX = std::min(minX, p.x());
		minY = std::min(minY, p.y());
		maxX = std::max(maxX, p.x());
		maxY = std::max(maxY, p.y());
	}
	const double w = maxX - minX, h = maxY - minY;
	if (!(w > 0.0) || !(h > 0.0) || !std::isfinite(w) || !std::isfinite(h))
		return false; // degenerate/invalid extents: nothing meaningful to test
	// long side ~2*sqrt(nf) texels: enough that per-triangle footprints are a
	// few texels wide, cheap enough to run inside every repair probe
	const int longSide =
	    gridLongSide > 0
	        ? gridLongSide
	        : std::clamp(static_cast<int>(2.0 * std::ceil(std::sqrt(static_cast<double>(nf)))), 64, 512);
	const double scale = static_cast<double>(longSide) / std::max(w, h);
	const int W = std::max(1, static_cast<int>(std::ceil(w * scale)));
	const int H = std::max(1, static_cast<int>(std::ceil(h * scale)));
	long covered = 0, overlaps = 0;
	if (colliding != nullptr) {
		// Diagnosis path: per-texel FIRST-OWNER triangle id (-1 = uncovered,
		// -2 = already-counted overlap) instead of a uint8 occupancy count.
		std::vector<int32_t> owner(static_cast<size_t>(W) * H, -1);
		for (size_t t = 0; t < nf; ++t) {
			if (src[t] < epsA)
				continue; // input-degenerate sliver: its UV placement is noise
			const auto& f = cm.faces[t];
			const Vec2 a((uv[f[0]].x() - minX) * scale, (uv[f[0]].y() - minY) * scale);
			const Vec2 b((uv[f[1]].x() - minX) * scale, (uv[f[1]].y() - minY) * scale);
			const Vec2 c((uv[f[2]].x() - minX) * scale, (uv[f[2]].y() - minY) * scale);
			RasterizeTriangleBary<double>(a, b, c, W, H, [&](int x, int y, const Vec3&) {
				int32_t& own = owner[static_cast<size_t>(y) * W + x];
				if (own == -1) {
					own = static_cast<int32_t>(t);
					++covered;
				} else if (own >= 0) {
					++overlaps;
					colliding->push_back(static_cast<int>(own));
					colliding->push_back(static_cast<int>(t));
					own = -2; // counted: further hits on this texel are a no-op
				}
			});
		}
	} else {
		std::vector<uint8_t> cover(static_cast<size_t>(W) * H, 0);
		for (size_t t = 0; t < nf; ++t) {
			if (src[t] < epsA)
				continue; // input-degenerate sliver: its UV placement is noise
			const auto& f = cm.faces[t];
			const Vec2 a((uv[f[0]].x() - minX) * scale, (uv[f[0]].y() - minY) * scale);
			const Vec2 b((uv[f[1]].x() - minX) * scale, (uv[f[1]].y() - minY) * scale);
			const Vec2 c((uv[f[2]].x() - minX) * scale, (uv[f[2]].y() - minY) * scale);
			RasterizeTriangleBary<double>(a, b, c, W, H, [&](int x, int y, const Vec3&) {
				uint8_t& cvr = cover[static_cast<size_t>(y) * W + x];
				if (cvr == 0) {
					cvr = 1;
					++covered;
				} else if (cvr == 1) {
					cvr = 2;
					++overlaps;
				}
			});
		}
	}
	// Repair probes (adaptive grid) use a relative threshold: bisecting a chart
	// over a cosmetic sub-0.1% graze would fragment the segmentation for no
	// visible gain. The shipping guard (fixed fine grid) is tighter but still
	// area-relative with a floor: a genuine fold covers a REGION (measured
	// pre-fix: 1249-2057 texels per folded mesh.ply chart), while a hairline
	// 1-texel band along a shared boundary is rasterization-scale grazing that
	// a fallback would "fix" at a real distortion cost.
	return gridLongSide > 0 ? overlaps > std::max<long>(8, covered / 500)
	                        : overlaps > std::max<long>(1, covered / 1000);
}

// -------------------------------------------------------------------------
// Flatten one chart: cut-to-disk if enabled, init (LSCM/Tutte), local/global
// refine. On success `uv` holds one Vec2 per (possibly duplicated) local vertex.
// Returns false (caller falls back to a planar projection) if it cannot be flattened.
// -------------------------------------------------------------------------
// Per-init-path chart tally for the flatten diagnostic summary. Counters and
// their increments compile out entirely unless the diagnostic build flag is set
// (cmake -DHALFMESH_ATLAS_DEBUG=ON), so release builds carry zero state/zero cost.
#ifdef HM_ATLAS_DEBUG
static std::atomic<int> gFlatLscm{0}, gFlatTutte{0}, gFlatPca{0}, gFlatCut{0},
    gFlatCached{0};
	#define HM_FLATTEN_TALLY(counter) (++(counter))
#else
	#define HM_FLATTEN_TALLY(counter) ((void)0)
#endif

bool FlattenChart(ChartMesh& cm, const ParametrizeParams& params,
                  std::vector<Vec2>& uv)
{
	if (cm.faces.empty())
		return false;
	int numLoops = 0;
	bool hasB = BuildBoundaryLoop(cm, numLoops);
	// Cut-to-disk (opt-in): a closed (no boundary) or multiply-connected / pinched
	// chart (numLoops != 1) is SLIT into a single-boundary disk so it flattens
	// flip-free as ONE chart, instead of being split into many upstream. Rebuild the
	// loop on the opened mesh. ChartFolds' fast path runs this IDENTICAL block, so
	// the repair verdict matches the shipped map (see ChartFolds).
	if (params.cutToDisk && (!hasB || numLoops != 1)) {
		if (CutChartToDisk(cm)) {
			hasB = BuildBoundaryLoop(cm, numLoops);
			HM_FLATTEN_TALLY(gFlatCut);
		}
	}
	if (!hasB)
		return false; // closed chart (no boundary): caller does PCA fallback

	// Initialize. LSCM (free-boundary conformal) only needs two pinned vertices,
	// NOT a topological disk, so it also handles multiply-connected / pinched
	// charts (numLoops != 1) — for which it produces a far better map than the
	// PCA planar fallback. Use it whenever it is flip-free (the SLIM/ARAP line
	// search assumes a flip-free start).
	bool initOk = false;
	if (params.initMethod == ParametrizeParams::InitMethod::LSCM && LscmInit(cm, uv)) {
		// Accept the LSCM map if it is flip-free on the REAL (non-sliver) triangles
		// — LscmInit skips input-degenerate slivers, whose orientation is noise.
		initOk = (CountRealFlips(cm, uv) == 0);
	}
	if (initOk) {
		HM_FLATTEN_TALLY(gFlatLscm);
	} else {
		// Tutte requires a topological disk (single boundary loop) for its
		// convex-boundary injectivity guarantee; non-disk charts with no usable
		// LSCM init fall through to the caller's PCA projection.
		if (numLoops != 1)
			return false;
		if (!TutteInit(cm, uv))
			return false;
		HM_FLATTEN_TALLY(gFlatTutte);
	}
	LocalGlobal(cm, uv, params.method, params.flattenIterations);
	for (const Vec2& p : uv)
		if (!std::isfinite(p.x()) || !std::isfinite(p.y()))
			return false;
	return true;
}

// -------------------------------------------------------------------------
// Fallback flattening for a chart with no usable boundary loop (a closed
// chart): project onto its best-fit plane (PCA of vertex positions). This is a
// minimal "cut to a disk" stand-in — it produces valid finite UVs but may
// distort/fold for non-planar closed charts. Noted as a documented limitation.
// -------------------------------------------------------------------------
void FlattenChartFallback(const ChartMesh& cm, std::vector<Vec2>& uv)
{
	const int n = static_cast<int>(cm.verts.size());
	uv.assign(n, Vec2::Zero());
	if (n == 0)
		return;
	Vec3 centroid = Vec3::Zero();
	for (const Vec3& v : cm.verts)
		centroid += v;
	centroid /= static_cast<double>(n);
	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	for (const Vec3& v : cm.verts) {
		const Vec3 d = v - centroid;
		cov += d * d.transpose();
	}
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
	// two largest eigenvectors span the best-fit plane; the smallest is the
	// plane normal.
	const Vec3 ax = es.eigenvectors().col(2);
	Vec3 ay = es.eigenvectors().col(1);
	const Vec3 planeN = es.eigenvectors().col(0);
	// Orient the (ax, ay) basis so it is RIGHT-HANDED w.r.t. the chart's average
	// face normal — otherwise every projected triangle would be globally flipped
	// (negative signed area). Compare ax×ay against the area-weighted normal.
	Vec3 avgN = Vec3::Zero();
	for (const auto& f : cm.faces) {
		const Vec3 fn = (cm.verts[f[1]] - cm.verts[f[0]])
		                    .cross(cm.verts[f[2]] - cm.verts[f[0]]);
		avgN += fn;
	}
	const Vec3 crossAxy = ax.cross(ay); // ≈ ±planeN
	if (avgN.norm() > 0.0 && crossAxy.dot(avgN) < 0.0)
		ay = -ay; // flip to make (ax, ay) match the surface orientation
	(void)planeN;
	for (int i = 0; i < n; ++i) {
		const Vec3 d = cm.verts[i] - centroid;
		uv[i] = Vec2(d.dot(ax), d.dot(ay));
	}
}

// -------------------------------------------------------------------------
// Flatten probe for the segmentation flip-repair (src/AtlasCharting.cpp): build a
// chart submesh and report whether it folds when flattened as shipped. Exposed via
// detail::ChartFacesFold so charting (Module A) needs no flattening internals.
// -------------------------------------------------------------------------

// Build a ChartMesh from an explicit global-face list. The list MUST be sorted by
// global face id so the local vertex numbering matches ExtractCharts (which walks
// faces in global order); that makes ChartFolds' verdict identical to what
// ParametrizeCharts ships for the same chart.
ChartMesh ExtractOneChart(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces)
{
	ChartMesh cm;
	std::unordered_map<Mesh::VIndex, int> map;
	for (Mesh::FIndex f : faces) {
		Eigen::Vector3i lf;
		for (int k = 0; k < 3; ++k) {
			const Mesh::VIndex gv = mesh.faces[f][k];
			auto it = map.find(gv);
			int lv;
			if (it == map.end()) {
				lv = static_cast<int>(cm.verts.size());
				map.emplace(gv, lv);
				cm.verts.push_back(mesh.vertices[gv].template cast<double>());
				cm.globalVid.push_back(gv);
			} else {
				lv = it->second;
			}
			lf[k] = lv;
		}
		cm.faces.push_back(lf);
		cm.globalFid.push_back(f);
	}
	return cm;
}

// Should this (flip-free) chart be split for OVER-DISTORTION? `uv` is the SHIPPED
// map (full init + SLIM, exactly what ParametrizeCharts writes). Returns true iff
// the chart's area-weighted symmetric-Dirichlet exceeds the budget τ AND it is not
// sliver-dominated. The sliver guard is mandatory: a near-zero-area INPUT triangle
// flips / blows up at any chart size, so bisection can never fix it — splitting on
// such a chart would shatter it to the runaway cap (catastrophic on the 15.3M mesh).
// We classify per source-triangle area against a relative threshold (epsA =
// meanArea·1e-6) so a real fold (normal-area triangle) is told apart from a degenerate
// sliver.
// `over`, when non-null, receives LOCAL triangle indices (appended, not
// cleared): every real-flip face AND every face whose OWN symmetric-Dirichlet
// exceeds tau (a face can appear via either or both conditions — duplicates
// are the caller's dedup responsibility). Passing a non-null collector never
// changes the returned verdict, the realFlips/sliver classification, or the
// area-weighted sd computed below — those are unchanged from the collector-
// free path.
bool ChartOverDistorted(const ChartMesh& cm, const std::vector<Vec2>& uv, float tau, std::vector<int>* over = nullptr)
{
	if (uv.size() != cm.verts.size())
		return false;
	std::vector<TriRef> trs;
	trs.reserve(cm.faces.size());
	double totA = 0.0;
	for (const auto& f : cm.faces) {
		trs.push_back(MakeTriRef(cm.verts[f[0]], cm.verts[f[1]], cm.verts[f[2]]));
		totA += trs.back().area;
	}
	const double meanA = totA / static_cast<double>(std::max<size_t>(1, cm.faces.size()));
	const double epsA = meanA * 1e-6; // relative sliver threshold
	double esum = 0.0, asum = 0.0;
	int realFlips = 0;
	bool sliver = false;
	for (size_t i = 0; i < cm.faces.size(); ++i) {
		const auto& f = cm.faces[i];
		const TriRef& tr = trs[i];
		const bool degen = tr.area < epsA;
		if (degen)
			sliver = true;
		if (SignedArea2x(uv[f[0]], uv[f[1]], uv[f[2]]) <= 0.0 && !degen) {
			++realFlips;
			if (over != nullptr)
				over->push_back(static_cast<int>(i));
		}
		if (tr.degenerate)
			continue;
		Mat2 Uu;
		Uu.col(0) = uv[f[1]] - uv[f[0]];
		Uu.col(1) = uv[f[2]] - uv[f[0]];
		const Mat2 J = Uu * tr.invX;
		const double e = SymDirEnergyJ(J);
		esum += tr.area * e;
		asum += tr.area;
		if (over != nullptr && e > static_cast<double>(tau))
			over->push_back(static_cast<int>(i));
	}
	if (realFlips > 0)
		return true; // a real fold survived the flip-free init → must split
	if (sliver)
		return false; // sliver-dominated → never split (unfixable degenerate input)
	const double sd = (asum > 0.0) ? esum / asum : 4.0;
	return sd > static_cast<double>(tau);
}

// Artifacts of an ACCEPTING ChartFolds verdict (the chart ships as-is), captured
// so the shipping flatten does not have to be recomputed by ParametrizeCharts:
//   - uv: the accepted per-local-vertex UVs (the caller keeps the matching,
//     possibly cut-to-disk, ChartMesh — its state after ChartFolds).
//   - finalUv: true = `uv` IS the shipped map. ChartFolds always judges the
//     fully-flattened map, so every accept deposits finalUv == true; the flag
//     is kept so the entry contract stays explicit if an init-only accept is
//     ever reintroduced (consumers assert on it instead of resuming).
//   - valid: set only at accept sites that computed a shippable uv (a <=1-face
//     early accept computes nothing → consumer recomputes).
struct FoldAccept
{
	std::vector<Vec2> uv;
	bool finalUv = false;
	bool valid = false;
};

// Convert LOCAL triangle indices (possibly unsorted, possibly duplicated —
// e.g. gathered from more than one collector) into detail::FoldDiagnosis:
// GLOBAL face ids, sorted ascending, deduplicated. `diag` must be non-null.
void FillFoldDiagnosis(const ChartMesh& cm, std::vector<int>& local, detail::FoldDiagnosis* diag)
{
	std::sort(local.begin(), local.end());
	local.erase(std::unique(local.begin(), local.end()), local.end());
	diag->badFaces.clear();
	diag->badFaces.reserve(local.size());
	for (int t : local)
		diag->badFaces.push_back(cm.globalFid[static_cast<size_t>(t)]);
	// cm.globalFid is monotonic in local index (ExtractCharts/ExtractOneChart
	// walk faces in global order, and CutAlongEdges never reorders faces), so
	// this is already sorted — re-sort/dedup anyway so the contract does not
	// silently depend on that invariant.
	std::sort(diag->badFaces.begin(), diag->badFaces.end());
	diag->badFaces.erase(std::unique(diag->badFaces.begin(), diag->badFaces.end()), diag->badFaces.end());
}

// Does this chart fold (OR, when a distortion budget is set, ship an over-stretched
// map) the SAME way ParametrizeCharts does? This is the predicate the segmentation
// flip-repair (RepairDevelopableFlips) drives its bisect-and-requeue split on.
//
// The chart is always flattened exactly as shipped (full init + SLIM via
// FlattenChart) and judged on THAT map: it folds if it has flipped triangles or
// self-overlaps globally, and — when a distortion budget is set
// (developableMaxUvDistortion > 0) — also if it is over-distorted (area-weighted
// symmetric-Dirichlet > budget), excluding sliver-dominated charts. Judging the
// SHIPPED map — not the init, whose local flip-freedom SLIM preserves but whose
// global injectivity it does not, and not the conformal LSCM energy, which reads
// high — is what makes the verdict match the atlas the user receives.
//
// When `out` is non-null and the verdict is "does not fold", the accepted UVs are
// deposited in *out (see FoldAccept) so the final accepting verdict per shipping
// chart can be reused instead of flattened a second time. Never affects the verdict.
//
// When `diag` is non-null and the verdict IS "folds", *diag is filled with the
// offending faces (see detail::FoldDiagnosis) so the repair (§6.1) can carve
// around them: on every return-true path below, the failing collector(s) are
// re-run (with their out-params) on the SAME judged map that produced the
// verdict, and the result converted to global ids. This is one extra collector
// pass, paid only for a folding chart — folding charts get re-flattened after
// the split anyway, so the cost is negligible, and the fast accept path above
// (and the verdict itself) is never touched by a non-null `diag`.
bool ChartFolds(ChartMesh& cm, const ParametrizeParams& params, FoldAccept* out = nullptr, detail::FoldDiagnosis* diag = nullptr)
{
	if (cm.faces.size() <= 1)
		return false;
	const auto mapFolds = [&](const std::vector<Vec2>& uv) {
		if (uv.size() != cm.verts.size())
			return true;
		// Exempt input-degenerate slivers (as FlattenChart's acceptance does), so
		// the repair verdict matches the map ParametrizeCharts actually ships.
		// Flip-freedom alone does NOT imply injectivity: a chart can fold back
		// over itself globally with zero flipped triangles, and the bake then
		// paints one region with the other's colors — treat global self-overlap
		// as a fold so the repair bisects such charts too (2026-08 review).
		// Probe at the SAME fixed 512 grid as the shipping guard: the adaptive
		// default (2*sqrt(nf), floor 64) gives a ~1000-face chart only ~4 cells
		// per face, so folds of individual fine-detail faces vanish — measured
		// on the challenge fixture, 95 charts shipped >1000-texel folds (up to
		// 15.7% of chart area) that the repair grid never saw and no
		// distortion-bounded rescue could fix downstream. Repair must detect
		// at the resolution shipping is judged, so the bisect happens HERE,
		// where charts can still be split.
		return CountRealFlips(cm, uv) != 0 || ChartUVSelfOverlaps(cm, uv, /*gridLongSide=*/512);
	};
	const auto accept = [&](std::vector<Vec2>& uv, bool finalUv) {
		if (out != nullptr) {
			out->uv = std::move(uv);
			out->finalUv = finalUv;
			out->valid = true;
		}
	};
	// diag support for a mapFolds() == true verdict: re-probe with collectors.
	const auto diagMapFolds = [&](const std::vector<Vec2>& uv) {
		if (diag == nullptr)
			return;
		std::vector<int> local;
		if (uv.size() == cm.verts.size()) {
			std::vector<int> flipped, colliding;
			CountRealFlips(cm, uv, &flipped);
			ChartUVSelfOverlaps(cm, uv, /*gridLongSide=*/512, &colliding);
			local.insert(local.end(), flipped.begin(), flipped.end());
			local.insert(local.end(), colliding.begin(), colliding.end());
		} else {
			// Unusable map (uv/vert count mismatch — a safety net, not expected
			// in practice): no reliable per-face signal, so the whole chart is
			// the carve target.
			local.resize(cm.faces.size());
			std::iota(local.begin(), local.end(), 0);
		}
		FillFoldDiagnosis(cm, local, diag);
	};

	// Judge the SHIPPED map (init + SLIM), always. The old fast path judged the
	// INIT only, assuming refinement preserves what matters — SLIM's barrier
	// does preserve LOCAL flip-freedom, but not GLOBAL injectivity, and on the
	// challenge fixture the refined maps of charts covering ~36% of the surface
	// folded back over themselves with flip-free inits (median Tutte-rescue
	// sym-Dir 554 — no downstream fallback can fix charts of that extent, they
	// must be BISECTED, which only the repair can do). The flatten computed
	// here is not wasted: an accepting verdict deposits it in the
	// ChartFlattenCache (finalUv=true), and ParametrizeCharts ships it without
	// re-flattening — the SLIM cost moves into the repair, it is not doubled.
	// A non-disk / un-flattenable chart (FlattenChart false) is the
	// PCA-fallback case — a fold there is a real fold to bisect.
	{
		std::vector<Vec2> uv;
		if (!FlattenChart(cm, params, uv)) {
			FlattenChartFallback(cm, uv);
			if (mapFolds(uv)) {
				diagMapFolds(uv);
				return true;
			}
			accept(uv, /*finalUv=*/true); // ships this PCA map (no SLIM after fallback)
			return false;
		}
		if (mapFolds(uv)) {
			diagMapFolds(uv);
			return true;
		}
		if (params.developableMaxUvDistortion > 0.0f && ChartOverDistorted(cm, uv, params.developableMaxUvDistortion)) {
			if (diag != nullptr) {
				std::vector<int> over;
				ChartOverDistorted(cm, uv, params.developableMaxUvDistortion, &over);
				FillFoldDiagnosis(cm, over, diag);
			}
			return true;
		}
		accept(uv, /*finalUv=*/true); // ships this full init+SLIM map
		return false;
	}
}

} // namespace

// Bridge for the segmentation flip-repair (src/AtlasCharting.cpp): build the chart
// submesh and report whether it folds when flattened as shipped. Lives at
// halfmesh::detail (external linkage) so AtlasCharting can call it, yet wraps the
// anonymous-namespace flattener internals above (same translation unit).
namespace detail {

// One accepted chart's flatten artifacts (declared opaque in ChartFlattenCache.h;
// ChartMesh is TU-local, so the definition lives here).
//   - cm: the chart submesh in EXACTLY the state FlattenChart would hold at the
//     accepted stage (cut-to-disk applied, boundary loop of the accepted init).
//     Required because CutChartToDisk duplicates vertices, so UVs alone cannot
//     be written back through an uncut re-extraction.
//     cm.globalFid is untouched by cutting → it doubles as the chart's identity
//     (== the sorted global face list the verdict was computed for).
//   - uv/finalUv: see FoldAccept.
struct ChartFlattenEntry
{
	ChartMesh cm;
	std::vector<Vec2> uv;
	bool finalUv = false;
};

ChartFlattenSlot::~ChartFlattenSlot()
{
	delete entry;
}

ChartFlattenSlot& ChartFlattenSlot::operator=(ChartFlattenSlot&& o) noexcept
{
	if (this != &o) {
		delete entry;
		entry = o.entry;
		o.entry = nullptr;
	}
	return *this;
}

void ChartFlattenCache::Store(ChartFlattenSlot&& slot)
{
	if (slot.entry == nullptr || slot.entry->cm.globalFid.empty())
		return;
	ASSERT(std::is_sorted(slot.entry->cm.globalFid.begin(),
	                      slot.entry->cm.globalFid.end()));
	// Chart identity = smallest global face id (face lists are sorted ascending).
	// The key is unique across live charts and invariant under chart-id
	// relabelling (Compact). Post-repair merge rounds mutate accepted charts'
	// face sets, so stale entries may exist; Find's full-face-list verification
	// makes them unreachable, at the cost of retaining dead entries until
	// segmentation returns.
	entries[slot.entry->cm.globalFid.front()] = std::move(slot);
}

ChartFlattenEntry* ChartFlattenCache::Find(const std::vector<Mesh::FIndex>& globalFid) const
{
	if (globalFid.empty())
		return nullptr;
	const auto it = entries.find(globalFid.front());
	if (it == entries.end() || it->second.entry == nullptr)
		return nullptr;
	// Re-verify full chart identity: the key is only the smallest face id.
	if (it->second.entry->cm.globalFid != globalFid)
		return nullptr;
	return it->second.entry;
}

bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params)
{
	return ChartFacesFold(mesh, faces, params, nullptr);
}

bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params, ChartFlattenSlot* out)
{
	return ChartFacesFold(mesh, faces, params, out, nullptr);
}

bool ChartFacesFold(const Mesh& mesh, const std::vector<Mesh::FIndex>& faces,
                    const ParametrizeParams& params, ChartFlattenSlot* out,
                    FoldDiagnosis* diag)
{
	ChartMesh cm = ExtractOneChart(mesh, faces);
	FoldAccept acc;
	if (ChartFolds(cm, params, out != nullptr ? &acc : nullptr, diag))
		return true;
	if (out != nullptr && acc.valid) {
		// The chart ships: hand the accepting verdict's artifacts to the caller.
		ChartFlattenSlot slot;
		slot.entry = new ChartFlattenEntry;
		slot.entry->cm = std::move(cm); // post-ChartFolds state (cut applied if any)
		slot.entry->uv = std::move(acc.uv);
		slot.entry->finalUv = acc.finalUv;
		*out = std::move(slot);
	}
	return false;
}

} // namespace detail

// ===========================================================================
// Public entry points (Module B).
// ===========================================================================

namespace detail {

// Cache-aware ParametrizeCharts (ChartFlattenCache.h). For every chart the
// flip-repair accepted (cache hit, verified against the FULL global face list)
// the accepting verdict's artifacts are reused: a final map is written back
// directly; an init-stage map resumes FlattenChart's tail (LocalGlobal + finite
// check → PCA fallback), which is bitwise the map FlattenChart would produce
// because the inputs (cut ChartMesh + accepted init) ARE FlattenChart's state at
// that point. A miss — no repair ran, a <=2-face chart, a runaway-capped wave,
// or any identity mismatch — recomputes from scratch, so the OUTPUT never
// depends on cache state; only the duplicate flatten work does.
void ParametrizeCharts(Mesh& mesh, const std::vector<unsigned>& faceChart,
                       unsigned numCharts, const ParametrizeParams& params,
                       ChartFlattenCache* cache)
{
	mesh.SyncFaces();
	const size_t nf = mesh.faces.size();
	mesh.faceTexcoords.assign(nf * 3, Mesh::TexCoord::Zero());
	if (nf == 0 || numCharts == 0)
		return;

	// Ensure adjacency exists (used for chart vertex/face structure; the
	// flattener itself works on the extracted local submesh).
	if (mesh.halfMesh.Empty())
		mesh.ListHalfEdges();

	std::vector<ChartMesh> charts = ExtractCharts(mesh, faceChart, numCharts);

#ifdef HM_ATLAS_DEBUG
	gFlatLscm = gFlatTutte = gFlatPca = gFlatCut = gFlatCached = 0;
#endif
	// Charts are flattened independently: cm and uv are chart-local, and the
	// faceTexcoords writeback slots are disjoint (every global face belongs to
	// exactly one chart). Chart flatten cost is superlinear in face count
	// (O(V^1.5) factorizations) and chart sizes are heavy-tailed, so a static
	// contiguous block partition lets the block holding the biggest chart bound
	// the wall time while other workers idle. Submit ONE task per non-empty chart
	// in descending face-count order (LPT / longest-processing-time scheduling,
	// 4/3-optimal makespan) so a freed worker always pulls the next-largest chart.
	// Each task touches only its own chart and disjoint writeback slots, so the
	// output is bitwise identical regardless of execution order (a pure function
	// of chart geometry — determinism preserved).
	std::vector<std::size_t> order;
	order.reserve(charts.size());
	for (std::size_t ci = 0; ci < charts.size(); ++ci)
		if (!charts[ci].faces.empty())
			order.push_back(ci);
	std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
		if (charts[a].faces.size() != charts[b].faces.size())
			return charts[a].faces.size() > charts[b].faces.size();
		return a < b; // stable tie-break (scheduling only; does not affect output)
	});
	// Charts whose self-overlapping refined map survived the fallback ladder
	// (no injective alternative within kFallbackMaxSymDir) — reported once below.
	std::atomic<unsigned> keptFolded{0};
	auto flattenOne = [&](std::size_t ci) {
		ChartMesh& ecm = charts[ci];
		// Look up the flip-repair's accepted artifacts for this chart. Find()
		// enforces the verify-before-reuse contract (key = smallest global face
		// id, hit confirmed against the FULL face list — globalFid survives
		// cutting untouched), so a stale or colliding entry can only ever
		// downgrade to a recompute, never corrupt. Concurrent Find() from
		// workers is safe, and each hit entry belongs to exactly one chart →
		// one task; the only writes to an entry are by its own task (the
		// injectivity fallback below and the trailing release).
		ChartFlattenEntry* cached = (cache != nullptr) ? cache->Find(ecm.globalFid) : nullptr;
		std::vector<Vec2> uv;
		ChartMesh* wb = &ecm; // writeback source (the cut cm on a cache hit)
		if (cached != nullptr) {
			// Every deposit carries the SHIPPED map (finalUv == true — ChartFolds
			// always judges the full flatten), so a hit needs no resume work.
			ASSERT(cached->finalUv);
			uv = std::move(cached->uv);
			HM_FLATTEN_TALLY(gFlatCached);
			wb = &cached->cm;
		} else if (!FlattenChart(ecm, params, uv)) {
			FlattenChartFallback(ecm, uv); // closed chart / degenerate: PCA project
			HM_FLATTEN_TALLY(gFlatPca);
		}

		// Global-injectivity guard on the SHIPPED map (2026-08 review): SLIM/ARAP
		// refinement can fold a flip-free init back over itself GLOBALLY (zero
		// flipped triangles, doubly-covered UV regions — the bake then paints one
		// region with the other's colors). If the final map self-overlaps, fall
		// back to the Tutte embedding — provably injective on a disk with convex
		// boundary (Tutte's theorem) — trading distortion for correctness on just
		// this chart. Non-disk charts where Tutte is unavailable keep the refined
		// map (the PCA-fallback cases never carried an injectivity guarantee).
		// Cache hits skip the probe: the repair's accepting verdict already ran
		// this exact ChartUVSelfOverlaps(cm, uv, 512) on the identical pair and
		// saw it clean — re-checking would be pure duplicate rasterization.
		if (cached == nullptr && ChartUVSelfOverlaps(*wb, uv, /*gridLongSide=*/512)) {
			// Fallback ladder, best distortion first:
			//   1. Tutte on the (cut-to-)disk chart — provably injective
			//      (convex-boundary embedding, Tutte's theorem).
			//   2. The un-refined LSCM init, if IT is overlap-clean — covers
			//      multi-boundary-loop charts CutChartToDisk cannot reduce
			//      (holey scan patches), where the fold was introduced by the
			//      SLIM/ARAP refinement on top of a clean conformal init.
			//   3. Keep the refined map (no injective alternative exists).
			// Work on a COPY for the cut: CutChartToDisk duplicates vertices, so
			// a failed Tutte attempt on the original would leave `uv`
			// inconsistent with the chart it is written back through. globalFid
			// survives cutting, so shipping the cut copy keeps the writeback
			// identity intact.
			// A fallback is only an improvement if it does not buy injectivity
			// with catastrophic stretch: cap its area-weighted sym-Dir (4 =
			// perfect isometry). 200 admits the heavily-stretched-but-usable
			// Tutte embeddings that rescue genuinely folding disk charts, while
			// refusing the runaway ones (unbounded Tutte on complex charts blew
			// the mesh-wide average sym-Dir pin by two orders of magnitude).
			// The repair loop judges the SHIPPED map at the same probe, so this
			// ladder is a backstop for numerical drift, not the primary defence.
			constexpr float kFallbackMaxSymDir = 200.f;
			ChartMesh cut = *wb;
			int numLoops = 0;
			bool hasB = BuildBoundaryLoop(cut, numLoops);
			if ((!hasB || numLoops != 1) && CutChartToDisk(cut))
				hasB = BuildBoundaryLoop(cut, numLoops);
			// Aspect-matched target boundary: the unit circle forces elongated
			// charts into a disk, and the scan fixture's thin-strip charts then
			// fail the distortion gate wholesale (measured: every one of the
			// 207 rescue attempts shipped its folds — Tutte-to-circle sym-Dir
			// above the cap, LSCM flip-free but self-overlapping). Pin the
			// boundary to an ellipse whose aspect is the chart's principal 3D
			// extent ratio instead — still strictly convex, so Tutte's theorem
			// is untouched — started at the boundary vertex extreme along the
			// principal axis so the strip's ends land on the ellipse's ends.
			double aspect = 1.0;
			int startK = 0;
			if (hasB && numLoops == 1 && !cut.boundary.empty()) {
				Eigen::Vector3d mean = Eigen::Vector3d::Zero();
				for (const Vec3& p : cut.verts)
					mean += p.cast<double>();
				mean /= static_cast<double>(cut.verts.size());
				Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
				for (const Vec3& p : cut.verts) {
					const Eigen::Vector3d d = p.cast<double>() - mean;
					cov += d * d.transpose();
				}
				const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
				const Eigen::Vector3d axis = es.eigenvectors().col(2); // largest
				const double l1 = es.eigenvalues()[2], l2 = es.eigenvalues()[1];
				if (l2 > 0.0)
					aspect = std::clamp(std::sqrt(l1 / l2), 1.0, 32.0);
				double best = std::numeric_limits<double>::lowest();
				for (int k = 0; k < static_cast<int>(cut.boundary.size()); ++k) {
					const double s =
					    axis.dot(cut.verts[cut.boundary[k]].cast<double>() - mean);
					if (s > best) {
						best = s;
						startK = k;
					}
				}
			}
			std::vector<Vec2> uvT;
			if (hasB && numLoops == 1 && TutteInit(cut, uvT, aspect, startK) && !ChartUVSelfOverlaps(cut, uvT, /*gridLongSide=*/512) && !ChartOverDistorted(cut, uvT, kFallbackMaxSymDir)) {
				*wb = std::move(cut); // on a cache hit this replaces the entry's
				// cm — safe: the entry is owned by this
				// task alone and released below
				uv = std::move(uvT);
			} else if (LscmInit(*wb, uvT) && CountRealFlips(*wb, uvT) == 0 && !ChartUVSelfOverlaps(*wb, uvT, /*gridLongSide=*/512) && !ChartOverDistorted(*wb, uvT, kFallbackMaxSymDir)) {
				uv = std::move(uvT);
			} else {
				// Keep the refined map — no injective alternative within the
				// distortion budget exists for this chart (documented residual;
				// reported once after the parallel flatten below).
				++keptFolded;
			}
		}

		// Write per-face-corner UVs back into mesh.faceTexcoords. Local vertex id
		// of corner k of local face lf maps to uv[lf[k]]; the global face id is
		// wb->globalFid[t]. Write in the SAME corner order as the original global
		// face's vertices.
		for (size_t t = 0; t < wb->faces.size(); ++t) {
			const Mesh::FIndex gf = wb->globalFid[t];
			const Eigen::Vector3i& lf = wb->faces[t];
			for (int k = 0; k < 3; ++k) {
				const Vec2& p = uv[lf[k]];
				mesh.faceTexcoords[gf * 3 + k] =
				    Mesh::TexCoord(static_cast<Mesh::Type>(p.x()),
				                   static_cast<Mesh::Type>(p.y()));
			}
		}
		if (cached != nullptr)
			cached->cm = ChartMesh{}; // release the consumed bulk early (value-only
		// mutation of this task's entry — no rehash)
	};
	BS::light_thread_pool pool;
	for (std::size_t ci : order)
		pool.detach_task([ci, &flattenOne]() { flattenOne(ci); });
	pool.wait();
	if (keptFolded.load() != 0)
		REPORT_WARNING("ParametrizeCharts: {} chart(s) kept a self-overlapping map — no injective "
		               "fallback within the distortion cap; expect localized double-painted texels "
		               "on those charts",
		               keptFolded.load());
#ifdef HM_ATLAS_DEBUG
	// Load the atomic tallies to plain int — a std::atomic cannot be passed through
	// fprintf's variadic ellipsis (its copy constructor is deleted). `cached`
	// counts charts reused from the flip-repair (their init tallies happened
	// during the repair probes, not here).
	std::fprintf(stderr, "[flatten] charts=%u lscm=%d tutte=%d cut2disk=%d pca_fallback=%d cached=%d\n",
	             numCharts, gFlatLscm.load(), gFlatTutte.load(), gFlatCut.load(), gFlatPca.load(),
	             gFlatCached.load());
#endif
}

} // namespace detail

void ParametrizeCharts(Mesh& mesh, const std::vector<unsigned>& faceChart,
                       unsigned numCharts, const ParametrizeParams& params)
{
	detail::ParametrizeCharts(mesh, faceChart, numCharts, params, nullptr);
}

unsigned Parametrize(Mesh& mesh, const ParametrizeParams& params)
{
	mesh.SyncFaces();
	// Segmentation and flattening share one flatten cache: the flip-repair's final
	// accepting verdict per shipping chart already computed its map (or its init),
	// so ParametrizeCharts resumes instead of flattening every chart twice.
	std::vector<unsigned> faceChart;
	detail::ChartFlattenCache cache;
	const unsigned numCharts = detail::SegmentCharts(mesh, params, faceChart, &cache);
	detail::ParametrizeCharts(mesh, faceChart, numCharts, params, &cache);
	return numCharts;
}

} // namespace halfmesh
