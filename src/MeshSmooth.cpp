/*
* MeshSmooth.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

#include <halfmesh/Mesh.h>
#include <halfmesh/Util/Accumulator.h>
#include <halfmesh/Util/Loop.h>

#include <vector>
#include <BS_thread_pool.hpp>

#include "ParallelFor.h"

namespace halfmesh {
namespace {

using detail::ParallelForPool;

} // namespace

// ---------------------------------------------------------------------------
// SmoothHCLaplacian
// HC Laplacian smoothing — Vollmer, Mencl & Mueller, "Improved Laplacian
// Smoothing of Noisy Surface Meshes", EUROGRAPHICS 1999.  Per iteration:
//   pass 1: avg_i = uniform one-ring average of the current positions
//   pass 2: dif_i = one-ring average of the neighbors' corrections
//           b_n = avg_n - p_n
//   pass 3: p_i' = avg_i - beta*(avg_i - p_i) - (1-beta)*dif_i,  beta = 0.5
// The correction pulls each vertex back toward its pre-step position, largely
// avoiding the volume shrinkage of plain Laplacian smoothing.
//
// Implementation notes (don't "fix" these without golden + quality coverage):
//  - the dif term is SUBTRACTED, per the paper.  Adding it instead makes the
//    filter degenerate toward plain Laplacian: measured on a noisy 16x24 UV
//    sphere, ~2.4%/iter mean-radius shrink with the radial spread regrowing
//    past the input noise after ~2 iterations, versus monotone denoising at
//    <=0.1%/iter here.
//  - iterating every directed halfedge once (boundary halfedges included)
//    yields the uniform one-ring average directly, with no border special-case.
//  - locked vertices keep their position but still contribute to their
//    neighbors' avg/dif.
// ---------------------------------------------------------------------------
void Mesh::SmoothHCLaplacian(int iterations, const std::vector<bool>* lockedVertices)
{
	static_assert(HALFMESH_TRIS, "mesh faces must be triangles");
	if (vertices.empty() || (faces.empty() && halfMesh.Empty()))
		return;
	SyncFaces();
	if (iterations <= 0)
		return;
	ASSERT(lockedVertices == NULL || lockedVertices->size() == vertices.size());
	ListHalfEdges();
	ASSERT(ValidateInvariants());
	// re-check post-build: repair (triggered when the input is non-manifold)
	// can weld/remove or add vertices, remapping indices, so the mask must
	// describe the post-build mesh, not the caller's original one
	ASSERT(lockedVertices == NULL || lockedVertices->size() == vertices.size());
	const HalfMesh& m = halfMesh;
	const Type beta(0.5f);
	const Type difWeight(Type(1) - beta);
	typedef WeightedAccumulator<Vertex> LaplacianInfo;
	std::vector<LaplacianInfo> sums(vertices.size());
	std::vector<LaplacianInfo> difs(vertices.size());
	std::vector<Vertex> avgs(vertices.size());
	// The halfedge SCATTER passes stay serial (their floating-point accumulation
	// order is the result); the per-vertex MAP passes below write only their own
	// slot from already-accumulated data, so they parallelize bit-identically.
	BS::light_thread_pool pool;
	for (int iter = 0; iter < iterations; ++iter) {
		// pass 1: uniform one-ring average for EVERY vertex (locked ones too,
		// as their average feeds the neighbors' correction term below); each
		// directed halfedge contributes its head to its tail exactly once
		std::fill(sums.begin(), sums.end(), LaplacianInfo(Vertex::Zero(), 0));
		FOREACHRAWIDX (HalfMesh::HIndex, iHe, m.HeSize())
			sums[m.HeTailVertex(iHe)].Add(vertices[m.HeHeadVertex(iHe)]);
		ParallelForPool(pool, vertices.size(), [&](std::size_t i) {
			const VIndex iV = static_cast<VIndex>(i);
			avgs[iV] = sums[iV].Empty() ? vertices[iV] : sums[iV].Normalized();
		});
		// pass 2: one-ring average of the neighbors' corrections b_n
		std::fill(difs.begin(), difs.end(), LaplacianInfo(Vertex::Zero(), 0));
		FOREACHRAWIDX (HalfMesh::HIndex, iHe, m.HeSize()) {
			const VIndex iN(m.HeHeadVertex(iHe));
			difs[m.HeTailVertex(iHe)].Add(avgs[iN] - vertices[iN]);
		}
		// pass 3: HC update; passes 1-2 read only pre-update positions, and
		// each vertex here reads only its own, so in-place update is safe
		ParallelForPool(pool, vertices.size(), [&](std::size_t i) {
			const VIndex iV = static_cast<VIndex>(i);
			if (sums[iV].Empty())
				return; // unreferenced vertex
			if (lockedVertices != NULL && (*lockedVertices)[iV])
				return;
			Vertex& p = vertices[iV];
			const Vertex& avg = avgs[iV];
			p = avg - (avg - p) * beta - difs[iV].Normalized() * difWeight;
		});
	}
	// vertex-only motion defeats the size-based freshness gate (faceNormals.size()
	// == faces.size() still holds, since faces are untouched) — explicitly
	// invalidate, mirroring CloseHoles's faceNormals.clear() (src/MeshHoles.cpp)
	// for any change that stales cached normals; Mesh has no vertexNormals-style
	// cache to mirror (ComputeVertexNormals() always recomputes fresh).
	faceNormals.clear();
	SyncFaces();
	ASSERT(ValidateInvariants());
}

// ---------------------------------------------------------------------------
// SmoothTaubin
// Taubin lambda|mu smoothing — Taubin, "A Signal Processing Approach To Fair
// Surface Design", SIGGRAPH 1995.  Per iteration, two uniform-Laplacian steps:
//   pass 1: p_i += lambda * (avg_i - p_i)   (lambda > 0, shrink)
//   pass 2: p_i += mu     * (avg_i - p_i)   (mu < -lambda, inflate)
// The per-frequency transfer function (1 - lambda*k)(1 - mu*k), k in [0, 2],
// is a band-pass: high frequencies (noise) are damped while low frequencies
// (shape/volume) keep gain slightly above 1 — hence no shrinkage, but very
// long runs slowly inflate (~1%/100 iterations measured), so prefer <= ~100
// iterations.  This is the "smooth aggressively without shrinking" smoother:
// much stronger denoising than HC at ~zero net shrink, given ~20-40x the
// (equally-priced) iterations.
// Each pass fully accumulates the one-ring averages BEFORE moving any vertex.
// Border vertices are smoothed along the boundary curve only — unlike the HC
// smoother's uniform ring, see its notes above.  Locked vertices keep their
// position but still contribute to their neighbors' averages.
// See also the chart-segmentation denoise in AtlasCharting.cpp: a separate,
// intentionally different Taubin pass that PINS border vertices (rather than
// smoothing them along the boundary curve as here) and runs on a temporary
// double-precision copy — the two are not meant to be consolidated.
// ---------------------------------------------------------------------------
void Mesh::SmoothTaubin(int iterations, Type lambda, Type mu, const std::vector<bool>* lockedVertices)
{
	static_assert(HALFMESH_TRIS, "mesh faces must be triangles");
	if (vertices.empty() || (faces.empty() && halfMesh.Empty()))
		return;
	SyncFaces();
	if (iterations <= 0)
		return;
	// documented parameter contract: lambda in (0, 1), mu negative with |mu| > lambda
	// (mu < -lambda) so the inflate step slightly overshoots the shrink step (no net shrink)
	ASSERT(lambda > Type(0) && lambda < Type(1) && mu < -lambda);
	ASSERT(lockedVertices == NULL || lockedVertices->size() == vertices.size());
	ListHalfEdges();
	ASSERT(ValidateInvariants());
	// re-check post-build: repair can weld/remove or add vertices, remapping
	// indices (see the identical note in SmoothHCLaplacian above)
	ASSERT(lockedVertices == NULL || lockedVertices->size() == vertices.size());
	const HalfMesh& m = halfMesh;
	typedef WeightedAccumulator<Vertex> LaplacianInfo;
	std::vector<LaplacianInfo> sums(vertices.size());
	// BORDER vertices smooth along the boundary curve only: their average is
	// seeded with the vertex itself and accumulates just the border-edge
	// neighbors.  Do NOT "unify" this with SmoothHCLaplacian's uniform ring —
	// the two filters deliberately differ on borders and each has its own
	// pinned tests.  Topology is constant while smoothing, so detect border
	// vertices once.
	std::vector<bool> borderVertices(vertices.size(), false);
	FOREACHRAWIDX (HalfMesh::HIndex, iHe, m.HeSize())
		if (m.HeIsBoundary(iHe)) {
			borderVertices[m.HeTailVertex(iHe)] = true;
			borderVertices[m.HeHeadVertex(iHe)] = true;
		}
	// the halfedge scatter stays serial (accumulation order = result); the
	// per-vertex update writes only its own slot (including the border
	// self-seed into sums[iV]) so it parallelizes bit-identically
	BS::light_thread_pool pool;
	const auto pass = [&](const Type weight) {
		// one-ring averages fully accumulated BEFORE any position moves:
		// uniform ring for interior vertices, border-edge neighbors only for
		// border vertices (locked ones still contribute)
		std::fill(sums.begin(), sums.end(), LaplacianInfo(Vertex::Zero(), 0));
		FOREACHRAWIDX (HalfMesh::HIndex, iHe, m.HeSize()) {
			const VIndex iTail(m.HeTailVertex(iHe));
			if (borderVertices[iTail] && !m.HeIsBoundary(iHe) && !m.HeIsBoundary(m.HeTwin(iHe)))
				continue; // border vertex: interior edges do not contribute
			sums[iTail].Add(vertices[m.HeHeadVertex(iHe)]);
		}
		ParallelForPool(pool, vertices.size(), [&](std::size_t i) {
			const VIndex iV = static_cast<VIndex>(i);
			if (sums[iV].Empty())
				return; // unreferenced vertex
			if (lockedVertices != NULL && (*lockedVertices)[iV])
				return;
			Vertex& p = vertices[iV];
			if (borderVertices[iV])
				sums[iV].Add(p); // the border average is seeded with the vertex itself
			p += (sums[iV].Normalized() - p) * weight;
		});
	};
	for (int iter = 0; iter < iterations; ++iter) {
		pass(lambda);
		pass(mu);
	}
	// vertex-only motion defeats the size-based freshness gate — see the
	// identical note at the end of SmoothHCLaplacian above
	faceNormals.clear();
	SyncFaces();
	ASSERT(ValidateInvariants());
}

// ---------------------------------------------------------------------------
// Unified smoothing entry point.  A thin dispatcher over the individual
// smoothers, each invoked with its per-function default parameters; the method
// selects the algorithm (see Mesh::SmoothMethod).  All the shared behavior
// (repair-on-build, no-op guards, locked-vertex and normal-cache semantics)
// lives in the callees, so nothing is duplicated here.
// ---------------------------------------------------------------------------
void Mesh::Smooth(int iterations, SmoothMethod method)
{
	// the switch stays exhaustive without a default arm so -Wswitch flags any
	// new enumerator; an out-of-range cast value falls through to the ASSERT
	// below instead of silently no-opping
	switch (method) {
	case SmoothMethod::Taubin:
		SmoothTaubin(iterations);
		return;
	case SmoothMethod::HCLaplacian:
		SmoothHCLaplacian(iterations);
		return;
	}
	ASSERT(false && "Mesh::Smooth: invalid SmoothMethod");
}

} // namespace halfmesh
