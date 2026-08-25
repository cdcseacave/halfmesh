/*
* MeshSimplify.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// Mesh::Simplify — QEM edge-collapse decimation.
// Uses Vector3f (float) for face errors via Mesh::Normal (Eigen::Matrix<Type,3,1>
// with Type=float). ComputeError() returns real (double); results are narrowed
// to Type via cast.

#include <halfmesh/Mesh.h>
#include <halfmesh/HalfMesh.h>
#include <halfmesh/Quadric.h>
#include <halfmesh/PriorityQueue.h>
#include <halfmesh/Util/Loop.h>
#include <halfmesh/Util/Assert.h>
#include <halfmesh/Util/Maths.h>
#include <halfmesh/Util/Log.h>

#include <vector>
#include <cmath>
#include <cstddef>
#include <BS_thread_pool.hpp>

#include "ParallelFor.h"

namespace halfmesh {
namespace {

using detail::ParallelForPool;

} // namespace

// Quadric Error Metric (QEM) edge-collapse decimation, after Garland & Heckbert,
// "Surface Simplification Using Quadric Error Metrics" (1997).
//
// Each vertex accumulates a 4x4 quadric Q that measures the sum of squared
// distances from a point to the supporting planes of its incident faces. The cost
// of collapsing an edge (v0,v1) is the quadric error of the merged quadric
// Q0+Q1 evaluated at the optimal collapse position (the point minimising that
// quadric, found by solving Q's linear system). Lower error => collapse first.
//
// Two strategies are offered, selected by `aggressiveness`:
//   * aggressiveness > 0 — "fast" mode: no global priority queue. Faces are swept
//     repeatedly; on each pass an error threshold is raised geometrically and every
//     edge below it is collapsed. Cheap and good enough for large meshes.
//   * aggressiveness == 0 — "exact" mode: a priority queue always collapses the
//     globally cheapest edge, re-costing affected edges after each collapse.
//
// Stopping: `decimateRatio` in (0,1] sets a target face count (ratio*faces); or,
// exclusively, `minEdgeLength` collapses every edge shorter than the given length.
void Mesh::Simplify(float decimateRatio, float minEdgeLength, float aggressiveness)
{
	// One stopping rule: decimateRatio (<1 = fraction of input, >1 = absolute
	// target face count) or minEdgeLength (>0); the two are mutually exclusive,
	// so decimateRatio must be 1 (identity, disabled) when minEdgeLength is set.
	// An empty mesh and the documented identity call (ratio == 1, no
	// minEdgeLength) are graceful no-ops in every build mode — matching the
	// early-return convention of the smoothers and CloseHoles.
	if (vertices.empty() || (faces.empty() && halfMesh.Empty()))
		return;
	// The identity guard precedes ListHalfEdges(): a build is not free, and on
	// non-manifold input the safe path repairs/manifoldizes in place, so probing
	// connectivity first would let an identity call mutate topology.
	if (minEdgeLength <= 0 && decimateRatio == 1.f) {
		SyncFacesOnPublicExit();
		return; // identity: nothing to decimate
	}
	ListHalfEdges();
	ASSERT(ValidateInvariants());
	ASSERT(decimateRatio > 0);
	ASSERT(minEdgeLength <= 0 || decimateRatio == 1.f);
	TIMER_START("Simplify");
	const size_t numFaces = halfMesh.FSize();
	const size_t numTargetFaces(minEdgeLength > 0     ? 1u
	                            : decimateRatio > 1.f ? static_cast<size_t>(std::llround(decimateRatio))
	                                                  : RoundCast<size_t>(numFaces * decimateRatio));
	BS::light_thread_pool pool; // persistent worker pool for the parallel setup phases
	// Build one quadric per vertex: the sum of the plane quadrics (n, d) of every
	// incident face, where the plane is n.x + d = 0 with unit normal n. Summing
	// these makes verticesQuadric[v] measure the squared distance of a candidate
	// point to v's local surface.
	typedef TQuadric<real> Quadric;
	const float discontinuityMultiplier = 3;
	std::vector<Quadric> verticesQuadric(vertices.size(), Quadric(0, 0, 0, 0)); // quadric approximation of the vertex local surface
	// The per-face plane normal and quadric are independent, so compute them in
	// parallel into face-sized arrays. The scatter-accumulation into the shared
	// verticesQuadric then runs sequentially in the original face/half-edge order,
	// keeping the floating-point summation order (hence the result) bit-identical —
	// determinism is a hard invariant.
	std::vector<Normal> facesNormal(numFaces);
	std::vector<Quadric> facesQuadric(numFaces);
	ParallelForPool(pool, numFaces, [&](std::size_t iF) {
		const Face face = halfMesh.F(static_cast<FIndex>(iF));
		const Normal normal = ComputeFaceNormal(face).normalized();
		facesNormal[iF] = normal;
		const Type d = -normal.dot(vertices[face(0)]);
		facesQuadric[iF] = Quadric(normal.x(), normal.y(), normal.z(), d);
	});
	for (FIndex iF = 0; iF < numFaces; ++iF) {
		const Normal& normal = facesNormal[iF];
		const Quadric& q = facesQuadric[iF];
		for (HIndex iHe : halfMesh.FAdjacentHalfedges(iF)) {
			const VIndex iV0 = halfMesh.HeTailVertex(iHe);
			verticesQuadric[iV0] += q;
			if (!halfMesh.EHeIsBoundary(iHe))
				continue;
			// Boundary edges have no face on the far side, so without help the
			// optimizer would freely slide them inward. Add a high-weight "virtual"
			// quadric for a plane perpendicular to the face through the boundary
			// edge; this pins boundary vertices and preserves the silhouette.
			// extend border surface with fake surface pulling the edge towards exterior
			const VIndex iV1 = halfMesh.HeHeadVertex(iHe);
			const Normal discNormal = (vertices[iV1] - vertices[iV0]).cross(normal).normalized() * discontinuityMultiplier;
			const Type discD = -discNormal.dot(vertices[iV1]);
			const Quadric qDiscontinuity(discNormal.x(), discNormal.y(), discNormal.z(), discD);
			verticesQuadric[iV0] += qDiscontinuity;
			verticesQuadric[iV1] += qDiscontinuity;
		}
	}
	if (aggressiveness > 0) {
		// Fast mode: no global priority queue. Precompute a per-face error (the
		// collapse error of each of the face's three edges) and sweep all faces over
		// up to 100 passes, raising an acceptance threshold geometrically each pass
		// (threshold = (iter+3)^aggressiveness * 1e-9 * diag^2). Any edge below the
		// current threshold whose collapse is valid is collapsed at once; affected
		// faces are re-costed and marked dirty so they are skipped for the rest of
		// the pass. Larger `aggressiveness` raises the threshold faster (fewer
		// passes, lower quality).
		// The quadric error is a squared distance (units length^2), so the raw 1e-9
		// base is anchored to the squared bounding-box diagonal, making the schedule
		// invariant under uniform model scaling (a mm-scale and a km-scale copy reach
		// the same face target with the same relative quality); computed once here,
		// O(V), before the sweep. compute face errors (one per edge of each face)
		const Eigen::AlignedBox<Type, 3> bbox = ComputeAABBox();
		const real bboxDiag2 = (bbox.max() - bbox.min()).cast<real>().squaredNorm();
		// Each face's three edge errors are independent (pure function of the two
		// endpoint quadrics/positions), so precompute them in parallel; every
		// facesError[iF] is written by exactly one task, so the result is
		// bit-identical to the sequential build.
		std::vector<Normal> facesError(numFaces);
		ParallelForPool(pool, numFaces, [&](std::size_t iF) {
			const Face face = halfMesh.F(static_cast<FIndex>(iF));
			Normal& faceError = facesError[iF];
			for (int i = 0; i < 3; ++i) {
				const VIndex iV0 = face(i), iV1 = face((i + 1) % 3);
				faceError(i) = static_cast<Type>((verticesQuadric[iV0] + verticesQuadric[iV1])
				                                     .ComputeError(vertices[iV0].cast<real>(), vertices[iV1].cast<real>()));
			}
		});
		vertexColors.clear();
		InvalidateFaces();
		// iterates over all faces, increasing the threshold at each iteration
		std::vector<bool> facesDirty(numFaces);
		const size_t maxNumIterations = 100;
		for (size_t iteration = 0; iteration < maxNumIterations; ++iteration) {
			const Type threshold = static_cast<Type>(std::pow(real(iteration + 3), aggressiveness) * real(1e-9) * bboxDiag2);
			std::fill(facesDirty.begin(), facesDirty.end(), false);
			// remove triangles with an edge error bellow accepted error threshold
			RFOREACH (iF, halfMesh.fHalfedges) {
				if (facesDirty[iF])
					continue;
				Normal::Index i;
				if (facesError[iF].minCoeff(&i) > threshold)
					continue;
				const Face face = halfMesh.F(iF);
				const VIndex iV0 = face(i), iV1 = face((i + 1) % 3);
				const EIndex iE = halfMesh.EEdge(iV0, iV1);
				if (!halfMesh.EIsCollapseValidTopologically(iE))
					continue;
				const Quadric quadric = verticesQuadric[iV0] + verticesQuadric[iV1];
				const Vertex p = quadric.ComputeOptimalPoint(vertices[iV0].cast<real>(), vertices[iV1].cast<real>()).cast<Vertex::Scalar>();
				if (!halfMesh.EIsCollapseValidGeometrically(iE, p, vertices))
					continue;
				// Collapse the edge. ERemove deletes one vertex, the edge, and its
				// incident faces, and renumbers the half-edge arrays by swapping the
				// removed slots with the last ones. We mirror that swap-with-last
				// compaction in our parallel arrays (vertices / verticesQuadric /
				// facesError / facesDirty) so indices stay in sync, then move the
				// surviving vertex to the optimal point p and store its merged quadric.
				HalfMesh::RemovedData removedData;
				const VIndex vertexMoved = halfMesh.ERemove(iE, removedData);
				ASSERT(removedData.numVerts == 1);
				vertices[removedData.verts[0]] = vertices.back();
				vertices.pop_back();
				verticesQuadric[removedData.verts[0]] = verticesQuadric.back();
				verticesQuadric.pop_back();
				vertices[vertexMoved] = p;
				verticesQuadric[vertexMoved] = quadric;
				for (uint8_t i2 = 0; i2 < removedData.numFaces; ++i2) {
					facesError[removedData.faces[i2]] = facesError.back();
					facesError.pop_back();
					facesDirty[removedData.faces[i2]] = facesDirty.back();
					facesDirty.pop_back();
				}
				for (FIndex iFAdj : halfMesh.VAdjacentFaces(vertexMoved)) {
					facesDirty[iFAdj] = true;
					const Face face2 = halfMesh.F(iFAdj);
					Normal& faceError = facesError[iFAdj];
					for (int j = 0; j < 3; ++j) {
						const VIndex jV0 = face2(j), jV1 = face2((j + 1) % 3);
						// Only the edges touching vertexMoved changed: its quadric and
						// position moved, but the opposite edge (neither endpoint is
						// vertexMoved) has unchanged endpoints and quadrics, so its
						// stored error is still current — recomputing it is redundant
						// (bit-identical) work. Skip it.
						if (jV0 != vertexMoved && jV1 != vertexMoved)
							continue;
						faceError(j) = static_cast<Type>((verticesQuadric[jV0] + verticesQuadric[jV1])
						                                     .ComputeError(vertices[jV0].cast<real>(), vertices[jV1].cast<real>()));
					}
				}
				if (halfMesh.fHalfedges.size() <= numTargetFaces)
					goto FinishSimplification;
				if (iF-- == 0)
					break;
			}
		}
	FinishSimplification: {
	}
	} else {
		// Exact mode: a priority queue keyed by edge collapse cost always collapses
		// the globally cheapest edge first. After each collapse, only the edges
		// incident to the surviving vertex change cost, so just those are re-costed
		// (queue.update); deleted edges are removed and the queue's index mapping is
		// kept consistent with the half-edge array's swap-with-last renumbering.
		vertexColors.clear();
		InvalidateFaces();
		// create priority queue
		// Edge costs are the double ComputeError results; keep them in double (real)
		// rather than narrowing to float queue keys. Near-planar/symmetric regions
		// produce many genuinely distinct costs that collapse to equal float keys,
		// where "globally cheapest" then degenerates into heap-layout order; double
		// keys break those ties by true cost. (Node grows 8->16 bytes, negligible.)
		typedef real EdgeCost;
		typedef TPriorityQueue<EIndex, EdgeCost> PriorityQueue;
		PriorityQueue queue;
		// Cache the optimal collapse point per edge (double, indexed by EIndex):
		// costing an edge already solves ComputeOptimalPoint, so store that point and
		// reuse it when the edge is popped instead of re-running the same solve. The
		// cached value stays exact because an edge's inputs (endpoint quadrics and
		// positions, resolved through the same representative half-edge EHalfedge(iE)
		// at every site) cannot change without the edge being re-costed, which
		// refreshes the entry; renumbering is mirrored below in the same order the
		// queue mirrors it.
		// Sized to the INITIAL edge count and intentionally never shrunk: the
		// pop/move relabel mirror below reads tail source slots (indices at/above
		// the shrinking ESize()), so those slots must stay live for the whole
		// decimation — do NOT "optimize" this to track ESize() or the relabel reads
		// go out of bounds.
		std::vector<Quadric::Point3> edgePoint(halfMesh.ESize());
		{
			const EIndex numEdges = halfMesh.ESize();
			queue.reserve(numEdges);
			// Pre-size the key->slot map to the full INITIAL edge range. In min-edge
			// mode the build-time filter below emplaces only a subset of the edges,
			// yet the pop/move relabel still addresses the whole key space
			// [0, numEdges) as the decimation renumbers tail edges into freed slots;
			// sizing key2heap up front keeps every move()/pop() index in-bounds (a
			// never-emplaced key reads as empty and its move is a correct no-op). This
			// is inert for the minEdgeLength==0 path, which emplaces every edge.
			queue.ReserveKeys(numEdges);
			std::vector<EdgeCost> edgeCost(numEdges);
			if (minEdgeLength > 0) {
				// Build-time length filter (min-edge mode only). Only edges already
				// within the threshold are cost-computed AND emplaced; long edges are
				// skipped entirely — this is the performance win (no QEM solve for the
				// long majority) and the reason exactly the SimplifyMinEdge golden is
				// regenerated: leaving the long edges out of the initial heap changes
				// the pop order among equal-cost ties (an equally valid result). The
				// per-edge length test + cost solve are independent, so run them in
				// parallel; mark survivors, then emplace them sequentially in EIndex
				// order (no parallel emplace) so the heap layout stays a deterministic
				// pure function of the input. A long edge that a later collapse
				// shortens is emplaced lazily by the neighborhood-update path below
				// (requirement 2b), so the set of collapsible candidates offered at any
				// pop is identical to the unfiltered algorithm's.
				std::vector<char> edgeShort(numEdges, 0);
				ParallelForPool(pool, numEdges, [&](std::size_t iE) {
					const auto [iV0, iV1] = halfMesh.EVertices(EIndex(iE));
					if ((vertices[iV0] - vertices[iV1]).norm() > minEdgeLength)
						return; // long edge: neither cost-computed nor emplaced
					edgeShort[iE] = 1;
					const Quadric quadric = verticesQuadric[iV0] + verticesQuadric[iV1];
					const Quadric::Point3 p = quadric.ComputeOptimalPoint(vertices[iV0].cast<real>(), vertices[iV1].cast<real>());
					edgePoint[iE] = p;
					edgeCost[iE] = static_cast<EdgeCost>(quadric * p);
				});
				for (EIndex iE = 0; iE < numEdges; ++iE)
					if (edgeShort[iE])
						queue.emplace(iE, edgeCost[iE]);
			} else {
				// minEdgeLength == 0: unchanged. The per-edge cost solve is
				// independent, so compute cost + optimal point in parallel into plain
				// arrays (pure map, deterministic), then emplace sequentially in EIndex
				// order: identical emplace sequence and values as the old inline loop,
				// so the heap layout — and thus the pop order among equal-cost edges —
				// is bit-identical (4 of the 5 Simplify goldens run this path and must
				// not change a byte).
				ParallelForPool(pool, numEdges, [&](std::size_t iE) {
					const auto [iV0, iV1] = halfMesh.EVertices(EIndex(iE));
					const Quadric quadric = verticesQuadric[iV0] + verticesQuadric[iV1];
					const Quadric::Point3 p = quadric.ComputeOptimalPoint(vertices[iV0].cast<real>(), vertices[iV1].cast<real>());
					edgePoint[iE] = p;
					edgeCost[iE] = static_cast<EdgeCost>(quadric * p);
				});
				for (EIndex iE = 0; iE < numEdges; ++iE)
					queue.emplace(iE, edgeCost[iE]);
			}
		}
		// remove edges in order
		while (halfMesh.FSize() > numTargetFaces && !queue.empty()) {
			const EIndex iE = queue.peek().key;
			const auto [iV0, iV1] = halfMesh.EVertices(iE);
			if (minEdgeLength > 0 && (vertices[iV0] - vertices[iV1]).norm() > minEdgeLength) {
				queue.pop();
				continue;
			}
			if (!halfMesh.EIsCollapseValidTopologically(iE)) {
				queue.pop();
				continue;
			}
			const Quadric quadric = verticesQuadric[iV0] + verticesQuadric[iV1];
			// reuse the optimal point solved when this edge was last costed: the exact
			// double a fresh ComputeOptimalPoint on the same inputs would return
			const Vertex p = edgePoint[iE].cast<Vertex::Scalar>();
			if (!halfMesh.EIsCollapseValidGeometrically(iE, p, vertices)) {
				queue.pop();
				continue;
			}
			// collapse edge
			HalfMesh::RemovedData removedData;
			const VIndex vertexMoved = halfMesh.ERemove(iE, removedData);
			ASSERT(removedData.numVerts == 1);
			vertices[removedData.verts[0]] = vertices.back();
			vertices.pop_back();
			verticesQuadric[removedData.verts[0]] = verticesQuadric.back();
			verticesQuadric.pop_back();
			vertices[vertexMoved] = p;
			verticesQuadric[vertexMoved] = quadric;
			// remove edges
			for (uint8_t i = 0; i < removedData.numEdges; ++i) {
				queue.pop(removedData.edges[i]);
				queue.move(halfMesh.ESize() + (removedData.numEdges - 1 - i), removedData.edges[i]);
				// Mirror the same relabeling into edgePoint, INTERLEAVED in the same
				// order as the queue: when a removed slot lies inside the moved-from
				// tail range the relabelings chain (tail -> removed slot -> earlier
				// removed slot), and only in-order sequential assignment reproduces
				// that chain (a source slot may hold a value moved into it by an
				// earlier iteration).
				edgePoint[removedData.edges[i]] = edgePoint[halfMesh.ESize() + (removedData.numEdges - 1 - i)];
			}
			// update cost for changed edges (and refresh their cached optimal point).
			// Only edges incident to the surviving vertex changed geometry, so an
			// edge's length changes iff it is in this set — which is why handling the
			// dynamic-length cases here keeps the min-edge candidate set correct.
			for (EIndex iEAdj : halfMesh.VAdjacentEdges(vertexMoved)) {
				const VIndex jV0 = halfMesh.EFirstVertex(iEAdj), jV1 = halfMesh.ESecondVertex(iEAdj);
				const Quadric quadricAdj = verticesQuadric[jV0] + verticesQuadric[jV1];
				const Quadric::Point3 pAdj = quadricAdj.ComputeOptimalPoint(vertices[jV0].cast<real>(), vertices[jV1].cast<real>());
				edgePoint[iEAdj] = pAdj;
				const EdgeCost costAdj = static_cast<EdgeCost>(quadricAdj * pAdj);
				if (minEdgeLength > 0 && (vertices[jV0] - vertices[jV1]).norm() > minEdgeLength) {
					// Min-edge mode, neighbor now longer than the threshold: it is not
					// a collapse candidate, so keep it OUT of the queue (preserving the
					// build-time filter's win). If a prior collapse had already queued
					// it while short and it just became long, it may still sit in the
					// queue; the pop-time length check discards it there — correct and
					// cheaper than removing it now (a discarded pop never reads its
					// stale cost/point).
					continue;
				}
				// Short neighbor (always taken when minEdgeLength==0). Keep the
				// candidate set identical to the unfiltered algorithm: refresh an
				// in-queue edge, or EMPLACE one that is not queued. The emplace branch
				// is what re-admits a formerly-long edge this collapse just shortened
				// (requirement 2b) — without it the candidate set would silently
				// shrink. For minEdgeLength==0 every adjacent edge is already queued,
				// so this reduces to the previous unconditional update(); the
				// !contains() case then only fires for an edge earlier popped as
				// invalid, which update() historically re-emplaced here too (behavior
				// unchanged, so the 4 non-min-edge goldens stay bit-identical).
				if (queue.contains(iEAdj)) {
					queue.update(iEAdj, costAdj);
				} else {
					queue.emplace(iEAdj, costAdj);
				}
			}
		}
	}
	// Both modes stop early when no remaining edge passes the collapse validity
	// checks (exact mode: candidate queue exhausted; fast mode: threshold sweep
	// ran out of passes). Distinguish that from success — a silent 20% miss on
	// needle-triangle-heavy CAD input was indistinguishable from reaching the
	// target (2026-08 review, pipes_textured: floor at 59.85% of faces for any
	// requested ratio). minEdgeLength mode targets 1 face by design, so the
	// warning applies only to ratio/count mode.
	if (minEdgeLength <= 0 && halfMesh.FSize() > numTargetFaces)
		REPORT_WARNING("Simplify: stopped at {} faces (target {}): no remaining edge passes the collapse "
		               "validity checks; for needle/T-junction-heavy input run RemoveDegenerateFaces(1e-5f) + "
		               "RemoveUnreferencedVertices() + FixNonManifold() before Simplify",
		               halfMesh.FSize(), numTargetFaces);
	SyncFacesOnPublicExit();
	ASSERT(ValidateInvariants());
	REPORT_STATUS_NOW("Mesh decimated: {} -> {} faces ({})", numFaces, halfMesh.FSize(), TIMER_STR());
}

} // namespace halfmesh
