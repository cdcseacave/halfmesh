/*
* Parametrize.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// halfmesh/Parametrize.h — public UV-parametrization API.
//
// This header declares the public surface for turning a triangle mesh into a
// texture atlas. The pipeline has three logical modules:
//
//   A. Chart segmentation  — partition the mesh into a small number of
//                            quasi-developable charts whose boundaries are
//                            snapped to feature edges (creases / mesh borders).
//                            Implemented here in src/Parametrize.cpp.
//   B. Per-chart flattening — compute a 2D UV layout per chart (SLIM / ARAP).
//                            Declared as a stub below.
//   C. Atlas packing       — pack the per-chart layouts into one texture.
//                            See halfmesh/AtlasCharting.h.
//
// Module A follows a Variational-Shape-Approximation (VSA) style planar-proxy
// Lloyd clustering with crease-snapped boundaries. Correctness is verified by
// properties (partition, connectivity, crease-respect) in tests.
#pragma once

#include <halfmesh/Mesh.h>
#include <halfmesh/Util/Maths.h>

#include <cmath>
#include <functional>
#include <vector>

namespace halfmesh {

// ---------------------------------------------------------------------------
// ParametrizeParams — tunables for the whole parametrization pipeline.
//
// Only the segmentation (Module A) fields are honored in this release; the
// flattening (Module B) fields are declared for forward context.
// ---------------------------------------------------------------------------
struct ParametrizeParams
{
	// ===== Segmentation (Module A) — developable D-Charts ===================
	// halfmesh partitions a mesh into the FEWEST flip-free flattenable charts via a
	// developable criterion (D-Charts, Julius/Kraevoy/Sheffer 2005): a chart may
	// span a whole plane / cylinder / cone because a cone proxy ⟨axis, half-angle⟩
	// fits the entire developable family (its fit error is λ_min of the chart's
	// weighted normal covariance — zero for any developable surface). The complete
	// method (cone-Lloyd → developable merge → flip repair) is documented atop
	// halfmesh/AtlasCharting.h.

	// Per-area cone-fit error a chart may reach (≈ how far from a perfect cone).
	// Larger ⇒ fewer/larger charts at slightly more distortion (the preferred
	// trade). Default 0.05.
	float developableMaxConeError = 0.05f;

	// Max absolute vertex angle defect (radians) a chart may ENCLOSE. A vertex whose
	// Gaussian curvature exceeds this (a cone/saddle point) is kept on a chart
	// boundary, never made interior — the anti-fold guarantee. Straight folds and
	// cylinders have ~0 defect and are spanned freely. Default 0.35 (~20°); smaller
	// ⇒ more, flatter charts.
	float developableMaxVertexDefect = 0.35f;

	// D-Charts distance-term exponent β for best-first chart growth (0 = off).
	// When > 0, a candidate face is ranked by (fit + 1e-8)·dist^β instead of the
	// raw cone-fit cost, where dist is the accumulated centroid-path distance
	// from the chart's seed along the flood tree (D-Charts F^α·D^β·N^γ with α=1,
	// D only; Julius/Kraevoy/Sheffer 2005 use β=0.7). Among near-equal fits the
	// CLOSER chart wins, so charts stay compact and seams short instead of
	// snaking along curvature ridges. The multiplicative form is scale-invariant
	// (uniform mesh scaling rescales every key by the same s^β, preserving pop
	// order), so no normalization constant is needed; at β=0 the ranking is
	// exactly the fit-only order.
	// Measured (2026-07-16, segment-quality harness calibration): at β=0.7 an
	// MVS-like scan improves (charts −12%, seam −3.2%) and a UV-sphere's seam
	// drops 8.4%, but anisotropically-developable surfaces can over-split
	// (torus: 2→4 charts) and segmentation wall-clock at 200k faces rises ~27%
	// — so the term is OPT-IN. Default 0 (off); β=0.7 (the paper value) is the
	// measured best on-value.
	// Measured on the default build; near-tie outcomes are build-flag
	// sensitive (e.g. -march=native flips the UV-sphere seam delta).
	float developableDistanceExponent = 0.0f;

	// Virtual GEOMETRY denoise: Taubin (λ|μ, no shrinkage) passes over a TEMPORARY
	// copy of the vertex positions, used only to recompute the face normals (cone
	// fit) and the angle defect (cap) during segmentation — the mesh and the
	// parametrization are untouched. Both derive from triangle geometry, so on a
	// noisy MVS/photogrammetry mesh both explode without it. ADAPTIVE: smoothing is
	// skipped on small meshes (noise only matters at scale) so synthetic / clean
	// inputs are never deformed. 0 disables. Default 4.
	unsigned developableSmoothIters = 4;

	// Maximum cone-Lloyd relaxation iterations (cone flood-assign + proxy recompute
	// + seed relocation). A handful converges. Default 8.
	unsigned maxIterations = 8;

	// Extra farthest-point seeds for the initial cone-Lloyd, as a multiple of the
	// region count (initial seeds ≈ (1 + this) × regions). Batched developability
	// seeding then splits only where genuinely needed. Default 1.0.
	float seedExtraMult = 1.0f;

	// Flip-free guarantee (> 0 on; 0 off). After segmentation every chart is
	// flattened and any that FOLDS (a flipped triangle — the only sure flattenability
	// test) is spatially bisected and its connected pieces re-checked until none fold
	// (≤2-face charts cannot fold, so it always converges). Incremental: settled
	// charts are never re-flattened. Catches what the cone error + cap cannot
	// (annulus/handle charts, accumulated sub-threshold curvature). Default 16.
	unsigned developableFlipRepairRounds = 16;

	// Post-repair merge↔repair rounds. Flip repair bisects folding charts but
	// nothing recombined the fragments afterward — on noisy MVS meshes that
	// leaves ~6-face charts (seam + padding blowup, quadratic packing input).
	// Each round re-runs DevelopableMerge over the post-repair partition (same
	// cone-error + vertex-defect gates) and then one flip-repair wave over the
	// merged charts only, so a merge that re-folds is split right back (never
	// a regression). Stops early when a round changes <1% of charts.
	// 0 restores the pre-2026-08 behavior.
	unsigned postRepairMergeRounds = 2;

	// Failure-localized repair splitting (0 = off, the default: blind PCA
	// bisection, the current behavior). When > 0, a folding chart is first
	// split by carving off the faces within this many TopoNeighbor rings of
	// the offending triangles (the FoldDiagnosis), so one localized failure
	// costs ONE small extra chart instead of a bisection cascade; the carve
	// falls back to the PCA bisection when the failure is not localized
	// (region ≥ half the chart) — the termination guarantee is unchanged.
	// 2 is the sane on-value.
	unsigned repairCarveRings = 0;

	// Distortion-bounded split (0 disables — the default; flip-only repair, the
	// current SOTA behavior). When > 0 it is a symmetric-Dirichlet cap τ that
	// EXTENDS the flip-repair: a chart that is flip-FREE but whose SHIPPED map
	// (full SLIM, area-weighted symmetric-Dirichlet) still exceeds τ is also
	// spatially bisected and its pieces re-checked, trading a few extra charts
	// for far lower per-chart stretch. τ = 4.0 is perfect isometry (the floor),
	// so any on-value must exceed 4; ~4.4 is the sane benchmark setting.
	// Sliver-dominated charts (near-zero-area input triangles, sym-Dir → ∞ at any
	// size) are EXCLUDED from the split — bisection cannot fix degenerate input
	// and would otherwise shatter the chart to the runaway cap.
	float developableMaxUvDistortion = 0.0f;

	// Per-face importance weight hook (default = face area). The cone moments are
	// area/weight-averaged by this; signal-aware (e.g. texture-detail) weighting
	// plugs in here. When empty (the default) the weight is the face area
	// (double-area / 2).
	std::function<float(Mesh::FIndex)> faceWeight;

	// ---- Flattening (Module B) ---------------------------------------------

	// Seamster cut-to-disk (Sheffer & Hart 2002; false = off, the default). When
	// false a closed / multiply-connected chart is made flattenable upstream by the
	// segmentation's disk guarantee, which BISECTS it into many topological disks —
	// robust, flip-free, but on hole-riddled MVS the chart count runs high (every
	// noise-hole forces a cut). When true such a chart is instead SLIT open into a
	// single-boundary disk at flatten time and stays ONE chart, slashing the chart
	// count on noisy meshes. Strictly additive and opt-in: the flip-repair safety
	// net still bisects any chart that folds after the cut (genus>0 handle, residual
	// sliver), so the hard flip-free guarantee is preserved.
	bool cutToDisk = false;

	// Fold-rescue slits (0 = off, the default). When > 0 and the flattened
	// chart folds (flipped triangles or global self-overlap), FlattenChart
	// cuts a slit from the worst interior vertex (largest quantized angle
	// defect; among fold-incident vertices when the failure is localized) to
	// the boundary and re-flattens, up to this many times — the chart ships
	// as ONE chart with one extra seam instead of being split into ≥2 padded
	// rects. Deterministic pure function of the chart geometry, so the repair
	// verdict and the shipped map agree on every path (the cutToDisk
	// contract). Charts still folding after the last slit fall through to the
	// repair's carve/bisect safety net — the flip-free guarantee holds. 2 is
	// the sane on-value.
	unsigned foldRescueSlits = 0;

	// Per-chart UV flattening method. After a Tutte (convex-boundary Laplacian)
	// injective initialization, the chart UVs are refined by a local/global solver:
	//   - ARAP (Liu 2008): per-triangle closest-rotation fit + constant
	//     cotangent-Laplacian solve. The robust MUST-HAVE foundation.
	//   - SLIM (Rabinovich 2017): isotropic-proximal symmetric-Dirichlet
	//     local/global with a flip-free line search; a simplified SLIM
	//     variant using a scalar (isotropic) per-triangle weight rather than
	//     the full anisotropic weight matrix. Energy-decreasing and flip-free
	//     starting from the Tutte init. The default — yields locally-injective
	//     (flip-free) maps with low, uniform distortion.
	enum class FlattenMethod { SLIM,
		                       ARAP };
	FlattenMethod method = FlattenMethod::SLIM;

	// Per-chart UV initialization (the starting map the local/global solver
	// refines). The choice strongly affects the converged distortion because
	// SLIM/ARAP descend to the nearest local minimum:
	//   - LSCM (default): a free-boundary least-squares conformal map. Starts
	//     near-isometric for elongated/curved charts → SLIM converges to far
	//     lower distortion (symmetric-Dirichlet near its floor).
	//   - Tutte: pins the boundary to a unit circle and solves a positive-weight
	//     (mean-value / Floater) Laplacian. Guaranteed injective but heavily
	//     distorts non-round charts; SLIM cannot fully recover (it stays trapped
	//     well above the floor).
	// LSCM falls back to Tutte automatically if it fails or produces flips, so
	// injectivity of the init is never lost.
	enum class InitMethod { LSCM,
		                    Tutte };
	InitMethod initMethod = InitMethod::LSCM;

	// Number of local/global refinement iterations performed after the Tutte
	// initialization. A handful suffices for ARAP; SLIM benefits from a few
	// more. Default 5.
	unsigned flattenIterations = 5;
};

// Module A — chart segmentation (`SegmentCharts`) lives in halfmesh/AtlasCharting.h.

// ---------------------------------------------------------------------------
// Module B — per-chart flattening.
//
// Flatten each chart produced by SegmentCharts into a 2D UV layout and write
// the result into `mesh.faceTexcoords` (sized faces*3: three UVs per face,
// one per face corner, in face-vertex order). Each chart is laid out in its
// OWN local UV frame — the per-chart maps are NOT packed, scaled, or
// normalized relative to one another (that is the atlas module).
//
// Per chart the pipeline is:
//   1. Extract the chart submesh (local vertices + reindexed faces) and its
//      boundary loop (the longest loop of chart-border edges).
//   2. Tutte initialization: pin the boundary loop onto a convex shape (a
//      circle, arc-length parameterized) and solve the cotangent Laplacian for
//      the interior → a guaranteed-injective starting map.
//   3. Local/global refinement (`params.flattenIterations`):
//        - ARAP: per-triangle closest-rotation fit, constant cotangent-
//          Laplacian solve (factored once via Eigen SimplicialLDLT).
//        - SLIM: symmetric-Dirichlet proximal reweighting + flip-free line
//          search on top of the ARAP scaffold (the default).
//
// Inputs:
//   - `faceChart`: per-face chart id in [0, numCharts) (from SegmentCharts).
//   - `numCharts`: number of charts.
//   - `params`: honors `method` and `flattenIterations`.
//
// Side effects: ensures mesh.halfMesh is populated; fills mesh.faceTexcoords
// (resized to faces*3). All produced UVs are finite. A flat (planar) chart is
// flattened near-isometrically (≈0 distortion); SLIM charts are flip-free.
void ParametrizeCharts(Mesh& mesh, const std::vector<unsigned>& faceChart,
                       unsigned numCharts, const ParametrizeParams& params);

// ---------------------------------------------------------------------------
// Convenience: run the full per-chart pipeline — segment (Module A) then
// flatten (Module B) — on `mesh`. Fills mesh.faceTexcoords with per-chart
// local UVs and returns the number of charts produced.
unsigned Parametrize(Mesh& mesh, const ParametrizeParams& params);

} // namespace halfmesh
