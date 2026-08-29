/*
* AtlasCharting.h
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// halfmesh/AtlasCharting.h — public atlas API: segmentation (A) + density (C) +
// the GenerateAtlas pipeline. Packing (D) is in halfmesh/AtlasPacking.h; per-chart
// UV flattening (B) is in halfmesh/Parametrize.h.
//
// =============================================================================
// The atlas pipeline (GenerateAtlas) turns a triangle mesh into a packed texture
// atlas. Three of its four modules live here (SEGMENT, DENSITY, PACK); per-chart
// UV FLATTENING is the one separable concern, in halfmesh/Parametrize.h.
//
//   A. SEGMENTATION  (SegmentCharts) — partition the surface into the fewest
//      flip-free, flattenable charts. (Method below.)
//   B. FLATTENING    (ParametrizeCharts, halfmesh/Parametrize.h) — per-chart UV
//      map: LSCM/Tutte init + SLIM/ARAP refinement.
//   C. DENSITY       (NormalizeChartDensity) — rescale each chart's UVs so every
//      chart has the same texels-per-world-unit (explicit, or auto from a target
//      resolution so Σ chart-UV-area ≈ resolution²).
//   D. PACKING       (PackAtlas) — rectangle-pack the charts into one page.
//
// -----------------------------------------------------------------------------
// Segmentation method: developable (D-Charts) charting — cone-Lloyd + merge + flip
// repair. halfmesh charts for the FEWEST flip-free, flattenable patches, following
// D-Charts (Julius/Kraevoy/Sheffer 2005): it bounds FLATTENABILITY, not flatness.
// A chart may span a whole plane / cylinder / cone because a single CONE proxy
// ⟨axis, half-angle⟩ fits the entire developable family — its fit error is λ_min of
// the chart's area-weighted normal covariance C = Σwᵢnᵢnᵢᵀ − (Σwᵢnᵢ)(Σwᵢnᵢ)ᵀ/Σwᵢ,
// exactly zero for any developable surface — so a curved-but-flattenable wall/pipe
// stays ONE chart where a planar (normal-deviation) metric would shatter it.
//
//   1. Cone-Lloyd (the scalable core). A global, simultaneous Lloyd relaxation on
//      cone proxies: seed a few charts (one per topological region + farthest-point
//      extras), flood-assign every face at once to the cheapest-fitting cone, then
//      iterate {recompute each chart's best-fit cone, relocate its seed, re-flood}
//      until stable; when charts exceed the cone-error budget, add in ONE BATCH a
//      seed at the worst face of every over-budget chart and re-Lloyd (batching
//      converges in O(log) rounds where one-seed-at-a-time grow is O(F²)).
//   2. Developable merge. A union-find pass over the chart graph stitches adjacent
//      charts while the merged per-area cone error stays in budget, with an angle-
//      defect (Gaussian-curvature) cap forbidding a merge that would make a cone/
//      saddle vertex chart-interior — the anti-fold guard.
//   3. Flip repair (the hard flip-free guarantee). The cone error + cap cannot see
//      every fold source (annulus/handle charts, accumulated sub-threshold
//      curvature), so each chart is actually FLATTENED and any that folds is
//      spatially bisected, its connected pieces re-checked, until none fold (≤2-face
//      charts cannot fold, so it converges; incremental — settled charts are not
//      re-flattened).
//
// Two opt-in extensions ride this same flatten/repair machinery (both OFF by default,
// so the flip-free, fewest-charts result above is unchanged until a knob is set; see
// docs/BENCHMARKS.md §4 for the attribution matrix):
//   - developableMaxUvDistortion (Parametrize.h): the flip repair (3) ALSO splits a
//     flip-FREE but over-stretched chart (shipped area-weighted sym-Dirichlet > budget),
//     trading a few extra charts for lower per-chart distortion; a mandatory sliver
//     guard excludes degenerate near-zero-area input so it cannot runaway-split.
//   - cutToDisk (Parametrize.h, Module B): a closed / multiply-connected chart is
//     SLIT into a single disk at flatten time (Seamster, Sheffer & Hart 2002) instead
//     of being bisected into many — far fewer charts on hole-riddled MVS. The flip
//     repair still bisects anything that folds AFTER the cut, so the guarantee holds.
// Both flow into the segmenter only through the existing detail::ChartFacesFold bridge
// (the cut and the distortion test live entirely in Parametrize.h's Module B).
//
// Robustness to MVS/photogrammetry noise: the face normals (cone fit) and angle
// defect (cap) both derive from triangle geometry, so a virtual Taubin geometry
// smoothing denoises a TEMPORARY copy of the positions for the analysis only (mesh
// and UVs untouched); it is adaptive — skipped on small meshes so synthetic/clean
// inputs are not deformed. Charts grow over topological adjacency (across creases),
// so a noisy normal can never spawn a chart — only true developability breaks and
// cone/saddle vertices cut the surface.
//
// Approaches that did NOT work, and why (kept so they are not re-tried):
//   - Planar-VSA Lloyd: an L2,1 normal-deviation proxy measures
//     FLATNESS, so it shatters curved-but-developable regions — 1.5 faces/chart.
//   - Per-vertex angle-defect cap alone: blind to crease folds (edge dihedral has
//     ~0 vertex defect) and to accumulated curvature → charts still fold.
//   - Per-area cone error alone (no flip repair): a localized fold averages away
//     over a large chart, so big charts pass the budget yet fold.
//   - Single-pass cone region-grow: every un-absorbable face becomes a seed → seam
//     singletons / over-segmentation (the batched Lloyd replaced it).
//   - Bottom-up planar-fine → developable merge: fine charts follow flatness, not
//     cone structure, so the merge plateaus far above the cone-Lloyd result.
//   - Normal-field smoothing (vs geometry smoothing): denoises the cone fit but not
//     the geometric angle defect, so the cap still over-segments on noise.
// See docs/BENCHMARKS.md §4 and docs/ATLAS_SEGMENTATION_DESIGN.md for the numbers.
// =============================================================================
#pragma once

#include <halfmesh/Mesh.h>
#include <halfmesh/Parametrize.h>

#include <vector>

namespace halfmesh {

// ---------------------------------------------------------------------------
// AtlasParams — knobs for Module C (density) and Module D (packing).
// ---------------------------------------------------------------------------
struct AtlasParams
{
	// ---- density -----------------------------------------------------------
	// Desired texels per world unit. 0 (the default) means auto-derive from
	// `resolution` so the total normalised UV area fills approximately one
	// atlas of that size.
	float texelsPerUnit = 0.f;

	// Target atlas side length in texels used when texelsPerUnit == 0.
	// The auto density is chosen so Σ(chart_uv_area) ≈ resolution².
	unsigned resolution = 1024;

	// ---- packing — consumed by PackAtlas; NormalizeChartDensity ignores these.
	unsigned padding = 2; // gutter texels between packed charts
	bool allowRotation = true; // permit 90° rotation during packing
	bool powerOfTwo = false; // round atlas dims up to next power of two
	bool square = false; // force a square atlas

	// Pre-orient each chart to its MINIMUM-AREA bounding rectangle (computed via
	// rotating calipers on the chart's UV convex hull) before packing, so the
	// axis-aligned bounding rect the packer uses is as tight as possible. The
	// rotation is rigid (preserves texel density and distortion) and is baked
	// into the output UVs. Default on (matches xatlas's rotateChartsToAxis).
	bool orientCharts = true;

	// Fit the whole atlas into ~one page of `resolution` texels: PackAtlas
	// applies a single global UV scale so the total padded chart area ≈ one
	// page, then packs.  Without it, the per-chart texel density is fixed and
	// padding + packing gaps spill many small charts across several pages at low
	// occupancy.  With it, the atlas fills one `resolution`×`resolution` page at
	// high occupancy (matching xatlas's "fit the requested resolution"
	// behaviour).  OFF for raw PackAtlas (exact density preserved); GenerateAtlas
	// turns it ON.  Honored only when packing the resolution is the goal.
	bool fitToResolution = false;
};

// ---------------------------------------------------------------------------
// Module A: developable chart segmentation (the method is documented atop this
// header). Partition `mesh` into charts: on return `faceChart` is sized to
// mesh.faces.size() and each face holds a chart id in [0, return_value); the
// return value is the chart count (>= 1 for a non-empty mesh, 0 for an empty one).
//
// Guarantees (see tests/ParametrizeTest.cpp):
//   - Partition:    every face gets exactly one chart id in [0, N).
//   - Connectivity: each chart is a single connected face set.
//   - Flattenable:  each chart flattens without folds (flip-free UV) — the cone
//                   budget + angle-defect cap keep charts developable and the flip
//                   repair bisects any residual folder. Charts MAY span creases
//                   when the surface is developable across them; only mesh-border /
//                   non-manifold / texblob-border edges always bound a chart.
//
// Side effects: ensures mesh.faceNormals and mesh.halfMesh are populated.
unsigned SegmentCharts(Mesh& mesh, const ParametrizeParams& params,
                       std::vector<unsigned>& faceChart);

// ---------------------------------------------------------------------------
// Module C: normalise per-chart UV scale to uniform texel density (in place).
//
// For each chart the function:
//   1. Computes the chart's 3-D (world) area and its current 2-D UV area.
//   2. Derives a per-chart scale = targetDensity / sqrt(worldArea / uvArea)
//      and applies it to every UV coordinate belonging to that chart.
//   3. Translates the chart's UVs so its bounding-box minimum is at the
//      origin (pre-packing canonical placement; PackAtlas moves them again).
//
// The INVARIANT guaranteed after the call:
//   For every non-degenerate chart c,
//     sqrt(uv_area_c / world_area_c) == returnedDensity   (up to float eps).
//
// Global density selection:
//   params.texelsPerUnit > 0  →  that value is used directly.
//   params.texelsPerUnit == 0 →  density is auto-derived so that the sum of
//                                   all post-scale chart UV areas equals
//                                   params.resolution².
//
// Inputs:
//   mesh        — Mesh whose faceTexcoords hold per-chart local UVs (from
//                 ParametrizeCharts). Modified in place.
//   faceChart  — per-face chart id in [0, numCharts) (from SegmentCharts).
//                 Size must equal mesh.faces.size().
//   numCharts  — number of charts.
//   params      — density and packing parameters.
//
// Returns: the global density (texels per world unit) applied.
// Returns 0 if the mesh has no faces or all charts are degenerate.
float NormalizeChartDensity(Mesh& mesh,
                            const std::vector<unsigned>& faceChart,
                            unsigned numCharts,
                            const AtlasParams& params = AtlasParams{});

// ---------------------------------------------------------------------------
// Module D: atlas packing result and functions.
// ---------------------------------------------------------------------------
struct AtlasResult
{
	unsigned width = 0; // page width  in texels
	unsigned height = 0; // page height in texels
	unsigned numPages = 1; // number of atlas pages (>1 on multi-atlas overflow)
	float occupancy = 0.f; // packed chart area (with padding) / total atlas area [0,1]
	// TRIANGLE coverage: Σ(UV triangle areas in final normalized atlas space) /
	// numPages ∈ [0,1] — the fraction of the texel budget actually under
	// geometry. `occupancy` above is PADDED-RECT occupancy: it contains the
	// per-chart bbox waste and the padding tax, so with many small charts it
	// reads high (~0.8) while coverage can be 4–13× lower. Consumers choosing an
	// atlas resolution for a target texel density must use THIS number.
	float coverage = 0.f;
	// fit-to-resolution probe packs performed (0 = fitToResolution off). A
	// converging fit takes 1-2; values near the internal cap (8) mean the
	// shrink loop struggled — a diagnosability hook for huge chart counts.
	unsigned fitAttempts = 0;
	// page index per chart (size == numCharts)
	std::vector<unsigned> chartPage;
	// per-face chart id (size == mesh.faces.size()); populated by PackAtlas /
	// GenerateAtlas so callers can verify atlas invariants without re-running
	// segmentation.
	std::vector<unsigned> faceChart;
};

// Module D: PackAtlas (pack density-normalized charts into atlas page(s)) lives in
// halfmesh/AtlasPacking.h.

// Convenience full pipeline: SegmentCharts + ParametrizeCharts +
// NormalizeChartDensity + PackAtlas in one call.
AtlasResult GenerateAtlas(Mesh& mesh,
                          const ParametrizeParams& pparams,
                          const AtlasParams& aparams = AtlasParams{});

} // namespace halfmesh
