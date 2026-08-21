# Unwrap atlas speedup — design spec

Date: 2026-08-21. Branch: `worktree-python-bindings` (per project owner: all halfmesh
changes land on the same branch as the Python bindings). Companion radiance work is
described in §6 and lands on the radiance `halfmesh-integration` branch.

## 1. Problem

Unwrapping the radiance PGSR pipeline's post-processed Truck mesh (1,000,199 faces,
161,627 charts) through `hm.unwrap(resolution=4096, padding=4)` takes **3h45m
(13,473 s)**. The target set by the project owner is **~10 minutes** for a ~1M-face
mesh. The unwrap must stay geometry-preserving (weld + minor manifold repairs only,
exactly as today).

### Measured evidence (2026-08-20 profiling spike, 8-core GCP VM)

Instrumented CLI (`HALFMESH_ATLAS_DEBUG`, RelWithDebInfo + perf, padding=2) and the
installed Release wheel on the same meshes:

| run | faces | charts | padding | wall time |
|---|---|---|---|---|
| wheel (production, radiance defaults) | 1.0M | 161,627 | 4 | 13,473 s |
| wheel (production, mcs25 control) | 1.0M | 136,901 | 4 | 8,846 s |
| CLI + perf | 1.0M | 161,627 | 2 | 1,164 s |
| CLI + perf | 320k | 112,047 | 2 | ~390 s |
| wheel | 320k | 112,047 | 2 | 353 s |
| wheel | 320k | 112,047 | 4 | 2,864 s |

Stage timestamps at 1M faces: load+repair+segmentation+flip-repair+flatten complete
in **~2m07s** (cone-Lloyd 91 s → 103,061 charts; flip-repair 32 s → 161,627 charts;
flatten 4 s, parallel + cache). Everything else is packing.

perf attribution (320k run): **78.1 % `SkylineBin::Insert` under `PackRects`**,
11.4 % `ChartUVSelfOverlaps` (parallel fold checks), ~3 % flatten. Segmentation does
not register.

### Root causes (src/AtlasPacking.cpp)

1. **Quadratic pack pass.** `SkylineBin::Insert` is a min-waste *full scan*: every
   insert walks every skyline node (`consider` → `RectFits`, which itself walks the
   spanned segments), twice with rotation. Cost ≈ O(N·S) with the segment count S
   growing with placed rects. Designed for ~10k charts ("packing 10k+ charts in well
   under a second" — comment at the top of `SkylineBin`); our fragmented input hands
   it 112k–161k rects, deep in the quadratic regime (~15 min per pass at 161k).
2. **Blind fit-to-resolution probe loop.** `PackAtlas` §1.5 probes a full `PackRects`
   and, while the result needs >1 page or overflows, shrinks the global scale by a
   fixed ×0.95 and repacks — up to 8 retries. Padding is a fixed per-rect area floor
   (at padding=4, a ≥9×9-texel footprint × 112k charts is more than half a 4096²
   page before any chart content), so the analytic first guess never fits and the
   loop runs to exhaustion: the measured **8.1×** (pad 4 vs pad 2, same code, same
   mesh) is the probe-loop multiplier.

Chart fragmentation (~6 faces/chart at 1M) is the input multiplier: flip repair
bisects 103k → 161k charts and no merge ever runs afterward
(`src/AtlasCharting.cpp`: `DevelopableMerge` is called once, before
`RepairDevelopableFlips`; nothing recombines the fragments).

## 2. Goals and non-goals

Goals, in priority order:

1. 1M-face unwrap at `resolution=4096, padding=4` in **≤ 10 min** end-to-end
   (stretch: ≤ 5 min), occupancy ≥ 75 %, 1 page, geometry unchanged.
2. Chart count goes **down**, not up (fewer seams + less padding waste → better
   textures and faster future bakes).
3. No radiance-side API change: same `hm.unwrap()` signature, wheel rebuilt from
   this branch.
4. Packing quality (occupancy) within a few points of the current packer on the
   existing test/bench meshes.

Non-goals:

- No new charting algorithm (the earlier "fast mode" idea is **contingency only**,
  see §7 — not designed or built unless the fixes below miss the target).
- No GPU work, no changes to segmentation/flatten algorithms.
- No change to the geometry contract of unwrap (remeshing is a *radiance
  postprocess* option, §6, never part of unwrap).

## 3. Fix 1a — two-tier pack pass (`src/AtlasPacking.cpp`)

Split `PackRects` placement into two tiers by padded rect size, threshold
`T = pageW / 32` texels (128 at 4096) on `max(paddedW, paddedH)`:

- **Head (≥ T)**: unchanged skyline min-waste insert. Head size is geometrically
  bounded: at most `fill·32²` ≈ ~840 rects per page can exceed T, so the O(N·S)
  scan is harmless here.
- **Tail (< T)**: **shelf packing**. Sort the tail by padded height descending
  (the FFD area sort already approximates this; re-sort the tail slice exactly).
  Allocate each shelf as a single wide pseudo-rect through the existing skyline
  `Insert` (height = tallest remaining tail rect, width = the widest span that
  fits), then fill it left-to-right O(1) per rect; open a new shelf when the row is
  full. Rotation: rotate each tail rect so height ≤ width before shelving (only
  when `allowRotation`, preserving the winding-safe 90° convention).
- Multi-page: shelves that fit no open page open a new page — same growing-bins
  logic as today. Degenerate 1-texel clamp and padding semantics unchanged.

Complexity: O(H·S + N) with H ≈ hundreds — the pack pass at 161k rects drops from
~15 min to seconds. Shelf waste (shelfH − rectH per rect) is small with the
height-descending order; acceptance bound in §8.

## 4. Fix 1b — analytic fit loop (`PackAtlas` §1.5)

Replace the fixed ×0.95 shrink ladder with an overflow-proportional step: after a
probe pack, compute the achieved packed area `A` versus the single-page budget
`targetFill·R²`; on failure set `k ← k · sqrt(targetFill·R² / A)` (clamped to
[0.8, 0.99] per step so a pathological probe cannot collapse the scale), keep the
existing attempt cap of 8 as a safety net. Expected convergence: ≤ 2–3 packs
instead of 9–10. Sort order is computed once and reused across probes (it is
scale-invariant — global k preserves the area ordering). With Fix 1a each probe is
cheap anyway; 1b restores the occupancy lost to over-shrinking (production packed
at 78.8 % / 77.2 % where 82 % fill was the target).

## 5. Fix 2 — post-repair re-merge (`src/AtlasCharting.cpp`)

After `RepairDevelopableFlips`, run a bounded merge↔repair loop (≤ 2 rounds):

1. `DevelopableMerge` over the current (post-repair) partition with the same gates
   as today — combined cone-fit error within `developableMaxConeError`, no enclosed
   vertex above `developableMaxVertexDefect`.
2. One flip-repair wave restricted to charts changed by the merge (the
   `ChartFlattenCache` already makes unchanged charts free): any merged chart that
   folds is bisected back — net-zero for that chart, never a regression.
3. Stop when a round merges < 1 % of charts, or after round 2.

Expected effect: recombine a large share of the 46k bisection fragments (320k mesh:
112,047 → target ≤ 78k, i.e. ≥ 30 % reduction; 1M mesh: 161,627 → proportionally
fewer). Fewer charts means fewer seams, less padding overhead (higher effective
texel density), and a smaller packing input. Invariants preserved: no folded chart
in the final partition (existing tests), every face charted, chart connectivity.

## 6. Track 2 — opt-in remesh in radiance postprocess (radiance repo)

`radiance.mesh.postprocess` gains `--remesh-edge-length FLOAT` (world units,
default 0 = off). When set, `hm.remesh(vertices, faces, edge_length, iterations=3)`
(already exposed by the wheel) runs as the **final** geometry step (after hole
closing / component filtering, before export), and provenance records it under the
existing `params` block. It is an experiment lever, not a default: enabling it in
any preset requires passing the same exact-distance bench gates the current
defaults passed (F1 ≥ 0.98× target / 0.95× floor, thin-structure metrics ≥ 0.95×,
via `scripts/bench_postprocess.py` — CPU-only, no retraining) **and** a measured
unwrap improvement (time or chart count) worth the geometry change. Otherwise it
ships documented as opt-in with the measured numbers.

## 7. Contingency — fast charting mode

Only if §3+§4 (+§5) miss the ≤ 10 min gate on the 1M mesh: add an opt-in one-pass
normal-cone charting mode (no Lloyd, no developability proxy, LSCM-only, no flip
repair) behind a new `ParametrizeParams` switch, leaving the current pipeline as
the quality default. Not designed further here; a separate mini-design is required
before building it. Given the stage timings (§1: everything except packing already
fits in ~2 min), reaching this contingency is unlikely.

## 8. Validation and acceptance

halfmesh (this branch):

- Existing unit/atlas/bench tests green (`AtlasTest` overlap-freedom and occupancy
  assertions; goldens updated where packing layout legitimately changes —
  layout-hash goldens are expected to change, occupancy/page-count bounds must not
  regress by more than 3 points on the bench meshes).
- New tests: two-tier boundary behavior (head/tail split, threshold edge, rotation
  in shelves, multi-page shelf spill, degenerate rects), analytic-shrink
  convergence (≤ 3 packs on a synthetic 100k-tiny-rect fixture), re-merge (chart
  count strictly drops on a fragmented fixture, no folds in output, ≤ 2 rounds).
- Perf gates on the two reference meshes (release build, padding=4, 4096):
  - 320k / Truck-derived: **≤ 2 min** end-to-end (from 2,864 s).
  - 1M / Truck-derived: **≤ 10 min** end-to-end (from 13,473 s), occupancy ≥ 75 %,
    pages = 1, charts ≤ 161,627.
- Wheel rebuilt from this branch, installed into the radiance conda env; radiance
  unit suite green with **zero radiance code changes** required.

radiance (halfmesh-integration branch):

- `--remesh-edge-length` unit tests (arg plumbing, provenance, off-by-default
  no-op) + docs.
- Remesh evaluation run on Truck: bench metrics + unwrap time/charts, verdict
  recorded (enable in preset or document as opt-in) with the same evidence style
  as the 2026-08-19 postprocess study.
- `docs/mesh-postprocess.md` known-limitation section updated: the 3h45m unwrap
  wall-time cause is fixed; new measured numbers cited.

## 9. Landing

All halfmesh changes on `worktree-python-bindings` (this branch), which carries the
Python bindings and is 21 commits ahead of `develop` (0 behind, contains all atlas
code). CHANGELOG entry under the pending 0.2.0 release (unreleased — the fixes fold
into it; the radiance Dockerfile already pins the 0.2.0 wheel URL). Wheel rebuild:
`pip install .` from this worktree into the radiance env (incremental build tree
already configured). Pushing/releasing remains the project owner's decision.
