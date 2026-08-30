# UV-Atlas Benchmarks: halfmesh vs SOTA

This document describes the `atlasbench` harness, the state-of-the-art libraries
halfmesh's UV-atlas pipeline is compared against, the measured results, and an
assessment of where halfmesh stands.

## 1. The harness (`tests/bench/atlasbench`)

A standalone CLI that runs halfmesh's atlas pipeline and selected SOTA baseline
engines on the same mesh, converts **every** engine's output into a
`halfmesh::Mesh` (per-corner UVs + per-face chart labels), and measures all of
them with the **identical** `hmtest::metrics` toolkit — so the comparison is
strictly apples-to-apples.

Build & run:
```bash
cmake -S . -B make -DHALFMESH_BUILD_BENCH=ON \
      -DHALFMESH_BENCH_WITH_LIBIGL=ON -DHALFMESH_BENCH_WITH_PMP=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build make --target atlasbench atlasbench_sanity_test -j

# end-to-end comparison (charting + flatten + pack)
./make/tests/bench/atlasbench --mesh tests/data/mesh.ply

# parametrization isolation — every flattener on the SAME chart partition
./make/tests/bench/atlasbench --mesh tests/data/mesh.ply --stage param

# large meshes: decimate first for fast iteration
./make/tests/bench/atlasbench --mesh tests/data/mesh_roi_crop_1.ply --decimate 200000
```
Writes `report.{json,md,csv}` to `--out` (default `.`). Engines: `halfmesh`,
`xatlas` (vendored), and with the opt-in flags `libigl-lscm`, `pmp-lscm`,
`pmp-harmonic`, `cgal-lscm`, `cgal-arap`, `cgal-sdf`, `bff`. Each baseline is
independently gated so the harness still builds when one is absent.
Self-checks: `ctest --test-dir make -L bench`.

### Metrics (all literature-standard)
- **Segmentation**: chart count, boundary-cut (seam) length, per-chart planarity
  error (normal deviation, rad), compactness (perimeter²/4πarea), coverage.
- **Parametrization**: flipped triangles, symmetric-Dirichlet (floor 4.0), Sander
  L2 stretch (floor 1.0), quasi-conformal σ_max/σ_min (1=conformal), area
  distortion (P95/P5 of per-face scale), finiteness.
- **Packing**: rectangle occupancy (engine-reported), triangle occupancy (true
  fill), page count, UV-bbox overlaps.
- **Cross-cutting**: per-stage wall time, peak RSS, determinism, validity.

**Measurement subtleties** (see `tests/bench/BenchMetrics.cpp`): parametrization
metrics are measured on a per-chart **unit-scale** normalization (so the
scale-sensitive energies bottom out at their isometric minima) and **before**
packing (packing's per-chart 90° "rotation" is actually a coordinate transpose =
a reflection that would otherwise miscount every rotated chart as flipped).
Occupancy is measured **after** packing. Per-chart orientation is normalized
uniformly before measuring (a global reflection is a free isometry; without it
an engine with arbitrary handedness would be miscounted as "all flipped").

## 2. SOTA libraries surveyed

| Task | Libraries evaluated | Used as baseline |
|---|---|---|
| Segmentation | xatlas (greedy charting), CGAL SDF, UVAtlas isochart | xatlas, **cgal-sdf** |
| Parametrization | xatlas (LSCM), **libigl** (LSCM), **pmp** (LSCM/harmonic), **CGAL** (LSCM/ARAP), **BFF** (geometry-central) | xatlas, libigl-lscm, pmp-lscm, pmp-harmonic, cgal-lscm, cgal-arap, bff |
| Packing | xatlas, RectangleBinPack (MaxRects) | xatlas |

All six third-party libraries are wired into `atlasbench` (CGAL 6.1 +
SuiteSparse from vcpkg behind the `bench` feature; geometry-central fetched via
`FetchContent`, its real `parameterizeBFF`). See §3 for the full
all-library × all-mesh matrix.

halfmesh uses **SLIM** (the algorithm libigl is known for) for refinement, so
for the flattening stage the question is implementation quality, not algorithm
choice.

## 3. Results — full comparison matrix (all libraries × all meshes × all stages)

Raw output: `tests/bench/results/full_matrix.txt`. Three meshes:
`mesh.ply` (3 786 F), **roi100k** = `mesh_roi_crop_1.ply` (9.1 M → 100 k),
**ours128k** = `mesh_ours_2pivots_texture_refined_999.ply` (15.3 M → 128 k).

> **Fixture note (2026-08):** every `mesh.ply` figure in this document was
> measured on the previous committed fixture — a clean 3,786-face scan. The
> current `tests/data/mesh.ply` is a 120,943-face challenge mesh (non-manifold
> spots, open boundaries, thin structures); rerun `atlasbench` for comparable
> numbers on it. The roi100k/ours128k meshes are large local files, not
> committed (see `.gitignore`).

Large meshes are **directly decimated** (QEM collapses the millions of thin
photogrammetry faces into well-shaped triangles; manifoldness is auto-repaired
during the half-edge build). *Pre*-repairing then decimating instead strips the
thin faces and yields sliver charts that blow up the energy for every engine,
so it is not used here. Parametrization is measured in **isolation**: all
flatteners run on the *same* halfmesh chart partition, so the deltas are the
flattener's alone.

### 3a. Parametrization — symmetric-Dirichlet (↓ better; floor 4.0), flips, finiteness

| engine | mesh.ply | roi100k | ours100k (noisy) |
|---|--:|--:|--:|
| **halfmesh** (LSCM init + SLIM) | **4.006**, 0fl ✓ | **4.007**, 0fl ✓ | **4.000**, 0fl ✓ |
| libigl-lscm | 4.045, 0fl ✓ | 5.622, 0fl ✓ | 4.002, 0fl **✗ non-finite** |
| pmp-lscm | 4.112, 0fl ✓ | 4.120, 0fl ✓ | 86 983, 1fl ✓ |
| pmp-harmonic | diverges (3e16) | diverges | diverges |
| cgal-lscm | 4.043, 0fl ✓ | 4.139, 1fl ✓ | 130 331, 3fl ✓ |
| cgal-arap | 4.008, 0fl ✓ | 3e12, 312fl ✓ | 5e13, 88fl ✓ |
| bff (geometry-central) | 10.5*, 1fl ✓ | 2 053, 224fl ✓ | 4 232, 9fl **✗ non-finite** |

- **halfmesh has the lowest symmetric-Dirichlet on every mesh, always 0 flips,
  always finite.** On clean inputs the LSCM baselines are close (cgal-arap
  4.008, cgal-lscm 4.043, libigl 4.045); halfmesh still edges them because SLIM
  refines the conformal init toward the isometric floor.
- On the **noisy** ours100k (real 15.3 M photogrammetry) halfmesh is the
  **only** engine that is both finite and low-distortion: libigl and BFF go
  **non-finite**, pmp/cgal LSCM explode on the worst charts (8e4–1e5), and
  cgal-arap / pmp-harmonic diverge. This is the robustness gap that matters on
  real data.
- `*` BFF on mesh.ply is genuinely conformal (L2 1.001, quasi-conformal 1.085);
  its 10.5 comes from a single near-degenerate triangle dominating the mean.
- Speed (roi100k flatten): bff 0.13 s ‹ pmp 0.17 s ‹ cgal-lscm 0.27 s ‹
  **halfmesh 0.56 s** ‹ libigl 1.07 s ‹ cgal-arap 9.3 s. halfmesh is mid-pack on
  speed but top on quality+robustness.

### 3b. Segmentation — charts (↓ fewer) / flip-free? / time

halfmesh charts with the **developable D-Charts** method (the only segmenter;
method in §4). It targets the FEWEST flattenable charts, so the right axis is
chart COUNT + flip-freeness, not per-chart flatness (developable charts are
curved by design — a whole cylinder is one chart).

| engine | mesh.ply | roi100k | ours128k |
|---|---|---|---|
| **halfmesh** (D-Charts) | **49** / flip-free / 0.06 s | **5 176** / flip-free / **2.4 s** | **49 376** / flip-free / **2.8 s** |
| xatlas | 95 / 46 071 flips* | 10 177 / 46 071 flips | 69 174 / 63 323 flips |
| cgal-sdf | 3 / — / 0.18 s | 348 / — / 18 s | 18 394 / — / 30 s |

\* xatlas flips are counted on its own LSCM (it charts-then-packs and does not
aim for flip-free islands). halfmesh produces **1.4–3.5× fewer charts than
xatlas on every mesh, all flip-free**, and segments **~35× faster on roi100k
and ~170× faster on the noisy ours128k** (2.8 s vs 475 s — xatlas's greedy
charting struggles badly on noisy input). CGAL SDF is a *part*-decomposition
(few, non-planar segments), not a developable-atlas charter; included for
completeness.

### 3c. Packing — rectangle occupancy / pages / time, and end-to-end

| | mesh.ply | roi100k | ours128k |
|---|---|---|---|
| halfmesh occ / pages / time | 82.0% / 1 / 0.000 s | 82.0% / 1 / **0.018 s** | 82.0% / 1 / 0.11 s |
| xatlas occ / pages / time | 72.0% / 1 / 0.046 s | 82.6% / 1 / 0.53 s | 73.6% / 1 / 2.61 s |
| **end-to-end** halfmesh vs xatlas | 0.07 s vs 0.22 s | **2.9 s vs 85 s** | **3.7 s vs 477 s** |

Single-page rectangle occupancy is on par with xatlas (82.0% vs 82.6% on roi;
halfmesh *higher* on mesh.ply and ours128k) at **~30× the packing speed**.
xatlas's *triangle* occupancy is higher (it packs silhouettes, not bounding
rects) — the one place halfmesh trails, addressed by the rasterized-silhouette
future-work item (§6).

### 3d. Verdict

**halfmesh is at or above SOTA on all stages across every test mesh.**
Segmentation (developable D-Charts) yields the **fewest charts of any engine,
always flip-free**, 1.4–3.5× fewer than xatlas; the flattener is the most
robust on noisy real photogrammetry (the only one of six libraries that stays
finite and low-distortion where the others diverge or explode — §3a); packing
rect-occupancy is on par; and end-to-end it is 30–130× faster than xatlas. The
trades: D-Charts accepts higher *per-chart* distortion on noisy meshes for far
fewer charts (§4), and *triangle* packing occupancy still trails xatlas's
silhouette packer.

§3a measures the FLATTENER in isolation (all flatteners on one common chart
partition). The end-to-end distortion under the D-Charts partition (fewer,
larger charts) is reported in §4.

## 4. Segmentation method — developable D-Charts (halfmesh's segmenter)

halfmesh's segmenter (the only one; `SegmentCharts` in
`AtlasCharting.{h,cpp}`) targets the *fewest* charts directly, following
D-Charts (Julius/Kraevoy/Sheffer 2005): it bounds **flattenability**, not
flatness. A wall, a whole cylinder, and a cone are each *one* chart because a
**cone proxy** ⟨axis, half-angle⟩ fits the entire developable family (the fit
error is λ_min of the weighted normal covariance — zero for any developable
surface), where a planar L2,1 metric would split a curved-but-flattenable
region into many patches.

Pipeline: **global cone-Lloyd** (multi-source cone flood-assign + seed
relocation + batched developability seeding) → **developable merge**
(consolidate adjacent charts while the combined cone error stays in budget,
with an angle-defect anti-fold cap) → **flip/topology repair** (flatten every
chart and spatially bisect any that folds, until none do — the hard flip-free
guarantee). Robustness to MVS noise comes from a virtual **Taubin geometry
smoothing** that denoises the normals *and* the angle defect used by the
analysis (the mesh itself is untouched). Full design:
`docs/ATLAS_SEGMENTATION_DESIGN.md`; approaches that did **not** work are
recorded atop `include/halfmesh/AtlasCharting.h` so they are not re-tried.

### Chart count (↓ better) / flips (↓; 0 = valid atlas) / sym-Dirichlet (↓; floor 4.0)

Columns: charts / flips / sym-Dirichlet / **end-to-end wall time** (segment +
flatten + pack). The retired planar-VSA Lloyd is shown for chart-count
reference (it was flip-free but over-segmented — curved-but-developable
regions shatter under a flatness metric).

| mesh (faces) | xatlas (charts / flips) | retired VSA (charts) | **halfmesh D-Charts (charts / flips / symD / time)** |
|---|---|---|---|
| mesh.ply (3 786) | 95 / 1 430 | 172 | **49 / 0 / 4.03 / 0.07 s** |
| roi100k (100 k) | 10 177 / 46 071 | 11 723 | **5 176 / 0 / 10.2 / 2.9 s** |
| ours128k (128 k) | 69 174 / 63 323 | 84 417 | **49 376 / 0 / 9.05 / 3.7 s** |
| **ours full (15.3 M)** | — (intractable) | — | **190 380 / 0 / 80.7 / 321 s** |

> The large-mesh **sym-Dirichlet** figures above (10.2 / 9.05 / 80.7) predate
> the Lévy-LSCM flatten-init upgrade, which dropped per-chart distortion to
> near the 4.0 floor; current halfmesh `base` sym-Dirichlet is ~4.4 (roi100k) /
> ~4.5 (ours128k) — see the opt-in subsection below for faithful inline
> numbers. Chart counts are unaffected.

D-Charts produces the **fewest charts on every mesh** — 1.4–3.5× fewer than
xatlas and 1.7–3.5× fewer than the retired VSA — while staying **flip-free**
(xatlas folds tens of thousands of triangles). End-to-end it is **30–130×
faster** than xatlas. The trade is higher sym-Dirichlet on noisy meshes: fewer,
larger charts over curved real surfaces stretch more — the intended *prefer
fewer charts even at some extra distortion* trade, tunable via
`developable_max_cone_error` (smaller ⇒ more, flatter charts).

### Distortion-bounded split & Seamster cut-to-disk (two opt-in knobs)

Two opt-in axes extend the segmenter, both **OFF by default** — the numbers
above are unchanged until a knob is flipped:

- `developable_max_uv_distortion` (τ, symmetric-Dirichlet cap;
  `--max-distortion`): a flip-FREE but over-stretched chart is *also*
  bisected-and-requeued by the existing flip-repair, judged on the **shipped
  SLIM map** (not the conformal LSCM init, which reads high) with a mandatory
  sliver guard so degenerate input is never shattered to the `4F` cap. τ = 4.0
  is perfect isometry; 4.4 is the sane on-value.
- `cut_to_disk` (Seamster, Sheffer & Hart 2002; `--cut-to-disk`): a closed /
  multiply-connected chart is **slit open into one disk** instead of being
  bisected into many — the chart-count reducer on hole-riddled MVS. The
  flip-repair safety net still bisects any chart that folds *after* the cut, so
  the hard flip-free guarantee is intact.

4-cell attribution (charts / sym-Dirichlet / end-to-end time; flips 0 unless
noted), all `--engines halfmesh --fix-manifold --seed 0`, faithful inline
`--decimate`:

| mesh | base | p1 `τ=4.4` | p2 `cut` | both `cut+τ` |
|---|---|---|---|---|
| roi100k (100 k) | 5 184 / 4.44 / 3.0 s | 5 671 / 4.02 / 3.2 s | 3 980 / 6.11 / 6.4 s | **4 185 / 4.02 / 6.5 s** |
| ours128k (128 k) | 32 292 / 4.55¹ / 2.6 s | 34 055 / 4.39¹ / 2.9 s | 30 486 / 24.95¹ / 2.7 s | 31 344 / 24.90¹ / 3.1 s |

¹ ours128k carries one pre-existing baseline flip (a decimation sliver, already
in `base`); the cut is flip-neutral (adds none).

**Attribution.** On the moderately-noisy **roi100k** the two knobs are
complementary and together **escape the chart-count ↔ distortion Pareto
frontier**: `cut` alone removes **23 %** of charts (5 184→3 980) by opening
hole-riddled regions into single disks, but stretches the larger charts
(sym-Dir 4.44→6.11); adding the distortion split (`both`) splits exactly those
over-budget charts back down — **19 % fewer charts than base AND sym-Dir at the
4.0 isometric floor, flip-free**, with higher packing occupancy (0.287→0.302).
The split alone (`p1`) is a pure quality move: 4.44→4.02 for +9 % charts.

On the **very-noisy ours128k** the cut is a weaker trade: it opens
**sliver-dominated** mega-charts (sym-Dir → ~25) for only **−6 %** charts, and
the distortion split cannot rescue them — the sliver guard refuses to bisect
degenerate input (splitting would shatter to the `4F` cap). Net guidance:
`cut_to_disk` pays off on moderately-noisy MVS, best left off on sliver-ridden
meshes; `developable_max_uv_distortion` is a safe, monotone quality knob
wherever distortion is curvature- (not sliver-) driven.

**Determinism.** Every config was run twice; chart_count and sym-Dirichlet came
back **byte-identical** on every mesh. The cut is a pure function of the chart
geometry (sorted loop-starts, sorted adjacency, sorted BFS seeds, index-order
duplicated-vertex ids), so it survives the determinism gate that
`unordered_map`/`set` iteration would otherwise fail.

### Cost / scale — the full 15.3 M-face mesh

Run on the **original 15.3 M-face** photogrammetry mesh with only
`FixNonManifold()` (the minimal manifold repair: it splits non-manifold
vertices/edges but keeps **all 15 331 464 faces**), the *entire* pipeline
(segment → flatten → pack) completes in **321 s (5.4 min) at a 4.68 GB peak**,
producing 190 380 charts, flip-free, 9 atlas pages. Stage split: segmentation
286 s, flatten 34 s, pack 1.5 s; the flip repair is incremental (only
re-checks the pieces it splits) so it adds little over the cone-Lloyd. (For
production, decimate first: million-chart atlases are rarely the goal, and
distortion/seam totals favour a chart budget matched to the texture
resolution.)

### Repair carve rings / fold-rescue slits / per-size padding — Task 9 sweep (defaults OFF)

Three more opt-in knobs landed in 0.3.1, all **OFF by default**:
`repair_carve_rings` (`--repair-carve-rings`, failure-localized
repair split — carve off the faces within N `TopoNeighbor` rings of a folding
chart's diagnosed failure instead of a blind PCA bisection), `fold_rescue_slits`
(`--fold-rescue-slits`, cut a slit from the worst interior vertex to the
boundary and re-flatten a folding chart in place, up to N times, instead of
splitting it), and `AtlasParams::tinyChartSide` / `debrisChartFaces`
(`--tiny-chart-side` / `--debris-chart-faces`, a 1-texel gutter instead of the
uniform `padding` for charts under a size/face-count trigger — packing only,
does not change the partition).

**Setup.** `tests/data/` on this machine contains only `mesh.ply` (63 049
vertices / 120 943 faces once repaired — 459 non-manifold issues fixed on
load; no `roi100k`, `ours128k`, or a Truck-class mesh are present here, so
those cross-checks and the spec §7 success-criterion mesh could not be run on
this machine — see "what to try next" below). All arms: `--engines halfmesh
--resolution 4096 --cut-to-disk` (padding stays at the `AtlasParams` default
of 2, unchanged by any arm). Wall-clock is indicative only — this machine was
under load from an unrelated training job during the sweep.

| arm | charts | coverage (tri) | seg. time (s) | flips | sym-Dirichlet |
|---|--:|--:|--:|--:|--:|
| baseline (all off) | 2 390 | 38.8 % | 43.9 | 0 | 45.8 |
| `carve=1` | 2 256 | 39.6 % | 46.4 | 0 | 51.4 |
| `carve=2` | 2 302 | 40.2 % | 54.3 | 0 | 485.8 |
| `carve=3` | 2 375 | 36.0 % | 57.5 | 0 | 27.6 |
| `slits=2` | 2 153 | 40.5 % | 72.5 | 0 | 52.4 |
| `carve=2, slits=2` | 2 022 | 41.6 % | 56.8 | **4** | **9 533 546** |
| `carve=2, slits=2, tiny=8` | 2 022 | 41.6 % | 54.9 | 4 | 9 533 546 |
| `carve=2, slits=2, tiny=8, debris=100` | 2 022 | 42.3 % | 54.0 | 4 | 9 533 546 |

(`tiny=8`/`debris=100` only touch Module D packing, so the partition and
parametrization above them are byte-identical to `carve=2, slits=2` — the
extra knobs just shrink the gutter on the smallest charts, nudging coverage
from 41.6 % to 42.3 %.)

**Reading the table:**

- **Carve alone** is a small, *non-monotonic* win: ring=1 drops charts
  2390→2256 (−5.6 %) cleanly, but ring=2 and ring=3 climb back toward baseline
  (2302, 2375) while ring=2's sym-Dirichlet (485.8) is a wild outlier next to
  ring=1 (51.4) and ring=3 (27.6) — no ring count is a clear standardization
  target yet.
- **Slits alone** is a clean win: 2390→2153 (−9.9 %) with flips still 0 and
  sym-Dirichlet in line with baseline (52.4).
- **Carve+slits combined is worse than either alone**, not just on chart
  count relative to slits-alone (2022 vs 2153 is actually fewer — the
  regression is on *quality*): flips go from 0 to **4**, and sym-Dirichlet
  balloons **~200 000×** (45.8 → 9 533 546). Re-run twice — byte-identical
  both times. This is exactly the spec's §7 warning materializing: fold-rescue
  slits opening a sliver-dominated chart, here compounded by carve fragments
  feeding it. A blind flip of both defaults would ship this regression to
  every caller.

**Defaults decision: stay OFF** (`repairCarveRings = 0`, `foldRescueSlits = 0`;
the padding knobs remain opt-in). Rationale:

1. The spec §7 success criterion (≤ 55 k charts on ~480 k faces, 0.11–0.13
   charts/face, coverage ≥ 0.30 at padding 2 / 4096²) is defined on a
   Truck-class mesh that is **not present on this machine** — there is no
   direct evidence the defaults help on the mesh class the criterion targets.
2. On the one mesh available, the combined configuration measured **worse
   than either knob alone** — real flipped triangles and a catastrophic
   distortion outlier, not merely a smaller-than-hoped win.
3. Carve-ring sensitivity is non-monotonic with no obviously-better fixed
   value.

No golden re-freeze: defaults are unchanged, so `tests/golden/` fixtures
still reflect the current (591/591-green) behavior.

**What to try next:**

- **Ridge-snapped carve boundaries**: route the carve-ring cut along a nearby
  ridge/curvature feature instead of a blind N-ring band, to avoid slicing
  through the sliver-thin geometry implicated in the sym-Dirichlet blowup
  above.
- **Enclose-test revision** (per Task 7's gate analysis): the post-repair
  merge's `wouldEnclose` rejects outnumber budget rejects ~3.3–3.7× on
  `mesh.ply`, yet folding pairs still pass both gates and ship as extra
  fragments — a geometry-aware (rather than blanket) enclose test might let
  more carve-created fragments re-merge instead of shipping as extra tiny
  charts.
- **The pending Truck-class sweep**: re-run this Step-1/Step-2 sweep (all
  arms above, plus the `mesh.ply` / `roi100k` / `ours128k` cross-checks) on a
  genuine Truck-class mesh (a Tanks-and-Temples splat reconstruction,
  QEM-decimated aggressiveness 7 to ~500 k faces) — via the Python wheel and
  the new `unwrap()` knobs (`docs/PYTHON.md`) — on the machine that has that
  data, the study's L4 box (spec §7). The ≤ 55 k charts / 0.11–0.13 charts-per-face
  / ≥ 0.30 coverage criterion is only meaningful measured there.

## 5. Assessment

| Stage | Verdict |
|---|---|
| Segmentation | **At/above SOTA**: fewest charts of any engine, always flip-free; ~35× faster than xatlas on roi100k, ~170× on noisy ours128k. |
| Parametrization | **Above SOTA**: lowest symmetric-Dirichlet of all 7 flatteners on every mesh, 0 flips, always finite; the only engine that stays finite+low-distortion on noisy input. |
| Packing | **At/above SOTA**: rect occupancy on par with xatlas in a single page, ~30× faster (triangle occupancy still trails — see §6). |
| Robustness | manifoldness auto-checked+repaired before any half-edge build; weld reconnects glTF sub-meshes; graceful malformed-input handling. |
| Scalability | segmentation merge O(E·log) + skyline packing → no O(n²) blowups; end-to-end 30–130× faster than xatlas on large meshes. |

## 6. Future work

- **Rasterized-silhouette packing**: pack chart bitmaps rather than bounding
  rects, to push triangle occupancy higher (xatlas's triangle occupancy is
  higher because it packs silhouettes, not just bboxes).
- **Lower-overhead half-edge** (perf, not correctness): a struct-of-arrays,
  32-bit-indexed layout with CSR vertex adjacency and derived (not stored)
  twins would cut the per-element cost and the many tiny heap allocations;
  useful for full-resolution meshes but not required for correctness.
