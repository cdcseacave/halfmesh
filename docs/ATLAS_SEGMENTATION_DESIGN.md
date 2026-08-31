# Atlas Generation — Chart Segmentation: Design & Implementation

As-built design note for `SegmentCharts` (Module A) and its companion flattener
(Module B). §0 and §3–§5 describe the pipeline that ships in
`src/AtlasCharting.cpp` / `src/Parametrize.cpp`; §1–§2 record the problem analysis
and method survey that motivated it.

**Goal (priority order):**

1. **As few charts as possible.**
2. **Low size variance** — charts that follow the surface, no single giant + a swarm of slivers.
3. **Fast** — offline, a few seconds on 100k–1M faces.
4. **Low per-chart parametrization distortion** — each chart flattens (UV) with little stretch.

**Target input:** photogrammetry / MVS reconstructions — dense, *noisy per-face normals*, organic surfaces. Robustness to normal noise is the dominant constraint, and the one a flatness-based segmenter gets wrong.

**Scope:** Module A (segmentation), designed jointly with Module B (flattening) because priority #4 can only be measured after flattening and the two close a loop through the `detail::ChartFacesFold` bridge. Density (C) and packing (D) are unchanged.

---

## 0. TL;DR — what shipped

The segmenter targets the **fewest flip-free, flattenable charts** by bounding
**developability**, not flatness, following **D-Charts** (Julius/Kraevoy/Sheffer 2005):
a wall, a whole cylinder, and a cone are each *one* chart because a single **cone proxy**
⟨axis, half-angle⟩ fits the entire developable family (its fit error is λ_min of the
chart's area-weighted normal covariance — exactly zero on any developable surface), where
a planar normal-deviation proxy would shatter a curved-but-flattenable region.

`SegmentCharts` is three phases over the half-edge mesh, preceded by a robust precompute:

> **Precompute** (adaptive Taubin geometry denoise + normals + angle defect) →
> **cone-Lloyd** (batched global relaxation) → **developable merge** (union-find, anti-fold
> capped) → **flip/topology repair** (flatten, bisect any folder, until none) → *(Module B)*
> **LSCM/Tutte init + SLIM/ARAP** flatten.

The two highest-leverage decisions, relative to a planar-VSA segmenter:

- **Dihedral creases are not walls.** Treating them as walls fractures the mesh into
  thousands of crease-bounded regions (one mandatory seed each) before growth even starts.
  The shipped code treats only mesh-border / non-manifold / material(texblob) edges as hard;
  the cone proxy absorbs curved-but-developable surface and a per-vertex **angle-defect cap**
  decides where a chart *must* stop. This removes the chart-count explosion at the root; no
  softened crease weight is needed, and there is no crease term at all.
- **A hard flip-free guarantee by flatten-and-bisect.** Every chart is actually flattened;
  any that folds is spatially bisected and its pieces re-checked until none fold (≤2-face
  charts cannot fold ⇒ it converges). Without such a disk-topology invariant, segmentation
  alone produces tens of thousands of flips.

Two extensions ride the same flatten/repair machinery (see `docs/BENCHMARKS.md` §4):

- **Distortion-bounded split** (`developable_max_uv_distortion`): the flip-repair *also* splits
  a flip-free-but-over-stretched chart, gated on the **measured shipped** symmetric-Dirichlet,
  sliver-guarded. Always on — the parameter *tightens* the internal ship-ability bar toward
  isometry rather than switching the check on, so `0` (the default) still splits charts
  stretched past all use.
- **Seamster cut-to-disk** (`cut_to_disk`): a closed / multiply-connected chart is *slit* into
  one disk at flatten time instead of being bisected into many — fewer charts on hole-riddled
  MVS, with the flip-repair still bisecting anything that folds after the cut.

---

## 1. The problem this design solves (planar-VSA segmentation)

A planar-VSA segmenter produces, on `ours100k` (1024², "all"):

| stage | symptom | value |
|---|---|---:|
| segmentation | charts | **34,689** |
| segmentation | seam length | 42,093 |
| segmentation | time | **489.99 s** |
| parametrization | flips | **25,682** |
| parametrization | sym-Dirichlet | 1.5e14 (diverged) |

Root causes (all four stated priorities, each a distinct failure):

**(a) Creases were HARD walls, evaluated on noisy normals.** An interior edge was marked a
wall when `cos(dihedral) < cos(40°)`. On MVS normals a large fraction of edges cross 40°
even after smoothing, so the flood fractured into thousands of tiny regions — and each region
forced ≥1 chart. The chart count was lower-bounded by the crease-region count before growth.

**(b) A threshold-driven split loop re-ran full Lloyd per split.** Each iteration found the
chart whose area-normalized planar L2,1 cost exceeded a fixed bound, inserted one seed, and
re-ran Lloyd over the whole mesh. *Count:* a planar-fit threshold is a curvature gate, so a
smoothly curved developable region (cylinder, rock face) is split forever — the loop's fixed
point is "one chart per near-planar facet." *Speed:* O(S·F log F) with S in the tens of thousands.

**(c) A planar L2,1 proxy is the wrong fit model.** Developable-but-curved geometry cannot be
represented by a single average normal, so it is over-segmented. The cone model existed but was
used only in the post-merge, after the explosion.

**(d) No disk-topology guarantee → the flips.** Flattening pinned the *longest* border loop; an
annulus / handle / pinched chart is not a disk, so the rest folded — 25,682 inverted triangles.

---

## 2. Method survey

Mapped to the four priorities. "Fit model" is the per-chart shape the method keeps each chart close to (this bounds flatten distortion).

| Method | Fit model / driver | Few charts | Equal size | Fast | Low distortion | Disk topology |
|---|---|:--:|:--:|:--:|:--:|:--:|
| **VSA** (Cohen‑Steiner 2004) | planar proxy, Lloyd L2,1 | ✗ (shatters curves) | ✓ (Lloyd) | ✓ | ~ | ✗ |
| **D‑Charts** (Julius 2005) — *the shipped fit model* | **cone** proxy + compactness; bounds developability at segmentation | ✓ | ✓ | ✓ | ✓ | ~ |
| **Multi‑Chart Geometry Images** (Sander 2003) | region-grow from seeds, merge | ~ | ~ | ✓ | ~ | ✓ (validated) |
| **Iso‑charts / UVAtlas** (Zhou 2004) | **stretch**-driven spectral (IsoMap); bounds stretch, count stays small; *no flips* | ✓ | ~ | ~ (per-chart spectral) | ✓ | ✓ |
| **xatlas / thekla** (practical baseline) | greedy growth: normalDeviation + **roundness** + **straightness** + **normalSeam**; **maxChartArea**; maxCost merge; **disk-validated**; LSCM | ✓ | ✓ (area cap) | ✓ | ✓ | ✓ |
| **OptCuts** (Li 2018) | joint cut+param, **hard distortion bound**, minimal seam length, no tuning | ✓✓ | — | ✗ (minutes) | ✓✓ | ✓ |
| **Variational Surface Cutting** (Sharp 2018) | minimal-distortion cuts via uniformization | ✓ | — | ✗ | ✓✓ | ✓ |
| **PartUV** (SIGGRAPH Asia 2025) | top-down recursive part decomposition; **per-chart distortion bound, minimize count**; robust on noisy/AI meshes; ABF++ | ✓✓ | ~ | ~ | ✓ | ✓ |
| Neural fields / SeamCrafter‑RL (2024–25) | learned seams / UV field | ✓✓ | — | ✗ (>30 min, GPU) | ✓ | ✓ |

What we took, for our constraints (offline-seconds, MVS, geometric C++ lib, no GPU/ML):

- **D‑Charts** supplied the *fit model* — a cone bounds developability, so cylinders/cones stay
  whole. The shipped code makes it drive the *grow* step (not just the post-merge), which is the
  fix for cause (c).
- **xatlas/thekla** supplied the *structural template* (we inherit its packer) and the lesson that
  **disk validity must be enforced**, not assumed — realized here as flatten-and-bisect repair.
- **Iso‑charts / OptCuts / PartUV** supplied the *principle*: bound **measured** per-chart
  distortion and minimize count, rather than gating on a curvature proxy. The shipped
  `developable_max_uv_distortion` split is exactly this, done cheaply (no spectral / global solve),
  and the `cut_to_disk` option borrows **Seamster**'s disk-slitting.
- **OptCuts** remains the quality north-star for a possible future "max-quality, minutes" mode.

---

## 3. The implemented pipeline

`SegmentCharts` (`src/AtlasCharting.cpp:1089`) runs, in order:

```
Precompute → ConeLloydSegment → EnforceConnectivity → DevelopableMerge
           → EnforceConnectivity → RepairDevelopableFlips        (gated on flip rounds)
```

then Module B (`ParametrizeCharts`) flattens each chart. Every stage is a thin layer over
`HalfMesh` primitives (§4).

### Phase 0 — Precompute (`Precompute`, :174)

- Build/validate the half-edge structure.
- **Adaptive Taubin geometry denoise.** λ|μ Laplacian smoothing (lambda 0.330, mu −0.331, so
  |μ|>λ ⇒ no shrinkage; boundary vertices pinned) over a **temporary** copy of the positions.
  Face normals (cone fit) and the per-vertex angle defect (cap) both derive from triangle
  geometry, so denoising once makes the whole developable analysis robust to MVS noise — the
  mesh and the UVs are never modified. **Adaptive:** skipped below 2000 faces, so a clean/
  synthetic mesh (e.g. a cube) is never deformed.
- Face normals from the denoised positions; per-face **weights = true-geometry area** (so density
  / packing measure the real surface), overridable via the `face_weight` hook.
- Hard boundaries a chart may never cross = **mesh border / non-manifold edge / material(texblob)
  seam** (`TopoNeighbor`). Dihedral creases are *not* walls.

### Phase 1 — Cone-Lloyd (`ConeLloydSegment`, :826)

The scalable core: a global, simultaneous Lloyd relaxation on cone proxies.

- **Seeds:** one per topological region (`ComputeRegions` — connected components across
  `TopoNeighbor`; mandatory) + farthest-point extras (`AddFarthestSeeds`, count ≈
  `(1 + seed_extra_mult) × regions`).
- **Relaxation (`ConeLloyd` → `ConeFloodAssign` + `RecomputeCones`):** flood-assign every face at
  once to the cheapest-fitting cone (best-first by cone-fit cost, by default), then iterate
  {recompute each chart's best-fit cone = λ_min of its weighted normal covariance + relocate its
  seed → re-flood}, for `max_iterations`.
  - **Optional distance term (`developable_distance_exponent` β, default 0 = off).** When β > 0
    the flood-assign cost becomes `(fit + 1e-8)·dist^β` (D-Charts' F¹·Dᵝ·N⁰ form; Julius/Kraevoy/
    Sheffer 2005 use β=0.7), where `dist` is the accumulated centroid-path distance from the
    chart's seed along the flood tree. Among near-tied fits the geometrically closer chart wins,
    so charts stop snaking along curvature ridges toward a farther-but-marginally-better fit.
    Multiplicative, not additive, so it is scale-invariant: uniformly scaling the mesh by `s`
    rescales every candidate's key by the same factor `s^β`, which preserves relative pop order —
    and therefore the segmentation — with no normalization constant needed; at β=0 the factor is
    1 and the ranking is exactly the fit-only order above. Measured on the segment-quality harness
    (`tests/SegmentQualityTest.cpp`, the evidence for this knob): at β=0.7 an MVS-like scan
    improves (mesh.ply: charts −12%, seam −3.2%) and a near-tie UV-sphere's seam drops 8.4%, but
    an anisotropically-developable surface can over-split (torus: 2→4 charts) and segmentation
    wall-clock at 200k faces rises ~27% — so the term ships **opt-in** (default β=0, off). Measured
    on the reference build; near-tie outcomes are build-flag sensitive (e.g. -march=native flips
    the UV-sphere seam delta).
- **Batched developability seeding:** when charts exceed the `developable_max_cone_error` budget,
  add — in ONE batch — a seed at the worst face of every over-budget chart and re-Lloyd. Batching
  converges in O(log) rounds where one-seed-at-a-time growth is O(F²); this is the fix for cause
  (b)'s speed and (a)'s count.

### Phase 2 — Developable merge (`DevelopableMerge`, :417)

A union-find pass over the chart adjacency graph stitches adjacent charts while the merged
per-area cone error stays within `developable_max_cone_error`, with an **angle-defect anti-fold
cap**: a merge that would make a cone/saddle vertex (|angle defect| > `developable_max_vertex_defect`)
chart-*interior* is forbidden. This consolidates co-developable neighbours into the fewest large
charts on the now-small graph, and the cap is the cheap guard that prevents most folds before
flattening.

### Phase 3 — Flip/topology repair (`RepairDevelopableFlips`, :1022; gated on `developable_flip_repair_rounds` > 0)

The hard flip-free guarantee — the disk-topology invariant, realized operationally. Each chart is **flattened** (Module B) and, if it folds, **spatially bisected**
(median split along its longest PCA axis), its connected pieces re-checked, until none fold. A
piece is strictly smaller than its parent (≥2 components per split), and ≤2-face charts cannot
fold, so it converges; a `4·F` runaway cap backstops degenerate input. It is **incremental** —
settled charts are never re-flattened — so cost is ~O(F·log) rather than rounds×all-charts. The
fold test is `detail::ChartFacesFold` → `ChartFolds` (`src/Parametrize.cpp`), the single bridge
between Modules A and B.

### Extensions riding the Phase-3 predicate

- **`developable_max_uv_distortion` (τ).** The repair predicate *also* flags a flip-FREE chart
  whose **shipped** (full SLIM) area-weighted symmetric-Dirichlet exceeds τ, so it is bisected
  too — trading a few extra charts for lower per-chart distortion. Measured on the *shipped* map
  (not the conformal LSCM init, which reads high), with a **mandatory sliver guard**
  (`epsA = meanArea·1e-6`) excluding degenerate near-zero-area input so it cannot runaway-split.
  τ = 4.0 is perfect isometry (the floor of the `s₀²+s₁²+s₀⁻²+s₁⁻²` energy); ~4.4 is the sane
  on-value. This is the iso-charts/OptCuts/PartUV principle done cheaply — a flat or developable
  chart is never split, only genuinely high-curvature regions earn more charts.

  **This check is always on.** τ = 0 (the default) does not disable it — it selects an internal
  ship-ability bar of 200, the same one the injectivity fallback in `ParametrizeCharts` refuses
  to ship above. Without that floor a flip-free chart shipped at any stretch: measured on a
  471 814-face Ignatius at defaults, 31 of 78 123 charts above 200 and the worst at 3.3e8.
  Setting τ tightens the bar toward isometry; it does not switch a check on.
- **`cut_to_disk` (Seamster, Sheffer & Hart 2002).** A closed (no boundary) or multiply-connected
  / pinched chart is **slit** into a single-boundary disk at flatten time (and stays ONE chart)
  instead of being bisected into many by the disk guarantee — far fewer charts on hole-riddled MVS.
  `FlattenChart` (the shipper) and `ChartFolds` (the oracle) run the IDENTICAL cut so the repair
  verdict matches the shipped map, and the flip-repair still bisects anything that folds *after*
  the cut, so the guarantee is preserved. The cut surgery duplicates only vertices (never faces),
  so the chart-id labelling and the texcoord writeback are untouched. Best on moderately-noisy MVS;
  on very-noisy sliver meshes it opens sliver-dominated charts the split cannot rescue (off by
  default for that reason). See `docs/BENCHMARKS.md` §4.

### Module B — per-chart flattening (`src/Parametrize.cpp`)

Per chart: extract a local submesh + boundary loop → initialize with **LSCM** (free-boundary
least-squares conformal, two-pin gauge — near-isometric, the default) when it is flip-free, else
**Tutte** (convex-boundary Laplacian, injective on a disk), else a **PCA** planar projection →
refine with **SLIM** (symmetric-Dirichlet proximal + flip-free line search; default) or **ARAP**
on a constant cotangent Laplacian factored once. The LSCM init is what pulled converged
sym-Dirichlet from ~10 down to near the 4.0 floor on noisy meshes.

---

## 4. Exploiting the half-edge structure

Every stage is a thin layer over primitives `HalfMesh` already exposes — no auxiliary adjacency.

| Need | HalfMesh primitive |
|---|---|
| Dual-graph step (face → neighbour across edge) | `HeTwin` (O(1), `iHe ^ 1`) + `HeFace`; `FAdjacentFaces`; `SegmentState::TopoNeighbor` (skips hard seams) |
| Topo-region / per-chart flood + connectivity | `ComputeRegions` / `EnforceConnectivity` over `TopoNeighbor` |
| Farthest-point seeding & compactness | hop/centroid BFS over half-edges; `PriorityQueue.h` (`Candidate` min-heap) |
| Boundary-loop extraction & count `b` | `BuildBoundaryLoop` walks the chart-border half-edge chain (Module B) |
| Cone proxy + merge | incremental normal-covariance moments with union-find — `RecomputeCones`, `DevelopableMerge` |
| Angle-defect cap | `ComputeVertexDefect` (Gaussian curvature from the denoised geometry) |
| Disk-slitting (opt-in cut) | `AllBoundaryLoops` / `ShortestCutEdges` / `CutAlongEdges` (deterministic, Module B) |
| Signal-aware weighting | `face_weight` hook → `SegmentState::weights` |

The cone-moment union-find and the `ChartFacesFold` flatten bridge are the two load-bearing
primitives.

---

## 5. Parameters (`ParametrizeParams`, as built)

### Module A — segmentation

| param | default | role |
|---|---:|---|
| `developable_max_cone_error` | 0.05 | per-area cone-fit error a chart may reach — the **primary count/distortion lever**. Drives batched seeding (Phase 1) and the merge gate (Phase 2). Smaller ⇒ more, flatter charts. |
| `developable_max_vertex_defect` | 0.35 rad (~20°) | angle-defect **anti-fold cap**: a vertex above it stays on a chart boundary, never interior. Smaller ⇒ more, flatter charts. |
| `developable_distance_exponent` | 0.0 (**opt-in**) | flood-assign distance-term exponent β (§3 Phase 1). 0 = fit-only. β=0.7 (paper value) measured best on-value: helps MVS-like/near-tie meshes, can over-split anisotropically-developable ones (see `tests/SegmentQualityTest.cpp`). |
| `developable_smooth_iters` | 4 | adaptive Taubin geometry-denoise passes (skipped < 2000 faces). 0 disables. |
| `max_iterations` | 8 | cone-Lloyd relaxation iterations. |
| `seed_extra_mult` | 1.0 | farthest-point extra seeds as a multiple of the region count. |
| `developable_flip_repair_rounds` | 16 | flip/topology repair (0 = off). |
| `developable_max_uv_distortion` | 0.0 | symmetric-Dirichlet split cap τ (floor 4.0; ~4.4 on-value). 0 selects the internal ship-ability bar (200), **not** off. |
| `cut_to_disk` | false (**opt-in**) | Seamster cut-to-disk instead of bisecting non-disk charts. |
| `face_weight` | area | per-face importance hook (signal-aware weighting). |

### Module B — flattening

| param | default | role |
|---|---:|---|
| `method` | `SLIM` | SLIM (symmetric-Dirichlet, flip-free line search) or ARAP. |
| `init_method` | `LSCM` | LSCM free-boundary conformal init (falls back to Tutte/PCA), or Tutte. |
| `flatten_iterations` | 5 | local/global refinement iterations. |

**Deliberately absent knobs:** `target_chart_area` / `target_charts` (chart count is driven by
the cone-error budget + merge — developability-driven, not area-budgeted) and any
crease/compactness/seam weights (developability + the angle-defect cap replace them).

---

## 6. Validation

`atlasbench` is the regression harness (`docs/BENCHMARKS.md` §3–§4). Current state vs. the
planar-VSA failures (§1):

- **charts** — fewest on every mesh, 1.4–3.5× below xatlas; the 34,689 → low thousands collapse
  came from Phases 0–2.
- **flips = 0** — the flatten-and-bisect repair makes this a guarantee, not a hope (25,682 → 0).
- **sym-Dirichlet** — near the 4.0 floor on clean meshes; ~4.4–4.5 base on noisy MVS after the
  LSCM-init upgrade (the opt-in split pulls it to the floor at the cost of a few charts).
- **time** — seconds on 100k (was minutes); the full 15.3 M-face mesh segments + flattens + packs
  end-to-end in minutes, flip-free.
- **determinism** — `--seed 0` runs are byte-identical (chart_count + sym-Dirichlet); the opt-in
  cut is hardened to a pure function of the chart geometry.
- **invariants** (unit tests): partition, per-chart connectivity, finite UVs, flip-free flatten on
  the test meshes; defaults-off opt-in knobs keep all existing fixtures unchanged.

Cross-checked against **xatlas**, **libigl**, **pmp**, **CGAL**, **BFF** on the same meshes for
charts / distortion / time (the full matrix is `docs/BENCHMARKS.md` §3).

---

## 7. Risks & open edges

- **Very-noisy sliver meshes.** `cut_to_disk` opens sliver-dominated mega-charts (sym-Dirichlet
  → tens) that the distortion split cannot rescue — the sliver guard correctly refuses to bisect
  degenerate input. Off by default; recommended only on moderately-noisy MVS. (`ours128k`: cut is a
  −6 % chart / +5× distortion trade, vs. `roi100k`'s clean −23 % / flip-free win.)
- **Genus > 0 charts.** The cut-to-disk handles holes/annuli, not handles; a residual genus>0 chart
  falls through to flip-repair bisection (documented limitation, rare on MVS).
- **Cone fitting on tiny/noisy charts.** Guarded by the adaptive geometry smoothing and area
  weighting; the angle-defect cap keeps high-curvature vertices on boundaries.
- **The 15.3 M-face mesh** with hundreds of holes can hit the 256-merge cut guard + `4·F` repair
  cap — watch wall-time and chart_count there specifically (the cut is best left off at that scale).
- **A "max-quality, minutes" mode** (OptCuts-style joint cut+param, or a PartUV-style learned part
  prior at the seeding stage) remains the natural future extension for hero assets; the shipped
  pipeline's *top-down, developability-bounded, count-minimizing* philosophy is deliberately the
  same, so such a mode could slot in without disturbing the rest.

---

## 8. References

- Cohen-Steiner, Alliez, Desbrun. *Variational Shape Approximation.* SIGGRAPH 2004.
- Julius, Kraevoy, Sheffer. *D-Charts: Quasi-Developable Mesh Segmentation.* Eurographics / CGF 2005.
- Sander, Wood, Gortler, Snyder, Hoppe. *Multi-Chart Geometry Images.* SGP 2003.
- Zhou, Synder, Guo, Hoppe. *Iso-charts: Stretch-driven Mesh Parameterization using Spectral Analysis.* SGP 2004. (Microsoft **UVAtlas**.)
- Lévy, Petitjean, Ray, Maillot. *Least Squares Conformal Maps for Automatic Texture Atlas Generation.* SIGGRAPH 2002.
- Sheffer, Hart. *Seamster: Inconspicuous Low-Distortion Texture Seam Layout.* Vis 2002. (disk-cutting — the `cut_to_disk` basis)
- Liu, Zhang, Xu, et al. *A Local/Global Approach to Mesh Parameterization (ARAP).* SGP 2008.
- Rabinovich, Poranne, Panozzo, Sorkine-Hornung. *Scalable Locally Injective Mappings (SLIM).* TOG 2017.
- Sawhney, Crane. *Boundary First Flattening (BFF).* TOG 2017. (fast flattening alternative)
- Li, Kaufman et al. *OptCuts: Joint Optimization of Surface Cuts and Parameterization.* SIGGRAPH Asia 2018.
- Sharp, Crane. *Variational Surface Cutting.* SIGGRAPH 2018.
- Wang, Liu, et al. *PartUV: Part-Based UV Unwrapping of 3D Meshes.* SIGGRAPH Asia 2025. arXiv:2511.16659.
- **xatlas** (jpcy) — practical chart-segmentation/packing reference (fork of thekla_atlas, "The Witness").
