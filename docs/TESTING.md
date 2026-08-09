# HalfMesh — Testing Strategy

How every component is validated. The guiding idea: **structural invariants and
independent libraries establish correctness; frozen known-good snapshots then keep
it from drifting.**

---

## 1. The oracle problem & four test layers

Every component is checked at up to four layers, strongest first:

1. **Structural invariants** — properties that must hold for *any* valid output,
   independent of any oracle (manifoldness, Euler characteristic, no flips, no
   NaNs, density uniformity…). These never go stale and catch the most bugs.
2. **Regression vs frozen golden snapshots** — committed known-good output fixtures
   (`tests/data/golden/`) capture correct behaviour once it is established. Tests
   recompute each op and compare to the frozen result. See §4.
3. **Independent third-party cross-checks** — libigl / CGAL / PMP /
   geometry-central / pymeshlab / Open3D / xatlas / glTF-Validator. Not bit-exact;
   confirms we're in the right ballpark and catches systematic errors. Essential for
   the UV/atlas modules, where no closed-form answer exists.
4. **Metamorphic / property / performance** — relations that must hold (round-trip
   identity, idempotence, transform-equivariance), randomized fuzzing, and
   timing/scaling regression guards (the code is "highly optimized" — protect it
   with numbers).

A component is "done" only when its layer-1 invariants pass **and** it matches the
frozen golden snapshot within tolerance (§5).

---

## 2. Test mesh corpus

Keep inputs small and committed; generate adversarial cases by corrupting clean
ones so the defect is known exactly.

**Synthetic / analytic (known answers).**
- Single triangle, quad, tetrahedron, cube, icosahedron — exact Euler/genus,
  trivial adjacency to hand-verify.
- Flat grid plane — must survive fairing/remeshing unchanged (planar) and
  parametrize with **zero** distortion.
- Open cylinder, cone — developable: unroll to a rectangle / circular sector with
  near-zero stretch (closed-form ground truth for UV).
- Sphere, torus — known genus (0, 1); curvature integrates to 4π / 0
  (Gauss–Bonnet sanity).

**Real meshes.**
- The repo fixture `tests/data/mesh.ply` — a 120,943-face scan chosen as a
  challenge input (non-manifold spots, duplicate faces, open boundaries, thin
  structures, planar and detailed regions), so every fixture-based test
  exercises a hard case, not a friendly one. (Until 2026-08 this was a
  3,786-face clean scan; measured values in older docs refer to that mesh.)
- 2–3 meshes of increasing size: one small for unit tests, one mid, one large
  (≥1M faces) for perf and scalability. Pick permissively-licensed or
  self-generated to keep the repo redistributable.

**Dirty meshes (synthesized from clean ones).**
- Non-manifold: bow-tie (vertex shared by two fans), three faces on one edge.
- Duplicate faces, degenerate/sliver faces, unreferenced vertices.
- Holes: delete N known faces from a closed mesh.
- Many small disconnected components plus one large.

A `tests/data/make_corpus` step documents/automates how each dirty mesh is
derived from its clean parent, so expected fix counts are exact.

---

## 3. Shared metrics toolkit (test it first)

A `tests/metrics/` helper library, itself unit-tested against analytic cases,
used by all suites:

- **Topology:** V/E/F counts, Euler χ, genus, boundary-loop count, manifold checks
  (edge- and vertex-manifold), watertightness, valence histogram.
- **Geometry:** surface area, AABB, volume (closed), per-triangle quality
  (min angle, aspect ratio, radius ratio), edge-length distribution.
- **Distance:** symmetric **Hausdorff** and mean surface distance between two
  meshes — implemented via `TriangleKdTree`, and **cross-checked against an O(n·m)
  brute-force** on small meshes so the metric itself is trusted.
- **UV:** per-triangle signed area (→ **flip count**), isometric / conformal /
  area distortion, symmetric-Dirichlet energy, **stretch L2** (Sander), texel
  density per chart, atlas occupancy, cross-chart overlap.
- **Robustness:** NaN/Inf scan over all buffers (fail hard on any).

Canonicalization helper: compare meshes up to vertex/face **relabeling** and
per-face cyclic rotation (same orientation), so regression tests don't trip over
ordering differences.

---

## 4. Differential regression: frozen golden snapshots

For each `(input mesh × op × parameters)` spec in `tests/golden/GoldenSpecs.h`,
two files are committed to `tests/data/golden/`:

| file | content |
|------|---------|
| `<mesh>__<op>.ply`  | frozen result mesh — binary PLY, lossless |
| `<mesh>__<op>.json` | scalar metrics: counts, op count, area, bbox, Hausdorff-to-input |

`golden_diff_test` (default build) recomputes the op on the deterministic input
and compares to the frozen fixture within the §5 tolerance for that op's
`CompareMode`.

**Regeneration policy.** The fixtures follow the algorithm, not the reverse:
behaviour changes from genuine algorithmic improvements are **expected and
sanctioned** — a fixture need not stay bit-identical to superseded code. Every
regeneration is still justified, reviewed, and recorded. The opt-in regen run
(`GOLDEN_REGEN=<substring|all>`) prints a **delta report** that quantifies what
changed (one `[golden-regen]` line per metric, with a `QUALITY REGRESSION` flag
if Hausdorff-to-input rose > 10%); paste that delta table into the commit message
alongside the regenerated `.ply`+`.json`, together with the code change that
caused it, and log it in `tests/data/golden/README.md`. What is **never**
relaxed: the determinism invariants (run-twice identical, serial == parallel at
any thread count) — those are product guarantees, not fixture locks.

```sh
cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON
cmake --build make -j --target golden_diff_test
# regenerate the matching fixture(s), REVIEW the printed [golden-regen] delta,
# then commit the .ply+.json with the delta table in the message:
GOLDEN_REGEN=RemeshIsotropic ./make/tests/golden_diff_test --gtest_filter='*Regenerate*'
git add tests/data/golden
```

---

## 5. Tolerance policy

- **Topology / integers** (counts, manifold flags, fix counts, flip counts): exact.
- **Float geometry, op order unchanged:** attempt bit-exact vs the frozen snapshot;
  fall back to relative ε ≈ 1e-5 (float positions) if the refactor reorders sums.
- **Accumulated quantities** (areas, energies, Hausdorff): looser relative ε
  (≈1e-4) — document per assertion.
- **Isotropic-remesh edge length** (the isotropy ratchet): `RemeshIsotropic`'s
  `edge_len_mean` within ±10% and `edge_len_min`/`edge_len_max` within ±25% of
  the frozen baseline — baseline-relative, named constants. Uniform edge length
  is `RemeshIsotropic`'s contract and the Hausdorff net cannot see isotropy loss.
- **Independent third-party checks:** qualitative / ranged (e.g., "our QEM
  Hausdorff ≤ 1.2× libigl qslim at equal face count"), never exact.
- **Lossy I/O** (JPEG textures): use PNG for exact pixel round-trips; allow PSNR
  threshold for JPEG.

Every tolerance is named and justified in the test; no silent magic numbers.

---

## 6. Per-component test matrix

### Half-edge core (`HalfMesh`: build, adjacency, twin/next/prev, boundary, collapse/flip/remove)
- **Independent oracle (no golden needed):** build a brute-force adjacency from
  the face list; assert `VAdjacentVertices/Faces/Edges`, `FAdjacentFaces`,
  boundary flags, and degrees match exactly for every element.
- **Invariants / `Validate()`:** write a structural validator (twin pairing
  `he^1`, `he_nexts` cycle consistency, `v_halfedges` even & boundary-pointing,
  `he_faces` NO_ID only on boundary) and run it **after every mutation** in tests.
- **Round-trip:** `Build → FFaces` reproduces input faces (up to cyclic rotation).
- **Euler / boundary:** `V−E+F` consistent with genus & boundary-loop count.
- **Mutations:** collapse/flip/remove on hand-built fans, boundaries, and the
  cube; assert exact resulting connectivity + invariants; check `EIsFlipValid` /
  `EIsCollapseValidTopologically` against the link-condition definition.
- **3rd-party:** compare adjacency/boundary to libigl
  (`igl::triangle_triangle_adjacency`) on the corpus.
- **Differential + perf:** vs golden snapshot; build-time scaling on the large mesh.

### Topology repair (FixNonManifold, RemoveDuplicate/Degenerate/Unreferenced, RemoveSmallComponents)
- **Adversarial exact counts:** the synthesized dirty meshes have known defects →
  assert exact number of fixes/removals.
- **Invariants after repair:** result is edge+vertex manifold (FixNonManifold), no
  duplicate faces, no faces below area threshold, no unreferenced vertices,
  correct surviving component count.
- **Idempotence:** running the op again changes nothing (zero further fixes).
- **Conservation:** surface area within ε (degenerate removal ≈ 0 area); bbox
  preserved.
- **3rd-party:** confirm manifoldness with CGAL `is_valid` / libigl manifold
  checks; component count vs `igl::facet_components` / Open3D.
- **Regression:** vs golden snapshot (deterministic → tight tolerance).

### QEM decimation (Simplify)
- **Determinism:** same input+params → identical output across runs.
- **Regression:** vs golden at fixed `decimate_ratio` / `min_edge_length` /
  `aggressiveness` — exact canonical mesh equality, Hausdorff ≈ 0.
- **Quality invariants:** reaches target face count (within tolerance); symmetric
  Hausdorff to the *input* mesh below a bound; no non-manifold/degenerate
  introduced; preserved boundaries; normals not flipped en masse.
- **Monotonicity (metamorphic):** smaller ratio ⇒ fewer faces and larger Hausdorff
  (trend holds across a sweep).
- **3rd-party ballpark:** error vs libigl `qslim` / CGAL / pymeshlab quadric
  decimation at equal face count — same order of magnitude.

### Isotropic remeshing (RemeshIsotropic) + fairing (VertexCoordLaplacian)
- **Regression:** vs golden snapshot (deterministic given iteration order).
- **Quality invariants:** edge-length histogram concentrated in
  [4/5, 4/3]·target; valence histogram peaked at 6 interior / 4 boundary;
  triangle quality (min angle, aspect ratio) improved vs input.
- **Surface fidelity:** every output vertex within `max_surf_dist` of the original
  surface (KD-tree distance) when `check_surf_dist`; symmetric Hausdorff bounded.
- **Crease preservation:** edges tagged by `th_crease_cos_angle` survive.
- **Fairing analytic test:** a noisy *planar* patch with cotangent weights returns
  to the plane (residual → 0); a smooth region's volume/area barely changes;
  energy/roughness metric decreases monotonically per iteration.

### Hole filling (CloseHoles)
- **Synthetic watertight test:** delete N known faces from a closed mesh, fill,
  assert `EnumerateHoles == 0`, face count grows as expected, no new non-manifold,
  patch normals consistent with the neighbourhood.
- **Boundary-length cap:** holes larger than `nCloseHoles` are left open (exact
  behaviour).
- **Regression:** vs golden snapshot.

### Triangle KD-tree (NearestPoint, IntersectedPoint)
- **Perfect independent oracle:** brute-force over all triangles. For thousands of
  random query points and rays, KD-tree nearest distance/point and ray hit must
  **equal** brute-force (exact, no golden needed).
- **Edge cases:** query exactly on a vertex/edge; ray grazing/parallel/missing;
  empty regions; degenerate triangles.
- **Perf:** build + query time and scaling vs brute force (must be the expected
  speedup) — guards the "optimized" promise.

### I/O (PLY/tinyply, glTF/tinygltf, OpenCV textures)
- **Round-trip identity:** `Load(Save(M)) == M` for vertices/faces/colors/normals/
  texcoords/texblobs within tolerance; both binary and ASCII; PNG textures exact,
  JPEG by PSNR.
- **Cross-reader:** parse the saved files with independent tools — PLY via
  trimesh/pymeshlab/Open3D, glTF via the **Khronos glTF-Validator** + pygltflib/
  Blender — and verify geometry & UVs match.
- **Regression:** compare against golden snapshots on the same mesh.
- **Texture path:** textured mesh (multiple texblobs) round-trips; per-pixel image
  compare after save/load.

### UV parametrization (segmentation + SLIM/ARAP)
No golden oracle → invariants + analytic + third-party.
- **No flips (the key test):** every parameterized triangle has positive signed
  area in a consistent orientation — **zero flipped triangles** (SLIM guarantee).
  Fail on any flip.
- **Analytic ground truth:** plane → isometric (distortion ≈ 0); cut cylinder →
  rectangle; cone → sector (compare to closed-form stretch ≈ 0). These are tight
  unit tests with known answers.
- **Distortion thresholds:** symmetric-Dirichlet energy, stretch L2, area
  distortion below per-mesh bounds on the corpus; report distributions.
- **Convergence:** SLIM energy decreases monotonically and converges within the
  iteration cap; Tutte initialization is injective (convex boundary).
- **Chart validity:** each chart is a topological disk; UV boundary is a simple
  (non-self-intersecting) polygon.
- **Segmentation:** few charts, near-developable (per-chart proxy-fit error
  bounded), boundaries snap to creases/material seams; deterministic seeding ⇒
  reproducible chart count.
- **3rd-party reference:** compare distortion to **libigl** SLIM/ARAP and
  **geometry-central** (BFF) on the same mesh — comparable magnitude, not identical.

### Atlas generation (density normalization + packing)
- **Uniform density invariant:** texels-per-world-unit variance across charts ≈ 0
  (the whole point of the module). Strong assertion.
- **No cross-chart overlap:** rasterize the atlas and assert no two charts touch
  the same texel beyond the padding band (plus analytic rectangle-overlap check).
  Critical — overlaps mean corrupted textures.
- **Packing:** occupancy ratio above threshold; padding/gutter respected;
  power-of-two/square honoured; multi-atlas overflow correct when charts exceed
  one page. Compare occupancy to **xatlas** on the same charts.
- **End-to-end bake:** bake a known checker/gradient into the atlas, sample across
  the surface, verify continuity (no seams beyond expected, correct density). Round
  the result through `Save`/`SaveGLTF` and re-open in a viewer/validator.
- **Determinism:** identical atlas across runs.

---

## 7. Cross-cutting practices

- **Sanitizers & robustness:** run the suite under ASan/UBSan (Job 2 in
  ci.yml, ubuntu); assert no NaN/Inf in any output buffer.
- **Determinism:** every op run twice must give identical output (guards against
  uninitialized memory, unordered containers leaking into results).
- **Property-based / fuzz:** random meshes and random *sequences* of operations
  (repair → decimate → remesh → fill → unwrap); after each, structural invariants
  must still hold. Tiny vertex perturbations must not crash or break manifold
  handling.
- **Metamorphic relations:** load→save→load identity; repair idempotence; rigid
  transform commutes with decimation/remeshing up to the same rigid transform;
  uniform scale leaves topology and UV distortion invariant.
- **Performance regression:** record op timings on the mid/large meshes; CI flags
  regressions beyond a margin vs a committed baseline.
- **CI matrix:** Linux/macOS/Windows, built via vcpkg; GoogleTest; the Python
  third-party cross-checks as a separate (looser, non-blocking-optional) job;
  clang-format + a static-analysis pass.
- **High-stakes verification:** for the trickiest ops (FixNonManifold,
  RemeshIsotropic, Simplify) diff the result against the golden fixture over the
  whole corpus in a dedicated job before declaring the module done.

---

## 8. Tooling summary

- **Framework:** GoogleTest (via vcpkg), CTest.
- **Regression snapshots:** frozen golden fixtures in `tests/data/golden/`
  (`golden_diff_test`).
- **C++ references:** libigl, CGAL, PMP, geometry-central (compare-only, not
  shipped deps).
- **Python cross-checks (CI job):** pymeshlab, trimesh, Open3D, libigl-python,
  pygltflib, Khronos **glTF-Validator**.
- **Atlas/UV references:** xatlas, Microsoft UVAtlas (stretch/occupancy comparison).
