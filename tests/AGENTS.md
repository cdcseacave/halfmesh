# AGENTS.md — tests

GoogleTest suite, discovered via `gtest_discover_tests` (PRE_TEST mode).
Build with `-DHALFMESH_BUILD_TESTS=ON`, run with `ctest --test-dir make`. Full strategy:
`docs/TESTING.md`. See root AGENTS.md for conventions.

## Test files (PascalCase `*Test.cpp`)
Per-component unit tests live at the top level: `HalfMeshTest`, `MeshCoreTest`,
`MeshIOTest`, `MeshRepairTest`, `MeshSimplifyTest`, `MeshRemeshTest`, `MeshSmoothTest`, `MeshHolesTest`,
`TriangleKDTreeTest`, `ParametrizeTest`, `FlattenTest`, `AtlasTest`, `OrientedBoundingBoxTest`,
`PriorityQueueTest`, `AccumulatorTest`, `GeometryTest`, `UtilTest`, plus
`HalfMeshInvariantsTest` (durable half-edge invariants + edge-collapse).

## Reusable infrastructure (static libs)
- `tests/metrics/` → `halfmesh_metrics` (`hmtest::metrics`): `ComputeTopology` (V/E/F, χ,
  genus, watertight, manifold, boundary loops), `ComputeDistanceKdTree` (symmetric Hausdorff),
  `ComputeAllTriangleQualities`, `ComputeEdgeLengthStats`, `ScanFinite`, `CanonicallyEqual`.
  Header-only `RemeshQuality.h` distils these into the remeshing-quality scalars (edge-length
  CoV, mean min-angle, sliver fraction, mean/irregular valence, surface fidelity) shared by the
  remesh unit tests and `remeshbench`.
- `tests/corpus/` → `halfmesh_corpus` (`hmtest::corpus`): analytic generators with `*_Known()`
  topology (Tetrahedron, Cube, Icosahedron, UVSphere, **TorusMesh** (genus 1), GridPlane,
  OpenCylinder, Cone, `LargeMesh(N)`) + dirty synthesizers.
- `tests/golden/` → frozen regression fixtures + `GoldenDiffTest`. See its AGENTS.md.

## Test layers
1. **Unit / per-component** — the `*Test.cpp` files.
2. **Invariants & accuracy** — `HalfMeshInvariantsTest` (twin/next/orbit, Euler, adjacency
   vs brute force) + `MeshSimplifyTest` accuracy (both modes, watertight/genus preserved,
   no slivers, Hausdorff bound, determinism). Use `halfmesh_metrics`/`halfmesh_corpus`.
3. **Golden regression** — `tests/golden/` (current output vs committed frozen fixtures).
4. **Cross-checks** — `tests/crosscheck/` (libigl; `-DHALFMESH_BUILD_CROSSCHECKS=ON`) and
   `tests/python/` (trimesh + Khronos glTF-Validator; optional/dev, skip-if-absent).
5. **Robustness** — `tests/robustness/` (determinism / property / metamorphic, fixed seeds).
6. **Perf** — `tests/perf/PerfHarness.cpp` (`-DHALFMESH_BUILD_PERF=ON`, Release): machine-
   independent **scaling** assertions for `HalfMesh::Build` and `Simplify` (sub-quadratic)
   and a KD-tree ≥10× speedup; absolute times only as an advisory regression alarm. Run the
   binary directly (not via ctest) so cross-test timing state is shared.
7. **Remesh benchmark** — `tests/bench/RemeshBench.cpp` → `remeshbench`
   (`-DHALFMESH_BUILD_BENCH=ON`, Release): sweeps `RemeshIsotropic` over the corpus at several
   target edge lengths, reporting the `RemeshQuality` scalars + wall-time as a table and
   `tests/bench/results/remesh_baseline.json` (gitignored; timing is machine-dependent). An
   opt-in baseline runs on identical inputs/params for a direct quality+time comparison:
   `-DHALFMESH_REMESH_WITH_PMP=ON` (pmp::uniform_remeshing, links libpmp built under
   `$HOME/Pro/pmp-library/build`). Registered with ctest as `remeshbench_smoke` (label
   "bench"); non-zero exit on an empty mesh or non-finite metric.

## Test data
`tests/data/mesh.ply` — the committed challenge fixture (63,049 V / 120,943 F
scan, deliberately dirty: 138 duplicate faces, 235 non-manifold edges, 4,750
border edges over 29 boundary loops incl. one 2,739-edge outer loop, 243
needle faces, thin structures, planar and detailed regions). Every
mesh.ply-dependent test skips if absent, but it is present. Half-edge
consumers manifoldize it on entry (documented in `Mesh::ListHalfEdges`), so
count pins compare against the manifoldized baseline. `tests/data/golden/` —
frozen golden fixtures.
Optional large real meshes can be dropped in `tests/data/` for the perf real-mesh timing
and python real-world-GLB crosscheck (skip when absent; never committed).

## Adding a test
Add `tests/FooTest.cpp`, register it in `tests/CMakeLists.txt` (mirror an existing
`add_executable` + `halfmesh_discover_tests`; link `halfmesh_corpus`/`halfmesh_metrics` if
you use them).
