# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Python bindings
- Pip-installable `halfmesh` package (`pip install .` via scikit-build-core +
  pybind11, `-DHALFMESH_BUILD_PYTHON=ON`): `repair`, `smooth`, `simplify`,
  `close_holes`, `remove_small_components`, `remesh` array ops (numpy
  float32/uint32 in, new arrays out; face indices validated before mesh
  construction; GIL released around native work; `repair` runs the library's
  full auto-repair sequence including duplicate-face removal), a `Mesh`
  facade class for file-based I/O, and file-based `unwrap` (UV-atlas
  generation). See [docs/PYTHON.md](docs/PYTHON.md).
- Self-contained manylinux_2_28 wheels (CPython 3.10–3.13, x86_64) built by
  `cibuildwheel` and attached to GitHub Releases on `v*` tags.

### UV pipeline
- Atlas packing: two-tier pack pass (skyline head + shelf-row tail) removes the
  quadratic regime at 100k+ charts; fit-to-resolution shrink is now
  overflow-proportional (was a blind ×0.95 ladder) and `AtlasResult` reports
  `fitAttempts`. Measured on a 1M-face MVS mesh (padding 4, 4096²):
  13,473 s → 172 s, 161,627 → 133,665 charts.
- Atlas segmentation: post-repair merge rounds
  (`ParametrizeParams::postRepairMergeRounds`, default 2) recombine
  flip-repair fragments — fewer charts, fewer seams, less padding waste.

### Interoperability
- openMVS interop: `ConvertMesh` now transfers per-vertex colors and
  per-face normals in both directions.

## [0.1.0]

Initial release.

### Half-edge core
- `HalfMesh` — compact half-edge connectivity for manifold triangle meshes (5 index
  arrays; twin = `h^1`, edge = `h/2`) with O(1) adjacency iterators, border
  connection, hole enumeration, connected components, and the edge
  collapse/split/flip primitives with their validity predicates. Non-manifold
  input is detected (never silently corrupted); `Mesh::ListHalfEdges`
  auto-repairs it (geometry-preserving, warning logged) so every half-edge
  consumer operates on a manifold mesh.
- `Mesh` — geometry container (vertices, faces, optional vertex colors, cached face
  normals, per-face-corner texture coordinates, texture blobs, diffuse images) plus
  normals, area, AABB, vertex/face adjacency, and the editing primitives
  (`RemoveFaces`, `RemoveUnreferencedVertices`, `RemoveFacesOutside`, …).
- `IsManifold()` — O(faces) edge-manifoldness query.

### I/O
- PLY import/export (binary + ASCII) via `tinyply`, and glTF 2.0 / GLB
  import/export via `tinygltf`. `Load`/`Save` dispatch on the file extension.
- The glTF loader flattens the node hierarchy into world space, concatenates
  triangle primitives, expands per-vertex UVs into per-face-corner
  `faceTexcoords`, and decodes embedded base-color images; UVs round-trip with
  `SaveGLTF`.
- Malformed or unsupported files are reported through the `bool` API — never as an
  escaping exception or an out-of-bounds read.
- `ExportSeamEdges` — seam-edge export helper.

### Repair
- `FixNonManifold` (splits non-manifold vertices/edges), `RemoveDuplicateFaces`,
  `RemoveDegenerateFaces`, `RemoveSmallComponents`, `RemoveUnreferencedVertices`.
- `RemoveDuplicateVertices(epsilon = 0)` — weld spatially-coincident vertices
  (exact bit-match or grid-snap) and remap faces, reconnecting meshes that a format
  split at seams (e.g. glTF's unwelded per-texture sub-meshes).

### Algorithms
- `Simplify` — QEM edge-collapse decimation, by ratio, absolute target face count,
  or minimum edge length; exact priority-queue mode and a fast threshold-sweep
  mode, with parallelized setup and a sliver-cull pre-pass.
- `RemeshIsotropic` — flip / collapse / relocate / refine with fairing, optional
  adaptive sizing field, tangential smoothing, and feature-aware crease handling
  (crease vertices slide along the feature curve instead of freezing).
- `CloseHoles` — Liepa minimum-weight triangulation + refinement + fairing, holes
  filled in parallel; the patch-refinement density is bounded so very large
  natural holes stay tractable.
- `SmoothHCLaplacian` — HC (anti-shrink) Laplacian smoothing, with an optional
  per-vertex lock mask.
- `SmoothTaubin` — Taubin lambda|mu band-pass smoothing: aggressive denoising at
  ~zero shrinkage (defaults `lambda = 0.65`, `mu = -0.69`).
- `Smooth(iterations, method)` — unified entry point over both smoothers
  (`SmoothMethod::Taubin` by default). Per-vertex passes run on the worker pool
  with bit-identical outputs.

### Spatial indices
- `TriangleKdTree` — median-split KD-tree with nearest-point and ray queries
  (branch-and-bound pruning, optional distance bound).
- `TriangleBVH` — binned-SAH BVH with a bounded traversal stack; queries take an
  optional distance bound and `NearestPoint` an optional warm-start face hint.
- `OBB` — oriented bounding box, used by `RemoveFacesOutside`.

### UV pipeline
- `SegmentCharts` — developable (D-Charts) chart segmentation: adaptive geometry
  denoise precompute, cone-Lloyd relaxation, developable merge, and
  flatten-and-bisect repair that guarantees flip-free, **globally injective**
  charts (a chart whose shipped map self-overlaps is bisected at segmentation
  time; a ship-time guard rescues refinement-introduced folds with a
  distortion-bounded Tutte/LSCM fallback).
- `ParametrizeCharts` / `Parametrize` — per-chart flattening (LSCM or Tutte init →
  SLIM or ARAP local/global), parallelized per chart in dependency-free waves.
- Opt-in extensions (both default off): distortion-bounded chart split
  (`developable_max_uv_distortion`) and Seamster cut-to-disk (`cut_to_disk`).
- `NormalizeChartDensity` — uniform texel-density normalization.
- `PackAtlas` — skyline (min-waste) first-fit multi-page packing, with optional
  padding, rotation, power-of-two and square constraints; fit-to-resolution
  honors the requested page dimensions.
- `GenerateAtlas` — the segment → flatten → normalize → pack pipeline.

### Texture baking
- `BakeAtlas`, `RebakeTexture`, `DefragmentTexture`, `AutoAtlasResolution` —
  resample textures onto a new UV layout, with selectable source-to-target
  correspondence. Missing per-corner UVs/textures fail fast (default result +
  warning) in every build mode.
- `BakeParams::maxDefragPatches` — `DefragmentTexture` refuses source atlases
  beyond this patch count (default 65,536; 0 disables) instead of spinning
  unbounded in the packer; such inputs are `RebakeTexture` territory.

### Diagnostics
- `halfmesh::SetStatusLog(std::ostream*)` — redirect or silence the library's
  progress logs (default: stdout).

### Interoperability
- `InteropOpenMVS.h` — opt-in `halfmesh::Mesh` ↔ `MVS::Mesh` converters, compiled
  only when `<MVS/Mesh.h>` is on the include path; no hard dependency.

### Build / packaging
- vcpkg manifest with `tests`, `tools`, `crosschecks` and `bench` features;
  dependency versions pinned via `builtin-baseline`.
- CMake install + `find_package(halfmesh CONFIG)` export; `Version()` is
  injected from the CMake project version.
- GoogleTest suite (`-DHALFMESH_BUILD_TESTS=ON`) with committed golden fixtures,
  robustness/determinism layers, an optional libigl cross-check, and a
  deliberately-dirty 120,943-face challenge fixture (`tests/data/mesh.ply`).
- Example CLIs — `decimate`, `remesh`, `smooth`, `unwrap`, `texturebake`
  (`-DHALFMESH_BUILD_TOOLS=ON`).
- Optional harnesses: performance/scaling guards (`HALFMESH_BUILD_PERF`), the
  UV-atlas and remeshing benchmarks (`HALFMESH_BUILD_BENCH`), ASan+UBSan
  (`HALFMESH_SANITIZE`), and verbose atlas diagnostics (`HALFMESH_ATLAS_DEBUG`).

[0.2.0]: https://github.com/cdcseacave/halfmesh/releases/tag/v0.2.0
[0.1.0]: https://github.com/cdcseacave/halfmesh/releases/tag/v0.1.0
