# AGENTS.md — halfmesh

Fast, compact **C++20 half-edge triangle-mesh processing library**. Standalone, MIT,
dependency-light.

## What it does
Half-edge connectivity + mesh repair, QEM decimation, isotropic remeshing,
HC-Laplacian and Taubin smoothing, Liepa hole-filling, a triangle KD-tree, PLY/glTF I/O,
and a UV pipeline (D-Charts chart
segmentation → SLIM/ARAP flattening → uniform-density + skyline-packed texture atlas).

## Standing conventions (read before editing)
- PascalCase files: `Mesh.h` / `Mesh.cpp` (headers `.h`, sources `.cpp`).
- `#pragma once` (no include-guard macros).
- Namespace `halfmesh::`; an internal utility namespace `math::` is used by the headers.
- Assertions: **`ASSERT(expr)` only** — there are deliberately *no* `ASSERT_EQ`/`_NE`/…
  comparison-variant macros (they collide with googletest). `ASSERT` is `#ifndef`-guarded
  so a host project's own `ASSERT` wins when both are compiled together.
- Formatting: the repo `.clang-format` (tabs, `ColumnLimit: 0`, opening brace on its own
  line for class/struct/function). Run `clang-format -i` before committing.
- Every source file starts with a short MIT header block (filename + copyright + MIT note).
- Topology authority: once `halfMesh` exists it is primary and `faces` is a derived snapshot.
  Native mutators keep connectivity live and invalidate/regenerate `faces`; array mutators clear
  connectivity. After editing public `faces` directly, call `InvalidateHalfMesh()` before any
  half-edge consumer. OpenMVS-style multi-stage processing uses `BeginHalfEdgePipeline()` /
  `EndHalfEdgePipeline()` to harvest `faces` once; ordinary public calls still return synced.
  Writing a new op: a half-edge-native one starts `ListHalfEdges()`, then optionally
  `InvalidateFaces()`, and ends `SyncFacesOnPublicExit()`; an array-native one starts
  `SyncFaces()` and ends `halfMesh.Clear()`; a read-only array consumer calls `SyncFaces()`
  (or `SyncFacesConst()` from a `const` context) and nothing else.
- Processing targets **untextured** meshes: half-edge mutators drop face-keyed attributes.
  A method that happens to keep them aligned is a documented bonus, never a contract —
  see the policy table in `docs/FEATURES.md`.
- Vertex representatives are written only through `HalfMesh::SetVHalfedge` (the `alwaysEven`
  choke point), never `vHalfedges[v] =`. `GuaranteeAlwaysEven()` is a full in-place rebuild:
  it is never the fix for a parity problem.

## Layout
- `include/halfmesh/` — public API headers (+ `Util/` helpers). See its AGENTS.md.
- `src/` — implementation TUs (the `Mesh` class is split across several `.cpp`). See its AGENTS.md.
- `tests/` — gtest suite + reusable test infra (corpus/metrics/golden) + perf + python crosscheck. See its AGENTS.md.
- `examples/` — five example CLIs (decimate / remesh / smooth / unwrap / texturebake). See its AGENTS.md.
- `cmake/` — `Utils.cmake` (warning flags helper) + `halfmeshConfig.cmake.in` (install/export).
- `docs/` — `FEATURES.md` (API tour + the representation/texture contracts),
  `TESTING.md` (layered testing strategy), `BENCHMARKS.md` (atlasbench harness
  + results vs xatlas/libigl/pmp/CGAL/BFF), `ATLAS_SEGMENTATION_DESIGN.md` (D-Charts
  segmenter design), `PYTHON.md` (bindings).
- `include/halfmesh/InteropOpenMVS.h` — opt-in converters to/from
  [openMVS](https://github.com/cdcseacave/openMVS)'s `MVS::Mesh`, guarded by
  `#if __has_include(<MVS/Mesh.h>)` (no hard dependency).
- This repo ships no vcpkg port of itself. A consumer (e.g. openMVS) carries its
  own `ports/halfmesh/` pinned to a released tag; that port must never patch the
  library — a fix lands as a commit here, then a tag, then a port bump there.

## Core types
- `halfmesh::Mesh` (`Mesh.h`) — geometry container: public `vertices`, `faces`,
  `faceTexcoords`, `texturesDiffuse`, …, plus a `HalfMesh halfMesh` member for
  connectivity. This is the main user-facing class (Load/Save/Simplify/Remesh/CloseHoles/…).
- `halfmesh::HalfMesh` (`HalfMesh.h`) — the compact half-edge structure (5 index arrays;
  twin = `h^1`, edge = `h/2`); O(1) adjacency iterators; `HALFMESH_TRIS=1` (triangles).

## Build & test
Set `$VCPKG_ROOT` (the toolchain auto-loads). Deps: eigen3, bshoshany-thread-pool,
opencv4 (no default features — only core/imgproc/imgcodecs are used), tinyply,
tinygltf.
```sh
cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON -DHALFMESH_BUILD_TOOLS=ON
cmake --build make -j
ctest --test-dir make --output-on-failure
```
Options: `HALFMESH_BUILD_TESTS`, `HALFMESH_BUILD_TOOLS`, `HALFMESH_BUILD_PERF`
(scaling guards, Release), `HALFMESH_BUILD_CROSSCHECKS` (libigl), `HALFMESH_SANITIZE`
(ASan+UBSan, Linux), `HALFMESH_ATLAS_DEBUG` (compile in verbose cone-Lloyd /
flip-repair / flatten-init diagnostics on stderr; off = zero cost, no runtime
gate). Opt-in perf builds (all default OFF, and each changes float results so it
breaks the bit-identical goldens): `HALFMESH_NATIVE` (`-march=native`),
`HALFMESH_ARCH=<isa>` (e.g. `x86-64-v3`), `HALFMESH_LTO` (IPO/LTO) — for the
perf / bench / tools builds only. Build dirs `make*/` are gitignored.

## Gotchas
- macOS is case-insensitive; **Linux CI is not** — keep `#include` casing exact (PascalCase).
- The `make/` build dir and `vcpkg_installed/` are gitignored.
- `.gitignore` is intentionally minimal: only build/generated artifacts are ignored;
  assume everything else is tracked.
