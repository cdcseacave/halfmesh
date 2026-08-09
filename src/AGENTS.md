# AGENTS.md — src (implementation)

Implementation TUs for the public API. Namespace `halfmesh::`; file-local helpers go in
anonymous namespaces. See root AGENTS.md for conventions.

**Numerics are load-bearing.** Every formula, threshold and iteration count here is
covered by golden fixtures and quality bounds, so BY DEFAULT preserve them and don't
casually "improve" numerics. A deliberate change needs the same level of evidence the
current value has: golden fixtures regenerated with a reviewed delta report, plus the
quality/invariant bounds in `tests/` (see `tests/AGENTS.md`).

## TU map
The `Mesh` class is split across several TUs by concern:
- `HalfMesh.cpp` — half-edge build/connectivity, adjacency, `ConnectBorders`,
  `EnumerateHoles`, `ConnectedComponents`, edge collapse (`ERemove`) + validity predicates.
- `Mesh.cpp` — `Mesh` core: normals, area, AABB, vertex/face adjacency, `ListHalfEdges`,
  edit primitives (RemoveFaces/RemoveUnreferencedVertices…).
- `MeshIO.cpp` — PLY/glTF load/save, texture & texcoord handling, seam export.
- `MeshRepair.cpp` — RemoveDuplicate/Degenerate faces, FixNonManifold, RemoveSmallComponents.
- `MeshSimplify.cpp` — `Mesh::Simplify`: QEM edge-collapse decimation (priority/exact mode
  and the fast threshold-sweep mode; setup is parallelized); the file has explanatory comments
  on the QEM math, the boundary discontinuity quadric, and the swap-with-last compaction.
- `MeshRemesh.cpp` — `RemeshIsotropic` (flip/collapse/relocate/refine + fairing); parallel
  tangential smoothing, surface queries run on the BVH. Shared helpers in `MeshRemeshShared.{h,cpp}`.
- `MeshHoles.cpp` — `CloseHoles` via Liepa min-weight triangulation (O(n^3) DP) + refine +
  fairing; holes are filled in parallel. **Keep** the Liepa / pmp-library algorithm
  attribution comment block at the top of the file.
- `MeshSmooth.cpp` — `Mesh::SmoothHCLaplacian`: HC (anti-shrink) Laplacian smoothing
  (Vollmer'99). `Mesh::SmoothTaubin`: Taubin'95 lambda|mu band-pass — smooths
  aggressively at ~zero shrinkage (needs more iterations).
  `Mesh::Smooth(iterations, method)`: unified dispatcher over both smoothers
  (Taubin default), each run with its per-function default parameters.
- `Parametrize.cpp` — UV flattening: per chart (LSCM/Tutte init → SLIM/ARAP),
  parallelized in dependency-free waves.
- `AtlasCharting.cpp` — developable D-Charts chart segmentation (`SegmentCharts`) —
  cone-Lloyd relaxation + developable merge + flatten-and-bisect flip repair.
- `AtlasPacking.cpp` — uniform texel-density normalization + skyline (min-waste)
  first-fit multi-page atlas packing.
- `TriangleKDTree.cpp` — median-split KD-tree (build + nearest-point/ray queries with
  branch-and-bound pruning); commented.
- `TriangleBVH.cpp` — binned-SAH BVH build with a bounded traversal stack; nearest-point/ray
  queries take an optional `max_dist` bound and `NearestPoint` a `hint_face` warm start.
- `ParallelFor.h` — internal shared parallel-for pool helper (`ParallelForPool`) used by the
  parallel phases above.
- `ChartFlattenCache.h` — internal flip-repair→`ParametrizeCharts` flatten-artifact
  bridge (cache keyed by chart face-identity); consumed inside `GenerateAtlas` /
  `Parametrize` only.
- `Version.cpp` — version strings.
- `TinyGLTFImpl.cpp` — the single TU that `#define`s `TINYGLTF_IMPLEMENTATION`.

## Conventions specific to src/
- Include own headers as `#include <halfmesh/Xxx.h>`.
- `ASSERT(expr)` for invariants (debug-only; no comparison-variant macros).
- After edits, build + run the matching `tests/*Test.cpp`; the algorithm files also have
  golden + quality coverage (see `tests/AGENTS.md`).
