# halfmesh — Feature Guide

A tour of every feature the library implements: what it does, the exact entry
points, the knobs that matter, and where to read further. All types and
functions live in `namespace halfmesh`; headers are under
[`include/halfmesh/`](../include/halfmesh). The example CLIs under
[`examples/`](../examples) demonstrate the canonical call sequence for each
major feature.

The full pipeline below — repair, smooth, simplify, close holes, remesh, and
UV-atlas generation — is also available from Python via the pip-installable
`halfmesh` package; see [`docs/PYTHON.md`](PYTHON.md).

Quick index:

| Feature | Entry point | Header | Example |
|---|---|---|---|
| [Half-edge core](#half-edge-core) | `HalfMesh` | `HalfMesh.h` | — |
| [Mesh container](#mesh-container) | `Mesh` | `Mesh.h` | all |
| [PLY / glTF I/O](#ply--gltf-io) | `Mesh::Load` / `Mesh::Save` | `Mesh.h` | all |
| [Repair & cleaning](#repair--cleaning) | `Mesh::RemoveDuplicateVertices`, `FixNonManifold`, … | `Mesh.h` | `Decimate.cpp` preamble |
| [QEM decimation](#qem-decimation) | `Mesh::Simplify` | `Mesh.h` | `Decimate.cpp` |
| [Isotropic remeshing](#isotropic-remeshing) | `Mesh::RemeshIsotropic` | `Mesh.h` | `Remesh.cpp` |
| [Smoothing](#smoothing) | `Mesh::Smooth`, `SmoothTaubin`, `SmoothHCLaplacian` | `Mesh.h` | `Smooth.cpp` |
| [Hole filling](#hole-filling) | `Mesh::CloseHoles` | `Mesh.h` | — |
| [Spatial indices](#spatial-indices) | `TriangleBVH`, `TriangleKdTree` | `TriangleBVH.h`, `TriangleKDTree.h` | `TextureBakeTool.cpp` |
| [UV parametrization](#uv-parametrization) | `Parametrize`, `SegmentCharts`, `ParametrizeCharts` | `Parametrize.h` | `Unwrap.cpp` |
| [Atlas packing](#atlas-packing) | `GenerateAtlas`, `PackAtlas`, `NormalizeChartDensity` | `AtlasCharting.h`, `AtlasPacking.h` | `Unwrap.cpp` |
| [Texture bake / rebake / defrag](#texture-bake--rebake--defrag) | `BakeAtlas`, `RebakeTexture`, `DefragmentTexture` | `TextureBake.h` | `TextureBakeTool.cpp` |
| [openMVS interop](#openmvs-interop) | `ConvertMesh` | `InteropOpenMVS.h` | — |
| [Utilities](#utilities) | `OBB`, raster/sampler/geometry helpers | `OrientedBoundingBox.h`, `Util/` | — |

See also: [Threading & determinism](#threading--determinism) and
[Performance](#performance) at the end.

---

## Half-edge core

`class HalfMesh` — manifold triangle-mesh connectivity in a compact five-array
SoA layout (`vHalfedges`, `fHalfedges`, `heNexts`, `heVertices`, `heFaces`).
The twin of half-edge `h` is `h ^ 1` and its edge is `h / 2`, so twin/edge
lookups cost no memory; a boundary is `heFaces[h|1] == NO_ID`.

- **Build**: `Build(const Mesh&)` or `Build(numVertices, faces)`. Rejects
  non-manifold *edges and vertices* (bow-ties included) by returning `false` —
  use `Mesh::ListHalfEdgesSafe()` to auto-repair first.
- **O(1) navigation**: `HeTwin/HeNext/HePrev`, `HeVertex/HeHeadVertex/HeTailVertex`,
  `HeFace`, `VHalfedge`, `FHalfedge`, `HeEdge`/`EHalfedge`, plus boundary
  predicates (`HeIsBoundary`, `EIsBoundary`, `VIsBoundary`).
- **Range-for circulators**: `VIncomingHalfedges`, `VOutgoingHalfedges`,
  `VAdjacentVertices`, `VAdjacentFaces`, `VAdjacentEdges`,
  `VBoundaryLoopVertices`, `FAdjacentHalfedges/Vertices/Faces/Edges`, and the
  edge-centric variants.
- **Euler operators** (triangle build, `HALFMESH_TRIS=1`): validated edge
  collapse (`EIsCollapseValidTopologically` — Hoppe'93 link condition —
  `EIsCollapseValidGeometrically`, `ERemove`), edge flip (`EIsFlipValid`,
  `EFlip`), and in-place edge split (`ESplit`).
- **Holes and components**: `EnumerateHoles`, `TriangulateHole`,
  `ConnectedComponents` (optionally with a custom edge validator),
  `ConnectBorders`.

Header: [`HalfMesh.h`](../include/halfmesh/HalfMesh.h) ·
implementation: `src/HalfMesh.cpp`.

## Mesh container

`class Mesh` — a plain aggregate; every member is public and directly
readable/writable:

```cpp
std::vector<Vertex>   vertices;       // Eigen::Vector3f
std::vector<Face>     faces;          // Eigen::Matrix<uint32_t,3,1>
std::vector<Pixel>    vertexColors;   // BGR uint8, per vertex (optional)
std::vector<Normal>   faceNormals;    // per-face cache (optional)
std::vector<TexCoord> faceTexcoords;  // per corner (faces*3) or per vertex
std::vector<FIndex>   faceTexblobs;   // per-face texture id (optional)
std::vector<Image3u>  texturesDiffuse;// one cv::Mat_<Pixel> per texture blob
HalfMesh halfMesh;                    // optional connectivity (built on demand)
```

Vertices, faces and UVs are contiguous and memcpy-compatible
(`static_assert`ed in `src/MeshIO.cpp`) — bulk import/export from external
buffers is a single copy in each direction. Geometry helpers include
`ComputeFaceNormals`, `ComputeSmoothFaceNormals`, `ComputeVertexNormals`,
`ComputeArea`, `ComputeAABBox`, and per-face/per-edge queries. Texture-layout
conversions: `ToTexCoordPerVertex()`, `ToTexCoordPerVertexUVOnly()`,
`ToOneMeshPerTexblob()`.

`faces` is a derived topology snapshot once `halfMesh` has been built. Call
`Mesh::InvalidateHalfMesh()` after editing `faces` directly; half-edge
consumers deliberately trust a non-empty `halfMesh` and cannot detect direct
array edits.

Header: [`Mesh.h`](../include/halfmesh/Mesh.h).

## PLY / glTF I/O

`Mesh::Load(path)` / `Mesh::Save(path, binary = true)` dispatch on the file
extension: `.glb`/`.gltf` → glTF 2.0 (via tinygltf), everything else → PLY
(via tinyply). Both binary and ASCII variants are supported for each format.

- **PLY** reads positions (any scalar type, narrowed with a warning), vertex
  colors, per-face texture coordinates (`face/texcoord`, 6 floats) and texture
  ids (`face/texnumber`), plus sidecar textures declared by
  `comment TextureFile <name>` headers (decoded in parallel). On save,
  textures are written as `<stem>_diffuseNN.jpg` and `texnumber` is written as
  INT32 for MeshLab compatibility.
- **glTF** flattens the node hierarchy into world space on load (`POSITION`,
  `TEXCOORD_0`, `COLOR_0`, base-color textures); on save it writes one
  primitive per texture blob with embedded JPEG textures, `KHR_materials_unlit`
  and a z-up → y-up root rotation.
- **UV conventions** (worth reading twice): in-memory textured meshes store
  *absolute pixel* UVs with no Y flip; PLY on disk stores normalized+Y-flipped;
  glTF stores `(pixel + 0.5)/size`. The `FTexcoords{Normalize,UnNormalize}[FlipY]`
  helpers convert, and `UVBlobsAreNormalized()` (in `TextureBake.h`) classifies
  a loaded mesh at runtime.
- glTF (and some PLY) input arrives *unwelded* — one vertex per corner. Run
  the standard cleanup preamble (`RemoveDuplicateVertices(0)` →
  `RemoveDegenerateFaces(0.f)` → `RemoveUnreferencedVertices()`) before any
  half-edge algorithm, or every edge counts as a boundary.

Implementation: `src/MeshIO.cpp`.

## Repair & cleaning

Everything needed to turn scanner/photogrammetry output into a mesh the
half-edge core accepts:

- `RemoveDuplicateVertices(epsilon)` — weld coincident vertices; `0` = exact
  bit match, `>0` = grid snap.
- `RemoveDegenerateFaces(thArea)` (+ iterated overload) — drop faces with
  repeated or near-coincident vertices.
- `RemoveDuplicateFaces()`, `RemoveUnreferencedVertices()`, `RemoveVertices()`,
  `RemoveFaces()`, `RemoveFacesOutside(const OBB&)`.
- `FixNonManifold(thMoveDuplicate)` — finds non-manifold edges/vertices and
  restores manifoldness by duplicating the offending vertices (optionally
  displaced toward the incident-face barycenter).
- `RemoveSmallComponents(minComponentSize)` — floater removal.
- `ListHalfEdgesSafe()` — the whole pipeline in one call, followed by the
  half-edge build.

Note that every half-edge consumer (`Simplify`, `RemeshIsotropic`,
`CloseHoles`, `SegmentCharts`, the smoothers) builds connectivity via
`ListHalfEdges()`, which detects non-manifold input and auto-repairs it with a
logged warning — a geometry-preserving *mutation* (weld, split bow-ties, drop
self-edge faces, prune unreferenced vertices). Callers holding external
per-vertex/per-face arrays must re-map them afterwards.

Implementation: `src/MeshRepair.cpp`.

## QEM decimation

```cpp
void Mesh::Simplify(float decimateRatio, float minEdgeLength = 0.f, float aggressiveness = 0.f);
```

Garland–Heckbert quadric-error-metric edge collapse with boundary/silhouette
preservation (discontinuity quadrics at ×3 weight).

- `decimateRatio` is dual-purpose by magnitude: `(0,1)` = keep that fraction
  of faces; `>1` = absolute target face count (clamped to the input); exactly
  `1` = identity.
- `minEdgeLength` collapses all edges below a length instead (exclusive with
  the ratio).
- `aggressiveness = 0` runs the **exact** variant (global mutable priority
  queue — best quality); `>0` (recommended 5–8) runs the **fast threshold
  sweep**, multiplying the error threshold by this factor per pass — the right
  choice for very large reductions.
- Topology is preserved: collapses that would break manifoldness are skipped,
  so adversarial input has a reachable floor above the target (a warning is
  logged). The header documents the repair pre-pass that dissolves the
  phantom triangles causing this.
- Destroys optional attributes (`ReleaseOptional()`): decimation is
  geometry-only; regenerate UVs afterwards with the atlas pipeline.

Supporting machinery: `TQuadric` ([`Quadric.h`](../include/halfmesh/Quadric.h))
and the indexed mutable heap `TPriorityQueue`
([`PriorityQueue.h`](../include/halfmesh/PriorityQueue.h)).
Implementation: `src/MeshSimplify.cpp` · example: `examples/Decimate.cpp`.

## Isotropic remeshing

```cpp
struct Mesh::RemeshParams { /* 18 fields, see Mesh.h */ };
void Mesh::RemeshIsotropic(RemeshParams params, RemeshStats* stats = nullptr);
```

Botsch–Kobbelt split / collapse / flip / tangential-smooth / project loop that
regularizes edge lengths and triangle aspect ratios. Key knobs:

- `SetEdgeLength(L)` — sets the min/max band to `L·4/5 … L·4/3`.
- `SetAdaptive(error, minMult, maxMult)` — curvature-adaptive target sizing.
- `SetCreaseAngle(degrees)` — dihedral threshold for feature/crease tagging;
  `featureCorners` pins corner vertices.
- `SetMaxSurfaceDistance(...)` / `checkSurfDist` — bounds drift from the input
  surface; projection runs against a `TriangleBVH` of the original geometry.
- Per-pass toggles (`doSplit/doCollapse/doFlip/doSmooth/doProject`),
  `iterations`, and tangential-smoothing controls.
- `RemeshStats` returns per-phase op counts and timings.

Changes vertex/face counts; UVs are not preserved (rebake afterwards).
Implementation: `src/MeshRemesh.cpp` · example: `examples/Remesh.cpp`.

## Smoothing

```cpp
void Mesh::Smooth(int iterations = 1, SmoothMethod method = SmoothMethod::Taubin); // dispatcher
void Mesh::SmoothHCLaplacian(int iterations = 1, const std::vector<bool>* lockedVertices = NULL);
void Mesh::SmoothTaubin(int iterations = 1, Type lambda = 0.65f, Type mu = -0.69f,
                        const std::vector<bool>* lockedVertices = NULL);
```

Two complementary methods, both attribute-preserving (positions move,
topology/UVs/colors stay; the `faceNormals` cache is invalidated):

- **HC Laplacian** (Vollmer/Mencl/Müller 1999) — a uniform-Laplacian step plus
  a correction pass pushing vertices back toward their prior position; largely
  shrink-free at low iteration counts. Good default for gentle cleanup.
- **Taubin λ|μ** (Taubin 1995) — alternating shrink (λ>0) / inflate (μ<−λ)
  steps forming a band-pass filter: aggressive high-frequency noise removal at
  ~zero volume change. Wants more iterations (typically 10–100); the defaults
  are the strongest stable pair under `|(1−2λ)(1−2μ)| ≤ 1`. Border vertices
  are smoothed along the boundary curve only.
- `lockedVertices` pins selected vertices (they still contribute to their
  neighbors' averages); size must match the post-repair vertex count.

Implementation: `src/MeshSmooth.cpp` · example: `examples/Smooth.cpp` (prints
bbox-diagonal and mean-radius shrinkage ratios so the methods can be compared
at a glance).

## Hole filling

```cpp
unsigned Mesh::CloseHoles(unsigned nCloseHoles = 200,
                          std::vector<std::vector<FIndex>>* holesFaces = NULL);
```

The Liepa 2003 pipeline (structure follows the pmp-library implementation):

1. **Triangulate** — minimum-weight dynamic-programming triangulation over the
   boundary loop, minimizing (max dihedral angle, area) lexicographically.
2. **Refine** — a local isotropic remesh of the patch (split/collapse/flip/
   relax, cap removal) so fill triangles match the surrounding density.
3. **Fair** — bi-Laplacian (k=2) cotangent fairing solved with Eigen
   `SimplicialLDLT`, falling back to a k=1 graph Laplacian.

Holes are processed **smallest-first by boundary vertex count**; `nCloseHoles`
caps *how many* are filled (there is no size threshold — leave the cap below
the hole count to skip the largest loops, e.g. a scan's open outer boundary).
An internal patch budget (`max(16384, 8·n)` triangles) keeps degenerate giant
boundaries from exploding the fill. Returns the number of holes closed;
optionally reports the new faces per hole. Lower-level building blocks are
also public: `HalfMesh::EnumerateHoles()` + `HalfMesh::TriangulateHole()`.

Implementation: `src/MeshHoles.cpp`.

## Spatial indices

Two triangle indices with the same query surface — nearest point and
ray intersection:

- **`TriangleBVH`** ([`TriangleBVH.h`](../include/halfmesh/TriangleBVH.h)) —
  binned-SAH, flattened node array, bounded traversal stack, and a `hintFace`
  warm start for coherent query streams. The default accelerator for texture
  baking ("markedly faster for the millions of queries a bake issues") and
  remeshing's surface projection.
- **`TriangleKdTree`** ([`TriangleKDTree.h`](../include/halfmesh/TriangleKDTree.h)) —
  median-split kd-tree; simpler, still ≥10× brute force (asserted by the perf
  harness).

Both return `NearestNeighbor { dist, nearest, idxFace }` and hold the mesh
**by reference** — it must outlive the index and stay unmutated.

## UV parametrization

```cpp
struct ParametrizeParams { /* segmentation + flattening knobs, see Parametrize.h */ };
unsigned Parametrize(Mesh&, const ParametrizeParams&);   // segment + flatten
void ParametrizeCharts(Mesh&, const std::vector<unsigned>& faceChart,
                       unsigned numCharts, const ParametrizeParams&);
```

Two modules that together turn a raw mesh into per-chart UV layouts:

- **Chart segmentation (D-Charts)** — partitions the mesh into the fewest
  flip-free *developable* charts (Julius/Kraevoy/Sheffer 2005): cone-proxy
  Lloyd clustering (a cone fits the whole developable family — planes,
  cylinders, cones), a developable union-find merge, and a flatten-and-bisect
  flip-repair pass that makes the flip-free property a hard guarantee.
  Notable knobs: `developableMaxConeError` (chart size vs distortion),
  `developableMaxVertexDefect` (the anti-fold cap — high-curvature vertices
  stay on chart boundaries), `developableSmoothIters` (virtual Taubin denoise
  of the *segmentation signal* on noisy MVS meshes — the geometry is
  untouched), and opt-ins `developableDistanceExponent` (compact charts,
  β=0.7) and `developableMaxUvDistortion` (symmetric-Dirichlet cap).
- **Per-chart flattening** — LSCM initialization (falling back to Tutte, then
  PCA) followed by **SLIM** (default) or **ARAP** iterations; flip-free by
  construction via the line search. `cutToDisk` optionally slits annuli/holed
  charts (Seamster) first.

Output: `mesh.faceTexcoords` per corner, each chart in its own local frame
(unpacked) — feed to the atlas packer next. Design record with measured
trade-offs: [`ATLAS_SEGMENTATION_DESIGN.md`](ATLAS_SEGMENTATION_DESIGN.md).

Headers: [`Parametrize.h`](../include/halfmesh/Parametrize.h),
[`AtlasCharting.h`](../include/halfmesh/AtlasCharting.h) (the definitive
method write-up lives in its header comment).

## Atlas packing

```cpp
struct AtlasParams { texelsPerUnit, resolution, padding, allowRotation,
                     powerOfTwo, square, orientCharts, fitToResolution };
AtlasResult GenerateAtlas(Mesh&, const ParametrizeParams&, const AtlasParams& = {});
AtlasResult PackAtlas(Mesh&, const std::vector<unsigned>& faceChart,
                      unsigned numCharts, const AtlasParams& = {});
float NormalizeChartDensity(Mesh&, const std::vector<unsigned>& faceChart,
                            unsigned numCharts, const AtlasParams& = {});
```

- `GenerateAtlas` is the one-call pipeline: `SegmentCharts` →
  `ParametrizeCharts` → `NormalizeChartDensity` (uniform texels-per-area
  across charts) → `PackAtlas`. Returns chart→page assignment, atlas
  dimensions, page count and occupancy; leaves normalized `[0,1]` UVs in
  `mesh.faceTexcoords`.
- `PackAtlas` is the packer alone: skyline bottom-left min-waste placement
  (xatlas-inspired) with per-chart minimum-area-rectangle pre-orientation
  (rotating calipers), optional 90° rotations (true rotations — winding
  survives), gutter `padding`, and multi-page overflow. For multi-page
  results, per-face pages come from `AtlasResult::chartPage[faceChart[f]]`.

Benchmarks against xatlas / libigl / pmp / CGAL / BFF (methodology + numbers):
[`BENCHMARKS.md`](BENCHMARKS.md). Headline: end-to-end atlas 30–130× faster
than xatlas at comparable pack occupancy, always flip-free, lowest
symmetric-Dirichlet distortion of the seven flatteners tested.

Implementation: `src/AtlasCharting.cpp`, `src/AtlasPacking.cpp` ·
example: `examples/Unwrap.cpp`.

## Texture bake / rebake / defrag

```cpp
struct BakeParams { resolution, maxResolution, multiPage, padding, supersample,
                    interp, correspondence, raySearchDist, numThreads,
                    accelerator, maxDefragPatches };
BakeResult BakeAtlas(Mesh& target, const std::vector<Image3u>& sourceImages,
                     const SourceResolver&, const BakeParams&);
BakeResult RebakeTexture(const Mesh& source, Mesh& target, const BakeParams&);
BakeResult DefragmentTexture(Mesh& mesh, const BakeParams&);
unsigned   AutoAtlasResolution(const Mesh& source, unsigned maxResolution = 8192);
```

Rasterizes every target-mesh face into its atlas UV footprint and fills each
texel by asking a **`SourceResolver`** where the color comes from — the
library's texturing extension point:

- `SameUVResolver` — same parametrization, different layout (defrag path).
- `RaycastResolver` — geometric correspondence to a *different* textured mesh
  (nearest-point or raycast, BVH- or kd-tree-accelerated, `hintFace` warm
  starts), used by `RebakeTexture` to re-texture a retopologized/decimated
  mesh from the original.
- Custom subclasses can implement any source: projecting camera images,
  procedural shading, vertex-color splatting — implement `Resolve(...)` once
  and `BakeAtlas` handles rasterization, supersampling, multi-page output and
  gutter dilation.

`DefragmentTexture` repacks an existing textured mesh's UV islands into a
fresh atlas and rebakes, recovering wasted texels. `BakeAtlas` expects
`target.faceTexcoords` in absolute-pixel space; `RebakeTexture` runs
`GenerateAtlas` internally first.

Header: [`TextureBake.h`](../include/halfmesh/TextureBake.h) ·
example: `examples/TextureBakeTool.cpp` (rebake / defrag / fidelity modes).

## openMVS interop

[`InteropOpenMVS.h`](../include/halfmesh/InteropOpenMVS.h) — header-only,
compiled only when `<MVS/Mesh.h>` is on the include path (a no-op otherwise):
`ConvertMesh(const MVS::Mesh&, halfmesh::Mesh&)` and the reverse transfer
geometry, per-vertex colors, per-face normals, per-corner UVs, per-face
texture indices and diffuse textures between the two libraries with zero
build-time coupling. openMVS's per-vertex normals have no halfmesh
counterpart and are not transferred.

## Utilities

- [`OrientedBoundingBox.h`](../include/halfmesh/OrientedBoundingBox.h) —
  `struct OBB` (rotation + AABB) with extend/contains/transform/corners; pairs
  with `Mesh::RemoveFacesOutside`.
- `Util/Geometry.h` — angles/cotangents, point–segment and point–triangle
  distances, Möller–Trumbore ray-triangle intersection, barycentric helpers.
- `Util/Raster.h` — `RasterizeTriangleBary` (top-left-rule triangle traversal
  with barycentric callback) and mask-guided `Dilate` (the bake gutter fill).
- `Util/Sampler.h` — bilinear/bicubic image sampling with texel-center
  convention (`uv*size − 0.5`).
- `Util/Accumulator.h`, `Util/PixelTraits.h` — weighted accumulation over
  image/pixel types.
- `Util/Log.h` — `SetStatusLog(std::ostream*)` to redirect or silence library
  output; `REPORT_*`/`TIMER_*` macros, all `#ifndef`-guarded so a host project
  can substitute its own. Uses `std::format`; define `HALFMESH_USE_FMT` to
  use `fmt::format` on toolchains without `<format>`.
- `Util/{Assert,Loop,Maths,Hash}.h` — assertion/iteration/math/hashing
  helpers (also `#ifndef`-guarded).

## Threading & determinism

Parallel phases (remeshing, simplification setup, smoothing, per-chart
flattening, hole filling, texture decode/encode, bake row bands) run on
locally-created `BS::light_thread_pool` pools; the bake additionally accepts
`BakeParams::numThreads`. **Determinism is a hard invariant**: every parallel
region is a pure map into disjoint slots, floating-point scatter/accumulation
stays serial to fix summation order, and holes/charts are processed in a
deterministic order regardless of thread timing. The golden-fixture tests
compare bit-exact output — which is also why `-march=native`
(`HALFMESH_NATIVE`/`HALFMESH_ARCH`) and LTO are opt-in build flags.

## Performance

Advisory single-machine baselines (`tests/perf/baseline.json`; the perf
harness asserts machine-independent *scaling*, not absolute times):

| Operation | 50 k faces | 200 k faces | 800 k faces |
|---|---:|---:|---:|
| `HalfMesh::Build` | 0.006 s | 0.028 s | 0.165 s |
| `Simplify` | 0.016 s | 0.068 s | 0.384 s |
| `RemeshIsotropic` | 0.220 s | 2.15 s | — |
| `CloseHoles` | 0.012 s | 0.050 s | — |
| `Parametrize` | 0.666 s | 3.10 s | — |
| `GenerateAtlas` | 0.673 s | 2.97 s | — |

Full-pipeline scale point: a 15.3 M-face scan runs the complete atlas pipeline
in 321 s at 4.68 GB peak RSS (190 k charts, 9 pages). Comparative numbers
against other libraries: [`BENCHMARKS.md`](BENCHMARKS.md). Testing strategy
(unit / property / golden / cross-check layers): [`TESTING.md`](TESTING.md).
