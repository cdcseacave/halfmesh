# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1]

### Fixed: one degenerate chart could collapse the whole atlas

`NormalizeChartDensity` bounded a chart's UV *area* but not its *extent*. A
chart with negligible area over a huge extent — what `cutToDisk` produces when
it slits a tube into a ribbon — was passed through unnormalized, because the
guard tested area compression (`sqrt(worldArea/uvArea) > 1e4`) and, when it
fired, skipped the chart entirely. `PackAtlas`'s degenerate rescue does not
catch that shape either (it tests for a zero-width or zero-height rect; a
ribbon has a large `w` and a small-but-positive `h`), so the chart reached
`fitToResolution`, whose **max-dimension** term then shrank every *other*
chart until the widest one fit. Nothing errored, and `occupancy` still read
plausibly — only `coverage` showed the damage.

The bound is now on the chart's scaled **extent**: the page when the atlas
must fit one, `D·sqrt(worldArea)` for a degenerate flatten. Measured at
4096²/padding 2, triangle coverage before → after:

| mesh / knobs | before | after |
|---|--:|--:|
| Ignatius (471 814 f), defaults | 0.1891 | **0.2484** |
| Ignatius, `foldRescueSlits=2` | 0.0200 | **0.2292** |
| Ignatius, `foldRescueSlits=1` | 0.2292 | 0.2292 |
| Ignatius, `cutToDisk` | 0.2668 | 0.2668 |
| Truck (476 690 f), defaults | 0.2334 | 0.2334 |
| Truck, `cutToDisk` | 0.2412 | 0.2412 |

Only the two arms that actually had an over-page chart move; the other four
are bit-identical, chart counts included.

The gate fires only when a chart does not *earn* its extent with area
(`rawExtent > R·sqrt(uvArea)`) — a mesh with few charts has charts that
legitimately span the page, and clamping those costs the 2-chart Cone 3.2 % of
occupancy. **`R` is 16**, calibrated over every chart of six 4096² arms across
two PGSR splat→mesh scenes plus the 5-mesh quality corpus, split by whether a
chart actually exceeds the page: charts that legitimately do measure
**1.4–1.9**; healthy charts sit at p50 2.1, p99.9 12–15; the ribbons that
collapse an atlas measure **55–606**. Since the ratio is `sqrt(aspect)`, `R`
is a bound on the square root of a chart's aspect — `16` admits 256 : 1.

Because the gate only ever judges charts already wider than the page, it is
not a universal aspect bound: `cutToDisk` emits thousands of legitimate small
high-ratio ribbons (max ratio 1 700 on Truck, 1 244 on Ignatius) that never
reach the clamp, and squashing those is what cost an aspect-8 bound 9.7 % of
Truck's coverage.

This removes a chart's ability to tax its siblings. It does not by itself
remove the ribbon, which still occupies a page-wide slot — the distortion bar
below is what dissolves it. Present in 0.3.0 as well; this release's
segmentation change is what first made a real mesh reach it.

### Fixed: a flip-free but unusably stretched chart shipped unchecked

The flip repair split a chart for over-distortion only when the caller set
`developableMaxUvDistortion`, whose default is `0` — and at that default it
applied no distortion check at all. Any map that was injective shipped,
however badly stretched. Meanwhile the injectivity fallback ladder in the same
file refuses to *ship* a map whose area-weighted symmetric-Dirichlet exceeds
200. The two disagreed about what is shippable, and the repair's acceptance is
the one that decides — so charts the ladder would have rejected reached the
atlas, where nothing downstream can fix them. Only a split can, and only
upstream, while the chart can still be divided.

Worst shipped per-chart symmetric-Dirichlet at 4096²/padding 2 (4.0 is perfect
isometry; the count in parentheses is charts above the bar):

| mesh / knobs | before | after |
|---|--:|--:|
| Ignatius (471 814 f), defaults | 3.3e8 (31) | **22 390** (12) |
| `mesh.ply`, `cutToDisk` | 18 288 (3) | **1 533** (1) |
| `mesh.ply`, `cutToDisk` + `carve=2` + `slits=2` | 5 373 (2) | **167** (0) |

On the Ignatius default arm that is a 14 900× reduction in the worst chart for
199 extra charts out of 78 123 (+0.25 %), and it *raises* coverage, 0.2484 →
**0.2515**. The widest chart in that atlas drops from 2 071 texels to **640**
of 4 096, and on the `foldRescueSlits=2` arm from 1 961 to **244**: the
page-spanning chart that survived the extent gate above was an over-stretched
chart all along. Splitting it dissolves the ribbon instead of merely capping
it, which is why coverage goes *up* — the charts being split were the ones
hogging the page.

`developableMaxUvDistortion` now *tightens* that bar rather than switching the
check on: `0` selects the internal ship-ability bar, and any explicit value
replaces it. The mandatory sliver guard is unchanged, so degenerate
near-zero-area input still cannot runaway-split.

The bar is a floor, not a guarantee: 12 charts on the Ignatius arm and 1 on
`mesh.ply` + `cutToDisk` still ship above it. The repair stops short by design
on input a split cannot fix — the sliver guard, the `4·F` runaway cap, and the
planar fallback for charts with no usable boundary loop all decline to
subdivide further. That residual is not diagnosed further here.

**Default-output change**, and not always a free one. Charts above the bar are
bisected, which costs charts and therefore global fit scale — and `coverage`
scales with the *square* of that scale. Where the split also frees page-hogging
extent, coverage rises; where it does not, the trade runs the other way:

| arm | coverage before | after | widest chart (of 4 096) |
|---|--:|--:|--:|
| Ignatius, defaults | 0.2484 | **0.2515** | 2 071 → **640** |
| Ignatius, `foldRescueSlits=2` | 0.2292 | **0.2534** | 1 961 → **244** |
| Ignatius, `cutToDisk` | 0.2668 | 0.2629 | → 293 |
| Truck, defaults | 0.2334 | 0.2334 | → 140 |
| `mesh.ply`, `cutToDisk` | 0.4561 | 0.3779 | → 1 233 |

Four of the five arms improve or hold. The `mesh.ply` row is the price of not
shipping texels that mean nothing: there the split adds charts without
releasing any page-hogging extent, so the fit scale — and with it coverage —
drops. Meshes with no chart above the bar are unaffected: every
`tests/data/golden/` fixture is byte-identical.

### Added: atlas layout diagnostics

`AtlasResult` gains four fields, exposed to Python from `unwrap()`. Each
answers a question the return value previously could not, and each was a real
integration cost — all three had to be reverse-engineered from a written PLY.

- **`fitScale`** (`fit_scale`) — the single global scale fit-to-resolution
  applied. Because the solve is `k = min(k_area, (resolution − 2·padding)/maxDim)`,
  a `fitScale` far below its area-driven value separates *"charts are small
  because there are many"* from *"charts are small because ONE chart forced a
  shrink"* — the failure mode fixed above.
- **`maxChartExtent`** (`max_chart_extent`) — widest unpadded chart side in
  texels. Read against `width`, it names the offending chart directly.
- **`minPadding` / `chartsPaddingReduced`** (`padding_applied{nominal, min,
  n_charts_reduced}`) — the narrowest gutter actually applied and how many
  charts got it. With the per-size padding knobs on, selected charts get 1
  texel while the nominal `padding` still reads 2. That difference decides
  whether the atlas can be mipmapped — halving resolution averages 2×2 texel
  blocks — and nothing in the result had reported it.

### Added: triangle coverage metric

`AtlasResult::coverage` — the fraction of the texel budget actually under UV
geometry, as opposed to `occupancy` (padded-rect fill, which reads high with
many small charts). Available from Python `unwrap()`; `atlasbench` reports the
same quantity as `occupancyTri`.

### Added: four opt-in segmentation and packing knobs, all default off

None of these changes default output. All four measured on both mesh classes
(`docs/BENCHMARKS.md` section 4); none earns a default.

- **`ParametrizeParams::repairCarveRings`** (default `0`) — a folding chart is
  first split by carving off the faces within N `TopoNeighbor` rings of the
  diagnosed failure, one small extra chart instead of a blind-bisection
  cascade, falling back to bisection when the failure is not localized (region
  ≥ half the chart). Termination guarantee unchanged. Moves the chart count
  −3.7 % on `tests/data/mesh.ply`, −0.2 % on a Truck-class mesh.
- **`ParametrizeParams::foldRescueSlits`** (default `0`) — a folding chart is
  cut from its worst interior vertex to the boundary and re-flattened, up to N
  times, before any split, so it can ship as ONE chart with an extra seam
  instead of ≥ 2 padded rects. Lives inside `FlattenChart`, so the repair
  verdict and the shipped map agree on every path. **The knob is not
  monotone and can cost charts.** The repair and the post-repair merge run as
  an iterative loop, so changing the split predicate perturbs which partition
  enters the next round, and the fixed point can land either way. Measured on
  `tests/data/mesh.ply` through the `GenerateAtlas` path at 4096²/padding 2,
  `foldRescueSlits` 0 → 1 → 2 → 3:

  | slits | 0 | 1 | 2 | 3 |
  |---|--:|--:|--:|--:|
  | charts | 2708 | 2734 | 2852 | 2766 |
  | coverage | 0.3732 | 0.3792 | **0.3388** | 0.3779 |

  Every value with the knob on costs charts, and `2` is a sharp local dip in
  both. It reduces charts on the Ignatius class and hurts here — measure on
  your own meshes before enabling it, and do not assume more slits is better.
- **`AtlasParams::tinyChartSide` / `debrisChartFaces`** (default `0`) — charts
  under an unpadded-bbox-side or face-count trigger get a 1-texel gutter
  instead of the uniform `padding`, so a uniform gutter stops being a
  multiplicative tax on exactly the charts that matter least. Packing-only —
  never changes the chart partition. On a Truck-class mesh these lift coverage
  0.2325 → 0.3200, but a global `padding=1` reaches 0.3334 at an identical
  partition: this mesh class has no mix of chart sizes for a size-triggered
  gutter to exploit. Coverage is texels, not bake quality — `padding` 2→1 is
  unmeasured against a bake.

### Changed: post-repair merge remembers pairs that re-fold

Always on, no knob. A candidate pair that demonstrably re-folds after merging
is memoized (keyed by the two sides' smallest global face ids, invariant under
relabelling) so later merge rounds never retry it, removing
accepted-then-resplit churn from the merge↔repair rounds. **This is a
default-segmentation-output change**: chart counts on meshes with fold/re-merge
churn may differ slightly (`tests/data/mesh.ply`: 2816 → 2779). `tests/data/golden/`
fixtures are unaffected — no re-freeze.

### Python and CLI

- `unwrap()` gains `repair_carve_rings`, `fold_rescue_slits`,
  `tiny_chart_side`, `debris_chart_faces`, plus `max_cone_error`
  (→ `developableMaxConeError`), `cut_to_disk` (→ `cutToDisk`) and
  `max_uv_distortion` (→ `developableMaxUvDistortion`). All default to the
  C++ defaults and are behavior-preserving. `cut_to_disk` is the chart-count
  reducer on hole-riddled MVS meshes.
- The binding's `padding` default drops 4 → 2, matching `AtlasParams::padding`.
  The binding was silently overriding the documented C++ default; the doubled
  gutter cost a texture consumer 1.26 dB of bake PSNR at 4096² (bug fix, not a
  tuning change).
- `atlasbench` gains `--repair-carve-rings`, `--fold-rescue-slits`,
  `--tiny-chart-side`, `--debris-chart-faces`.

## [0.3.0]

### Half-edge–primary mesh processing

The half-edge is now the working representation and `faces` is a derived
snapshot, so a multi-stage pipeline pays **one** `HalfMesh::Build` instead of
one per family transition. Measured on a 5 M-face mesh, a native clean
(spurious → spikes → holes → unref) performs exactly one build and one face
harvest; `tests/perf` asserts both counts.

- **Representation contract.** `Mesh` has exactly three valid states —
  arrays-only, half-edge-only, both-and-consistent. New:
  `Mesh::InvalidateFaces()`, `Mesh::SyncFaces()`, `Mesh::SyncFacesConst()`,
  `Mesh::InvalidateHalfMesh()`, `Mesh::ValidateInvariants()`,
  `Mesh::ValidateHalfMesh()`, and the `BeginHalfEdgePipeline()` /
  `EndHalfEdgePipeline()` scope that defers per-stage snapshots to a single
  final harvest. Public methods still return with `faces` populated.
- **New `HalfMesh` primitives.** `FRemoveBulk` removes an arbitrary face set in
  one pass, splitting pinch vertices and reporting the vertex swap-pops and
  split sources so `Mesh::vertices`/`vertexColors` follow in lockstep;
  `FAddDisk` attaches a pre-triangulated patch in dependency order;
  `VRemoveUnreferenced` sweeps unreferenced vertices natively. `FAdd` now
  accepts isolated corners (two new edges), propagates a border-relink failure
  as `NO_ID`, and unwinds through an undo log so a rejected add — or a rejected
  whole patch — leaves every array byte-identical.
- **Border relinking is parity-agnostic.** `ConnectBorders` identifies the
  boundary representative through `heFaces` instead of half-edge parity, so it
  works on a structure mutated in place by `EFlip`/`ESplit`/`ERemove`, not only
  on a freshly built one.
- **Native repair/hole stages.** `RemoveSpuriousComponents`,
  `RemoveSmallComponents`, `RemoveSpikes`, `RemoveDegenerateFaces`,
  `RemoveUnreferencedVertices`, `CloseHoles` and `RemoveVerticesAndFill` mutate
  the live half-edge and no longer clear it. `CloseHoles` also dropped its
  defensive entry rebuild, so a call that fills nothing now costs ~0, and
  `RemoveVerticesAndFill` classifies loops straight off the connectivity, so it
  neither reads nor produces a face snapshot.
  `FixNonManifold` short-circuits to a no-op once `halfMesh` exists — the
  structure cannot represent what it fixes.
- **Representation dispatch.** `RemoveUnreferencedVertices`,
  `RemoveDegenerateFaces` and `RemoveSpikes` pick an arm by current state and
  never force a transition; both arms are public as `…Arrays` / `…HalfEdge`.
  The array arms keep working on non-manifold soup (no silent manifoldization)
  and keep preserving attributes.
- **`Simplify` and `RemeshIsotropic` read topology from `halfMesh` only**, so
  neither needs a face snapshot on entry and neither maintains one mid-pass.

### Performance

- `HalfMesh::Build` pairs twin half-edges through a flat open-addressing table
  keyed by a packed `(min,max)` vertex pair instead of
  `std::unordered_map`: **3.24×** faster (0.608 s vs. 1.969 s) at 495.8 MiB vs.
  904.2 MiB peak working set on a 5,003,552-face mesh (Release, MSVC 14.51,
  x64, i7-13700KF). Half-edge numbering, rejection behavior and goldens are
  unchanged.

### New

- `Mesh::RemoveSpuriousComponents(factor)` — reconstruction-debris removal
  relative to the mesh's own edge-length distribution (long-edge faces, then
  components with a small bounding-box diagonal).
- `Mesh::RemoveSpikes(maxIterations)` — cascade removal of vertices incident to
  at most one face.
- `Mesh::RemoveVerticesAndFill(vertexRemoves)` — remove vertices and span only
  the boundary loops that removal created, without refining, so the vertex
  count is guaranteed to shrink.
- `Mesh::RemoveFacesHalfEdge`, `Mesh::ComputeMeanEdgeLength`.
- **`Mesh::vertexNormals`** — authored per-vertex normals, transported rather
  than maintained. Parallel to `vertices` under the same empty-or-exact contract
  as `vertexColors`, but explicitly *not* a cache: nothing recomputes it, so an
  empty array means "derive them yourself". It is remapped through every
  operation that only renumbers vertices (duplicate/unreferenced-vertex removal,
  non-manifold bow-tie splits, face removal, edge collapse) and cleared by every
  operation that *moves* one (`Simplify`, `SmoothHCLaplacian`, `SmoothTaubin`,
  `RemeshIsotropic`, degenerate-face collapse, hole filling) via the new
  `Mesh::InvalidateVertexNormals()`. `Mesh::VertexAttributesSwapPop()` /
  `VertexAttributesAppendFrom()` let a remover name the per-vertex attribute
  *set* instead of each array. `SavePLY` prefers stored normals over derived
  ones; the openMVS bridge carries them in both directions, so a round trip no
  longer silently replaces an artist's normals with geometric ones.
- **`BakeOntoAtlas(source, target, params)`** — `RebakeTexture`'s counterpart
  for a target whose UV layout is authored and must be preserved: it reads
  `target.faceTexcoords` instead of generating a new atlas, so only the texels
  change. Preconditions (including the target texture's size against
  `params.resolution`, the only evidence of the pixel space the UVs are in) are
  checked in every build mode.
- **`BakeParams::faceMask`** — restrict rasterization to selected target faces.
  Under a mask each page is seeded from `target.texturesDiffuse` rather than
  black, so unselected texels keep what they had and the bake is a true in-place
  edit; a wrong-sized mask is refused with a warning. Honored by `BakeAtlas` and
  `BakeOntoAtlas`; `RebakeTexture` and `DefragmentTexture` repack into a new
  layout in which the old texels mean nothing, so they ignore it.
- [`RectPacking.h`](docs/FEATURES.md#rectangle-packing-mesh-independent) —
  `PackRectangles` / `EstimateSquareTextureSize`: the atlas packer's two-tier
  skyline+shelf core exposed over integer pixel rectangles, for lightmaps,
  sprite sheets and texture repacking with no mesh involved. `PackAtlas` and
  `PackRectangles` are now two thin wrappers over one implementation.
- `HalfMesh::BuildCount()` / `FFacesCount()` (+ `Reset…`) — process-wide
  counters that let a pipeline assert it rebuilds connectivity once.

### glTF: exception-free tinygltf, OpenCV image codec

- **tinygltf no longer needs exceptions.** Built with `TINYGLTF_NOEXCEPTION` +
  `JSON_NOEXCEPTION`, so glTF JSON is parsed by checking the parser's return
  value rather than by catching, and nlohmann/json is compiled in its
  no-exception mode. Both are `PRIVATE` to the library -- they are only read
  behind `TINYGLTF_IMPLEMENTATION` -- so `JSON_NOEXCEPTION` cannot leak onto a
  consumer that uses nlohmann/json itself, where it would turn its exceptions
  into `abort()`. OpenCV and tinyply still throw, so the library as a whole is
  not yet buildable with exceptions disabled.
- **glTF textures are decoded and encoded with OpenCV, not stb.** Built with
  `TINYGLTF_NO_STB_IMAGE` + `TINYGLTF_NO_STB_IMAGE_WRITE`; the bundled stb
  codecs are no longer compiled into the build at all, and glTF images now take
  the same libjpeg-turbo / libpng path as every other image the library reads or
  writes. JPEG export keeps stb's quality 100, so file output is unchanged.
  Internal `src/GltfImageCodec.{h,cpp}`: with those macros set tinygltf ships no
  default codec, so every `tinygltf::TinyGLTF` context must call
  `SetGltfImageCodec()` before touching images. `Mesh::LoadGLTF` /
  `Mesh::SaveGLTF` do it themselves -- callers see no change.

### Changed

- **`LoadPLY` resets the optional arrays it does not replace.** Loading into a
  reused `Mesh` kept the previous mesh's `vertexColors` / `faceTexcoords` /
  textures whenever the new file carried none, leaving an array of the wrong
  length against the new vertex or face count. Both loaders now clear the
  optional set up front, so the empty-or-parallel contract holds across a load.
- **`Mesh::CloseHoles` changed meaning.** The first parameter is now
  `maxHoleEdges` (default **30**), a *size* threshold: every boundary loop
  spanned by at most that many edges is filled, in place of the old
  `nCloseHoles = 200` *count* of smallest-first loops. A scanned surface's
  large open boundary now stays open by default instead of being patched.
  Python: `close_holes(v, f, max_hole_edges=30)`.
- **Processing targets untextured meshes.** Half-edge mutators drop face-keyed
  attributes through `InvalidateFaces()` (one-time warning when texture data is
  actually discarded). `docs/FEATURES.md` marks every public processing method
  `untextured-only` or `attribute-preserving (bonus)`.
- `Mesh::ListHalfEdges()`'s freshness gate is now exact (`halfMesh` non-empty ⇒
  valid, by contract) instead of a vertex/face count heuristic. **A caller that
  hand-edits the public `faces` array must call `Mesh::InvalidateHalfMesh()`**;
  the old heuristic happened to catch count-changing edits.
- **`vertexColors` is a contractual parallel array** — either empty, or exactly
  as long as `vertices` — and `Mesh::ValidateInvariants()` now enforces it, so
  every mutator's existing assertions catch a desync at its source. Mutators
  that change the vertex set maintain it: `ECollapse` and `RemeshIsotropic`
  mirror the swap-pop (and interpolate at an edge split), `FixNonManifold` and
  `FRemoveBulk` duplicate the source colour on a vertex split, `CloseHoles`
  gives patch interiors the mean colour of the loop they span, and `Simplify`
  clears the array up front because its collapses have no mapping to offer.
  `ECollapse` also refreshes the face snapshot; scope a collapse loop in
  `BeginHalfEdgePipeline` to avoid the per-call harvest.
- Atlas packing: `PackAtlas` and the fit-to-resolution probe run through the
  shared packer; behavior matches 0.2.0.
- **`faceTexblobs` is `std::vector<Mesh::TexIndex>` (uint8)** rather than
  `FIndex`, matching openMVS's `MVS::Mesh::TexIndex` — whose texture list is
  indexed by that same type, so 255 blobs was always the ceiling. The array is
  4x smaller and `ConvertMesh` becomes a straight copy in both directions;
  `Mesh::MAX_TEXBLOBS` spells out where the limit comes from, and `LoadGLTF`
  now caps and says so instead of silently folding the extra textures onto
  blob 0. The PLY on-disc property stays `int32`, so existing files and their
  readers are unaffected.
- **`Mesh::SaveGLTF` takes `imageFormat` and `embedImages`**, defaulted to the
  previous embedded-JPEG behavior; `Mesh::ImageFormat` maps onto the glTF
  mimeType, which is what tinygltf picks its encoder from. Non-embedded images
  are named per blob (`<stem>_diffuse<NN>`, matching `SavePLY`) instead of
  every blob overwriting and referencing one `image` file.
- **openMVS interop gained consuming overloads.** `ConvertMesh` now has an
  rvalue form in each direction that frees every source array as soon as it is
  copied — peak footprint is the larger mesh plus one array instead of both at
  once — and the halfmesh->MVS side also drops the half-edge structure and the
  incident-face cache.
- **glTF save → load is now an identity.** `SaveGLTF` has always written the
  vertex buffer in halfmesh's z-up frame and put the z-up → y-up rotation on
  the root node, but `LoadGLTF` applied that node matrix like any other and
  returned the mesh rotated 90° about X — so the library could not read back
  what it had just written, and glTF was unusable as a lossless interchange
  format. `LoadGLTF` now converts back to z-up after flattening the hierarchy.
  The rotation stays in the *file*, so exports remain upright in Blender,
  three.js and Cesium.

  **halfmesh is z-up in memory and its glTF files are y-up** is now a stated
  contract (`Mesh::Load` header comment, `docs/FEATURES.md`) rather than
  something implied by a `const bool` inside `SaveGLTF`. It is fixed, not
  selectable — a knob would let the two sides be configured into disagreement.
  **A y-up glTF from any exporter now loads correctly; a glTF carrying z-up
  data under an identity node — non-conformant, but emitted by some writers,
  including openMVS's own pre-halfmesh exporter — will load rotated and must be
  corrected by its producer.** PLY is unaffected: it has no up-axis convention
  and is still read and written raw.
- **vcpkg `builtin-baseline` advanced** to `0ac8df3b98e3afcd8bf075fa74a6bd2c32613345`
  (2026-08-24), which carries the corrected `tinygltf` 3.0.0#1 archive hash.
  `vcpkg-overlay-ports/` (the tinygltf hotfix plus a local-source `halfmesh`
  port) and `vcpkg-configuration.json` are removed — a cold-cache build now
  resolves every dependency straight from the registry, and the halfmesh port
  lives only in the consumer that needs it.
- **Not ABI-compatible with 0.2.0** — recompile against the new headers.

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

### Changed
- **Default atlas output changes for every caller.** The two-tier pack pass
  changes atlas layout for all `PackAtlas` users (the shelf-tier threshold
  is a fixed `pageW/32`; there is no opt-out flag), and the new post-repair
  merge rounds change the default chart partition and therefore the shipped
  UVs. Frozen UV/layout goldens will move. Set
  `ParametrizeParams::postRepairMergeRounds = 0` to restore the pre-0.2.0
  segmentation behavior; the packing change has no revert switch.
- **Not ABI-compatible with 0.1.0**: `AtlasResult::fitAttempts` and
  `ParametrizeParams::postRepairMergeRounds` are mid-struct field
  insertions — recompile against the new headers.

### Interoperability
- openMVS interop: `ConvertMesh` now transfers per-vertex colors and
  per-face normals in both directions.
- `include/halfmesh/Types.h` no longer specializes
  `cv::DataDepth<halfmesh::Pixel::Scalar>` (behaviorally identical via
  OpenCV's generic trait; unbreaks `InteropOpenMVS.h` against openMVS's
  `Common/Types.h`).

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

[0.3.0]: https://github.com/cdcseacave/halfmesh/releases/tag/v0.3.0
[0.2.0]: https://github.com/cdcseacave/halfmesh/releases/tag/v0.2.0
[0.1.0]: https://github.com/cdcseacave/halfmesh/releases/tag/v0.1.0
