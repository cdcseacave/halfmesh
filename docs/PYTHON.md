# halfmesh — Python bindings

A pip-installable `halfmesh` package over the C++20 library: numpy in, numpy
out, plus a thin file-based facade for I/O and UV-atlas generation. The native
extension is `halfmesh._halfmesh` (built from [`python/binding.cpp`](../python/binding.cpp));
the pure-Python package [`python/halfmesh/__init__.py`](../python/halfmesh/__init__.py)
re-exports it.

## Install

**From a release (recommended)** — self-contained manylinux_2_28 wheels
(CPython 3.10–3.13, x86_64) are attached to every tagged
[GitHub Release](https://github.com/cdcseacave/halfmesh/releases):

```sh
pip install https://github.com/cdcseacave/halfmesh/releases/download/v0.3.0/halfmesh-0.3.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

Pick the `cpXY-cpXY` tag matching your interpreter (`cp310`, `cp311`, `cp312`,
`cp313`). The wheel bundles every dependency (Eigen, OpenCV, tinyply,
tinygltf) statically — no `VCPKG_ROOT`, no compiler, no system OpenCV needed
at install time. Only `numpy>=1.24` is pulled in.

**From source** — building against a checkout:

```sh
git clone https://github.com/cdcseacave/halfmesh && cd halfmesh
VCPKG_ROOT=/path/to/vcpkg pip install .
```

This drives the same root `CMakeLists.txt` via `scikit-build-core`, with
`-DHALFMESH_BUILD_PYTHON=ON`. Notes:

- **Cold build time**: vcpkg has to compile Eigen, OpenCV, tinyply and
  tinygltf from source on the first invocation — expect on the order of
  15–30 minutes uncached (the same cost the `wheels.yml` CI job pays;
  pybind11 itself comes from pip, not vcpkg). After
  that, vcpkg's binary cache makes dependency setup near-instant, and because
  `pyproject.toml` sets a persistent `build-dir` (`build/{wheel_tag}`),
  repeat `pip install .` runs are true incremental C++ rebuilds rather than
  full recompiles.
- **Triplet auto-selection**: on Linux, when you have not already set
  `VCPKG_TARGET_TRIPLET` (env or CMake cache) — and have not set
  `VCPKG_DEFAULT_TRIPLET` either, which also suppresses the auto-selection —
  the build automatically overlays and selects `x64-linux-pic`
  (`cmake/triplets/x64-linux-pic.cmake`) — a static+`-fPIC` triplet, required
  because every static archive linked into the shared extension module must
  be position-independent. You do not need to pass this yourself; it only
  matters if you're mixing this build with another vcpkg triplet in the same
  install tree, or if a tool in your setup (e.g. `lukka/run-vcpkg` in CI)
  sets `VCPKG_DEFAULT_TRIPLET` for you — in that case, set it to
  `x64-linux-pic` yourself and also set `VCPKG_OVERLAY_TRIPLETS` to
  `cmake/triplets`.
- Add the `test` extra (`pip install '.[test]'`) to also pull in `pytest` for
  running `python/tests`.

## API reference

All array-taking functions accept `vertices: float32[N,3]` and
`faces: uint32[M,3]` numpy arrays (other numeric dtypes and non-contiguous
layouts are force-cast/copied on entry) and **return new arrays** — always
float32/uint32, C-contiguous. This is not a style choice: halfmesh's
half-edge construction may weld, re-index, or otherwise repair non-manifold
input, so **the input and output index spaces are never guaranteed to match
— do not assume vertex/face index `i` in means index `i` out.** Face indices
that are out of range for the given vertex array raise `ValueError` before
any C++ work happens. The GIL is released around all native computation (see
[Threading and the GIL](#threading-and-the-gil)).

### `version() -> str`

The halfmesh library version string (`"0.3.0"`), single-sourced from
`project(halfmesh VERSION …)` in `CMakeLists.txt`. Also exposed as
`halfmesh.__version__`.

### `repair(vertices, faces) -> (v, f)`

Weld exact-duplicate vertices, drop duplicate faces (keeping one copy) and
degenerate faces, drop vertices left unreferenced by that, then fix
non-manifold topology. This is the recommended pre-pass before any other
operation — the same sequence the library's own auto-repair
(`ListHalfEdgesSafe`) runs, since it dissolves the phantom topology that
blocks edge collapses and makes later half-edge builds non-mutating.

### `smooth(vertices, faces, iterations, method) -> (v, f)`

Smooth vertex positions in place topology-wise (face count/connectivity is
unchanged; only positions move). `method` is `"taubin"` or `"hc"` — any
other value raises `ValueError`.

- `"taubin"` — Taubin lambda|mu band-pass smoothing: aggressive noise
  reduction with ~zero shrinkage (library defaults `lambda=0.65`,
  `mu=-0.69`). Preferred when you want to denoise without the mesh visibly
  shrinking.
- `"hc"` — HC (anti-shrink) Laplacian smoothing. A different anti-shrink
  strategy; pick whichever gives a better result on your data — neither
  dominates the other for all inputs, so if `"taubin"` output still looks
  off on a given mesh, try `"hc"` (and vice versa) rather than only tuning
  `iterations`.

`iterations` is an integer pass count; more iterations means more
smoothing/less noise at the cost of more (recoverable) surface flattening.
`iterations <= 0` raises `ValueError`.

### `simplify(vertices, faces, target, aggressiveness=0.0) -> (v, f)`

QEM (quadric error metric) edge-collapse decimation. `target` is
**dual-magnitude**:

- `target` in `(0, 1)` — a **keep-fraction**: `0.5` means "decimate to
  roughly half the input face count".
- `target > 1` — an **absolute face count**: `500.0` means "decimate to
  roughly 500 faces".
- `target <= 0` raises `ValueError`.

`aggressiveness` trades exactness for speed: `0.0` is exact priority-queue
QEM (collapses evaluated and applied in strict error order); larger values
(the codebase's examples use up to `~7.0`) switch to a fast threshold-sweep
mode that can overshoot the target face count somewhat in exchange for much
higher throughput on large meshes. Use `0.0` when you need the target hit
precisely; raise it when you're decimating large meshes and an approximate
result is fine.

### `close_holes(vertices, faces, max_hole_edges=30) -> (v, f, closed)`

Liepa minimum-weight-triangulation hole filling (fill → refine → fair) of every
hole spanned by at most `max_hole_edges` boundary edges, so the big open
boundary of a scanned surface stays open while its small gaps are patched.
`closed` (an `int`) is the number of holes filled. Pass a large cap to fill
every hole.

### `remove_small_components(vertices, faces, min_faces) -> (v, f, removed)`

Drop every connected component with fewer than `min_faces` triangles (and
the vertices that fall unreferenced as a result). `removed` is the number of
components dropped.

### `remesh(vertices, faces, edge_length, iterations=3) -> (v, f)`

Isotropic remeshing (flip/collapse/relocate/refine) toward a uniform target
`edge_length`, in the same world units as the input vertices — not a ratio.
`iterations` is the number of remeshing passes (more passes converge closer
to uniform edge length; 3 is a reasonable default for moderately
non-uniform input). Both `edge_length <= 0` and `iterations <= 0` raise
`ValueError`.

### `class Mesh`

A thin facade over `halfmesh::Mesh` for file-based I/O and interop with the
array ops above.

- `Mesh()` — empty mesh.
- `Mesh.from_arrays(vertices, faces) -> Mesh` (static) — build a `Mesh` from
  the same `[N,3]`/`[M,3]` arrays the array ops use (same validation: an
  out-of-range face index raises `ValueError`).
- `to_arrays() -> (v, f)` — `(float32[N,3], uint32[M,3])` copies of the
  current geometry.
- `load(path)` — load `.ply` / `.gltf` / `.glb` (format from the extension).
  Raises `RuntimeError` on failure (missing file, malformed data).
- `save(path, binary=True)` — save as `.ply` / `.gltf` / `.glb` (format from
  the extension). `binary=False` writes ASCII PLY. Raises `RuntimeError` on
  failure.
- `n_vertices: int`, `n_faces: int` — read-only counts.
- `has_texcoords: bool` — whether the mesh carries per-face-corner UVs
  (read-only).

### `unwrap(input_path, output_path, resolution=4096, padding=2, allow_rotation=True, max_cone_error=0.05, cut_to_disk=False, max_uv_distortion=0.0, repair_carve_rings=0, fold_rescue_slits=0, tiny_chart_side=0.0, debris_chart_faces=0) -> dict`

File-based UV-atlas generation: load `input_path` → weld/clean prelude
(`RemoveDuplicateVertices` + `RemoveDegenerateFaces` +
`RemoveUnreferencedVertices` — unwelded input makes every edge a boundary
and fragments segmentation into one chart per face) → `GenerateAtlas` →
save to `output_path`. Deliberately file-based rather than array-based: it
reuses the exact `examples/Unwrap.cpp` flow and keeps every UV-convention
subtlety (absolute-pixel atlas space in memory, normalized UVs in PLY,
half-texel offset in glTF) inside halfmesh's own `Save`, instead of
re-deriving it at the Python boundary.

- `resolution` — target atlas page size in texels (square pages).
- `padding` — texel padding between packed charts (same default as `AtlasParams::padding`).
- `allow_rotation` — whether the packer may rotate charts for a tighter fit.
- `max_cone_error` — segmentation cone-fit budget
  (`ParametrizeParams::developableMaxConeError`): larger ⇒ fewer, larger
  charts at slightly more distortion.
- `cut_to_disk` — Seamster cut-to-disk (`ParametrizeParams::cutToDisk`):
  slit closed / multiply-connected charts into one disk instead of
  bisecting them; the chart-count reducer on hole-riddled MVS meshes.
- `max_uv_distortion` — symmetric-Dirichlet cap
  (`ParametrizeParams::developableMaxUvDistortion`); `0` disables, on-values
  must exceed the 4.0 isometry floor (4.4 is the sane setting). See
  `docs/BENCHMARKS.md` §4 for the attribution matrix of these knobs.
- `repair_carve_rings` — failure-localized repair split
  (`ParametrizeParams::repairCarveRings`): `0` disables (the default — blind
  PCA bisection of folding charts); when `> 0`, a folding chart is first split
  by carving off the faces within this many `TopoNeighbor` rings of the
  diagnosed failure, falling back to PCA bisection when the failure isn't
  localized. `2` is the sane on-value. **Default `0` (off)** — see
  `docs/BENCHMARKS.md` §4 for the Task 9 sweep and why it stayed off.
  Combined with `fold_rescue_slits` this measured *worse* than either knob
  alone on the one mesh available for the Task 9 sweep (see
  `docs/BENCHMARKS.md` §4); do not enable both without re-checking that
  sweep on your mesh.
- `fold_rescue_slits` — fold-rescue slit count
  (`ParametrizeParams::foldRescueSlits`): `0` disables (the default); when
  `> 0`, a folding chart is slit from its worst interior vertex to the
  boundary and re-flattened, up to this many times, instead of being split
  into multiple charts. `2` is the sane on-value. **Default `0` (off)** —
  combined with `repair_carve_rings` this measured *worse* than either knob
  alone on the one mesh available for the Task 9 sweep (see
  `docs/BENCHMARKS.md` §4); do not enable both without re-checking that
  sweep on your mesh.
- `tiny_chart_side` — per-size padding trigger, max unpadded chart bounding
  side in texels (`AtlasParams::tinyChartSide`): charts at or under this size
  get a 1-texel gutter instead of `padding`. `0` disables (the default).
  Packing-only — never changes the chart partition.
- `debris_chart_faces` — per-size padding trigger, chart face-count
  (`AtlasParams::debrisChartFaces`): charts with this many faces or fewer get
  a 1-texel gutter instead of `padding`. `0` disables (the default).
  Packing-only — never changes the chart partition.

Returns a `dict`:

| Key | Meaning |
|---|---|
| `charts` | number of UV charts produced by segmentation |
| `pages` | number of atlas pages the charts were packed into |
| `width`, `height` | final atlas page dimensions in texels |
| `occupancy` | fraction of atlas area covered by charts, `[0, 1]` (0 only for a degenerate empty atlas) |
| `coverage` | fraction of the texel budget under actual UV triangles, `[0, 1]` — the honest density number (`occupancy` is padded-rect fill and reads far higher with many small charts) |
| `fit_attempts` | number of fit-to-resolution packing probes it took to fit the target page size |
| `vertices`, `faces` | vertex/face counts of the (welded) output mesh |

Raises `RuntimeError` if `input_path` fails to load or `output_path` fails
to save.

## Worked example

Load a noisy scan, run the standard cleanup pipeline, and generate a UV
atlas from the result:

```python
import halfmesh as hm

# Load
mesh = hm.Mesh()
mesh.load("noisy_scan.ply")
v, f = mesh.to_arrays()

# Clean → denoise → decimate → patch small holes
v, f = hm.repair(v, f)
v, f = hm.smooth(v, f, iterations=10, method="taubin")
v, f = hm.simplify(v, f, target=0.5)          # keep ~50% of faces
v, f, closed = hm.close_holes(v, f, max_hole_edges=50)
print(f"closed {closed} holes")

# Save the cleaned mesh, then generate a packed UV atlas from it
hm.Mesh.from_arrays(v, f).save("clean.ply")
meta = hm.unwrap("clean.ply", "clean_uv.ply", resolution=2048)
print(f"{meta['charts']} charts across {meta['pages']} page(s), "
      f"{meta['occupancy']:.1%} occupancy")
```

Remember: `v, f` are reassigned after every array-op call because each call
returns new arrays — the mesh may have been re-indexed under the hood.

## Threading and the GIL

Every array op and `Mesh.load`/`Mesh.save`/`unwrap` release the GIL for the
duration of the native call (`py::gil_scoped_release`), so a single
long-running halfmesh call does not block other Python threads. That said,
**halfmesh already parallelizes its own heavy phases internally** — remeshing,
simplification setup, smoothing, hole filling, and chart flattening all run
on halfmesh's own worker-pool (`BS::light_thread_pool`) instances, with
deterministic (order-independent) results. Do not additionally wrap calls to
`hm.simplify`/`hm.remesh`/etc. in a Python-level `ThreadPoolExecutor` or
`multiprocessing.Pool` expecting extra speedup from data parallelism within
a single mesh — you'll mostly just add scheduling overhead and, across
processes, pay full mesh-copy cost. Running independent *meshes* through a
process pool is fine and does help if you have many small meshes to process.

A `Mesh` instance is not safe to share across threads: `Mesh.load` and
`Mesh.save` release the GIL while mutating/reading the instance, so two
Python threads touching the same `Mesh` concurrently can race into a crash.
Give each thread its own instance.

## Deliberately absent

The following are recorded as out of scope for this package, not
oversights:

- **Torch tensor overloads** — the boundary is numpy only, on purpose: it
  keeps this extension free of any dependency on torch's C++ ABI. If you
  work in torch, convert at the boundary yourself with
  `torch.from_numpy(v)` / `v.numpy()`.
- **Texture baking** (`BakeAtlas`, `RebakeTexture`, source-image resolvers)
  — not bound yet. It needs image-array marshalling and a
  Python-subclassable source resolver, and is planned as its own follow-up
  designed together with the texturing-stage consumer.
- **openMVS interop** (`InteropOpenMVS.h`) — a C++-side concern (conversion
  between `halfmesh::Mesh` and `MVS::Mesh`); out of scope for the Python
  package.
- **macOS / Windows wheels** — only Linux x86_64 manylinux_2_28 wheels are
  built today; no current consumer needs the others.
