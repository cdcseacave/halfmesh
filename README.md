# halfmesh

[![CI](https://github.com/cdcseacave/halfmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/cdcseacave/halfmesh/actions/workflows/ci.yml)

Fast, compact C++20 half-edge triangle mesh processing library.

halfmesh is a standalone, dependency-light library for loading, manipulating, and
saving triangle meshes. It exposes a half-edge data structure optimised for
spatial queries and mesh editing operations, together with a complete UV
parametrization and texture atlas pipeline.

## Features

- **Half-edge core** — manifold triangle mesh connectivity with fast adjacency queries
- **Mesh repair** — fix non-manifold edges/vertices, remove duplicates and degenerate faces
- **QEM decimation** — edge-collapse simplification with quadric error metric (fast iterative variant)
- **Isotropic remeshing** — regularise edge lengths and triangle aspect ratios via flip/collapse/relocate
- **Smoothing** — HC Laplacian (anti-shrink) and Taubin lambda|mu band-pass
- **Liepa hole-filling** — fill boundary holes using the minimum-area advancing-front approach
- **Triangle KD-tree** — spatial index for ray-triangle intersection and closest-point queries
- **PLY / glTF I/O** — load and save binary/ASCII PLY and glTF/GLB files
- **UV parametrization** — developable (D-Charts) chart segmentation + per-chart SLIM/ARAP flattening
- **Texture atlas** — uniform-density normalisation + skyline (min-waste) packing into one or more atlas pages

Each feature is described in depth — entry points, key parameters, gotchas,
and pointers to the examples — in [`docs/FEATURES.md`](docs/FEATURES.md).

## Dependencies (all via vcpkg)

| Package | Role |
|---------|------|
| `eigen3` | Linear algebra (sparse solvers, geometry) |
| `bshoshany-thread-pool` | Worker pool for the parallel phases (remeshing, simplification setup, smoothing, per-chart flattening) |
| `opencv4` (no default features; eigen, fs, intrinsics, jpeg, png, thread) | Image types used by the texture layer |
| `tinyply` | PLY I/O |
| `tinygltf` | glTF/GLB I/O (header-only) |
| `gtest` | Unit tests (test build only) |

## Build

Set `$VCPKG_ROOT` to your vcpkg installation; the CMake toolchain is loaded
automatically. Dependency versions are pinned by the `builtin-baseline` in
`vcpkg.json`.

halfmesh depends on `opencv4` directly rather than the `opencv` metaport, with
default features off: only `core`, `imgproc` and `imgcodecs` are used, so the
GUI/video/DNN modules that OpenCV enables by default (and the GTK stack they
drag in on Linux) are not built. This cuts the dependency graph from 79 packages
to 14.

```sh
git clone https://github.com/cdcseacave/halfmesh && cd halfmesh
cmake -S . -B make
cmake --build make --config Release -j
```

### Build options

| Option | Default | Description |
|--------|---------|-------------|
| `HALFMESH_BUILD_TESTS` | `OFF` | Build unit tests (requires `gtest` via vcpkg) |
| `HALFMESH_BUILD_TOOLS` | `OFF` | Build the example CLIs |
| `HALFMESH_BUILD_PERF` | `OFF` | Build the performance/scaling harness (Release) |
| `HALFMESH_BUILD_CROSSCHECKS` | `OFF` | Build the libigl cross-check tests (dev) |
| `HALFMESH_SANITIZE` | `OFF` | ASan + UBSan instrumentation (MSVC: ASan only — it ships no UBSan) |
| `HALFMESH_ATLAS_DEBUG` | `OFF` | Compile in verbose atlas-pipeline diagnostics |
| `HALFMESH_NATIVE` | `OFF` | Build for the host CPU (`-march=native`); opt-in, breaks bit-identical goldens |
| `HALFMESH_ARCH` | *(empty)* | Target CPU arch for `-march=<isa>` (e.g. `x86-64-v3`); opt-in, breaks bit-identical goldens |
| `HALFMESH_LTO` | `OFF` | Enable LTO/IPO on optimized builds; opt-in, may shift float rounding (breaks goldens) |

```sh
cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON -DHALFMESH_BUILD_TOOLS=ON
cmake --build make --config Release -j
ctest --test-dir make -C Release --output-on-failure
```

`--config` / `-C` are what make this work with every generator. Multi-config
generators (Visual Studio, Ninja Multi-Config, Xcode) ignore `CMAKE_BUILD_TYPE`
and pick the configuration at build/test time, so without them you silently get a
`Debug` build; single-config generators (Makefiles, Ninja) ignore `--config`/`-C`
and use `CMAKE_BUILD_TYPE`, which defaults to `Release` here. Multi-config
generators also place binaries in a per-config subdirectory
(`make/examples/Release/unwrap`).

### Install

```sh
cmake --install make
```

## Quick API example

```cpp
#include <halfmesh/Mesh.h>

halfmesh::Mesh mesh;
mesh.Load("input.ply");

// QEM decimation — keep 50 % of faces
mesh.Simplify(0.5f);

mesh.Save("output.ply");
```

### Isotropic remeshing

```cpp
halfmesh::Mesh::RemeshParams params;
params.SetEdgeLength(0.02f);   // target edge length in world units
mesh.RemeshIsotropic(params);
```

### UV atlas generation

```cpp
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/Parametrize.h>

halfmesh::AtlasParams ap;
ap.resolution = 1024;

halfmesh::AtlasResult r = halfmesh::GenerateAtlas(mesh, {}, ap);
// mesh.faceTexcoords now holds packed atlas-space UVs
```

## Example CLIs

Build with `-DHALFMESH_BUILD_TOOLS=ON`.

```sh
# Decimate: reduce to 50 % of the original face count
# (target > 1 = absolute face count; optional 4th arg: aggressiveness, 0 = exact QEM)
./make/examples/decimate tests/data/mesh.ply /tmp/dec.ply 0.5

# Remesh: isotropic remeshing (auto edge length from bbox diagonal / 50)
./make/examples/remesh tests/data/mesh.ply /tmp/rem.ply

# Unwrap: generate a 1024-texel UV atlas
./make/examples/unwrap tests/data/mesh.ply /tmp/uv.ply 1024

# Smooth: 20 Taubin iterations (algo: taubin | hc, default hc)
./make/examples/smooth tests/data/mesh.ply /tmp/smooth.ply 20 taubin

# Texture bake: decimate to 30 % and rebake the source texture onto a fresh atlas
./make/examples/texturebake rebake --input textured.glb --output rebaked.glb \
    --decimation 0.3 --texture-size 4096

# Texture defrag: repack the existing UV atlas into a tighter layout
./make/examples/texturebake defrag --input textured.glb --output defragged.glb

# Fidelity: layout-independent PSNR between two textured meshes
./make/examples/texturebake fidelity --ref textured.glb --test rebaked.glb
```

All CLIs load and save both PLY and glTF/GLB — the format is picked from the
file extension (e.g. `unwrap tests/data/mesh.ply /tmp/uv.glb` writes a GLB).

## Consuming the library with CMake

After `cmake --install`, downstream projects use:

```cmake
find_package(halfmesh CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE halfmesh::halfmesh)
```

The package exports version compatibility files (`halfmeshConfigVersion.cmake`)
so `find_package(halfmesh 0.1.0 CONFIG REQUIRED)` works as expected.

## Python bindings

A pip-installable `halfmesh` package wraps the core mesh ops (repair, smooth,
simplify, close holes, remove small components, remesh) plus a `Mesh` facade
and UV-atlas `unwrap`, all numpy in/out:

```sh
pip install https://github.com/cdcseacave/halfmesh/releases/download/v0.2.0/halfmesh-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

(replace `cp312-cp312` with your interpreter's tag — wheels are published for
`cp310` through `cp313`)

```python
import halfmesh as hm

mesh = hm.Mesh()
mesh.load("input.ply")
v, f = mesh.to_arrays()
v, f = hm.repair(v, f)
v, f = hm.simplify(v, f, target=0.5)      # keep 50% of faces
hm.Mesh.from_arrays(v, f).save("output.ply")
```

See [docs/PYTHON.md](docs/PYTHON.md) for the full install matrix, API
reference, and a worked cleanup-pipeline example.

## Documentation

- [docs/PYTHON.md](docs/PYTHON.md) — the Python bindings: install, full API reference, worked example
- [docs/TESTING.md](docs/TESTING.md) — the layered testing strategy (invariants, frozen goldens, third-party cross-checks, perf guards)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — the `atlasbench` harness and results vs xatlas / libigl / pmp / CGAL / BFF
- [docs/ATLAS_SEGMENTATION_DESIGN.md](docs/ATLAS_SEGMENTATION_DESIGN.md) — design of the developable D-Charts chart segmenter
- [CHANGELOG.md](CHANGELOG.md)

## Algorithm credits

- **QEM decimation** — Garland & Heckbert (1997) "Surface Simplification Using Quadric Error Metrics".
- **Isotropic remeshing** — Botsch & Kobbelt (2004) "A Remeshing Approach to Multiresolution Modelling".
- **Liepa hole-filling** — Peter Liepa (2003) "Filling Holes in Meshes"; the min-weight
  triangulation + refine/fair structure follows [pmp-library](https://www.pmp-library.org/) (MIT).
- **HC Laplacian smoothing** — Vollmer, Mencl & Mueller (1999) "Improved Laplacian Smoothing of Noisy Surface Meshes".
- **Taubin smoothing** — Taubin (1995) "A Signal Processing Approach To Fair Surface Design".
- **Chart segmentation** — Julius, Kraevoy & Sheffer (2005) "D-Charts: Quasi-Developable Mesh Segmentation", with Lloyd relaxation after Cohen-Steiner, Alliez & Desbrun (2004) "Variational Shape Approximation".
- **ARAP flattening** — Liu et al. (2008) "A Local/Global Approach to Mesh Parameterization".
- **SLIM flattening** — Rabinovich et al. (2017) "Scalable Locally Injective Mappings" (simplified isotropic variant).
- **Atlas packing** — skyline (min-waste) bin-packing inspired by [xatlas](https://github.com/jpcy/xatlas) (MIT).

## License

MIT — see [LICENSE](LICENSE).
