# halfmesh Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an official, pip-installable `halfmesh` Python package (pybind11 extension over the existing static library) so any Python consumer — DroneDeploy's `radiance` first — can use the library directly instead of writing custom bindings.

**Architecture:** A `python/` directory holds one pybind11 binding TU plus a thin pure-Python package. A new `HALFMESH_BUILD_PYTHON` CMake option (OFF by default, like every other optional target) builds the extension module `halfmesh._halfmesh` against the existing `halfmesh::halfmesh` static library and the existing vcpkg manifest (new `python` feature → `pybind11`). A root `pyproject.toml` (scikit-build-core backend) drives that same CMake for `pip install .`, so there is exactly one build system. CI gains a bindings job (build + pytest on every push) and a `wheels.yml` workflow that builds self-contained manylinux wheels with cibuildwheel on version tags and attaches them to the GitHub Release — the artifact downstream consumers install.

**Tech Stack:** pybind11 (via vcpkg, new `python` manifest feature), scikit-build-core ≥0.10, numpy, pytest, cibuildwheel (manylinux_2_28, Linux x86_64), GitHub Actions.

**Spec:** This document's "API contract" section below is the spec; the API design was settled in review against the two known consumers: `radiance`'s mesh post-processing stage (see `/home/wii/radiance/docs/superpowers/plans/2026-08-19-halfmesh-postprocess-integration.md`, which consumes this package) and the future texturing stage. Prior art studied: `~/openMVS`'s `pyOpenMVS` (`libs/MVS/PythonWrapper.cpp`) — we keep its thin-facade idea and its opt-in build gate, and replace Boost.Python with pybind11 and "in-tree only" with pip-installable wheels.

## API contract (the spec)

Module `halfmesh` (pure-Python package re-exporting from the native `halfmesh._halfmesh`):

- `version() -> str` and `__version__` — the library version (single-sourced from `project(halfmesh VERSION …)` in `CMakeLists.txt`).
- **Array ops** — free functions on numpy arrays. Input `vertices` float32 `[N,3]`, `faces` uint32 `[M,3]` (other dtypes/layouts are force-cast). Every op returns **new** arrays (float32/uint32, C-contiguous) because half-edge construction may auto-repair (weld, re-index) non-manifold input — index stability is never promised. The GIL is released around all C++ work.
  - `repair(vertices, faces) -> (v, f)` — weld exact duplicates, drop degenerate faces, drop unreferenced vertices, fix non-manifold topology.
  - `smooth(vertices, faces, iterations, method) -> (v, f)` — `method` in `{"taubin", "hc"}`; unknown method raises `ValueError`.
  - `simplify(vertices, faces, target, aggressiveness=0.0) -> (v, f)` — `target` in (0,1) = keep-fraction, >1 = absolute face count, ==1 = no-op; `aggressiveness` 0 = exact QEM, ~7 = fast threshold sweep. `target <= 0` raises `ValueError`.
  - `close_holes(vertices, faces, max_holes=200) -> (v, f, closed: int)` — Liepa fill, smallest holes first; `max_holes` is a **count**, not a size threshold.
  - `remove_small_components(vertices, faces, min_faces) -> (v, f, removed: int)`.
  - `remesh(vertices, faces, edge_length, iterations=3) -> (v, f)` — isotropic remeshing toward the target edge length; `edge_length <= 0` raises `ValueError`.
- **`Mesh` class** — thin facade over `halfmesh::Mesh` for file-based flows and the future texturing stage:
  - `Mesh()` constructor; `Mesh.from_arrays(vertices, faces) -> Mesh` (static); `to_arrays() -> (v, f)`.
  - `load(path)` / `save(path, binary=True)` — PLY + glTF/GLB by extension; failures raise `RuntimeError`.
  - Read-only properties `n_vertices: int`, `n_faces: int`, `has_texcoords: bool`.
- **`unwrap(input_path, output_path, resolution=4096, padding=4, allow_rotation=True) -> dict`** — file-based UV-atlas generation (Load → weld prelude → `GenerateAtlas` → Save), returning `{"charts", "pages", "width", "height", "occupancy", "vertices", "faces"}`. Deliberately file-based: it reuses the proven `examples/Unwrap.cpp` flow and keeps every UV-convention subtlety (absolute-pixel in memory, normalized in PLY, half-texel offset in glTF) inside halfmesh's own `Save`.
- **Never torch.** The package depends on numpy only; torch users convert with `torch.from_numpy()` themselves, and the module never couples to the torch C++ ABI.

## Global Constraints

- `HALFMESH_BUILD_PYTHON` defaults **OFF**; the default C++ build (`cmake -S . -B make && cmake --build make`) is byte-for-byte unaffected by this plan.
- One build system: `pip install .` must drive the existing root `CMakeLists.txt` via scikit-build-core — no parallel setup.py compile line.
- Version is single-sourced from `project(halfmesh VERSION x.y.z)`; `pyproject.toml` uses scikit-build-core's regex metadata provider, never a second hardcoded number.
- All vcpkg deps linked into the extension must be position-independent: Linux builds of the python feature use the overlay triplet `cmake/triplets/x64-linux-pic.cmake` (static + `-fPIC`), and the `halfmesh` static-lib target gets `POSITION_INDEPENDENT_CODE ON` when `HALFMESH_BUILD_PYTHON=ON`.
- No `-march=*` flags anywhere in the python build (halfmesh's bit-determinism guarantee; same rule as the C++ goldens).
- Python ≥ 3.10; numpy ≥ 1.24 is the only runtime dependency.
- Wheels: Linux x86_64 manylinux_2_28, CPython 3.10–3.13, fully self-contained (vcpkg static deps compiled in — `auditwheel` must report no external libs beyond the manylinux allowlist).
- `python/binding.cpp` follows `.clang-format` (run `clang-format -i` on it; Task 4 extends the CI lint sweep to cover `python/`).
- This machine currently has **no g++/cmake installed** (`gcc-12` only, no C++ frontend): every local build step starts by installing prerequisites (Step 0 of Task 1). Ubuntu 24.04's `g++` (13+) has `<format>`, so `HALFMESH_USE_FMT` is *not* needed here or in manylinux_2_28 (gcc-toolset ≥ 13); it remains the documented escape hatch for older toolchains only.

---

### Task 1: Package skeleton — CMake option, pyproject, `version()` binding

**Files:**
- Modify: `vcpkg.json` (add `python` feature)
- Modify: `CMakeLists.txt` (option + feature wiring + `add_subdirectory(python)`)
- Create: `cmake/triplets/x64-linux-pic.cmake`
- Create: `python/CMakeLists.txt`
- Create: `python/binding.cpp`
- Create: `python/halfmesh/__init__.py`
- Create: `pyproject.toml`
- Test: `python/tests/test_version.py`

**Interfaces:**
- Consumes: the existing `halfmesh::halfmesh` static-lib target and `halfmesh_set_warnings()` from `cmake/Utils.cmake`.
- Produces: importable package `halfmesh` with `version() -> str` and `__version__: str`; the native module is `halfmesh._halfmesh` (Tasks 2–3 grow the same `python/binding.cpp`). `pip install .` works from the repo root with `VCPKG_ROOT` set.

- [ ] **Step 0: Install local build prerequisites**

```bash
sudo apt-get update && sudo apt-get install -y g++ cmake ninja-build python3-venv
python3 -m venv ~/.venvs/halfmesh && source ~/.venvs/halfmesh/bin/activate
pip install pytest numpy
# VCPKG_ROOT must point at a bootstrapped vcpkg checkout (the repo's CI does the same).
```

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_version.py`:

```python
import re

import halfmesh


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", halfmesh.version())


def test_dunder_version_matches_native():
    assert halfmesh.__version__ == halfmesh.version()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest python/tests/test_version.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'halfmesh'`.

- [ ] **Step 3: Add the `python` feature to `vcpkg.json`**

In the `"features"` object, add:

```json
"python": {
  "description": "Build the Python bindings module",
  "dependencies": ["pybind11"]
}
```

- [ ] **Step 4: Wire the option in `CMakeLists.txt`**

In the options block (before `project()`, next to `HALFMESH_BUILD_TOOLS`), add:

```cmake
option(HALFMESH_BUILD_PYTHON "Build the Python bindings module (pybind11)" OFF)
```

In the feature-wiring block (next to the `HALFMESH_BUILD_TOOLS` wiring), add:

```cmake
if(HALFMESH_BUILD_PYTHON)
	list(APPEND VCPKG_MANIFEST_FEATURES "python")
	# The extension is a shared module, so every static archive linked into it
	# (halfmesh itself and the vcpkg deps: opencv, tinyply, libjpeg, libpng, zlib)
	# must be PIC. The overlay triplet builds the vcpkg side with
	# -DCMAKE_POSITION_INDEPENDENT_CODE=ON; the halfmesh target is handled below.
	if(NOT DEFINED VCPKG_TARGET_TRIPLET AND NOT DEFINED ENV{VCPKG_DEFAULT_TRIPLET}
	   AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
		set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets" CACHE STRING "")
		set(VCPKG_TARGET_TRIPLET "x64-linux-pic" CACHE STRING "")
	endif()
endif()
```

After the library target section (right before the `Tests` section), add:

```cmake
# -----------------------------------------------------------------------
# Python bindings (pybind11 module halfmesh._halfmesh)
# -----------------------------------------------------------------------
if(HALFMESH_BUILD_PYTHON)
	set_property(TARGET halfmesh PROPERTY POSITION_INDEPENDENT_CODE ON)
	add_subdirectory(python)
endif()
```

- [ ] **Step 5: Create the PIC overlay triplet**

Create `cmake/triplets/x64-linux-pic.cmake`:

```cmake
# x64-linux (static) with position-independent code, so the static vcpkg
# archives can link into the shared Python extension module.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
```

- [ ] **Step 6: Create `python/CMakeLists.txt`**

```cmake
find_package(pybind11 CONFIG REQUIRED)

# Target name == module name: pybind11_add_module derives the PyInit__halfmesh
# entry point and the _halfmesh.<abi>.so output name from it.
pybind11_add_module(_halfmesh binding.cpp)
target_link_libraries(_halfmesh PRIVATE halfmesh::halfmesh)
halfmesh_set_warnings(_halfmesh)

# COMPONENT python: pyproject.toml installs only this component into the wheel,
# so the static lib / headers / CMake config never leak into site-packages.
install(TARGETS _halfmesh DESTINATION halfmesh COMPONENT python)
```

- [ ] **Step 7: Create the minimal `python/binding.cpp`**

```cpp
/*
* binding.cpp
*
* Copyright (c) 2026 cDc <cdc.seacave@gmail.com>
*
* This file is part of the halfmesh library, distributed under the MIT License.
* See the LICENSE file in the project root for the full license text.
*/

// pybind11 bindings for the halfmesh library — the native half of the
// pip-installable `halfmesh` package (see python/halfmesh/__init__.py).
// numpy-only by design: consumers using torch convert via torch.from_numpy().
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <halfmesh/Mesh.h>
#include <halfmesh/Version.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_halfmesh, m)
{
	m.doc() = "halfmesh — fast half-edge triangle mesh processing "
	          "(repair / smooth / simplify / holes / remesh / UV atlas)";
	m.def("version", []() { return std::string(halfmesh::Version()); },
	      "halfmesh library version string");
}
```

- [ ] **Step 8: Create `python/halfmesh/__init__.py`**

```python
"""halfmesh — fast, compact half-edge triangle mesh processing.

Python bindings over the C++20 halfmesh library. All array-taking functions
accept float32 [N,3] vertices and uint32 [M,3] faces (numpy) and return NEW
arrays: halfmesh's half-edge construction auto-repairs non-manifold input, so
vertex/face indices are never guaranteed stable across a call.
"""

from ._halfmesh import version

__version__ = version()
__all__ = ["version", "__version__"]
```

- [ ] **Step 9: Create `pyproject.toml`** (repo root)

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.12"]
build-backend = "scikit_build_core.build"

[project]
name = "halfmesh"
dynamic = ["version"]
description = "Fast, compact half-edge triangle mesh processing library"
readme = "README.md"
license = { file = "LICENSE" }
authors = [{ name = "cDc", email = "cdc.seacave@gmail.com" }]
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.urls]
Homepage = "https://github.com/cdcseacave/halfmesh"

[project.optional-dependencies]
test = ["pytest"]

[tool.scikit-build]
minimum-version = "0.10"
cmake.version = ">=3.21"
cmake.args = ["-DHALFMESH_BUILD_PYTHON=ON"]
wheel.packages = ["python/halfmesh"]
install.components = ["python"]

# Version is single-sourced from project(halfmesh VERSION x.y.z) in CMakeLists.
[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "CMakeLists.txt"
regex = 'project\(halfmesh VERSION (?P<value>[0-9]+\.[0-9]+\.[0-9]+)'
```

- [ ] **Step 10: Build, install, and run the test**

```bash
pip install . -v          # VCPKG_ROOT must be exported; first build compiles the vcpkg deps (~10-30 min cold)
pytest python/tests/test_version.py -v
```

Expected: both tests PASS; `python -c "import halfmesh; print(halfmesh.__version__)"` prints `0.1.0`.
If the link fails with `relocation ... recompile with -fPIC`, a vcpkg dep was built with a non-PIC triplet — check that CMake configured with `VCPKG_TARGET_TRIPLET=x64-linux-pic` (delete the vcpkg `buildtrees`/`installed` cache for the old triplet and re-run).

- [ ] **Step 11: Verify the default C++ build is untouched**

```bash
cmake -S . -B /tmp/hm-default-check   # no HALFMESH_BUILD_PYTHON
grep -c pybind11 /tmp/hm-default-check/vcpkg-manifest-install.log || true
```

Expected: configure succeeds without requesting the `python` vcpkg feature (no pybind11 install).

- [ ] **Step 12: Commit**

```bash
git add vcpkg.json CMakeLists.txt cmake/triplets/x64-linux-pic.cmake python pyproject.toml
git commit -m "feat(python): pip-installable package skeleton (pybind11 + scikit-build-core)"
```

---

### Task 2: Array ops — repair / smooth / simplify / close_holes / remove_small_components / remesh

**Files:**
- Modify: `python/binding.cpp`
- Modify: `python/halfmesh/__init__.py`
- Test: `python/tests/test_ops.py`

**Interfaces:**
- Consumes: Task 1's module skeleton; C++ API: `Mesh::RemoveDuplicateVertices(0) / RemoveDegenerateFaces(0.f) / RemoveUnreferencedVertices() / FixNonManifold()`, `Mesh::SmoothTaubin(int) / SmoothHCLaplacian(int)`, `Mesh::Simplify(float, float, float)`, `Mesh::CloseHoles(unsigned) -> unsigned`, `Mesh::RemoveSmallComponents(unsigned) -> unsigned`, `Mesh::RemeshIsotropic(RemeshParams)` with `RemeshParams::SetEdgeLength(float)` / `.iterations`.
- Produces: the six array ops from the API contract, importable as `halfmesh.repair` etc. Task 3 reuses the `MeshFromArrays` / `ArraysFromMesh` helpers.

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_ops.py`:

```python
import numpy as np
import pytest

import halfmesh as hm


def _grid_mesh(n=64, noise=0.05, seed=0):
    """n x n vertex grid over [0,1]^2 with gaussian z-noise; 2*(n-1)^2 triangles."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    z = rng.normal(0.0, noise, size=xs.shape)
    v = np.stack([xs, ys, z], axis=-1).reshape(-1, 3).astype(np.float32)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces.append((i, i + 1, i + n))
            faces.append((i + 1, i + n + 1, i + n))
    return v, np.asarray(faces, dtype=np.uint32)


def _cube_mesh():
    """Unit cube, 8 vertices / 12 triangles, closed and manifold."""
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float32,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.uint32,
    )
    return v, f


def _euler_characteristic(v, f):
    edges = {
        tuple(sorted(e))
        for tri in f
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    }
    return len(v) - len(edges) + len(f)


def test_repair_welds_duplicate_vertices():
    v, f = _cube_mesh()
    v2 = np.concatenate([v, v]).astype(np.float32)
    f2 = f.copy()
    f2[6:] += 8  # half the faces point at the duplicate block
    rv, rf = hm.repair(v2, f2.astype(np.uint32))
    assert rv.shape == (8, 3) and rf.shape == (12, 3)
    assert rv.dtype == np.float32 and rf.dtype == np.uint32
    assert rv.flags["C_CONTIGUOUS"] and rf.flags["C_CONTIGUOUS"]


def test_ops_accept_castable_dtypes():
    v, f = _cube_mesh()
    rv, rf = hm.repair(v.astype(np.float64), f.astype(np.int64))
    assert rv.dtype == np.float32 and rf.dtype == np.uint32


def test_smooth_taubin_reduces_noise_without_shrinking():
    v, f = _grid_mesh()
    sv, sf = hm.smooth(v, f, 20, "taubin")
    assert sf.shape == f.shape  # smoothing moves vertices, never topology
    assert sv[:, 2].std() < 0.5 * v[:, 2].std()
    assert abs(sv[:, 0].max() - 1.0) < 0.05


def test_smooth_rejects_unknown_method():
    v, f = _grid_mesh(n=8)
    with pytest.raises(ValueError):
        hm.smooth(v, f, 5, "banana")


def test_simplify_hits_a_ratio_target():
    v, f = _grid_mesh(noise=0.0)
    sv, sf = hm.simplify(v, f, 0.5, 0.0)
    assert len(sf) <= int(0.55 * len(f))
    assert len(sf) >= int(0.30 * len(f))


def test_simplify_hits_an_absolute_target():
    v, f = _grid_mesh(noise=0.0)
    sv, sf = hm.simplify(v, f, 500.0, 7.0)
    assert len(sf) <= 800  # fast sweep may stop above target, but nearby


def test_simplify_rejects_nonpositive_target():
    v, f = _cube_mesh()
    with pytest.raises(ValueError):
        hm.simplify(v, f, 0.0)


def test_close_holes_makes_an_open_cube_watertight():
    v, f = _cube_mesh()
    rv, rf, closed = hm.close_holes(v, f[:-1].copy(), 200)
    assert closed == 1
    assert _euler_characteristic(rv, rf) == 2  # closed genus-0 surface


def test_remove_small_components_drops_a_floater():
    v, f = _cube_mesh()
    fv = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]], dtype=np.float32)
    v2 = np.concatenate([v, fv])
    f2 = np.concatenate([f, np.array([[8, 9, 10]], dtype=np.uint32)])
    rv, rf, removed = hm.remove_small_components(v2, f2, 4)
    assert removed == 1
    assert len(rf) == 12


def test_remesh_coarsens_toward_target_edge_length():
    v, f = _grid_mesh(noise=0.0)  # grid spacing 1/63
    rv, rf = hm.remesh(v, f, 4.0 / 63.0, 3)
    assert 0 < len(rf) < 0.5 * len(f)  # ~4x longer edges => far fewer faces
    assert _euler_characteristic(rv, rf) == 1  # still one disk-topology patch


def test_remesh_rejects_nonpositive_edge_length():
    v, f = _cube_mesh()
    with pytest.raises(ValueError):
        hm.remesh(v, f, 0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest python/tests/test_ops.py -v`
Expected: FAIL with `AttributeError: module 'halfmesh' has no attribute 'repair'` (import-time, from `__init__.py` once updated — write the binding first, then the re-export; at this point the errors are `AttributeError` on `hm.repair`).

- [ ] **Step 3: Implement the ops in `python/binding.cpp`**

Add below the includes (keep `version()` in the module block):

```cpp
#include <cstring>
#include <stdexcept>

using halfmesh::Mesh;

namespace {

using VertArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using FaceArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

Mesh MeshFromArrays(const VertArray& v, const FaceArray& f)
{
	if (v.ndim() != 2 || v.shape(1) != 3)
		throw py::value_error("vertices must have shape [N,3] (float32)");
	if (f.ndim() != 2 || f.shape(1) != 3)
		throw py::value_error("faces must have shape [M,3] (uint32)");
	Mesh m;
	// Mesh::Vertex / Mesh::Face are static_asserted memcpy-compatible
	// (src/MeshIO.cpp), so bulk-copy the buffers.
	m.vertices.resize(static_cast<size_t>(v.shape(0)));
	std::memcpy(m.vertices.data(), v.data(), sizeof(float) * 3 * m.vertices.size());
	m.faces.resize(static_cast<size_t>(f.shape(0)));
	std::memcpy(m.faces.data(), f.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return m;
}

py::tuple ArraysFromMesh(const Mesh& m)
{
	py::array_t<float> v({static_cast<py::ssize_t>(m.vertices.size()), py::ssize_t(3)});
	std::memcpy(v.mutable_data(), m.vertices.data(), sizeof(float) * 3 * m.vertices.size());
	py::array_t<uint32_t> f({static_cast<py::ssize_t>(m.faces.size()), py::ssize_t(3)});
	std::memcpy(f.mutable_data(), m.faces.data(), sizeof(uint32_t) * 3 * m.faces.size());
	return py::make_tuple(std::move(v), std::move(f));
}

// The recommended pre-pass from the Simplify header docs: dissolves the phantom
// topology that blocks collapses and makes every later half-edge build
// non-mutating.
void RepairInPlace(Mesh& m)
{
	m.RemoveDuplicateVertices(0);
	m.RemoveDegenerateFaces(0.f);
	m.RemoveUnreferencedVertices();
	m.FixNonManifold();
}

} // namespace
```

Add inside `PYBIND11_MODULE(_halfmesh, m)`:

```cpp
	m.def("repair", [](const VertArray& v, const FaceArray& f) {
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			RepairInPlace(mesh);
		}
		return ArraysFromMesh(mesh);
	}, py::arg("vertices"), py::arg("faces"),
	   "Weld duplicates, drop degenerate faces and unreferenced vertices, fix non-manifold topology.");

	m.def("smooth", [](const VertArray& v, const FaceArray& f, int iterations,
	                   const std::string& method) {
		if (method != "taubin" && method != "hc")
			throw py::value_error("smooth method must be 'taubin' or 'hc', got '" + method + "'");
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			if (method == "taubin")
				mesh.SmoothTaubin(iterations);
			else
				mesh.SmoothHCLaplacian(iterations);
		}
		return ArraysFromMesh(mesh);
	}, py::arg("vertices"), py::arg("faces"), py::arg("iterations"), py::arg("method"),
	   "Smooth vertex positions: 'taubin' (band-pass, ~zero shrink) or 'hc' (anti-shrink Laplacian).");

	m.def("simplify", [](const VertArray& v, const FaceArray& f, float target,
	                     float aggressiveness) {
		if (target <= 0.f)
			throw py::value_error("simplify target must be > 0 (fraction in (0,1) or absolute count > 1)");
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			mesh.Simplify(target, /*minEdgeLength=*/0.f, aggressiveness);
		}
		return ArraysFromMesh(mesh);
	}, py::arg("vertices"), py::arg("faces"), py::arg("target"), py::arg("aggressiveness") = 0.f,
	   "QEM edge-collapse decimation. target in (0,1) = keep-fraction, > 1 = absolute face count.");

	m.def("close_holes", [](const VertArray& v, const FaceArray& f, unsigned max_holes) {
		Mesh mesh = MeshFromArrays(v, f);
		unsigned closed = 0;
		{
			py::gil_scoped_release release;
			closed = mesh.CloseHoles(max_holes);
		}
		py::tuple vf = ArraysFromMesh(mesh);
		return py::make_tuple(vf[0], vf[1], closed);
	}, py::arg("vertices"), py::arg("faces"), py::arg("max_holes") = 200u,
	   "Liepa hole filling (fill + refine + fair), smallest holes first; max_holes is a count.");

	m.def("remove_small_components", [](const VertArray& v, const FaceArray& f,
	                                    unsigned min_faces) {
		Mesh mesh = MeshFromArrays(v, f);
		unsigned removed = 0;
		{
			py::gil_scoped_release release;
			removed = mesh.RemoveSmallComponents(min_faces);
			mesh.RemoveUnreferencedVertices();
		}
		py::tuple vf = ArraysFromMesh(mesh);
		return py::make_tuple(vf[0], vf[1], removed);
	}, py::arg("vertices"), py::arg("faces"), py::arg("min_faces"),
	   "Remove connected components with fewer than min_faces faces.");

	m.def("remesh", [](const VertArray& v, const FaceArray& f, float edge_length,
	                   int iterations) {
		if (edge_length <= 0.f)
			throw py::value_error("remesh edge_length must be > 0");
		if (iterations <= 0)
			throw py::value_error("remesh iterations must be > 0");
		Mesh mesh = MeshFromArrays(v, f);
		{
			py::gil_scoped_release release;
			Mesh::RemeshParams params;
			params.SetEdgeLength(edge_length);
			params.iterations = iterations;
			mesh.RemeshIsotropic(params);
		}
		return ArraysFromMesh(mesh);
	}, py::arg("vertices"), py::arg("faces"), py::arg("edge_length"), py::arg("iterations") = 3,
	   "Isotropic remeshing toward a uniform target edge length (world units).");
```

Note: exceptions thrown while the GIL is released are fine — pybind11 re-acquires the GIL to translate them; `py::value_error` maps to Python `ValueError`.

- [ ] **Step 4: Update `python/halfmesh/__init__.py`**

```python
from ._halfmesh import (
    close_holes,
    remesh,
    remove_small_components,
    repair,
    simplify,
    smooth,
    version,
)

__version__ = version()
__all__ = [
    "close_holes",
    "remesh",
    "remove_small_components",
    "repair",
    "simplify",
    "smooth",
    "version",
    "__version__",
]
```

(Keep the module docstring from Task 1 above the imports.)

- [ ] **Step 5: Rebuild, test, format, commit**

```bash
pip install . -v --no-build-isolation
pytest python/tests -v
clang-format -i python/binding.cpp && git diff --exit-code python/binding.cpp
git add python/binding.cpp python/halfmesh/__init__.py python/tests/test_ops.py
git commit -m "feat(python): array ops - repair, smooth, simplify, close_holes, components, remesh"
```

Expected: all tests PASS. If `test_simplify_hits_a_ratio_target`'s floor assert trips, loosen the floor (0.30 → 0.20), not the ceiling — the ceiling is the contract.

---

### Task 3: `Mesh` class + file-based `unwrap`

**Files:**
- Modify: `python/binding.cpp`
- Modify: `python/halfmesh/__init__.py`
- Test: `python/tests/test_mesh.py`

**Interfaces:**
- Consumes: Task 2's `MeshFromArrays` / `ArraysFromMesh` helpers; C++ API: `Mesh::Load/Save`, `Mesh::HasTextureCoordinates()`, `GenerateAtlas(Mesh&, ParametrizeParams, AtlasParams) -> AtlasResult` (`halfmesh/Parametrize.h`, `halfmesh/AtlasCharting.h`).
- Produces: `halfmesh.Mesh` (constructor, `from_arrays`, `to_arrays`, `load`, `save`, `n_vertices`, `n_faces`, `has_texcoords`) and `halfmesh.unwrap(input_path, output_path, resolution=4096, padding=4, allow_rotation=True) -> dict` — the exact function radiance's `radiance/mesh/unwrap.py` calls.

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_mesh.py`:

```python
import numpy as np
import pytest

import halfmesh as hm


def _cube_arrays():
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float32,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.uint32,
    )
    return v, f


def test_mesh_roundtrips_arrays_and_ply(tmp_path):
    v, f = _cube_arrays()
    mesh = hm.Mesh.from_arrays(v, f)
    assert mesh.n_vertices == 8 and mesh.n_faces == 12
    assert not mesh.has_texcoords

    path = str(tmp_path / "cube.ply")
    mesh.save(path)
    loaded = hm.Mesh()
    loaded.load(path)
    lv, lf = loaded.to_arrays()
    np.testing.assert_allclose(lv, v)
    assert lf.shape == (12, 3)


def test_mesh_load_raises_on_missing_file(tmp_path):
    mesh = hm.Mesh()
    with pytest.raises(RuntimeError):
        mesh.load(str(tmp_path / "nope.ply"))


def test_unwrap_generates_a_packed_atlas(tmp_path):
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)

    out = str(tmp_path / "cube_uv.ply")
    meta = hm.unwrap(src, out, resolution=1024, padding=2)
    assert meta["charts"] >= 1
    assert meta["pages"] >= 1
    assert 0.0 < meta["occupancy"] <= 1.0
    assert meta["faces"] == 12

    unwrapped = hm.Mesh()
    unwrapped.load(out)
    assert unwrapped.has_texcoords


def test_unwrap_raises_on_missing_input(tmp_path):
    with pytest.raises(RuntimeError):
        hm.unwrap(str(tmp_path / "nope.ply"), str(tmp_path / "out.ply"))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest python/tests/test_mesh.py -v`
Expected: FAIL with `AttributeError: module 'halfmesh' has no attribute 'Mesh'`.

- [ ] **Step 3: Add the bindings to `python/binding.cpp`**

Add the includes:

```cpp
#include <halfmesh/Parametrize.h>
#include <halfmesh/AtlasCharting.h>
#include <halfmesh/AtlasPacking.h>
```

Add inside `PYBIND11_MODULE(_halfmesh, m)`:

```cpp
	py::class_<Mesh>(m, "Mesh",
	                 "Triangle mesh facade over halfmesh::Mesh (PLY / glTF / GLB I/O).")
	    .def(py::init<>())
	    .def_static("from_arrays", [](const VertArray& v, const FaceArray& f) {
		    return MeshFromArrays(v, f);
	    }, py::arg("vertices"), py::arg("faces"))
	    .def("to_arrays", [](const Mesh& self) { return ArraysFromMesh(self); },
	         "Return (vertices float32 [N,3], faces uint32 [M,3]) copies.")
	    .def("load", [](Mesh& self, const std::string& path) {
		    bool ok;
		    {
			    py::gil_scoped_release release;
			    ok = self.Load(path);
		    }
		    if (!ok)
			    throw std::runtime_error("Mesh.load: failed to load '" + path + "'");
	    }, py::arg("path"), "Load a .ply / .gltf / .glb mesh (format from extension).")
	    .def("save", [](const Mesh& self, const std::string& path, bool binary) {
		    bool ok;
		    {
			    py::gil_scoped_release release;
			    ok = self.Save(path, binary);
		    }
		    if (!ok)
			    throw std::runtime_error("Mesh.save: failed to save '" + path + "'");
	    }, py::arg("path"), py::arg("binary") = true,
	       "Save as .ply / .gltf / .glb (format from extension).")
	    .def_property_readonly("n_vertices", [](const Mesh& self) { return self.vertices.size(); })
	    .def_property_readonly("n_faces", [](const Mesh& self) { return self.faces.size(); })
	    .def_property_readonly("has_texcoords", &Mesh::HasTextureCoordinates)
	    .def("__repr__", [](const Mesh& self) {
		    return "<halfmesh.Mesh: " + std::to_string(self.vertices.size()) + " vertices, " +
		           std::to_string(self.faces.size()) + " faces>";
	    });

	m.def("unwrap", [](const std::string& input_path, const std::string& output_path,
	                   unsigned resolution, unsigned padding, bool allow_rotation) {
		Mesh mesh;
		unsigned charts = 0;
		halfmesh::AtlasResult result;
		{
			py::gil_scoped_release release;
			if (!mesh.Load(input_path))
				throw std::runtime_error("unwrap: failed to load '" + input_path + "'");
			// Weld + clean first (examples/Unwrap.cpp preamble): unwelded input
			// makes every edge a boundary and SegmentCharts fragments into one
			// chart per face. Lossless: the atlas regenerates the UVs anyway.
			mesh.RemoveDuplicateVertices(0);
			mesh.RemoveDegenerateFaces(0.f);
			mesh.RemoveUnreferencedVertices();

			halfmesh::ParametrizeParams pparams; // defaults tuned for MVS-like meshes
			halfmesh::AtlasParams aparams;
			aparams.resolution = resolution;
			aparams.padding = padding;
			aparams.allowRotation = allow_rotation;
			result = halfmesh::GenerateAtlas(mesh, pparams, aparams);
			charts = static_cast<unsigned>(result.chartPage.size());

			if (!mesh.Save(output_path))
				throw std::runtime_error("unwrap: failed to save '" + output_path + "'");
		}
		py::dict meta;
		meta["charts"] = charts;
		meta["pages"] = result.numPages;
		meta["width"] = result.width;
		meta["height"] = result.height;
		meta["occupancy"] = result.occupancy;
		meta["vertices"] = mesh.vertices.size();
		meta["faces"] = mesh.faces.size();
		return meta;
	}, py::arg("input_path"), py::arg("output_path"), py::arg("resolution") = 4096u,
	   py::arg("padding") = 4u, py::arg("allow_rotation") = true,
	   "Generate a packed UV atlas: load -> weld -> GenerateAtlas -> save. "
	   "Returns {charts, pages, width, height, occupancy, vertices, faces}.");
```

- [ ] **Step 4: Extend the re-exports in `python/halfmesh/__init__.py`**

Add `Mesh` and `unwrap` to the import list and `__all__` (alphabetical order, `Mesh` first as the sole class).

- [ ] **Step 5: Rebuild, test, format, commit**

```bash
pip install . -v --no-build-isolation
pytest python/tests -v
clang-format -i python/binding.cpp && git diff --exit-code python/binding.cpp
git add python/binding.cpp python/halfmesh/__init__.py python/tests/test_mesh.py
git commit -m "feat(python): Mesh facade class + file-based unwrap (GenerateAtlas)"
```

---

### Task 4: CI — bindings job on every push, manylinux wheels on tags

**Files:**
- Modify: `.github/workflows/ci.yml` (add one job; extend the clang-format sweep to `python/`)
- Create: `.github/workflows/wheels.yml`
- Modify: `pyproject.toml` (add `[tool.cibuildwheel]`)

**Interfaces:**
- Consumes: Tasks 1–3 (a package that builds and passes pytest).
- Produces: green `python-bindings` CI job; on pushing a tag `v*`, `wheels.yml` attaches `halfmesh-<ver>-cp3XX-cp3XX-manylinux_2_28_x86_64.whl` files (CPython 3.10–3.13) to the GitHub Release — the artifact the radiance plan's Task 1 installs by URL.

- [ ] **Step 1: Add the bindings job to `ci.yml`**

Mirror the repo's existing "Set up vcpkg" step (see the clang-tidy job) for checkout/bootstrap/caching, then:

```yaml
  # Job: python-bindings — build the pip package against vcpkg deps and run
  # its pytest suite. Ubuntu 24.04: g++ >= 13 has <format>, no fmt needed.
  python-bindings:
    name: Python bindings (pip install + pytest)
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      # ... copy the existing vcpkg setup/cache steps from the lint job here,
      #     exporting VCPKG_ROOT ...
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build and install the package
        run: python -m pip install -v '.[test]'
      - name: Run the binding tests
        run: python -m pytest python/tests -q
```

- [ ] **Step 2: Extend the lint sweep**

In the ci.yml clang-format job, add `python` to the `find src include tests examples …` directory list so `python/binding.cpp` is format-checked like every other TU.

- [ ] **Step 3: Add the cibuildwheel config to `pyproject.toml`**

```toml
[tool.cibuildwheel]
build = ["cp310-manylinux_x86_64", "cp311-manylinux_x86_64", "cp312-manylinux_x86_64", "cp313-manylinux_x86_64"]
manylinux-x86_64-image = "manylinux_2_28"
test-requires = ["pytest"]
test-command = "pytest {project}/python/tests -q"

[tool.cibuildwheel.linux]
# vcpkg bootstraps inside the manylinux container; deps build static+PIC via
# the x64-linux-pic overlay triplet (auto-selected by CMakeLists.txt).
before-all = """
dnf install -y curl zip unzip tar git perl-IPC-Cmd nasm &&
git clone --depth 1 https://github.com/microsoft/vcpkg /opt/vcpkg &&
/opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics
"""
environment = { VCPKG_ROOT = "/opt/vcpkg" }
```

- [ ] **Step 4: Create `.github/workflows/wheels.yml`**

```yaml
# Build self-contained manylinux wheels on version tags and attach them to the
# GitHub Release. Consumers (e.g. radiance) install these by URL — see README
# "Python bindings". Linux x86_64 only for now (the one deployment target);
# macOS/Windows wheels are a recorded follow-up.
name: Wheels

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build-wheels:
    name: manylinux_2_28 x86_64 wheels
    runs-on: ubuntu-24.04
    timeout-minutes: 180   # cold vcpkg build of opencv dominates
    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.21.3

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-linux-x86_64
          path: wheelhouse/*.whl

      - name: Attach wheels to the GitHub Release
        if: startsWith(github.ref, 'refs/tags/')
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release view "$GITHUB_REF_NAME" --repo "$GITHUB_REPOSITORY" >/dev/null 2>&1 \
            || gh release create "$GITHUB_REF_NAME" --repo "$GITHUB_REPOSITORY" --generate-notes
          gh release upload "$GITHUB_REF_NAME" wheelhouse/*.whl --clobber --repo "$GITHUB_REPOSITORY"
```

- [ ] **Step 5: Verify locally what CI will do, then commit and watch CI**

```bash
# fast local proxy for the CI job (the wheel workflow itself is verified on the tag):
pip install '.[test]' -v && pytest python/tests -q
git add .github/workflows/ci.yml .github/workflows/wheels.yml pyproject.toml
git commit -m "ci: python-bindings job + manylinux wheel builds on version tags"
git push && gh run watch
```

Expected: the `python-bindings` job passes. Then dry-run the wheel workflow once via `gh workflow run wheels.yml` (workflow_dispatch path — builds and uploads artifacts without needing a tag) and confirm:
- all four wheels build and their in-container pytest passes;
- `auditwheel show wheelhouse/*cp312*.whl` (run locally on the downloaded artifact) reports `manylinux_2_28` compliance with no vendored external libs — everything is statically linked.

---

### Task 5: Docs, version bump, and the release that radiance consumes

**Files:**
- Create: `docs/PYTHON.md`
- Modify: `README.md` (Python section + docs list)
- Modify: `docs/FEATURES.md` (pointer line)
- Modify: `CHANGELOG.md`, `CMakeLists.txt` (version 0.1.0 → 0.2.0), `vcpkg.json` (version field)

**Interfaces:**
- Consumes: everything above.
- Produces: tagged release `v0.2.0` with wheels attached — the exact artifact the radiance integration plan pins.

- [ ] **Step 1: Write `docs/PYTHON.md`**

Content requirements:
1. Install matrix: `pip install <wheel-url from the GitHub release>` (primary), `pip install .` from a checkout with `VCPKG_ROOT` set (source build; note the cold vcpkg build time and the `x64-linux-pic` triplet auto-selection).
2. Full API reference: every function/class from the "API contract" section above with signatures, parameter semantics (`simplify` dual-magnitude target, `close_holes` count-not-size, Taubin-vs-HC iteration guidance), and the "returns new arrays, indices are not stable" warning.
3. A worked example: load noisy PLY → `repair` → `smooth` → `simplify` → `close_holes` → save → `unwrap`.
4. Threading/GIL note: heavy ops release the GIL; halfmesh's own thread pool parallelizes internally — don't wrap calls in Python-level thread pools.
5. What is deliberately absent (torch tensors, texture baking — see out-of-scope).

- [ ] **Step 2: Update `README.md` and `docs/FEATURES.md`**

- README: add a `## Python bindings` section after "Consuming the library with CMake" with the pip install one-liner, a 6-line quick example (`import halfmesh as hm; v, f = hm.repair(v, f); …`), and a link to `docs/PYTHON.md`. Add `docs/PYTHON.md` to the Documentation list.
- `docs/FEATURES.md`: one pointer line in the intro: the full pipeline is also available from Python — see `docs/PYTHON.md`.

- [ ] **Step 3: Bump the version and changelog**

- `CMakeLists.txt`: `project(halfmesh VERSION 0.2.0 LANGUAGES CXX)` (pyproject picks it up via the regex provider — nothing else to touch).
- `vcpkg.json`: `"version": "0.2.0"`.
- `CHANGELOG.md`: `0.2.0 — Python bindings: pip-installable halfmesh package (repair/smooth/simplify/close_holes/remove_small_components/remesh, Mesh class, unwrap); manylinux wheels on releases.`

- [ ] **Step 4: Full gate, tag, release**

```bash
pip install '.[test]' -v && pytest python/tests -q
cmake -S . -B make -DHALFMESH_BUILD_TESTS=ON && cmake --build make -j && ctest --test-dir make -C Release --output-on-failure
git add -A && git commit -m "docs(python): PYTHON.md + README section; release 0.2.0"
git push
git tag v0.2.0 && git push origin v0.2.0     # triggers wheels.yml → wheels attached to the release
gh run watch
```

Expected: release `v0.2.0` exists with four `.whl` assets. Record the wheel URL pattern
`https://github.com/cdcseacave/halfmesh/releases/download/v0.2.0/halfmesh-0.2.0-cp3XX-cp3XX-manylinux_2_28_x86_64.whl`
— the radiance plan's Task 1 consumes it.

---

## Deliberately out of scope (record, don't do)

- **PyPI publishing**: wheels ship via GitHub Releases for now; adding a `pypa/gh-action-pypi-publish` step later is trivial once the name is registered.
- **macOS / Windows wheels**: no current consumer; cibuildwheel makes them an incremental `build` matrix addition when needed.
- **Texture-bake bindings** (`BakeAtlas` / `RebakeTexture` / `SourceResolver`): needs image-array marshalling and a Python-subclassable resolver — design it together with radiance's texturing stage, in its own plan.
- **Torch tensor overloads**: never — the numpy boundary is the point (no torch ABI coupling).
- **openMVS interop bindings** (`InteropOpenMVS.h`): C++-side concern.
- **vcpkg binary caching in the wheel workflow** (GH Actions cache → ~5 min warm builds instead of ~45 min cold): worthwhile optimization, separate change.
