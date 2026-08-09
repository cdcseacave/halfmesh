# halfmesh Python cross-check

Independent, non-blocking validation of halfmesh I/O output using external
Python and Node.js tools (docs/TESTING.md §1 layer-3, §7 CI job).

## What it does

| Step | Tool | Checks |
|------|------|--------|
| 1 | halfmesh CLIs | Run `unwrap` → `uv.glb`; `decimate` → `dec.ply` |
| 2 | Khronos glTF-Validator (`npx gltf-validator`) | **0 errors** on `uv.glb` (key conformance check) |
| 3 | pygltflib | GLB has mesh/primitive, POSITION + TEXCOORD_0 accessors, positive counts |
| 4 | trimesh | PLY vertex/face counts > 0; decimated < original; bbox consistent; no NaN/Inf |
| 5 | pymeshlab / open3d | (optional) vertex/face counts > 0 |

Each tool is **skip-if-absent** — the script exits 0 when all available tools pass.

## Quick start

```bash
# From repo root — venv in make/ (already gitignored)
python3 -m venv make/python-venv
make/python-venv/bin/pip install -r tests/python/requirements.txt
make/python-venv/bin/python tests/python/crosscheck.py
```

The script auto-detects the repo root, build directory (`make/`), and data
directory (`tests/data/`).  Override with flags or environment variables:

```bash
python crosscheck.py \
  --build-dir /path/to/build \
  --data-dir  /path/to/data \
  --tmp-dir   /tmp/halfmesh-xcheck
```

Or via environment:

```bash
HALFMESH_BUILD_DIR=/path/to/build \
HALFMESH_DATA_DIR=/path/to/data \
python tests/python/crosscheck.py
```

## Prerequisites

- **C++ build with tools**: `cmake -S . -B make -DHALFMESH_BUILD_TOOLS=ON && cmake --build make -j`
- **tests/data/mesh.ply**: the committed challenge fixture mesh
- **Node.js ≥ 18 + npx**: for the Khronos glTF-Validator
- **Python 3.10+**: 3.12 recommended in CI; 3.14 on local dev machines
- Optional: `pymeshlab`, `open3d` (may lack wheels on Python 3.14)

## Real-world validation

If `tests/data/mesh_roi_crop_1.textured.glb` is present (gitignored, not committed),
the script also validates it with the glTF-Validator and pygltflib.

## CI integration

Wired as the `python-crosscheck` job in `.github/workflows/ci.yml` — a
**separate, non-blocking** job (`continue-on-error: true`) that runs after the
C++ build.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All available tools passed |
| 1 | At least one tool reported a failure |
