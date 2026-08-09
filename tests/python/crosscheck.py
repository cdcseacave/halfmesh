#!/usr/bin/env python3
"""
halfmesh Python cross-check job  (docs/TESTING.md §1 layer-3, §7 CI)
=====================================================================
Independently validates halfmesh I/O output using external Python/Node tools.
Each tool is skip-if-absent — the script exits 0 when all *available* tools pass.

Usage
-----
  python crosscheck.py [--build-dir PATH] [--data-dir PATH] [--tmp-dir PATH]

Defaults:
  --build-dir  <repo>/make
  --data-dir   <repo>/data
  --tmp-dir    <repo>/make/crosscheck-tmp

Environment variables (override flags):
  HALFMESH_BUILD_DIR, HALFMESH_DATA_DIR, HALFMESH_TMP_DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS   = "[PASS]"
FAIL   = "[FAIL]"
SKIP   = "[SKIP]"
INFO   = "[INFO]"
WARN   = "[WARN]"

_failures: list[str] = []
_skips:    list[str] = []


def _log(tag: str, msg: str) -> None:
    print(f"{tag} {msg}", flush=True)


def record_fail(msg: str) -> None:
    _log(FAIL, msg)
    _failures.append(msg)


def record_skip(reason: str) -> None:
    _log(SKIP, reason)
    _skips.append(reason)


def record_pass(msg: str) -> None:
    _log(PASS, msg)


def assert_true(cond: bool, msg: str) -> None:
    if cond:
        record_pass(msg)
    else:
        record_fail(msg)


def run_cmd(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Locate the repo root by walking up from this file.

    The repo root is the highest ancestor directory containing both
    CMakeLists.txt and a recognisable project marker (vcpkg.json, .git, or
    the halfmesh include tree).  We can't just stop at the first
    CMakeLists.txt because tests/ has one too.
    """
    here = Path(__file__).resolve()
    best: Optional[Path] = None
    for p in [here, *here.parents]:
        cml = p / "CMakeLists.txt"
        if not cml.exists():
            continue
        # Prefer the topmost CMakeLists that looks like a project root
        is_root = (
            (p / "vcpkg.json").exists()
            or (p / ".git").exists()
            or (p / "include" / "halfmesh").is_dir()
        )
        if is_root:
            best = p
    if best:
        return best
    return here.parent.parent.parent  # fallback: tests/python/../../..


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--build-dir",
        default=os.environ.get("HALFMESH_BUILD_DIR", str(repo / "make")),
    )
    p.add_argument(
        "--data-dir",
        default=os.environ.get("HALFMESH_DATA_DIR", str(repo / "tests" / "data")),
    )
    p.add_argument(
        "--tmp-dir",
        default=os.environ.get("HALFMESH_TMP_DIR", str(repo / "make" / "crosscheck-tmp")),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Produce halfmesh artifacts
# ---------------------------------------------------------------------------

def step_produce_artifacts(
    build_dir: Path,
    data_dir: Path,
    tmp_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Run the halfmesh CLIs to produce:
      - uv.glb  (unwrap tests/data/mesh.ply)
      - dec.ply (decimate tests/data/mesh.ply 0.5)

    Returns (glb_path, ply_path); either may be None if the CLI failed.
    """
    print(f"\n{'='*60}")
    print("STEP 1 — Produce halfmesh artifacts")
    print(f"{'='*60}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    mesh_ply = data_dir / "mesh.ply"
    unwrap_bin = build_dir / "examples" / "unwrap"
    decimate_bin = build_dir / "examples" / "decimate"

    if not mesh_ply.exists():
        record_fail(f"Input mesh not found: {mesh_ply}")
        return None, None

    glb_path: Optional[Path] = tmp_dir / "uv.glb"
    ply_path: Optional[Path] = tmp_dir / "dec.ply"

    # --- unwrap → GLB ---
    if not unwrap_bin.exists():
        record_skip(f"unwrap binary not found at {unwrap_bin}; skip GLB generation")
        glb_path = None
    else:
        _log(INFO, f"Running: {unwrap_bin} {mesh_ply} {glb_path}")
        try:
            r = run_cmd([str(unwrap_bin), str(mesh_ply), str(glb_path)], timeout=300)
            if r.returncode != 0:
                record_fail(f"unwrap exited {r.returncode}: {r.stderr.strip()}")
                glb_path = None
            else:
                _log(INFO, r.stdout.strip())
                assert_true(glb_path.exists() and glb_path.stat().st_size > 0,
                            f"GLB produced: {glb_path} ({glb_path.stat().st_size if glb_path.exists() else 0} bytes)")
        except subprocess.TimeoutExpired:
            record_fail("unwrap timed out after 300s")
            glb_path = None

    # --- decimate → PLY ---
    if not decimate_bin.exists():
        record_skip(f"decimate binary not found at {decimate_bin}; skip PLY generation")
        ply_path = None
    else:
        _log(INFO, f"Running: {decimate_bin} {mesh_ply} {ply_path} 0.5")
        try:
            r = run_cmd([str(decimate_bin), str(mesh_ply), str(ply_path), "0.5"], timeout=120)
            if r.returncode != 0:
                record_fail(f"decimate exited {r.returncode}: {r.stderr.strip()}")
                ply_path = None
            else:
                _log(INFO, r.stdout.strip())
                assert_true(ply_path.exists() and ply_path.stat().st_size > 0,
                            f"Decimated PLY produced: {ply_path}")
        except subprocess.TimeoutExpired:
            record_fail("decimate timed out after 120s")
            ply_path = None

    return glb_path, ply_path


# ---------------------------------------------------------------------------
# Step 2 — Khronos glTF-Validator
# ---------------------------------------------------------------------------

def _find_node_validator() -> Optional[Path]:
    """
    Find the gltf_validate.js Node.js wrapper bundled next to this script.
    Also confirms `node` is on PATH.
    Returns Path to the wrapper, or None if node/wrapper unavailable.
    """
    wrapper = Path(__file__).with_name("gltf_validate.js")
    if not wrapper.exists():
        record_skip(f"gltf_validate.js not found at {wrapper}; glTF-Validator skipped")
        return None
    node = shutil.which("node")
    if node is None:
        record_skip("node not on PATH; glTF-Validator skipped")
        return None
    # Verify node_modules/gltf-validator is installed next to the wrapper
    nm = wrapper.parent / "node_modules" / "gltf-validator"
    if not nm.exists():
        record_skip(
            f"node_modules/gltf-validator not found at {nm}. "
            "Run `npm install` in tests/python/ first."
        )
        return None
    return wrapper


def _run_gltf_validator(wrapper: Path, glb_path: Path, label: str) -> bool:
    """
    Run the Node.js gltf_validate.js wrapper on *glb_path*.
    Returns True if 0 errors (warnings/hints allowed), False on any spec error.
    """
    _log(INFO, f"Validating {label}: {glb_path}")
    try:
        r = run_cmd(["node", str(wrapper), str(glb_path)], timeout=120)
    except FileNotFoundError:
        record_skip("node not found; glTF-Validator skipped for this file")
        return True
    except subprocess.TimeoutExpired:
        record_fail(f"glTF-Validator timed out for {label}")
        return False

    # Exit code 2 = validator tool not available (treat as skip)
    if r.returncode == 2:
        record_skip(f"glTF-Validator library unavailable for {label}")
        return True

    stdout = r.stdout.strip()
    stderr = r.stderr.strip()
    errors = 0
    warnings = 0
    parsed = False

    if stdout:
        try:
            report = json.loads(stdout)
            issues = report.get("issues", {})
            errors   = issues.get("numErrors",   0)
            warnings = issues.get("numWarnings", 0)
            hints    = issues.get("numHints",    0)
            infos    = issues.get("numInfos",    0)
            parsed = True
            _log(INFO, (
                f"glTF-Validator [{label}]: "
                f"errors={errors} warnings={warnings} hints={hints} infos={infos} "
                f"(validator v{report.get('validatorVersion', '?')})"
            ))
            if errors > 0:
                for msg in issues.get("messages", [])[:10]:
                    _log(WARN, f"  glTF error: {msg}")
        except (json.JSONDecodeError, AttributeError):
            parsed = False

    if not parsed:
        combined = stdout + "\n" + stderr
        _log(INFO, f"glTF-Validator raw output [{label}]:\n{combined[:1000]}")
        lower = combined.lower()
        if "error" in lower and "0 error" not in lower:
            errors = 1

    ok = errors == 0
    if ok:
        record_pass(
            f"glTF-Validator [{label}]: 0 errors "
            f"(warnings={warnings})"
        )
    else:
        record_fail(
            f"glTF-Validator [{label}]: {errors} error(s) — GLB is not spec-conformant"
        )
    return ok


def step_gltf_validator(glb_path: Optional[Path], real_glb: Optional[Path]) -> None:
    print(f"\n{'='*60}")
    print("STEP 2 — Khronos glTF-Validator")
    print(f"{'='*60}")

    wrapper = _find_node_validator()
    if wrapper is None:
        # Already logged a skip reason in _find_node_validator
        if glb_path is None:
            record_skip("No halfmesh GLB to validate (Step 1 failed)")
        if real_glb is None:
            record_skip("Real-world GLB tests/data/mesh_roi_crop_1.textured.glb absent — skipping")
        return

    if glb_path is None:
        record_skip("No halfmesh GLB to validate (Step 1 failed)")
    else:
        _run_gltf_validator(wrapper, glb_path, "halfmesh-uv.glb")

    if real_glb is not None:
        # The real-world GLB is a user-provided reference asset, not produced by
        # halfmesh.  Validation errors here are informational — they characterise
        # the input asset, not halfmesh's output.  We report but do NOT fail.
        ok = _run_gltf_validator(wrapper, real_glb, "mesh_roi_crop_1.textured.glb")
        if not ok:
            # Downgrade to a warning: this is the *input* GLB, not halfmesh output.
            failed_msg = _failures.pop() if _failures else None
            if failed_msg:
                _log(WARN, (
                    f"Real-world GLB has validation issues (informational only, "
                    f"not halfmesh output): {failed_msg}"
                ))
    else:
        record_skip("Real-world GLB tests/data/mesh_roi_crop_1.textured.glb absent — skipping")


# ---------------------------------------------------------------------------
# Step 3 — pygltflib parse
# ---------------------------------------------------------------------------

def step_pygltflib(glb_path: Optional[Path], real_glb: Optional[Path]) -> None:
    print(f"\n{'='*60}")
    print("STEP 3 — pygltflib structural parse")
    print(f"{'='*60}")

    try:
        import pygltflib  # type: ignore
        _log(INFO, f"pygltflib version: {pygltflib.__version__}")
    except ImportError:
        record_skip("pygltflib not installed; step skipped")
        return

    def _check_glb(path: Path, require_texcoords: bool, label: str) -> None:
        _log(INFO, f"Parsing [{label}]: {path}")
        try:
            glb = pygltflib.GLTF2().load(str(path))
        except Exception as exc:
            record_fail(f"pygltflib failed to load {label}: {exc}")
            return

        # Must have at least one mesh with one primitive
        meshes = glb.meshes or []
        assert_true(len(meshes) >= 1, f"[{label}] at least 1 mesh (got {len(meshes)})")

        for mi, mesh in enumerate(meshes):
            prims = mesh.primitives or []
            assert_true(len(prims) >= 1, f"[{label}] mesh[{mi}] has at least 1 primitive")
            for pi, prim in enumerate(prims):
                attrs = prim.attributes
                # POSITION accessor must exist
                has_pos = hasattr(attrs, "POSITION") and attrs.POSITION is not None
                assert_true(has_pos, f"[{label}] mesh[{mi}].prim[{pi}] has POSITION accessor")

                if require_texcoords:
                    has_uv = (
                        hasattr(attrs, "TEXCOORD_0")
                        and attrs.TEXCOORD_0 is not None
                    )
                    # The halfmesh unwrap CLI now emits TEXCOORD_0 for UV-only
                    # meshes (atlas UVs without an embedded texture image).
                    # A missing TEXCOORD_0 in the unwrap GLB is a real failure.
                    if has_uv:
                        record_pass(
                            f"[{label}] mesh[{mi}].prim[{pi}] has TEXCOORD_0"
                        )
                    else:
                        record_fail(
                            f"[{label}] mesh[{mi}].prim[{pi}] has no TEXCOORD_0 "
                            "— unwrap GLB must carry UV coordinates"
                        )

        # Accessor sanity: all counts must be positive integers
        accessors = glb.accessors or []
        for ai, acc in enumerate(accessors):
            assert_true(
                isinstance(acc.count, int) and acc.count > 0,
                f"[{label}] accessor[{ai}] count={acc.count} is positive",
            )

        _log(INFO, (
            f"[{label}] {len(meshes)} mesh(es), "
            f"{sum(len(m.primitives or []) for m in meshes)} prim(s), "
            f"{len(accessors)} accessor(s)"
        ))

    if glb_path is None:
        record_skip("No halfmesh GLB for pygltflib (Step 1 failed)")
    else:
        _check_glb(glb_path, require_texcoords=True, label="halfmesh-uv.glb")

    if real_glb is not None:
        _check_glb(real_glb, require_texcoords=False, label="mesh_roi_crop_1.textured.glb")
    else:
        record_skip("Real-world GLB absent — pygltflib real-world check skipped")


# ---------------------------------------------------------------------------
# Step 4 — trimesh PLY cross-read
# ---------------------------------------------------------------------------

def _bbox_size(mesh_obj) -> tuple[float, float, float]:  # type: ignore[type-arg]
    import numpy as np  # type: ignore
    verts = mesh_obj.vertices
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    sz = hi - lo
    return float(sz[0]), float(sz[1]), float(sz[2])


def step_trimesh(data_dir: Path, dec_ply: Optional[Path]) -> None:
    print(f"\n{'='*60}")
    print("STEP 4 — trimesh PLY cross-read")
    print(f"{'='*60}")

    try:
        import trimesh  # type: ignore
        import numpy as np  # type: ignore
        _log(INFO, f"trimesh version: {trimesh.__version__}")
    except ImportError:
        record_skip("trimesh not installed; step skipped")
        return

    mesh_ply = data_dir / "mesh.ply"
    if not mesh_ply.exists():
        record_skip(f"tests/data/mesh.ply not found ({mesh_ply}); trimesh step skipped")
        return

    # Load original mesh
    orig = trimesh.load(str(mesh_ply), force="mesh", process=False)
    orig_verts = len(orig.vertices)
    orig_faces = len(orig.faces)
    assert_true(orig_verts > 0, f"Original PLY: {orig_verts} vertices (>0)")
    assert_true(orig_faces > 0, f"Original PLY: {orig_faces} faces (>0)")
    orig_bbox = _bbox_size(orig)
    _log(INFO, f"Original PLY: {orig_verts} verts, {orig_faces} faces, bbox={orig_bbox}")

    # Verify no NaN/Inf in vertices
    import numpy as np  # type: ignore  (re-import for clarity)
    verts_arr = orig.vertices
    assert_true(
        bool(np.isfinite(verts_arr).all()),
        "Original PLY: all vertex coords are finite (no NaN/Inf)",
    )

    if dec_ply is None:
        record_skip("No decimated PLY to cross-read (Step 1 failed)")
        return

    dec = trimesh.load(str(dec_ply), force="mesh", process=False)
    dec_verts = len(dec.vertices)
    dec_faces = len(dec.faces)
    assert_true(dec_verts > 0, f"Decimated PLY: {dec_verts} vertices (>0)")
    assert_true(dec_faces > 0, f"Decimated PLY: {dec_faces} faces (>0)")
    assert_true(
        dec_faces < orig_faces,
        f"Decimated PLY has fewer faces than original ({dec_faces} < {orig_faces})",
    )
    _log(INFO, f"Decimated PLY: {dec_verts} verts, {dec_faces} faces")

    dec_bbox = _bbox_size(dec)
    _log(INFO, f"Decimated bbox={dec_bbox}, original bbox={orig_bbox}")
    # Bounding-box should be roughly consistent (within 10% on largest axis)
    max_orig = max(orig_bbox)
    if max_orig > 0:
        for axis_label, o, d in zip("XYZ", orig_bbox, dec_bbox):
            rel_diff = abs(o - d) / max_orig
            assert_true(
                rel_diff < 0.10,
                f"bbox {axis_label}-extent consistent orig={o:.4f} dec={d:.4f} (diff {rel_diff*100:.1f}% < 10%)",
            )

    dec_verts_arr = dec.vertices
    assert_true(
        bool(np.isfinite(dec_verts_arr).all()),
        "Decimated PLY: all vertex coords are finite (no NaN/Inf)",
    )


# ---------------------------------------------------------------------------
# Step 5 — Optional: pymeshlab / open3d
# ---------------------------------------------------------------------------

def step_optional_tools(dec_ply: Optional[Path]) -> None:
    print(f"\n{'='*60}")
    print("STEP 5 — Optional: pymeshlab / open3d")
    print(f"{'='*60}")

    if dec_ply is None:
        record_skip("No decimated PLY; optional geometry tools skipped")
        return

    # pymeshlab
    try:
        import pymeshlab  # type: ignore
        _log(INFO, f"pymeshlab version: {pymeshlab.__version__}")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(dec_ply))
        m = ms.current_mesh()
        vc = m.vertex_number()
        fc = m.face_number()
        assert_true(vc > 0, f"pymeshlab: {vc} vertices in decimated PLY")
        assert_true(fc > 0, f"pymeshlab: {fc} faces in decimated PLY")
    except ImportError:
        record_skip("pymeshlab not installed (may lack Python 3.14 wheels)")
    except Exception as exc:
        record_fail(f"pymeshlab error: {exc}")

    # open3d
    try:
        import open3d as o3d  # type: ignore
        _log(INFO, f"open3d version: {o3d.__version__}")
        mesh_o3d = o3d.io.read_triangle_mesh(str(dec_ply))
        vc = len(mesh_o3d.vertices)
        fc = len(mesh_o3d.triangles)
        assert_true(vc > 0, f"open3d: {vc} vertices in decimated PLY")
        assert_true(fc > 0, f"open3d: {fc} faces in decimated PLY")
    except ImportError:
        record_skip("open3d not installed (may lack Python 3.14 wheels)")
    except Exception as exc:
        record_fail(f"open3d error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    build_dir = Path(args.build_dir)
    data_dir  = Path(args.data_dir)
    tmp_dir   = Path(args.tmp_dir)

    print(f"halfmesh Python cross-check")
    print(f"  build_dir : {build_dir}")
    print(f"  data_dir  : {data_dir}")
    print(f"  tmp_dir   : {tmp_dir}")
    print(f"  python    : {sys.version}")

    real_glb_path = data_dir / "mesh_roi_crop_1.textured.glb"
    real_glb: Optional[Path] = real_glb_path if real_glb_path.exists() else None
    if real_glb:
        _log(INFO, f"Real-world GLB found: {real_glb}")
    else:
        _log(INFO, "Real-world GLB not present (tests/data/mesh_roi_crop_1.textured.glb) — will skip")

    # Run all steps
    glb_path, dec_ply = step_produce_artifacts(build_dir, data_dir, tmp_dir)
    step_gltf_validator(glb_path, real_glb)
    step_pygltflib(glb_path, real_glb)
    step_trimesh(data_dir, dec_ply)
    step_optional_tools(dec_ply)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Passed  : {PASS}")
    print(f"Skipped : {len(_skips)}")
    for s in _skips:
        print(f"  - {s}")
    print(f"Failures: {len(_failures)}")
    for f in _failures:
        print(f"  - {f}")

    if _failures:
        print(f"\nRESULT: FAIL ({len(_failures)} failure(s))")
        return 1
    print(f"\nRESULT: PASS (all available tools passed; {len(_skips)} tool(s) skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
