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


def test_repair_drops_duplicate_faces():
    v, f = _cube_mesh()
    f2 = np.concatenate([f, f[:1]])  # one exact duplicate face
    rv, rf = hm.repair(v, f2)
    assert rf.shape == (12, 3)  # one copy survives (geometry-preserving)


def test_ops_accept_empty_mesh():
    v = np.empty((0, 3), dtype=np.float32)
    f = np.empty((0, 3), dtype=np.uint32)
    rv, rf = hm.repair(v, f)
    assert rv.shape == (0, 3) and rf.shape == (0, 3)
    assert rv.dtype == np.float32 and rf.dtype == np.uint32


def test_repair_rejects_out_of_range_face_index():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    f = np.array([[0, 1, 100]], dtype=np.uint32)
    with pytest.raises(ValueError):
        hm.repair(v, f)


def test_smooth_rejects_out_of_range_face_index():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    f = np.array([[0, 1, 100]], dtype=np.uint32)
    with pytest.raises(ValueError):
        hm.smooth(v, f, 1, "taubin")


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


def test_smooth_rejects_nonpositive_iterations():
    v, f = _grid_mesh(n=8)
    with pytest.raises(ValueError):
        hm.smooth(v, f, 0, "taubin")


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
