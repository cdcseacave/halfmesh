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


def test_mesh_roundtrips_ascii_ply(tmp_path):
    v, f = _cube_arrays()
    path = str(tmp_path / "cube_ascii.ply")
    hm.Mesh.from_arrays(v, f).save(path, binary=False)
    loaded = hm.Mesh()
    loaded.load(path)
    lv, lf = loaded.to_arrays()
    np.testing.assert_allclose(lv, v)
    assert lf.shape == (12, 3)


def test_mesh_roundtrips_glb(tmp_path):
    v, f = _cube_arrays()
    path = str(tmp_path / "cube.glb")
    hm.Mesh.from_arrays(v, f).save(path)
    loaded = hm.Mesh()
    loaded.load(path)
    assert loaded.n_faces == 12
    lv, lf = loaded.to_arrays()
    assert lv.shape[1] == 3 and lf.shape == (12, 3)


def test_empty_mesh_roundtrips_arrays():
    mesh = hm.Mesh()
    assert mesh.n_vertices == 0 and mesh.n_faces == 0
    v, f = mesh.to_arrays()
    assert v.shape == (0, 3) and f.shape == (0, 3)


def test_mesh_save_raises_on_bad_path(tmp_path):
    v, f = _cube_arrays()
    mesh = hm.Mesh.from_arrays(v, f)
    with pytest.raises(RuntimeError):
        mesh.save(str(tmp_path / "no_such_dir" / "cube.ply"))


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
    assert set(meta) == {"charts", "pages", "width", "height", "occupancy", "fit_attempts", "vertices", "faces"}
    assert meta["charts"] >= 1
    assert meta["pages"] >= 1
    assert 0.0 < meta["occupancy"] <= 1.0
    assert meta["fit_attempts"] >= 1
    assert meta["faces"] == 12

    unwrapped = hm.Mesh()
    unwrapped.load(out)
    assert unwrapped.has_texcoords


def test_unwrap_raises_on_missing_input(tmp_path):
    with pytest.raises(RuntimeError):
        hm.unwrap(str(tmp_path / "nope.ply"), str(tmp_path / "out.ply"))


def test_unwrap_raises_on_unwritable_output(tmp_path):
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)
    with pytest.raises(RuntimeError):
        hm.unwrap(src, str(tmp_path / "no_such_dir" / "out.ply"))
