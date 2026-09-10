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
    assert set(meta) == {
        "charts", "pages", "width", "height", "occupancy", "coverage",
        "fit_attempts", "fit_scale", "max_chart_extent", "padding_applied",
        "vertices", "faces",
    }
    assert meta["charts"] >= 1
    assert meta["pages"] >= 1
    assert 0.0 < meta["occupancy"] <= 1.0
    assert 0.0 < meta["coverage"] <= meta["occupancy"] + 1e-3
    assert meta["fit_attempts"] >= 1
    assert meta["faces"] == 12

    # Layout diagnostics: the widest chart must fit the page it was packed into,
    # and with no per-size padding knob on, the applied gutter is the nominal one.
    assert meta["fit_scale"] > 0.0
    assert 0.0 < meta["max_chart_extent"] <= meta["width"]
    assert meta["padding_applied"] == {"nominal": 2, "min": 2, "n_charts_reduced": 0}

    unwrapped = hm.Mesh()
    unwrapped.load(out)
    assert unwrapped.has_texcoords


@pytest.mark.parametrize(
    "knobs",
    [
        dict(max_cone_error=0.1, cut_to_disk=True, max_uv_distortion=4.4),
        dict(repair_carve_rings=2, fold_rescue_slits=2, tiny_chart_side=8.0, debris_chart_faces=100),
    ],
    ids=["segmentation", "repair+padding"],
)
def test_unwrap_accepts_knobs(tmp_path, knobs):
    """Every unwrap() knob is keyword-addressable and keeps the atlas valid.

    A cube is developable and tiny, so the knobs cannot change its chart count
    much -- this pins the argument names and value plumbing, not the
    segmentation/packing behavior (docs/BENCHMARKS.md section 4 covers that).
    """
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)

    out = str(tmp_path / "cube_uv.ply")
    meta = hm.unwrap(src, out, resolution=1024, padding=2, **knobs)
    assert meta["charts"] >= 1
    unwrapped = hm.Mesh()
    unwrapped.load(out)
    assert unwrapped.has_texcoords


@pytest.mark.parametrize(
    "knobs, message",
    [
        (dict(resolution=0), "resolution must be > 0"),
        (dict(resolution=64, padding=32), "2\\*padding < resolution"),
        (dict(max_cone_error=0.0), "max_cone_error"),
        (dict(max_cone_error=float("nan")), "max_cone_error"),
        (dict(max_uv_distortion=1.0), "max_uv_distortion"),
        (dict(max_uv_distortion=4.0), "max_uv_distortion"),
        (dict(max_uv_distortion=-1.0), "max_uv_distortion"),
        (dict(fold_rescue_slits=17), "fold_rescue_slits"),
        (dict(repair_carve_rings=17), "repair_carve_rings"),
        (dict(tiny_chart_side=-1.0), "tiny_chart_side"),
    ],
    ids=[
        "resolution-zero", "padding-fills-page", "cone-error-zero", "cone-error-nan",
        "distortion-below-floor", "distortion-at-floor", "distortion-negative",
        "slits-over-cap", "carve-over-cap", "tiny-side-negative",
    ],
)
def test_unwrap_rejects_unhonourable_knobs(tmp_path, knobs, message):
    """A knob no run could honour raises instead of being worked from.

    max_uv_distortion is the one that matters: 4.0 is a perfectly isometric
    map, so a budget at or below it is met by no chart and used to bisect the
    mesh toward one chart per triangle with nothing reported.
    """
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)
    args = dict(resolution=1024, padding=2)
    args.update(knobs)
    with pytest.raises(ValueError, match=message):
        hm.unwrap(src, str(tmp_path / "cube_uv.ply"), **args)


def test_unwrap_accepts_the_edges_of_the_valid_domain(tmp_path):
    """The bounds reject only what is out of domain: 0 and just-above-4 are
    both legal distortion budgets, and the iteration caps are inclusive."""
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)
    for knobs in (
        dict(max_uv_distortion=0.0),
        dict(max_uv_distortion=4.001),
        dict(fold_rescue_slits=16, repair_carve_rings=16),
        dict(tiny_chart_side=0.0),
    ):
        meta = hm.unwrap(src, str(tmp_path / "cube_uv.ply"), resolution=1024, padding=2, **knobs)
        assert meta["charts"] >= 1


def test_unwrap_explicit_defaults_match_omitted(tmp_path):
    """Passing every knob at its default reproduces omitting them all: the
    keyword args are additive and match the C++ ParametrizeParams/AtlasParams
    defaults."""
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)

    meta_default = hm.unwrap(src, str(tmp_path / "default.ply"), resolution=1024, padding=2)
    meta_explicit = hm.unwrap(
        src,
        str(tmp_path / "explicit.ply"),
        resolution=1024,
        padding=2,
        max_cone_error=0.05,
        cut_to_disk=False,
        max_uv_distortion=0.0,
        repair_carve_rings=0,
        fold_rescue_slits=0,
        tiny_chart_side=0.0,
        debris_chart_faces=0,
    )
    assert meta_explicit == meta_default


def test_unwrap_raises_on_missing_input(tmp_path):
    with pytest.raises(RuntimeError):
        hm.unwrap(str(tmp_path / "nope.ply"), str(tmp_path / "out.ply"))


def test_unwrap_raises_on_unwritable_output(tmp_path):
    v, f = _cube_arrays()
    src = str(tmp_path / "cube.ply")
    hm.Mesh.from_arrays(v, f).save(src)
    with pytest.raises(RuntimeError):
        hm.unwrap(src, str(tmp_path / "no_such_dir" / "out.ply"))
