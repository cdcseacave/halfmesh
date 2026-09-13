import numpy as np
import pytest

import halfmesh as hm


def _sizes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 64, size=(n, 2), dtype=np.int64)


def _assert_valid_layout(result, sizes, padding):
    """Every packed rect keeps its input extents (swapped when rotated) and, grown by
    its gutter, stays inside its page without overlapping any other rect there."""
    rects, page, rotated = result["rects"], result["page"], result["rotated"]
    packed = result["packed"]
    w = np.where(rotated, sizes[:, 1], sizes[:, 0])
    h = np.where(rotated, sizes[:, 0], sizes[:, 1])
    assert np.array_equal(rects[packed, 2], w[packed])
    assert np.array_equal(rects[packed, 3], h[packed])
    x0, y0 = rects[:, 0] - padding, rects[:, 1] - padding
    x1, y1 = rects[:, 0] + rects[:, 2] + padding, rects[:, 1] + rects[:, 3] + padding
    assert np.all(x0[packed] >= 0) and np.all(y0[packed] >= 0)
    assert np.all(x1[packed] <= result["width"]) and np.all(y1[packed] <= result["height"])
    idx = np.flatnonzero(packed)
    for a in range(len(idx)):
        i = idx[a]
        j = idx[a + 1:]
        j = j[page[j] == page[i]]
        overlap = (x0[i] < x1[j]) & (x0[j] < x1[i]) & (y0[i] < y1[j]) & (y0[j] < y1[i])
        assert not overlap.any()


def test_pack_rectangles_grow_fits_everything_on_one_page():
    sizes = _sizes()
    result = hm.pack_rectangles(sizes, page_size=64, padding=2)
    assert result["pages"] == 1 and result["n_packed"] == len(sizes)
    assert result["packed"].all()
    assert result["width"] >= 64 and result["height"] >= 64  # grown from the initial page
    assert result["rects"].shape == (len(sizes), 4) and result["rects"].dtype == np.int32
    assert 0.0 < result["occupancy"] <= 1.0
    _assert_valid_layout(result, sizes, padding=2)


def test_pack_rectangles_multi_opens_fixed_size_pages():
    sizes = _sizes()
    result = hm.pack_rectangles(sizes, page_size=(128, 96), mode="multi", padding=1)
    assert result["pages"] > 1 and result["packed"].all()
    assert (result["width"], result["height"]) == (128, 96)  # never resized
    assert set(np.unique(result["page"])) == set(range(result["pages"]))
    _assert_valid_layout(result, sizes, padding=1)


def test_pack_rectangles_single_reports_what_does_not_fit():
    sizes = np.array([[8, 8], [8, 8]], dtype=np.int32)
    result = hm.pack_rectangles(sizes, page_size=8, mode="single", padding=0)
    assert result["pages"] == 1 and result["n_packed"] == 1
    assert result["packed"].sum() == 1
    loser = np.flatnonzero(~result["packed"])[0]
    assert np.all(result["rects"][loser] == 0)


def test_pack_rectangles_grow_honours_max_page_size():
    sizes = np.array([[8, 8], [8, 8]], dtype=np.int32)
    result = hm.pack_rectangles(sizes, page_size=8, max_page_size=(8, 8), padding=0)
    assert (result["width"], result["height"]) == (8, 8)
    assert result["n_packed"] == 1


def test_pack_rectangles_leaves_zero_sizes_unpacked():
    sizes = np.array([[0, 5], [4, 4], [5, 0]], dtype=np.int32)
    result = hm.pack_rectangles(sizes)
    assert list(result["packed"]) == [False, True, False]


def test_pack_rectangles_without_rotation_never_rotates():
    sizes = np.array([[40, 4]] * 20, dtype=np.int32)
    result = hm.pack_rectangles(sizes, page_size=48, allow_rotation=False, padding=0)
    assert not result["rotated"].any()
    _assert_valid_layout(result, sizes, padding=0)


def test_pack_rectangles_power_of_two_and_square_pages():
    sizes = _sizes(n=50)
    result = hm.pack_rectangles(sizes, page_size=(100, 60), mode="multi", power_of_two=True, square=True)
    assert result["width"] == result["height"] == 128


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sizes": np.array([[4.0, 4.0]])}, "integer"),  # would truncate
        ({"sizes": np.ones((2, 2), dtype=bool)}, "integer"),  # would read as 1s
        ({"sizes": np.array([4, 4])}, r"\[N,2\]"),
        ({"sizes": np.array([[4, 4, 4]])}, r"\[N,2\]"),
        ({"sizes": np.array([[-1, 4]])}, "0, 2"),
        ({"sizes": np.array([[2**31, 4]], dtype=np.int64)}, "0, 2"),
        ({"mode": "banana"}, "mode"),
        ({"page_size": 0}, "page_size"),
        ({"page_size": (64, -1)}, "page_size"),
        ({"mode": "single", "max_page_size": 64}, "only caps"),
        ({"max_page_size": (-1, 64)}, "max_page_size"),
    ],
)
def test_pack_rectangles_rejects_bad_arguments(kwargs, message):
    kwargs.setdefault("sizes", np.array([[4, 4]], dtype=np.int32))
    with pytest.raises(ValueError, match=message):
        hm.pack_rectangles(**kwargs)


def test_pack_rectangles_accepts_an_empty_input():
    result = hm.pack_rectangles(np.empty((0, 2), dtype=np.int32), mode="multi")
    assert result["rects"].shape == (0, 4) and result["pages"] == 0


def test_estimate_square_texture_size_rounds_up():
    sizes = np.array([[10, 10], [10, 10]], dtype=np.int32)
    assert hm.estimate_square_texture_size(sizes, multiple=8, target_occupancy=1.0) == 16
    assert hm.estimate_square_texture_size(sizes, target_occupancy=1.0) == 16
    assert hm.estimate_square_texture_size(sizes, multiple=5, target_occupancy=0.5) == 20


def test_estimate_square_texture_size_is_a_workable_page_size():
    sizes = _sizes()
    side = hm.estimate_square_texture_size(sizes, multiple=0, target_occupancy=0.8)
    result = hm.pack_rectangles(sizes, page_size=side, mode="single", padding=0)
    assert result["packed"].all()


def test_estimate_square_texture_size_rejects_bad_arguments():
    sizes = np.array([[10, 10]], dtype=np.int32)
    for occupancy in (0.0, 1.5, np.nan):
        with pytest.raises(ValueError, match="target_occupancy"):
            hm.estimate_square_texture_size(sizes, target_occupancy=occupancy)
    with pytest.raises(ValueError, match="multiple"):
        hm.estimate_square_texture_size(sizes, multiple=-8)
