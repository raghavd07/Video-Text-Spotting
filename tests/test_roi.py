import itertools

import cv2
import numpy as np
import pytest

from logo.lsc.roi_rotate import crop_many, crop_quad, order_vertices

OUT_HW = (32, 128)
BOX = (100, 168, 300, 232)  # x0, y0, x1, y1 of the rendered word: 200 x 64 px


def _word_image() -> np.ndarray:
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.rectangle(img, BOX[:2], (BOX[2] - 1, BOX[3] - 1), (0, 0, 200), 3)
    cv2.putText(img, "LOGO", (112, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (20, 120, 0), 5, cv2.LINE_AA)
    return img


def _box_quad() -> np.ndarray:
    x0, y0, x1, y1 = BOX
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def _rotated(angle: float):
    """Word image rotated by `angle` degrees (cv2 sign: positive = counter-clockwise) + its quad."""
    img = _word_image()
    m = cv2.getRotationMatrix2D((200.0, 200.0), angle, 1.0)
    rot = cv2.warpAffine(img, m, (400, 400), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    q = _box_quad()
    q = (np.c_[q, np.ones(4)] @ m.T)
    return rot, q


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_order_vertices_any_input_order():
    q = _box_quad()
    for perm in itertools.permutations(range(4)):
        np.testing.assert_array_equal(order_vertices(q[list(perm)]), q)
        np.testing.assert_array_equal(order_vertices(q[list(perm)].ravel().tolist()), q)


@pytest.mark.parametrize("angle", [0, 15, -30, 60, 90, -90, 135, 180])
def test_shape_dtype_and_permutation_invariance(angle):
    img, q = _rotated(angle)
    ref = crop_quad(img, q.ravel().tolist(), OUT_HW)
    assert ref.shape == (*OUT_HW, 3) and ref.dtype == np.uint8
    for perm in itertools.permutations(range(4)):
        np.testing.assert_array_equal(crop_quad(img, q[list(perm)], OUT_HW), ref)


def test_axis_aligned_equals_plain_resize():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    # same size: bit-exact
    x0, y0 = 37, 51
    q = [x0, y0, x0 + 128, y0, x0 + 128, y0 + 32, x0, y0 + 32]
    np.testing.assert_array_equal(crop_quad(img, q, OUT_HW), img[y0:y0 + 32, x0:x0 + 128])
    # scaled: equal to cv2.resize up to warpPerspective's 1/32-px interpolation table
    smooth = cv2.GaussianBlur(img, (0, 0), 2.0)
    for w, h in [(256, 64), (200, 50), (64, 16)]:
        q = [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
        got = crop_quad(smooth, q, OUT_HW).astype(int)
        want = cv2.resize(smooth[y0:y0 + h, x0:x0 + w], OUT_HW[::-1], interpolation=cv2.INTER_LINEAR)
        diff = np.abs(got - want.astype(int))
        if w >= OUT_HW[1]:
            assert diff.max() <= 2
        else:  # upsampling: border pixels see neighbours outside the crop, resize replicates
            assert diff[2:-2, 2:-2].max() <= 2
        assert diff.mean() < 0.5


@pytest.mark.parametrize("angle", [10, -25, 40, -90])
def test_rotated_word_matches_upright_crop(angle):
    upright = crop_quad(_word_image(), _box_quad(), OUT_HW)
    img, q = _rotated(angle)  # -90: clockwise, the word reads top-to-bottom (h/w = 3.1)
    crop = crop_quad(img, q, OUT_HW)
    assert crop.shape == (*OUT_HW, 3)
    assert _corr(crop, upright) > 0.85
    assert _corr(crop, upright[::-1, ::-1]) < 0.5  # not upside down


def test_vertical_is_rotated():
    img, q = _rotated(-90)
    w = np.linalg.norm(order_vertices(q)[1] - order_vertices(q)[0])
    h = np.linalg.norm(order_vertices(q)[3] - order_vertices(q)[0])
    assert h > 1.5 * w
    assert crop_quad(img, q, (48, 96)).shape == (48, 96, 3)


def test_crop_many_and_edge_cases():
    img, q = _rotated(20)
    out = crop_many(img, [q.ravel(), _box_quad().ravel()], OUT_HW)
    assert out.shape == (2, *OUT_HW, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[0], crop_quad(img, q, OUT_HW))
    assert crop_many(img, [], OUT_HW).shape == (0, *OUT_HW, 3)
    # degenerate (zero-area) and out-of-image quads do not crash
    assert crop_quad(img, [5, 5, 5, 5, 5, 5, 5, 5]).shape == (32, 128, 3)
    assert crop_quad(img, [10, 10, 50, 10, 90, 10, 130, 10]).shape == (32, 128, 3)
    assert crop_quad(img, [-50, -20, 450, -20, 450, 30, -50, 30]).shape == (32, 128, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = crop_quad(gray, q)
    assert g.shape == (32, 128, 3) and (g[..., 0] == g[..., 2]).all()
    with pytest.raises(ValueError):
        crop_quad(img, [0, 0, 1, 1, 2, 2])
