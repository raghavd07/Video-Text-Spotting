import pytest

from logo.utils.geometry import is_self_intersecting, order_quad, quad_iou, signed_area

SQ = [0, 0, 10, 0, 10, 10, 0, 10]  # TL, TR, BR, BL (clockwise in image coords)


def test_clockwise_kept():
    assert order_quad(SQ) == [float(v) for v in SQ]
    assert signed_area(SQ) > 0


def test_ccw_reversed_keeps_start():
    assert order_quad([0, 0, 0, 10, 10, 10, 10, 0]) == [float(v) for v in SQ]


def test_icdar_bowtie_fixed():
    bow = [762, 169, 762, 187, 781, 169, 781, 187]  # real ICDAR13 order: x1y1,x1y2,x2y1,x2y2
    assert is_self_intersecting(bow)
    q = order_quad(bow)
    assert not is_self_intersecting(q)
    assert q == [762.0, 169.0, 781.0, 169.0, 781.0, 187.0, 762.0, 187.0]


def test_iou():
    assert quad_iou(SQ, SQ) == pytest.approx(1.0)
    assert quad_iou(SQ, [20, 20, 30, 20, 30, 30, 20, 30]) == 0.0
    half = [5, 0, 15, 0, 15, 10, 5, 10]
    assert quad_iou(SQ, half) == pytest.approx(50 / 150)
    assert quad_iou(SQ, [0, 0, 0, 10, 10, 0, 10, 10]) == pytest.approx(1.0)  # bowtie input
