# tests/test_metrics.py
import math
from cascade.scoring.metrics import ausd, ttf, median_ttf


def test_ausd_is_trapezoidal_mean():
    # flat 1.0 -> mean 1.0
    assert ausd([1.0, 1.0, 1.0]) == 1.0
    # linear 1->0 over 3 points: trapezoid area / span = 0.5
    assert math.isclose(ausd([1.0, 0.5, 0.0]), 0.5, rel_tol=1e-9)


def test_ausd_single_point():
    assert ausd([0.7]) == 0.7


def test_ttf_finds_first_zero_one_indexed():
    assert ttf([1.0, 0.5, 0.0, 0.0]) == 3
    assert ttf([0.0, 1.0]) == 1
    assert ttf([1.0, 0.5, 0.2]) is None  # never fails


def test_median_ttf_handles_censoring():
    # failures at turns 2 and 4, one censored (None)
    res = median_ttf([2, 4, None])
    assert res["median_ttf"] == 3.0       # median of observed [2,4]
    assert res["n_failed"] == 2
    assert res["n_censored"] == 1
    assert math.isclose(res["failure_rate"], 2 / 3, rel_tol=1e-9)


def test_median_ttf_all_censored():
    res = median_ttf([None, None])
    assert res["median_ttf"] is None
    assert res["failure_rate"] == 0.0
