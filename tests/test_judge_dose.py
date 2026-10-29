"""tools/judge_dose：「任一」的定义、高 − 低差、API 组只报观测、d5 只报观测。"""
import numpy as np

from tools import judge_dose as JD


def test_fails_of_any_needs_all_four():
    assert JD.fails_of({"d1": 2, "d2": 0, "d3": 2, "d4": 2, "d5": 2})["any"] == 1
    assert JD.fails_of({"d1": 2, "d2": 1, "d3": 2, "d4": 2, "d5": 1})["any"] == 0
    assert JD.fails_of({"d1": 2, "d2": None, "d3": 2, "d4": 2, "d5": 2})["any"] is None


def _idx(group):
    lo = [(8.0, {o: 0 for o in JD.OUTCOMES})] * 4
    hi = [(8.0, {o: 1 for o in JD.OUTCOMES})] * 2 + [(8.0, {o: 0 for o in JD.OUTCOMES})] * 2
    return {(group, "fixed", "低"): {"p1": lo}, (group, "fixed", "中"): {"p1": lo}, (group, "fixed", "高"): {"p1": hi}}


def test_estimate_high_minus_low_and_policies():
    bs = {o: np.array([-4.0, 6.0, 0.0, 0.0]) for o in JD.OUTCOMES}
    edges = [7.0, 9.0]
    idx = _idx("本地")
    gate = JD.gate_of(idx, bs, edges)
    r = JD.estimate(idx, bs, edges, gate, {("本地", "fixed"): ["p1"]})
    assert r[("本地", "fixed", "any")]["diff"][0] == 0.5 and r[("本地", "fixed", "any")]["diff"][1] is not None
    assert r[("本地", "fixed", "d5")]["diff"][1] is None                     # d5 只报观测
    idx = _idx("API")
    r = JD.estimate(idx, bs, edges, JD.gate_of(idx, bs, edges), {("API", "fixed"): ["p1"]})
    assert r[("API", "fixed", "d1")]["diff"] == (0.5, None)                  # API 组只报观测
