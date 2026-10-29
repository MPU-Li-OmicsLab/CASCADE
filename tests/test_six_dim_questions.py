"""tools/six_dim_questions：失败定义（含 d6）、复读未打分轮的处理、多模型按轮数合成。"""
import numpy as np

from tools import six_dim_questions as SQ


def test_fails_definitions():
    f = SQ._fails({"d1": 0, "d2": 2, "d3": 1, "d4": 2, "d5": 1, "d6": 1})
    assert f == {"d1": 1, "d2": 0, "d3": 0, "d4": 0, "d5": 1, "d6": 0, "d6le1": 1}


def test_unscored_loop_counts_only_d5():
    loop = "我在这里陪着你，别害怕。" * 4
    t = SQ._turn("m", "zh_p01", loop, None)
    assert t[3]["d5"] == 1 and t[3]["d1"] is None and t[3]["d6"] is None
    assert SQ._turn("m", "zh_p01", "短", None) is None


def test_rates_pools_models_by_turns_and_skips_d6_correction():
    idx = SQ._index([("a", "p1", 8.0, {"d1": 1, "d6": 1}), ("a", "p1", 8.0, {"d1": 0, "d6": 0}),
                     ("b", "p1", 8.0, {"d1": 0, "d6": 0})])
    bs = {d: np.array([-4.0, 6.0, 0.0, 0.0]) for d in SQ.DIMS5}       # 高 Youden，可校正
    edges = [7.0, 9.0]
    obs, corr = SQ.rates(idx, ["a", "b"], ["p1"], "d1", bs, edges)
    assert obs == 1 / 3 and corr is not None
    obs6, corr6 = SQ.rates(idx, ["a", "b"], ["p1"], "d6", bs, edges)
    assert obs6 == 1 / 3 and corr6 is None


def test_rates_d5_observed_only_and_gate():
    idx = SQ._index([("a", "p1", 8.0, {"d1": 1, "d5": 1}), ("a", "p1", 8.0, {"d1": 0, "d5": 0})])
    bs = {d: np.array([-4.0, 6.0, 0.0, 0.0]) for d in SQ.DIMS5}
    edges = [7.0, 9.0]
    obs5, corr5 = SQ.rates(idx, ["a"], ["p1"], "d5", bs, edges)
    assert obs5 == 0.5 and corr5 is None                                  # d5 只报观测
    assert SQ.rates(idx, ["a"], ["p1"], "d5", bs, edges, observed_only=())[1] is not None
    assert SQ.rates(idx, ["a"], ["p1"], "d1", bs, edges, gate=set())[1] is None   # 门槛外
