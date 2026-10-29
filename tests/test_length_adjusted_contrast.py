"""tools/length_adjusted_contrast：se/sp 随长度、分段校正能找回真实失败率、无方差返回 None。"""
import numpy as np
import pytest

from tools import length_adjusted_contrast as LA


def test_se_sp_formula():
    b = np.array([-2.0, 3.0, 0.5, 0.0])
    se, sp = LA.se_sp(b, 2.0, 0.0)
    assert se == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert sp == pytest.approx(1 - 1 / (1 + np.exp(1.0)))


def test_corrected_rate_recovers_truth_under_length_dependent_error():
    rng = np.random.default_rng(0)
    b = np.array([-3.0, 3.5, 0.3, 0.0])                      # 越长越易判失败
    truth = 0.2
    turns = []
    for _ in range(40000):
        length = float(rng.uniform(4, 11))
        g = rng.random() < truth
        se, sp = LA.se_sp(b, length, 0.0)
        p = se if g else 1 - sp
        turns.append((length, int(rng.random() < p)))
    edges = LA.band_edges([x for x, _ in turns], n=10)
    got = LA.corrected_rate(turns, "d1", b, edges, 0.0)
    assert got == pytest.approx(truth, abs=0.02)
    assert abs(LA.observed(turns) - truth) > 0.03            # 不校正是偏的


def test_fit_error_none_without_variance():
    rows = [("zh_p01", 0.0, 1.0, 5.0, 1.0, 0.0)] * 20
    assert LA.fit_error(rows) is None


def test_se_sp_interaction_only_shifts_sensitivity():
    b4 = np.array([-2.0, 3.0, 0.5, 0.4])
    b5 = np.array([-2.0, 3.0, 0.5, 0.4, -1.0])
    assert LA.se_sp(b5, 2.0, 0.0) == pytest.approx(LA.se_sp(b4, 2.0, 0.0))     # 本地组不受交互项影响
    se4, sp4 = LA.se_sp(b4, 2.0, 1.0)
    se5, sp5 = LA.se_sp(b5, 2.0, 1.0)
    assert se5 < se4 and sp5 == pytest.approx(sp4)


def test_usable_requires_every_band_above_threshold():
    b = np.array([-6.0, 4.0, 0.4, 0.0])                      # 短回复灵敏度低
    edges = [6.0, 9.0]
    long_only = [(10.0, 0)] * 10
    with_short = long_only + [(3.0, 0)] * 10
    se_s, sp_s = LA.se_sp(b, 3.0, 0.0)
    se_l, sp_l = LA.se_sp(b, 10.0, 0.0)
    assert se_l + sp_l - 1 >= 0.5 > se_s + sp_s - 1
    assert LA.usable(long_only, b, edges, 0.0)
    assert not LA.usable(with_short, b, edges, 0.0)
    assert not LA.usable(long_only, None, edges, 0.0)


def test_contrasts_d5_observed_only_and_gate():
    b = {d: np.array([-4.0, 6.0, 0.0, 0.0]) for d in LA.DIMS5}  # 高 Youden，全部可校正
    by_key = {("mindchat", "fixed"): [(8.0, {d: 1 for d in LA.DIMS5})] * 3 + [(8.0, {d: 0 for d in LA.DIMS5})],
              ("qwen15-4b-base", "fixed"): [(8.0, {d: 0 for d in LA.DIMS5})] * 4}
    pairs = (("mindchat", "qwen15-4b-base"),)
    out = LA.contrasts(by_key, b, [7.0, 9.0], pairs, ["fixed"])
    raw5, cor5 = out[("mindchat", "qwen15-4b-base", "fixed", "d5")]
    assert raw5 == pytest.approx(0.75) and cor5 is None                    # d5 只报观测
    assert out[("mindchat", "qwen15-4b-base", "fixed", "d1")][1] is not None
    ref = LA.contrasts(by_key, b, [7.0, 9.0], pairs, ["fixed"], observed_only=())
    assert ref[("mindchat", "qwen15-4b-base", "fixed", "d5")][1] is not None
    gated = LA.contrasts(by_key, b, [7.0, 9.0], pairs, ["fixed"], gate=set())  # 门槛外 → 不可校正
    assert all(v[1] is None for v in gated.values())
