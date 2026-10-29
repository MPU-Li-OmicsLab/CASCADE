"""tools/judge_dose_turn：调和平均加权合成、人设匹配、串联、段内轮次斜率。"""
import numpy as np

from tools import judge_dose_turn as JT

O = JT.OUTCOMES


def _f(v):
    return {o: v for o in O}


def _row(persona, band, turn, fail, model="m", profile=None):
    return {"model": model, "arm": "fixed", "persona": persona, "profile": profile or f"{persona}-c1",
            "stage": "x", "band": band, "turn": turn, "length": 8.0, "fails": _f(fail)}


def test_hm_weight_and_pool():
    assert JT.hm_weight(2, 2) == 2.0
    assert JT.hm_weight(0, 5) == 0.0
    # 两轮：差 0.2（权重 2·1·3/4 = 1.5）与 0.6（权重 2·3·3/6 = 3）→ (0.3 + 1.8) / 4.5
    assert abs(JT.pool([(1, 3, 0.2), (3, 3, 0.6)]) - 2.1 / 4.5) < 1e-12
    assert JT.pool([(0, 3, 0.9), (2, 2, 0.1)]) == 0.1          # 一段为空的轮不计
    assert JT.pool([(2, 2, None), (2, 2, 0.1)]) is None         # 有样本却算不出 → None
    assert JT.pool([]) is None


def _toy():
    rows = []
    # 第 3 轮：低段只有 pA（失败 0），中段 pA（失败 1）与 pB（失败 0 ×3）
    rows += [_row("pA", "低", 3, 0), _row("pA", "中", 3, 1)] + [_row("pB", "中", 3, 0)] * 3
    # 第 7 轮：中段 pA（0），高段 pA（1）、pB（1）
    rows += [_row("pA", "中", 7, 0), _row("pA", "高", 7, 1), _row("pB", "高", 7, 1)]
    # 其他轮让三段都有数据
    rows += [_row("pA", "低", 1, 0), _row("pB", "低", 1, 0)]
    return rows


def test_matched_personas_and_shared_turns():
    idx = JT.cell_index(_toy())
    assert JT.shared_turns(idx, ("低", "中")) == [3]
    assert JT.shared_turns(idx, ("中", "高")) == [7]
    assert JT.matched_personas(idx, ("低", "中"), 3) == {"pA"}
    assert JT.matched_personas(idx, ("中", "高"), 7) == {"pA"}


def test_estimate_stratified_vs_matched_vs_unstrat():
    idx = JT.cell_index(_toy())
    r = JT.estimate(idx, {o: None for o in O}, [7.0, 9.0], set(), ["pA", "pB"])["any"]
    lm, mh = r["pairs"][("低", "中")], r["pairs"][("中", "高")]
    assert abs(lm["obs"] - 0.25) < 1e-12        # 中 1/4 − 低 0/1
    assert lm["m_obs"] == 1.0                   # 只留 pA：1 − 0
    assert mh["obs"] == 1.0 and mh["m_obs"] == 1.0
    assert abs(r["chain"][0] - 1.25) < 1e-12 and r["chain"][2] == 2.0
    assert lm["cor"] is None                    # 无门槛 → 不校正
    # 未分层：高 2/2 − 低 0/3
    assert r["unstrat"] == (1.0, None)
    # 只抽到 pB：中−低 第 3 轮低段为空 → 无可比轮
    r2 = JT.estimate(idx, {o: None for o in O}, [7.0, 9.0], set(), ["pB"])["any"]
    assert r2["pairs"][("低", "中")]["obs"] is None and r2["chain"][0] is None


def test_estimate_corrected_respects_gate_and_d5():
    idx = JT.cell_index(_toy())
    bs = {o: np.array([-4.0, 6.0, 0.0, 0.0]) for o in O}
    gate = {(b, o) for b in JT.BANDS for o in O}
    r = JT.estimate(idx, bs, [7.0, 9.0], gate, ["pA", "pB"])
    assert r["any"]["pairs"][("中", "高")]["cor"] is not None
    assert r["d5"]["pairs"][("中", "高")]["cor"] is None       # d5 只报观测


def test_slopes_pooled_vs_within():
    # 两段对话：对话 1 全在早轮且全失败，对话 2 全在晚轮且全不失败 → 合并斜率为负，对话内斜率为 0
    rows = [_row("p1", "中", t, 1, profile="p1-a") for t in (3, 4)] + [_row("p2", "中", t, 0, profile="p2-a") for t in (9, 10)]
    by_p = JT.trend_index(rows, "中", (3, 12))
    pooled, within = JT.slopes(by_p, ["p1", "p2"], "any")
    assert pooled < 0 and abs(within) < 1e-12
    # 对话内随轮上升：每轮 +0.5
    rows = [_row("p1", "中", 3, 0, profile="p1-a"), _row("p1", "中", 5, 1, profile="p1-a")]
    pooled, within = JT.slopes(JT.trend_index(rows + rows[:1], "中", (3, 12)), ["p1"], "any")
    assert within is not None and within > 0
