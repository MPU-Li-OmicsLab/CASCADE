"""tools/judge_error_by_model：混淆计数、O/E 汇总、泊松二项精确 p、Holm、心理微调对比集合。"""
from __future__ import annotations

import math

import pytest

from tools.judge_error_by_model import (confusion, contrast_sets, diff_point, error_rows, holm, oe_summary,
                                        poibin_p)

OK = {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2}


def _s(**kw):
    return {**OK, **kw}


def test_confusion_counts_and_rates():
    ids = ["mA|fixed|zh_p01-c1#t1", "mA|fixed|zh_p01-c1#t2", "mA|fixed|zh_p02-c1#t1", "mA|fixed|zh_p02-c1#t2",
           "mB|fixed|zh_p01-c1#t1"]
    gold = {ids[0]: _s(d2=0), ids[1]: _s(d2=0), ids[2]: _s(), ids[3]: _s(), ids[4]: _s(d2=0)}
    pred = {ids[0]: _s(d2=0), ids[1]: _s(d2=1), ids[2]: _s(d2=0), ids[3]: _s(), ids[4]: _s(d2=0)}
    w = {i: 1.0 for i in ids}
    w[ids[0]] = 3.0
    c = confusion(ids, gold, pred, w, lambda i: i.split("|")[0], "d2")
    a = c["mA"]
    assert (a["tp"], a["fn"], a["fp"], a["tn"], a["n"]) == (1, 1, 1, 1, 4)
    assert a["fpr"] == 0.5 and a["fnr"] == 0.5
    assert a["prev_w"] == pytest.approx(4 / 6, abs=1e-4)      # 医师失败权重 3 + 1
    assert a["judge_w"] == pytest.approx(4 / 6, abs=1e-4)     # 判官失败权重 3 + 1
    assert c["mB"]["fpr"] is None and c["mB"]["fnr"] == 0.0


def test_confusion_any_and_d5_threshold():
    ids = ["m|fixed|zh_p01-c1#t1", "m|fixed|zh_p01-c1#t2"]
    gold = {ids[0]: _s(d4=0, d5=1), ids[1]: _s(d5=2)}
    pred = {ids[0]: _s(d5=1), ids[1]: _s(d1=0, d5=1)}
    w = {i: 1.0 for i in ids}
    any_ = confusion(ids, gold, pred, w, lambda i: "m", "any")["m"]
    assert (any_["tp"], any_["fn"], any_["fp"], any_["tn"]) == (0, 1, 1, 0)
    d5 = confusion(ids, gold, pred, w, lambda i: "m", "d5")["m"]
    assert (d5["tp"], d5["fp"]) == (1, 1)                     # d5 失败 = ≤1


def test_error_rows_shape_and_api_flag():
    ids = ["openai/x|fixed|zh_p01-c1#t1", "local|fixed|zh_p02-c1#t1"]
    gold = {ids[0]: _s(d2=0), ids[1]: _s()}
    pred = {ids[0]: _s(), ids[1]: _s(d2=0)}
    rows = error_rows(ids, gold, pred, {i: 2.0 for i in ids}, {ids[0]: "a" * 7, ids[1]: ""}, "d2")
    assert rows[0][:3] == ("zh_p01", 0.0, 1.0) and rows[0][3] == pytest.approx(3.0) and rows[0][5] == 1.0
    assert rows[1][:3] == ("zh_p02", 1.0, 0.0) and rows[1][5] == 0.0 and rows[1][6] == ids[1]


def test_oe_summary_layers_and_counts():
    # (人设, 判官失败, 医师失败, 长度, 权重, API, id)
    rows = [("p", 1.0, 1.0, 5, 1, 0, "A|x"), ("p", 0.0, 1.0, 5, 1, 0, "A|y"), ("p", 1.0, 0.0, 5, 1, 0, "A|z"),
            ("p", 0.0, 0.0, 5, 1, 0, "B|z")]
    probs = [0.5, 0.5, 0.25, 0.25]
    r = oe_summary(rows, probs, lambda i: i.split("|")[0])
    a = r[("A", "all")]
    assert a["n"] == 3 and a["O"] == 2 and a["E"] == pytest.approx(1.25)
    assert a["oe"] == pytest.approx(1.6)
    assert a["z"] == pytest.approx((2 - 1.25) / math.sqrt(0.25 + 0.25 + 0.1875), abs=0.01)
    assert r[("A", "医师失败")]["O"] == 1 and r[("A", "医师失败")]["E"] == pytest.approx(1.0)
    assert r[("A", "医师通过")]["O"] == 1 and r[("A", "医师通过")]["E"] == pytest.approx(0.25)
    assert r[("B", "all")]["O"] == 0 and r[("B", "all")]["oe"] == 0.0


def test_poibin_p_matches_binomial():
    # 4 个 p=0.5：P(X=4)=1/16 → 双侧 2/16
    assert poibin_p(4, [0.5] * 4) == pytest.approx(2 / 16)
    assert poibin_p(2, [0.5] * 4) == 1.0
    assert poibin_p(0, [0.1, 0.1]) == pytest.approx(1.0)      # 2·0.81 截到 1


def test_holm_monotone_and_skips_none():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert "d" not in adj


def test_contrast_sets_and_diff_point():
    sets = contrast_sets()
    assert len(sets) == 4 and sets[0][1] == {"cpsycoun", "mindchat", "soulchat"}
    ids = ["cpsycoun|fixed|zh_p01-c1#t1", "internlm2-base|fixed|zh_p01-c1#t1", "internlm2-base|fixed|zh_p02-c1#t1"]
    score = {ids[0]: _s(), ids[1]: _s(d2=0), ids[2]: _s()}
    w = {ids[0]: 1.0, ids[1]: 3.0, ids[2]: 1.0}
    d, pa, pb = diff_point(ids, score, w, frozenset(["cpsycoun"]), frozenset(["internlm2-base"]), "d2")
    assert pa["p"] == 0.0 and pb["p"] == pytest.approx(0.75) and d == pytest.approx(-0.75)
    assert (pb["n"], pb["n_fail"]) == (2, 1)
