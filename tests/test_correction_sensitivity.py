"""tools/correction_sensitivity：金标按 split 拆分、权重按子样本重算、切换拟合样本后误判表与结果抽取。"""
from collections import Counter

import numpy as np
import pytest

from tools import correction_sensitivity as CS
from tools import judge_dose as JD


def _inp():
    """12 段合成金标：dev 8 段（判官与医师一致）、test 4 段（判官全漏判）。两个抽样格。"""
    ids, gold, pred, replies, split, cells = [], {}, {}, {}, {}, {}
    for k in range(12):
        model = "gpt/x" if k % 3 == 0 else "local"
        i = f"{model}|fixed|zh_p{k % 4:02d}-c01#t{k}"
        ids.append(i)
        s = "dev" if k < 8 else "test"
        split[i] = s
        g_fail = k % 2 == 0
        gold[i] = {d: (0 if g_fail else 2) for d in ("d1", "d2", "d3", "d4")} | {"d5": 1 if g_fail else 2}
        j_fail = g_fail if s == "dev" else False
        pred[i] = {d: (0 if j_fail else 2) for d in ("d1", "d2", "d3", "d4")} | {"d5": 1 if j_fail else 2}
        replies[i] = "字" * (20 + 10 * k)
        cells[i] = (("d1", 0), k % 2 == 0)
    return {"ids": ids, "gold": gold, "pred": pred, "replies": replies, "split": split, "cells": cells,
            "pop": Counter({(("d1", 0), True): 600, (("d1", 0), False): 400})}


def test_sample_ids_splits_and_all():
    inp = _inp()
    assert CS.sample_ids(inp["ids"], inp["split"], "all") == inp["ids"]
    dev = CS.sample_ids(inp["ids"], inp["split"], "dev")
    test = CS.sample_ids(inp["ids"], inp["split"], "test")
    assert len(dev) == 8 and len(test) == 4 and not set(dev) & set(test)
    assert set(dev) | set(test) == set(inp["ids"])
    with pytest.raises(ValueError):
        CS.sample_ids(inp["ids"], inp["split"], "train")


def test_split_missing_only_in_all():
    inp = _inp()
    inp["split"][inp["ids"][0]] = None
    assert inp["ids"][0] in CS.sample_ids(inp["ids"], inp["split"], "all")
    assert inp["ids"][0] not in CS.sample_ids(inp["ids"], inp["split"], "dev")


def test_gold_tables_reweight_within_subsample():
    """子样本的权重总和 = 全语料轮数（按子样本重算），且两套金标行格式一致。"""
    inp = _inp()
    for s in CS.SAMPLES:
        sub = CS.sample_ids(inp["ids"], inp["split"], s)
        la, dose, diag = CS.gold_tables(sub, inp)
        assert diag["n"] == len(sub) and diag["uncovered_turns"] == 0
        assert abs(sum(r[4] for r in dose["d1"]) - 1000) < 1e-6
        assert set(dose) == set(JD.OUTCOMES) and set(la) == set(JD.DIMS5)
        assert [r[:4] for r in la["d1"]] == [r[:4] for r in dose["d1"]]


def test_switching_fit_sample_changes_error_rates():
    """dev 判官全对 → 灵敏度高；test 判官全漏 → 灵敏度低。拟合样本切换必须反映出来。"""
    inp = _inp()
    se = {}
    for s in ("dev", "test"):
        _, dose, _ = CS.gold_tables(CS.sample_ids(inp["ids"], inp["split"], s), inp)
        e = CS.error_table(dose)["d1"]
        assert e["n_fail"]["API"] + e["n_fail"]["本地"] == sum(r[2] for r in dose["d1"])
        se[s] = None if e["grid"] is None else e["grid"]["本地"]["127字"][0]
    assert se["test"] is None or se["dev"] > se["test"]


def test_error_table_unfittable_without_variance():
    rows = {"d1": [("p", 0.0, 1.0, 5.0, 1.0, 0.0)] * 12}
    assert CS.error_table(rows)["d1"]["grid"] is None


def test_views_and_ci_flag():
    dose_res = {"point": {("本地", "fixed", o): {"diff": (0.1, 0.2)} for o in JD.OUTCOMES},
                "ci": {("本地", "fixed", o): ((0.05, 0.15), (-0.01, 0.3)) for o in JD.OUTCOMES}}
    v = CS.c2_view(dose_res)
    assert v["any"]["cor"] == 0.2 and CS.ci_excludes_zero(v["any"]["cor_ci"]) is False
    assert CS.ci_excludes_zero((0.01, 0.3)) is True and CS.ci_excludes_zero(None) is None
    la_res = {"point": {("a", "c", "fixed", "d1"): (-0.1, None)}, "ci": {("a", "c", "fixed", "d1"): ((-0.2, -0.05), None)}}
    assert CS.c5_view(la_res)[("a", "c", "fixed", "d1")]["cor"] is None
    assert "不可校正" in CS._cell(CS.c5_view(la_res)[("a", "c", "fixed", "d1")])
    assert CS._shrink({"cor": 0.2}, {"cor": 0.1}) == pytest.approx(0.5)
    assert CS._shrink({"cor": 0.2}, {"cor": None}) is None


def test_analyse_runs_on_each_subsample():
    """端到端：合成语料上三种拟合样本都能跑 judge_dose.analyse，不抛错。"""
    inp = _inp()
    crows = []
    rng = np.random.default_rng(0)
    for p in ("zh_p00", "zh_p01", "zh_p02", "zh_p03"):
        for band in ("低", "中", "高"):
            for _ in range(5):
                f = int(rng.random() < (0.5 if band == "高" else 0.2))
                fails = {o: f for o in JD.OUTCOMES}
                crows.append(("local", "fixed", p, band, float(rng.uniform(5, 9)), fails))
    for s in CS.SAMPLES:
        _, dose, _ = CS.gold_tables(CS.sample_ids(inp["ids"], inp["split"], s), inp)
        r = JD.analyse(dose, crows, n_boot=5, seed=0)
        v = CS.c2_view(r)
        assert "any" in v and v["any"]["obs"] is not None
