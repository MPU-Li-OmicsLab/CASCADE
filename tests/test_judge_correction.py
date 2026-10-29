"""tools/judge_correction：抽样格权重（含空格并入）、加权灵敏度 / 特异度、Rogan–Gladen、配对差、自助法可复现。"""
from collections import Counter

import pytest

from tools import judge_correction as JC


def test_cell_weights_basic_and_empty_cell_merge():
    cells = {"a": (("d1", 0), True), "b": (("d1", 0), True), "c": (("d2", 1), False)}
    pop = Counter({(("d1", 0), True): 10, (("d1", 0), False): 6, (("d2", 1), False): 50, (("d3", 2), False): 4})
    w, diag = JC.cell_weights(["a", "b", "c"], cells, pop)
    assert w == {"a": 8.0, "b": 8.0, "c": 50.0}          # (d1,0,False) 的 6 轮并入 (d1,0,True)
    assert diag["uncovered_turns"] == 4 and diag["pop_turns"] == 70
    assert sum(w.values()) == 66


def test_weighted_rates():
    r = JC.weighted_rates([(0, 0, 1.0), (2, 0, 3.0), (0, 2, 1.0), (2, 2, 9.0)], "d1")
    assert r["sens"] == 0.25 and r["spec"] == 0.9 and r["n_fail"] == 2 and r["prev_w"] == round(4 / 14, 4)
    r5 = JC.weighted_rates([(1, 1, 1.0), (2, 0, 1.0), (2, 2, 1.0)], "d5")     # d5 失败 = ≤1
    assert r5["sens"] == 0.5 and r5["spec"] == 1.0


def test_rogan_gladen_and_diff():
    assert JC.rogan_gladen(0.2, 0.8, 0.9) == pytest.approx(0.1 / 0.7)
    assert JC.rogan_gladen(0.05, 0.8, 0.9) == 0.0                           # 截到 0
    assert JC.rogan_gladen(0.3, 0.5, 0.52) is None                          # Youden 太低
    assert JC.corrected_diff(0.3, 0.1, 0.8, 0.9) == pytest.approx(0.2 / 0.7)
    assert JC.corrected_diff(0.5, 0.1, 0.55, 0.6) == 1.0                    # 放大后截到 ±100%


def test_group_of():
    assert JC.group_of("openai/gpt-5.6-sol") == "API" and JC.group_of("cpsycoun") == "本地"


def test_corpus_table_and_rate(tmp_path):
    import json
    p = tmp_path / "c.jsonl"
    rows = [{"target_model_id": "m", "arm": "fixed", "case_profile": {"id": "zh_p01-c01"},
             "turns": [{"turn": 1, "score": {"d1": 0, "d2": 2, "d3": 2, "d4": 2, "d5": 1}},
                       {"turn": 2, "score": {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2}},
                       {"turn": 3, "score": None}]}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    t = JC.corpus_table(p)
    assert t[("m", "fixed")] == [("zh_p01", {d: 2 for d in JC.DIMS5}, {"d1": 1, "d2": 0, "d3": 0, "d4": 0, "d5": 1})]
    assert JC.rate(t[("m", "fixed")], "d1") == 0.5


def test_corpus_table_imputes_d5_for_unscored_loops(tmp_path):
    import json
    loop = "我在这里陪着你，别害怕。" * 4
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"target_model_id": "m", "arm": "fixed", "case_profile": {"id": "zh_p01-c01"},
                             "turns": [{"turn": 1, "score": {d: 2 for d in JC.DIMS5}, "target_response": "好"},
                                       {"turn": 2, "score": None, "target_response": loop},
                                       {"turn": 3, "score": None, "target_response": "短"}]}), encoding="utf-8")
    st = {}
    (per, n, k), = JC.corpus_table(p, stats=st)[("m", "fixed")]
    assert per == "zh_p01"
    assert n["d5"] == 2 and k["d5"] == 1 and n["d1"] == 1
    assert st == {"loop_imputed_d5": 1, "unscored": 1}


def _toy():
    ids, gold, pred, cells = [], {}, {}, {}
    for p in range(1, 9):
        for n in range(6):
            i = f"m{p % 2}|fixed|zh_p{p:02d}-c{n:02d}#t1"
            ids.append(i)
            g = 0 if n < 2 else 2
            gold[i] = {d: g for d in JC.DIMS5}
            pred[i] = {d: (g if n != 2 else 0) for d in JC.DIMS5}
            cells[i] = (("d1", 0), True) if n < 3 else (("d2", 1), False)
    pop = Counter({(("d1", 0), True): 100, (("d2", 1), False): 900})
    corpus = {(f"m{k}", "fixed"): [(f"zh_p{j:02d}", {d: 10 for d in JC.DIMS5}, {d: 2 + k for d in JC.DIMS5})
                                     for j in range(8)]
              for k in (0, 1)}
    return ids, gold, pred, cells, pop, corpus


def test_analyse_reproducible_and_contrast_present():
    ids, gold, pred, cells, pop, corpus = _toy()
    a = JC.analyse(ids, gold=gold, pred=pred, cells=cells, pop=pop, corpus=corpus,
                   pairs=(("m1", "m0"),), n_boot=50, seed=1)
    b = JC.analyse(ids, gold=gold, pred=pred, cells=cells, pop=pop, corpus=corpus,
                   pairs=(("m1", "m0"),), n_boot=50, seed=1)
    assert a["ci"] == b["ci"]
    assert a["gold"]["weighted"]["d1"]["sens"] == 1.0
    raw, cor = a["contrasts"][("m1", "m0", "fixed", "d1")]
    assert raw == pytest.approx(0.1) and cor is not None and cor > raw
