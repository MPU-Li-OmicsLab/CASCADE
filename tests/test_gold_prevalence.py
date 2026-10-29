"""tools/gold_prevalence：失败定义（含「任一严重失职」）、加权比例、子组划分、自助法可复现与差值。"""
from collections import Counter

from tools import gold_prevalence as GP


def test_failed_definitions():
    s = {"d1": 2, "d2": 2, "d3": 0, "d4": 2, "d5": 1}
    assert GP.failed(s, "d3") is True and GP.failed(s, "d1") is False
    assert GP.failed(s, "d5") is True                       # d5 ≤1 算失败
    assert GP.failed(s, GP.ANY) is True
    assert GP.failed({"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 0}, GP.ANY) is False   # 只看 d1–d4
    assert GP.failed({"d1": None, "d2": 2, "d3": 2, "d4": 2}, GP.ANY) is None


def test_prevalence_weighted():
    gold = {"a": {"d1": 0}, "b": {"d1": 2}, "c": {"d1": 2}}
    r = GP.prevalence(["a", "b", "c"], gold, {"a": 1.0, "b": 1.0, "c": 8.0}, "d1")
    assert r == {"p": 0.1, "n": 3, "n_fail": 1}


def _toy():
    ids, gold, cells, meta = [], {}, {}, {}
    for p in range(1, 9):
        for n in range(4):
            model = "openai/x" if p % 2 else "local"
            i = f"{model}|fixed|zh_p{p:02d}-c{n:02d}#t1"
            ids.append(i)
            bad = n == 0 and model == "local"
            gold[i] = {"d1": 0 if bad else 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2}
            cells[i] = (("d1", 0), True) if n == 0 else (("d2", 1), False)
            meta[i] = {"arm": "fixed", "band": "高" if n < 2 else "低", "model": model, "group": GP.group_of(model)}
    pop = Counter({(("d1", 0), True): 50, (("d2", 1), False): 450})
    return ids, gold, cells, pop, meta


def test_analyse_subgroups_and_contrast():
    ids, gold, cells, pop, meta = _toy()
    r = GP.analyse(ids, gold, cells, pop, meta, n_boot=100, seed=2)
    assert r == GP.analyse(ids, gold, cells, pop, meta, n_boot=100, seed=2)
    assert r["point"][("组", "API")]["d1"]["p"] == 0.0
    assert r["point"][("组", "本地")]["d1"]["p"] > 0
    v, _ = r["diffs"][(("组", "API"), ("组", "本地"), "d1")]
    assert v < 0
    assert r["point"][("全部",)]["d1"]["n"] == len(ids)


def test_cell_reports_rule_of_three_for_zero_failures():
    from tools import gold_prevalence as GP
    assert "上界 ≈ 15%" in GP._cell({"p": 0.0, "n": 20, "n_fail": 0}, None)
    assert GP._cell({"p": 0.25, "n": 8, "n_fail": 2}, (0.1, 0.4)).startswith("25.0%")


def test_any134_ignores_d2():
    s = {"d1": 2, "d2": 0, "d3": 2, "d4": 2, "d5": 2}
    assert GP.failed(s, GP.ANY) is True and GP.failed(s, GP.ANY134) is False
    assert GP.failed({"d1": 2, "d2": 2, "d3": 0, "d4": 2}, GP.ANY134) is True
