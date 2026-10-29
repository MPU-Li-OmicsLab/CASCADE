"""tools/judge_select：预先写定的判官版本选择规则（（内部文档））。"""
import json

import pytest

from tools import judge_select as JS


def _gold(tmp_path, rows, split="dev"):
    p = tmp_path / f"gold_{split}.jsonl"
    p.write_text("\n".join(json.dumps({"id": i, "split": split, "loop": loop, "consensus": c}, ensure_ascii=False)
                           for i, c, loop in rows) + "\n", encoding="utf-8")
    return p


def _scored(tmp_path, name, scores):
    """scores = {turn: {d1..d6}}，都挂在同一段对话 m|fixed|p 上。"""
    rec = {"target_model_id": "m", "arm": "fixed", "case_profile": {"id": "p"},
           "turns": [{"turn": t, "score": s} for t, s in scores.items()]}
    p = tmp_path / f"{name}.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return p


def _c(*v):
    return dict(zip(JS.DIMS, v))


def test_choose_prefers_v6cap_within_tie_and_ignores_uncapped():
    assert JS.choose({"v5cap": 0.62, "v6cap": 0.60, "v5": 0.99}) == "v6cap"     # 差 0.02 < 0.03
    assert JS.choose({"v5cap": 0.64, "v6cap": 0.60}) == "v5cap"
    assert JS.choose({"v5": 0.9}) is None
    assert JS.choose({"v6cap": 0.3, "claudecap": 0.9}) == "v6cap"                # 诊断判官不参与
    assert JS.choose({"claudecap": 0.9}, candidates=()) is None


def test_total_counts_undefined_kappa_as_zero():
    res = {d: {"zero_kappa": 0.5, "wkappa": 0.5} for d in JS.DIMS}
    res["d3"]["zero_kappa"] = None
    assert JS.total(res) == pytest.approx(0.4)


def test_thresholds_floor_and_relative():
    human = {"d1": {"zero_kappa": 0.9}, "d2": {"zero_kappa": 0.3}, "d3": {"zero_kappa": 0.6},
             "d4": {"zero_kappa": 0.8}, "d5": {"wkappa": 0.9}}
    th = JS.thresholds(human)
    assert th["d1"] == pytest.approx(0.72) and th["d2"] == pytest.approx(0.40) and th["d5"] == pytest.approx(0.72)


def test_evaluate_matches_ids_and_excludes_loop_segments(tmp_path):
    g = _gold(tmp_path, [("m|fixed|p#t1", _c(0, 2, 1, 2, 2, 2), False),
                         ("m|fixed|p#t2", _c(2, 0, 1, 2, 1, 2), True),
                         ("m|fixed|p#t3", _c(2, 2, 0, 1, 2, 2), False)])
    s = _scored(tmp_path, "v6cap", {1: _c(0, 2, 1, 2, 2, 2), 2: _c(2, 2, 1, 2, 1, 2), 3: _c(2, 2, 0, 1, 2, 2)})
    r = JS.run(g, [("v6cap", s)])
    d2 = r["judges"]["v6cap"]["all"]["d2"]
    assert d2["n"] == 3 and d2["judge_non0_gold0"] == 1 and d2["judge0_gold_non0"] == 0
    assert r["judges"]["v6cap"]["no_loop"]["d2"]["n"] == 2
    assert r["judges"]["v6cap"]["all"]["d1"]["zero_kappa"] == 1.0
    assert r["chosen"] == "v6cap"


def test_test_split_requires_final(tmp_path):
    g = _gold(tmp_path, [("m|fixed|p#t1", _c(2, 2, 2, 2, 2, 2), False)], split="test")
    s = _scored(tmp_path, "v6cap", {1: _c(2, 2, 2, 2, 2, 2)})
    with pytest.raises(SystemExit):
        JS.main(["--gold", str(g), "--judge", f"v6cap={s}", "--out", str(tmp_path / "o")])
    assert JS.main(["--gold", str(g), "--judge", f"v6cap={s}", "--out", str(tmp_path / "o"), "--final",
                    "--report-json", str(tmp_path / "none.json")]) == 0


def test_warns_when_candidates_have_different_n(tmp_path):
    gold = _gold(tmp_path, [("m|fixed|p#t1", _c(0, 1, 2, 0, 1, 2), False), ("m|fixed|p#t2", _c(1, 1, 1, 1, 1, 1), False)])
    full = _scored(tmp_path, "a", {1: _c(0, 1, 2, 0, 1, 2), 2: _c(1, 1, 1, 1, 1, 1)})
    part = _scored(tmp_path, "b", {1: _c(0, 1, 2, 0, 1, 2)})                  # 第 2 轮失败 = 没分
    r = JS.run(gold, [("v6cap", full), ("v7cap", part)], None)
    assert r["n_mismatch"]["d1"] == {"v6cap": 2, "v7cap": 1}
    assert "样本数 n 不同" in JS.render(r)
    same = JS.run(gold, [("v6cap", full)], None)
    assert same["n_mismatch"] == {} and "样本数 n 不同" not in JS.render(same)
