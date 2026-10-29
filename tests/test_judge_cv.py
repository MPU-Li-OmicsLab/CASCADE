"""tools/judge_cv：按人设分组折、折内选规则、折外预测汇总、失败判定的灵敏度 / 特异度、人设聚类自助法。"""
import pytest

from tools import judge_cv as CV


def _id(persona, n):
    return f"m|fixed|{persona}-c{n:02d}#t{n % 5 + 1}"


def test_persona_and_dialogue_of():
    tid = "cpsycoun|responsive|zh_p12-c27#t4"
    assert CV.persona_of(tid) == "zh_p12"
    assert CV.dialogue_of(tid) == "cpsycoun|responsive|zh_p12-c27"


def test_group_folds_keep_groups_together_and_cover_all():
    ids = [_id(f"zh_p{p:02d}", n) for p in range(1, 11) for n in range(p)]
    folds = CV.group_folds(ids, k=5, seed=3, key=CV.persona_of)
    assert set(folds) == set(ids) and set(folds.values()) == set(range(5))
    for p in {CV.persona_of(i) for i in ids}:
        assert len({folds[i] for i in ids if CV.persona_of(i) == p}) == 1
    assert folds == CV.group_folds(ids, k=5, seed=3, key=CV.persona_of)      # 同种子可复现


def test_group_folds_rejects_too_few_groups():
    with pytest.raises(ValueError):
        CV.group_folds([_id("zh_p01", 1), _id("zh_p02", 2)], k=5, seed=0, key=CV.persona_of)


def _gold(ids, d1):
    return {i: {"d1": v, "d2": 2, "d3": 2, "d4": 2, "d5": 2} for i, v in zip(ids, d1)}


def test_choose_rules_picks_best_and_prefers_frozen_on_tie():
    ids = [_id("zh_p01", n) for n in range(6)]
    gold = _gold(ids, [0, 0, 0, 2, 2, 2])
    good = {i: {"d1": g["d1"], "d2": 2, "d3": 2, "d4": 2, "d5": 2} for i, g in gold.items()}
    bad = {i: {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2} for i in ids}
    table = {"r1": good, "r5": bad, "r6": bad}
    got = CV.choose_rules(ids, table, gold, ["r1", "r5", "r6"])
    assert got["d1"] == "r1"
    assert got["d2"] == "r6"            # 三版在 d2 上一样（都无方差）→ 取定版


def test_failure_rates_d5_uses_le1():
    pairs = [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
    r = CV.failure_rates(pairs, "d1")
    assert r == {"n_gold_fail": 3, "n_gold_ok": 2, "sens": round(1 / 3, 4), "spec": 0.5}
    r5 = CV.failure_rates(pairs, "d5")                     # d5：≤1 算失败
    assert r5["sens"] == round(2 / 3, 4) and r5["spec"] == 0.5


def test_cross_validate_oof_covers_every_id_and_uses_train_only():
    ids = [_id(f"zh_p{p:02d}", n) for p in range(1, 7) for n in range(4)]
    gold = _gold(ids, [0 if n % 2 else 2 for n in range(len(ids))])
    perfect = {i: dict(g) for i, g in gold.items()}
    flat = {i: {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2} for i in ids}
    oof, choices = CV.cross_validate(ids, {"r0": flat, "r6": perfect}, gold, k=3, seed=1,
                                     key=CV.persona_of, rulesets=["r0", "r6"])
    assert set(oof) == set(ids) and len(choices) == 3
    assert all(c["d1"] == "r6" for c in choices)
    assert all(oof[i]["d1"] == gold[i]["d1"] for i in ids)


def test_bootstrap_ci_is_reproducible_and_brackets_point():
    ids = [_id(f"zh_p{p:02d}", n) for p in range(1, 9) for n in range(5)]
    gold = _gold(ids, [0 if (n * 7) % 3 == 0 else 2 for n in range(len(ids))])
    pred = {i: dict(g) for i, g in gold.items()}
    pred[ids[0]]["d1"] = 2 if gold[ids[0]]["d1"] == 0 else 0
    a = CV.bootstrap(pred, gold, ids, n_boot=200, seed=5)
    assert a == CV.bootstrap(pred, gold, ids, n_boot=200, seed=5)
    point = CV.evaluate(pred, gold, ids)["dims"]["d1"]["zero_kappa"]
    lo, hi = a["d1"]["metric"]
    assert lo <= point <= hi
