"""tools/misclass_correction_clustered.py：金标按人设整组重抽的逻辑。零 GPU、零真数据。"""
import numpy as np
import pytest

from tools import misclass_correction as mc
from tools import misclass_correction_clustered as mcc


def test_persona_and_case_from_real_id():
    rid = "zh_p08-c23#openai/gpt-5.4-mini#fixed#t8"
    assert mcc.persona_of(rid) == "zh_p08"
    assert mcc.case_of(rid) == "zh_p08-c23"


def test_cluster_index_groups_and_unmapped_become_singletons():
    labels = np.array(["p1", "p2", "p1", "", "p2", ""], dtype=object)
    names, members = mcc.cluster_index(labels)
    assert len(names) == 4                         # p1, p2 + 两个映射不到的单元各成一簇
    assert members["p1"].tolist() == [0, 2]
    assert members["p2"].tolist() == [1, 4]
    singles = [k for k in names if k.startswith("__unit")]
    assert sorted(members[k].tolist()[0] for k in singles) == [3, 5]


def test_cluster_resample_keeps_whole_clusters():
    labels = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2, dtype=object)
    names, members = mcc.cluster_index(labels)
    rng = np.random.default_rng(1)
    sizes = set()
    for _ in range(200):
        gi = mcc.cluster_resample(names, members, rng)
        picked = labels[gi]
        # 每个被抽中的簇必须整组出现（次数 × 簇大小），不会只抽到簇里的一部分
        for k, idx in members.items():
            cnt = int((picked == k).sum())
            assert cnt % len(idx) == 0
            if cnt:
                assert sorted(gi[picked == k].tolist()) == sorted(idx.tolist() * (cnt // len(idx)))
        assert sum(int((picked == k).sum()) // len(members[k]) for k in members) == len(names)
        sizes.add(len(gi))
    assert len(sizes) > 1                          # 样本量随簇大小浮动


def _synthetic(seed=0, n_personas=6, per=30):
    rng = np.random.default_rng(seed)
    n = n_personas * per
    labels = np.array([f"p{i // per}" for i in range(n)], dtype=object)
    loglen = rng.uniform(3, 9, n)
    gold = (rng.random(n) < 0.3).astype(float)
    # 人设层面的随机效应让误判在簇内相关
    eff = {f"p{i}": rng.normal(0, 1.2) for i in range(n_personas)}
    z = -2.5 + 4.5 * gold + np.array([eff[l] for l in labels])
    machine = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    tab = {"loglen": loglen, "w": np.ones(n), "n": n, "abandon": (gold, machine)}

    def tt(rate):
        out = {}
        for p in range(8):
            ll = rng.uniform(3, 9, 40)
            out[f"q{p}"] = {"loglen": ll, "abandon": (rng.random(40) < rate).astype(float)}
        return out
    return tab, labels, tt(0.15), tt(0.35)


def test_point_estimate_identical_to_original_contrast():
    tab, labels, ta, tb = _synthetic()
    orig = mc.contrast(tab, ta, tb, "abandon", 50, 0)
    new = mcc.contrast_clustered(tab, labels, ta, tb, "abandon", 50, 0)
    assert new["corrected"]["diff"] == orig["corrected"]["diff"]
    assert new["observed"]["diff"] == orig["observed"]["diff"]
    assert new["n_gold_clusters"] == 6


def test_each_unit_its_own_cluster_matches_unit_bootstrap_width():
    """每单元一簇时等价于逐条重抽（随机数序列不同，宽度应接近）。"""
    tab, _, ta, tb = _synthetic(seed=3)
    singles = np.array([f"u{i}" for i in range(tab["n"])], dtype=object)
    w_unit = np.diff(mc.contrast(tab, ta, tb, "abandon", 400, 0)["corrected"]["ci"])[0]
    w_single = np.diff(mcc.contrast_clustered(tab, singles, ta, tb, "abandon", 400, 0)["corrected"]["ci"])[0]
    assert w_single == pytest.approx(w_unit, rel=0.35)


def test_compare_modes_unit_calls_original():
    tab, labels, ta, tb = _synthetic()
    out = mcc.compare_modes(tab, {"persona": labels}, ta, tb, "abandon", 30, 0, ("unit", "persona"))
    assert out["unit"] == mc.contrast(tab, ta, tb, "abandon", 30, 0)
    assert "n_gold_clusters" in out["persona"]


def test_crosses_zero_and_flag():
    assert mcc.crosses_zero((-0.1, 0.05))
    assert not mcc.crosses_zero((-0.3, -0.01))
    assert mcc._flag((-0.3, -0.01), (-0.4, 0.02)) == "⚠️ 新跨 0"
    assert mcc._flag((-0.3, -0.01), (-0.4, -0.02)) == "稳健"
    assert mcc._flag((-0.3, 0.1), (-0.4, 0.2)) == "原已跨 0"


def test_check_against_original_detects_mismatch():
    res = {"contrasts": {"x": {"abandon": {"unit": {"corrected": {"diff": -0.1, "ci": (-0.2, -0.05)}}}}}, "dose": {}}
    good = {"contrasts": {"x": {"abandon": {"corrected": {"diff": -0.1, "ci": [-0.2, -0.05]}}}}}
    bad = {"contrasts": {"x": {"abandon": {"corrected": {"diff": -0.1, "ci": [-0.2, -0.04]}}}}}
    assert mcc.check_against_original(res, good)["all_match"]
    assert not mcc.check_against_original(res, bad)["all_match"]
