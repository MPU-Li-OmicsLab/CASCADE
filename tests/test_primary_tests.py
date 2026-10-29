"""tools/primary_tests：自助百分位反转 p、Holm、退化判定、主假设定义。"""
import numpy as np
import pytest

from tools.primary_tests import HYPOTHESES, PA, PB, boot_p, degenerate, family_holm
from tools.multiplicity import holm


def test_boot_p_known_distribution():
    # 1999 个，正值 1899、负值 100：P(≤0) = 101/2000，P(≥0) = 1900/2000 → p = 2·101/2000
    d = [1.0] * 1899 + [-1.0] * 100
    r = boot_p(d)
    assert r["B"] == 1999
    assert r["p"] == pytest.approx(2 * 101 / 2000)


def test_boot_p_all_one_side_never_zero():
    r = boot_p([0.3] * 999)
    assert r["p"] == pytest.approx(2 / 1000)
    assert r["p"] > 0


def test_boot_p_symmetric_is_one():
    assert boot_p([-1.0, 1.0] * 500)["p"] == 1.0


def test_boot_p_zeros_count_both_sides():
    # 端点恰为 0 的点同时计入 ≤0 与 ≥0
    d = [0.0] * 30 + [0.5] * 970
    r = boot_p(d)
    assert r["le0"] == pytest.approx(31 / 1001)
    assert r["p"] == pytest.approx(2 * 31 / 1001)


def test_boot_p_drops_missing():
    r = boot_p([None, np.nan, 1.0, 1.0, -1.0])
    assert r["B"] == 3
    assert r["p"] == pytest.approx(2 * 2 / 4)


def test_boot_p_matches_normal_for_symmetric_case():
    rng = np.random.default_rng(0)
    d = rng.normal(0.2, 0.1, 200000)          # 真 p ≈ 2·Φ(−2) = 0.0455
    assert boot_p(d)["p"] == pytest.approx(0.0455, abs=0.003)


def test_holm_known():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_holm_monotone_and_capped():
    adj = holm([0.5, 0.6, 0.001])
    assert adj[2] == pytest.approx(0.003)
    assert adj[0] == pytest.approx(1.0) and adj[1] == pytest.approx(1.0)


def test_degenerate():
    assert degenerate(0, 10, [0.1] * 100) == "某组金标有效段数为 0"
    assert degenerate(10, 10, [0.0] * 100) == "自助差值分布全为 0"
    assert degenerate(10, 10, [0.1] * 10 + [None] * 90).startswith("有效重抽")
    assert degenerate(10, 10, [0.0] * 99 + [0.1]) is None


def test_family_holm_excludes_degenerate_and_splits_families():
    rows = [{"id": "a", "family": "F1", "p": 0.01, "degenerate": None},
            {"id": "b", "family": "F2", "p": 0.02, "degenerate": None},
            {"id": "c", "family": "F2", "p": 0.03, "degenerate": None},
            {"id": "d", "family": "F2", "p": None, "degenerate": "自助差值分布全为 0"}]
    fam = family_holm(rows, "p", lambda r: r["family"])
    assert fam[("F1", "a")] == pytest.approx(0.01)
    assert fam[("F2", "b")] == pytest.approx(0.04)
    assert fam[("F2", "c")] == pytest.approx(0.04)
    assert ("F2", "d") not in fam
    merged = family_holm(rows, "p", lambda r: "M")
    assert merged[("M", "a")] == pytest.approx(0.03)
    assert merged[("M", "c")] == pytest.approx(0.04)


def test_hypotheses_definition():
    ids = [h[0] for h in HYPOTHESES]
    assert len(ids) == len(set(ids)) == 7
    assert sum(h[1] == PA for h in HYPOTHESES) == 1
    assert sum(h[1] == PB for h in HYPOTHESES) == 6
    pa = HYPOTHESES[0]
    assert pa[2] == ("设计组", "旗舰") and pa[3] == ("组", "本地") and pa[4] == "any"
    pairs = {(h[2][1], h[3][1]) for h in HYPOTHESES if h[2][0] == "模型"}
    assert pairs == {("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base")}
