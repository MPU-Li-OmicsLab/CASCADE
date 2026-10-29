from tools.multiplicity import FAMILIES, P1, P2, bh, holm, p_from_ci, run
from tools.multiplicity import test as mk


def test_holm_known_values():
    assert [round(x, 4) for x in holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]


def test_bh_known_values():
    assert [round(x, 4) for x in bh([0.01, 0.04, 0.03, 0.20])] == [0.04, 0.0533, 0.0533, 0.2]


def test_p_from_ci_matches_ci_exclusion():
    assert p_from_ci(0.10, (0.02, 0.30)) < 0.05          # CI 不含 0
    assert p_from_ci(0.10, (-0.01, 0.30)) > 0.05         # CI 含 0
    assert abs(p_from_ci(0.10, (0.0, 0.30)) - 0.05) < 1e-3   # 端点恰在 0 → 边界
    assert p_from_ci(-0.10, (-0.30, -0.02)) < 0.05       # 负方向用上端半宽
    assert p_from_ci(0.0, (-0.1, 0.1)) == 1.0
    assert p_from_ci(0.1, (0.1, 0.3)) is None            # 退化


def test_run_families_and_strict_version():
    T = [mk(P1, "a", 0.1, (0.01, 0.2), source="x"), mk(P1, "b", 0.1, (0.03, 0.2), source="x"),
         mk(P2, "c", -0.1, (-0.2, -0.05), source="x"), mk(FAMILIES[3][0], "d", 0.1, (-0.1, 0.3), source="x"),
         mk(FAMILIES[8][0], "e", -0.07, (-0.12, -0.01), p=0.044, source="x")]
    res = run(T)
    by = {t["label"]: t for t in res["tests"]}
    assert by["e"]["p_source"] == "报告" and by["e"]["p"] == 0.044
    assert by["a"]["sig_raw"] and not by["d"]["sig_raw"]
    assert by["a"]["p_strict"] >= by["a"]["p_adj"]       # 合并成一族只会更严
    assert "p_strict" not in by["d"]
    assert by["c"]["p_adj"] == by["c"]["p"]              # 族内只有一个检验
