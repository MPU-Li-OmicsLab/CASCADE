from scipy import stats

from tools.judge_spearman import TOTAL, correlations, item_value, wspearman


def test_unweighted_matches_scipy_with_ties():
    x = [0, 1, 2, 2, 1, 0, 2, 2, 1]
    y = [0, 2, 2, 1, 1, 0, 2, 2, 0]
    assert abs(wspearman(x, y) - stats.spearmanr(x, y).statistic) < 1e-12


def test_integer_weights_equal_duplication():
    x, y, w = [0, 1, 2, 2, 1], [0, 2, 2, 1, 1], [1, 3, 1, 2, 1]
    xd = [v for v, k in zip(x, w) for _ in range(k)]
    yd = [v for v, k in zip(y, w) for _ in range(k)]
    assert abs(wspearman(x, y, w) - wspearman(xd, yd)) < 1e-12


def test_constant_side_is_none():
    assert wspearman([2, 2, 2, 2], [0, 1, 2, 1]) is None


def test_total_requires_all_dims():
    s = {"d1": 2, "d2": 1, "d3": 2, "d4": 0, "d5": 1}
    assert item_value(s, TOTAL) == 6
    assert item_value({**s, "d3": None}, TOTAL) is None


def test_correlations_keys_and_identity():
    rows = []
    for i, v in enumerate([0, 1, 2, 2, 1, 0]):
        s = {d: v for d in ("d1", "d2", "d3", "d4", "d5")}
        rows.append({"id": f"m|fixed|zh_p0{i}-c01#t1", "gold": s, "a": s, "b": s, "judges": {"J": s}})
    w = {r["id"]: 1.0 for r in rows}
    res = correlations(rows, w)
    assert abs(res["d1"]["医师a vs 医师b|不加权"] - 1.0) < 1e-12
    assert abs(res[TOTAL]["J vs 单个医师|加权"] - 1.0) < 1e-12
