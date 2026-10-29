"""tools/length_bias_v7：长度效应方向能被找回、无方差返回 None、三分位表与复读剔除。"""
import numpy as np

from tools import length_bias_v7 as LB


def _rows(n=240, slope=0.0, seed=0):
    """(人设, 判官失败, 医师失败, log2 长度, 权重, 复读, API)。判官失败几率随长度按 slope 变化。"""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        g = float(k % 4 == 0)
        ln = float(rng.uniform(5, 11))
        p = 1 / (1 + np.exp(-(-1.5 + 3.0 * g + slope * (ln - 8))))
        rows.append((f"zh_p{k % 12:02d}", float(rng.random() < p), g, ln, 1.0, k % 30 == 0, float(k % 2)))
    return rows


def test_recovers_negative_length_effect():
    res = LB.analyse({"d1": _rows(slope=-0.9)}, n_boot=100, seed=1)["d1"]
    assert res["main"]["or"] < 1 and res["main"]["ci_or"][1] < 1
    assert res["n_loop"] == 8 and res["no_loop"]["n"] == 232


def test_no_effect_ci_covers_one():
    res = LB.analyse({"d1": _rows(slope=0.0, seed=3)}, n_boot=100, seed=1)["d1"]
    lo, hi = res["main"]["ci_or"]
    assert lo < 1 < hi


def test_fit_coef_none_without_variance():
    X = np.ones((10, 1))
    assert LB.fit_coef(X, np.zeros(10), np.ones(10), 0) is None


def test_tertiles_cover_all_rows():
    rows = _rows()
    t = LB.analyse({"d1": rows}, n_boot=20, seed=0)["d1"]["tertiles"]
    assert sum(t[k]["n_fail"] + t[k]["n_ok"] for k in range(3)) == len(rows)
