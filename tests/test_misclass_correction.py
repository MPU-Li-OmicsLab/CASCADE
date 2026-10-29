import numpy as np

from tools import misclass_correction as MC


def _simulate(n, true_rate, rng, loglen=None):
    """按已知的长度依赖 se/sp 生成 (gold, machine, loglen)：长回复的灵敏度更低。"""
    loglen = rng.uniform(4, 10, n) if loglen is None else loglen
    gold = (rng.random(n) < true_rate).astype(float)
    se = 1 / (1 + np.exp(-(4.0 - 0.45 * loglen)))          # 长度越大越漏判
    sp = np.full(n, 0.97)
    machine = np.where(gold == 1, rng.random(n) < se, rng.random(n) > sp).astype(float)
    return gold, machine, loglen


def test_error_model_recovers_length_dependent_sensitivity():
    rng = np.random.default_rng(0)
    g, m, ll = _simulate(6000, 0.4, rng)
    b = MC.fit_error_model(g, m, ll, np.ones(len(g)))
    se, sp = MC.se_sp(b, np.array([5.0, 9.0]))
    assert se[0] > se[1] + 0.2                      # 短回复灵敏度明显更高
    assert abs(sp[0] - 0.97) < 0.04 and abs(sp[1] - 0.97) < 0.04


def test_rogan_gladen_undoes_length_driven_bias():
    rng = np.random.default_rng(1)
    g, m, ll = _simulate(8000, 0.4, rng)
    b = MC.fit_error_model(g, m, ll, np.ones(len(g)))
    # 两组真实率相同（0.30），但 B 组回复更长 → 观测率被压低；校正后应重新对齐
    ga, ma, la = _simulate(5000, 0.30, rng, loglen=rng.uniform(4, 6, 5000))
    gb, mb, lb = _simulate(5000, 0.30, rng, loglen=rng.uniform(8, 10, 5000))
    assert ma.mean() - mb.mean() > 0.08             # 纯由长度造成的假差异
    ca, _ = MC.rogan_gladen(ma, la, b); cb, _ = MC.rogan_gladen(mb, lb, b)
    assert abs(ca - cb) < 0.05 and abs(ca - 0.30) < 0.05


def test_rogan_gladen_skips_uninformative_band_and_handles_empty():
    b = np.array([0.0, 0.2, 0.0, 0.0])              # se≈0.55, sp=0.5 → se+sp−1≈0.05 < MIN_DENOM
    obs = np.array([1.0, 0.0, 1.0, 1.0] * 10); ll = np.linspace(4, 9, 40)
    rate, skipped = MC.rogan_gladen(obs, ll, b)
    assert skipped == 1.0 and rate == obs.mean()
    assert np.isnan(MC.rogan_gladen(np.array([]), np.array([]), b)[0])


def test_contrast_reports_observed_and_corrected_with_ci():
    rng = np.random.default_rng(2)
    g, m, ll = _simulate(1500, 0.4, rng)
    tab = {"loglen": ll, "w": np.ones(len(g)), "n": len(g), "abandon": (g, m)}
    def group(lo, hi, rate):
        out = {}
        for i in range(10):
            _, mm, l2 = _simulate(300, rate, rng, loglen=rng.uniform(lo, hi, 300))
            out[f"zh_p{i:02d}"] = {"loglen": l2, "abandon": mm}
        return out
    res = MC.contrast(tab, group(4, 6, 0.30), group(8, 10, 0.30), "abandon", 60, 0)
    assert res["n_personas"] == 10
    assert res["observed"]["diff"] < -0.05                       # 假差异
    assert res["corrected"]["ci"][0] < 0 < res["corrected"]["ci"][1]   # 校正后 CI 含 0
    md = MC.render({"meta": {"n_gold": 1, "reweighted": False, "n_bands": 4, "min_denom": 0.15, "n_boot": 60},
                    "error_model": {"abandon": {"20字": {"se": 0.9, "sp": 0.9}}},
                    "contrasts": {"x": {"abandon": res, "empathy": res}}})
    assert "校正差" in md


def test_low_youden_share_counts_turns_in_weak_bands():
    b = np.array([-2.0, 0.2, 0.0, 0.0])          # se ≈ sp 互补 → Youden ≈ 0.05，全部带都弱
    ll = np.log2(np.array([20, 70, 250, 800] * 5, float) + 1)
    assert MC.low_youden_share(ll, b) == 1.0
    strong = np.array([-4.0, 8.0, 0.0, 0.0])     # Youden ≈ 0.98
    assert MC.low_youden_share(ll, strong) == 0.0
    assert MC.low_youden_share(np.array([]), strong) == 0.0
