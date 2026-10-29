"""tools/gold_prevalence_raked：raking 后加权边际等于目标、权重为正；带上限版不超上限；缺类别时去掉并放大；整条流程可跑。"""
import numpy as np

from tools import gold_prevalence_raked as GR


def _toy_margins(seed=0, n=60):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, n)
    b = rng.integers(0, 4, n)
    c = rng.integers(0, 2, n)
    for k in range(3):                      # 每个类别至少一个样本
        a[k] = k
    for k in range(4):
        b[k] = k
    c[0], c[1] = 0, 1
    total = 1000.0
    ta = np.array([500.0, 300.0, 200.0])
    tb = np.array([100.0, 200.0, 300.0, 400.0])
    tc = np.array([650.0, 350.0])
    w0 = rng.uniform(5, 30, n)
    return w0, [(a, ta), (b, tb), (c, tc)], total


def _margins_hit(w, margins, tol=1e-8):
    return all(np.allclose(np.bincount(codes, weights=w, minlength=len(t)), t, rtol=tol, atol=0) for codes, t in margins)


def test_rake_matches_all_margins_and_positive():
    w0, margins, total = _toy_margins()
    w, info = GR.rake(w0, margins)
    assert info["converged"]
    assert np.all(w > 0)
    assert _margins_hit(w, margins)
    assert abs(w.sum() - total) < 1e-6


def test_rake_leaves_already_calibrated_weights():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    w0 = np.array([1.0, 2.0, 3.0, 4.0])
    margins = [(a, np.array([3.0, 7.0])), (b, np.array([4.0, 6.0]))]
    w, info = GR.rake(w0, margins)
    assert info["converged"] and np.allclose(w, w0)


def test_trim_rake_respects_cap_and_margins():
    w0, margins, _ = _toy_margins(seed=3)
    cap = 5 * np.median(w0)
    w, info = GR.trim_rake(w0, margins, cap=cap)
    assert info["converged"] and info["cap_met"]
    assert np.all(w > 0) and w.max() <= cap * (1 + 1e-9)
    assert _margins_hit(w, margins)


def test_rake_rejects_nonpositive_start():
    w0, margins, _ = _toy_margins()
    w0[0] = 0
    try:
        GR.rake(w0, margins)
    except ValueError:
        return
    raise AssertionError("应拒绝非正初始权重")


def test_build_margin_drops_absent_category_and_rescales():
    codes, t, dropped = GR.build_margin(["x", "x", "y"], {"x": 60, "y": 20, "z": 20}, 100.0)
    assert dropped == ["z"]
    assert np.isclose(t.sum(), 100.0) and np.allclose(t, [75.0, 25.0])
    assert list(codes) == [0, 0, 1]


def test_all_weights_on_toy_sampling_frame():
    """构造一个小抽样框：抽样格 2 个、模型 2 个、臂×风险段 2 个；raking 后三边际都还原全语料。"""
    frame, cells, model, ab = [], {}, {}, {}
    rng = np.random.default_rng(7)
    for k in range(400):
        i = f"t{k}"
        frame.append(i)
        cells[i] = (("d1", 0), bool(k % 5 == 0))
        model[i] = "m1" if k % 3 else "m2"
        ab[i] = ("fixed", "高") if rng.random() < 0.3 else ("responsive", "低")
    from collections import Counter
    sample = [frame[k] for k in rng.choice(400, 40, replace=False)]
    W, info = GR.all_weights(sample, cells, model, ab, Counter(cells[i] for i in frame),
                             Counter(model[i] for i in frame), Counter(ab[i] for i in frame))
    assert info["raked"]["converged"]
    for s in ("raked", "trim"):
        w = W[s]
        assert np.all(w > 0)
        for key in (cells, model, ab):
            pop = Counter(key[i] for i in frame)
            got = Counter()
            for i, wi in zip(sample, w):
                got[key[i]] += wi
            for c in got:
                assert abs(got[c] - pop[c]) < 1e-6 * pop[c]
    assert abs(W["orig"].sum() - 400) < 1e-9
