"""统计小工具：按人设的 cluster bootstrap、配对 Wilcoxon、单向 ICC、Spearman、k-means + silhouette。
只依赖 numpy / scipy（conda env 无 sklearn / statsmodels）。"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def cluster_boot_ci(values_by_cluster: dict[str, list[float]], seed: int = 0,
                    n_boot: int = 1000) -> tuple[float, float]:
    """按簇（人设）有放回重抽，簇内全取；统计量 = 所有观测的均值。"""
    keys = [k for k, v in values_by_cluster.items() if v]
    if not keys:
        return (float("nan"), float("nan"))
    arrs = [np.asarray(values_by_cluster[k], dtype=float) for k in keys]
    if len(arrs) == 1:
        m = float(arrs[0].mean())
        return (m, m)
    rng = np.random.default_rng(seed)
    sums = np.array([a.sum() for a in arrs]); ns = np.array([len(a) for a in arrs])
    idx = rng.integers(0, len(arrs), size=(n_boot, len(arrs)))
    means = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def paired_wilcoxon(a: list[float], b: list[float], seed: int = 0, n_boot: int = 2000) -> dict:
    """按簇配对（a[i], b[i] 同一人设）。返回 n / mean_diff / ci（差值均值的 bootstrap CI）/ p。"""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    d = a - b
    n = int(len(d))
    out = {"n": n, "mean_diff": round(float(d.mean()), 4) if n else None, "ci": None, "p": None}
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    out["ci"] = tuple(round(float(x), 4) for x in np.percentile(bm, [2.5, 97.5]))
    if n >= 6 and np.any(d != 0):
        out["p"] = round(float(_st.wilcoxon(d, zero_method="wilcox").pvalue), 5)
    return out


def icc_1(cells: list[list[float]]) -> dict:
    """单向随机效应 ICC(1,1) 与 ICC(1,k)（Shrout & Fleiss），要求各单元等长 k。"""
    X = np.asarray(cells, dtype=float)
    n, k = X.shape
    grand = X.mean()
    ms_b = k * ((X.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_w = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    if ms_b + (k - 1) * ms_w == 0:
        return {"icc11": 1.0, "icc1k": 1.0, "k": k, "n": n}
    icc11 = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    icc1k = (ms_b - ms_w) / ms_b if ms_b else 0.0
    return {"icc11": round(float(icc11), 4), "icc1k": round(float(icc1k), 4), "k": k, "n": n}


def spearman(a, b) -> tuple[float | None, float | None]:
    """常量输入（如某维全被测同分）时相关无定义 → 返回 (None, None)，JSON 可写。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _st.spearmanr(a, b)
    rho, p = float(r.statistic), float(r.pvalue)
    return ((None if np.isnan(rho) else round(rho, 4)), (None if np.isnan(p) else round(p, 5)))


def quadratic_kappa(pairs: list[tuple[int, int]], k: int = 3) -> float | None:
    """二次加权 κ（0..k-1 序数）。"""
    n = len(pairs)
    if not n:
        return None
    w = lambda i, j: ((i - j) / (k - 1)) ** 2
    po = sum(w(i, j) for i, j in pairs) / n
    pa = [sum(1 for i, _ in pairs if i == v) / n for v in range(k)]
    pb = [sum(1 for _, j in pairs if j == v) / n for v in range(k)]
    pe = sum(w(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k))
    if pe == 0:
        return 1.0
    return round(1 - po / pe, 4)


def auc_mann_whitney(low: list[float], high: list[float]) -> float | None:
    """P(high > low) + 0.5·P(tie)。AUC=1 表示 high 组分数全高于 low 组。"""
    if not low or not high:
        return None
    lo = np.asarray(low, dtype=float)[:, None]; hi = np.asarray(high, dtype=float)[None, :]
    return round(float(((hi > lo).sum() + 0.5 * (hi == lo).sum()) / (len(low) * len(high))), 4)


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 10, iters: int = 100):
    """朴素 Lloyd k-means，多次随机初始化取簇内平方和最小。返回 (labels, centroids)。"""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        cents = X[rng.choice(len(X), size=k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            new = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else cents[j]
                            for j in range(k)])
            if np.allclose(new, cents):
                break
            cents = new
        inertia = float(((X - cents[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), cents.copy())
    return best[1], best[2]


def silhouette(X: np.ndarray, labels) -> float:
    X = np.asarray(X, dtype=float); labels = np.asarray(labels)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    ks = np.unique(labels)
    if len(ks) < 2:
        return 0.0
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == i * 0 + labels[i]
        own_n = own.sum() - 1
        a = D[i, own].sum() / own_n if own_n > 0 else 0.0
        b = min(D[i, labels == c].mean() for c in ks if c != labels[i])
        s[i] = 0.0 if own_n == 0 else (b - a) / max(a, b)
    return round(float(s.mean()), 4)


def choose_k(X: np.ndarray, ks=(3, 4, 5, 6), seed: int = 0) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in ks:
        if k >= len(X):
            continue
        lab, _ = kmeans(X, k, seed=seed)
        scores[k] = silhouette(X, lab)
    best = max(scores, key=scores.get)
    return best, scores
