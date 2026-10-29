# cascade/goldeval/agreement.py
"""Pure agreement statistics for gold-eval (Plan 4.4 §5). No model calls, no IO —
label pairs in, numbers out. Pairs where either side is None (model unparseable /
human blank) are excluded, matching risk_labeler's None-on-unparseable contract.
Ordinal callers must pass labels that are all in `levels`; out-of-scale human labels
(e.g. risk 'none') are filtered by report.py BEFORE calling weighted_kappa."""
from __future__ import annotations
from itertools import combinations


def _clean(pairs):
    return [(a, b) for a, b in pairs if a is not None and b is not None]


def cohen_kappa(pairs) -> float | None:
    pairs = _clean(pairs)
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({x for p in pairs for x in p})
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == l) / n) * (sum(1 for _, b in pairs if b == l) / n)
             for l in labels)
    return 1.0 if pe >= 1.0 else round((po - pe) / (1 - pe), 4)


def weighted_kappa(pairs, levels, weights: str = "quadratic") -> float | None:
    pairs = _clean(pairs)
    n = len(pairs)
    k = len(levels)
    if n == 0 or k < 2:
        return None
    idx = {l: i for i, l in enumerate(levels)}

    def w(i, j):
        d = abs(i - j) / (k - 1)
        return d * d if weights == "quadratic" else d

    hi = [idx[a] for a, _ in pairs]
    mi = [idx[b] for _, b in pairs]
    po = sum(w(i, j) for i, j in zip(hi, mi)) / n
    ph = [hi.count(x) / n for x in range(k)]
    pm = [mi.count(x) / n for x in range(k)]
    pe = sum(w(i, j) * ph[i] * pm[j] for i in range(k) for j in range(k))
    return 1.0 if pe == 0 else round(1 - po / pe, 4)


def accuracy(pairs, levels) -> dict | None:
    pairs = _clean(pairs)
    n = len(pairs)
    if n == 0:
        return None
    idx = {l: i for i, l in enumerate(levels)}
    exact = sum(1 for a, b in pairs if a == b) / n
    within1 = sum(1 for a, b in pairs if abs(idx[a] - idx[b]) <= 1) / n
    return {"exact": round(exact, 4), "within1": round(within1, 4), "n": n}


def prf1(pairs, positive=1) -> dict | None:
    pairs = _clean(pairs)
    if not pairs:
        return None
    tp = sum(1 for h, m in pairs if m == positive and h == positive)
    fp = sum(1 for h, m in pairs if m == positive and h != positive)
    fn = sum(1 for h, m in pairs if m != positive and h == positive)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "n": len(pairs)}


def confusion(pairs, levels) -> list[list[int]]:
    pairs = _clean(pairs)
    idx = {l: i for i, l in enumerate(levels)}
    k = len(levels)
    m = [[0] * k for _ in range(k)]
    for h, mm in pairs:
        if h in idx and mm in idx:
            m[idx[h]][idx[mm]] += 1
    return m


def pairwise_iaa(annotations: dict, kind: str, levels) -> float | None:
    ann_ids = sorted(annotations)
    if len(ann_ids) < 2:
        return None
    ks = []
    for a, b in combinations(ann_ids, 2):
        shared = sorted(set(annotations[a]) & set(annotations[b]))
        pairs = [(annotations[a][i], annotations[b][i]) for i in shared]
        k = cohen_kappa(pairs) if kind == "binary" else weighted_kappa(pairs, levels)
        if k is not None:
            ks.append(k)
    return round(sum(ks) / len(ks), 4) if ks else None
