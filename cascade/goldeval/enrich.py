"""Enriched sampling for the C-evaluator gold (Plan 4.4). compare_evaluators showed
empathy/deesc have ~no discrimination and abandon is the battlefield + the only real
disagreement dim, so a RANDOM 300 would burn clinician time on easy empathy=1 turns.
This over-samples the informative-but-scarce strata (evaluator disagreement, abandon
candidates), soft-quotas high-risk turns, fills the rest with baseline, then splits
dev/test. PURE logic: the caller precomputes each turn's stratum flags. Deterministic
under `seed`."""
from __future__ import annotations
import random

# strata in descending priority; a turn is assigned its top-matching stratum (no double-count)
_PRIORITY = ("disagree", "abandon", "connect_hard", "high_risk", "baseline")


def scatter_order(items, *, seed: int = 0, key: str = "dlg") -> list[dict]:
    """Order items so that consecutive ones come from different dialogues (round-robin over
    dialogue groups, deterministic under `seed`). The annotation sheets are sorted by unit
    id, so if ids are minted in source order the doctor reads several exchanges of the SAME
    conversation back to back — that re-introduces exactly the within-dialogue rater
    carry-over the per-dialogue cap exists to limit, and lets the escalating trajectory
    prime the label. Adjacency is guaranteed broken unless one dialogue holds more than
    half the items (impossible under any sane cap)."""
    rng = random.Random(seed)
    groups: dict = {}
    for it in items:
        groups.setdefault(it.get(key), []).append(it)
    ordered_keys = sorted(groups, key=lambda k: (k is None, k))     # stable before shuffling
    rng.shuffle(ordered_keys)
    for k in ordered_keys:
        rng.shuffle(groups[k])
    rank = {k: i for i, k in enumerate(ordered_keys)}               # tie-break: shuffled order
    out: list[dict] = []
    prev = object()
    remaining = sum(len(g) for g in groups.values())
    while remaining:
        # greedy: take from the largest group that isn't the one we just used
        cand = [k for k in ordered_keys if groups[k] and k != prev] or \
               [k for k in ordered_keys if groups[k]]
        k = min(cand, key=lambda k: (-len(groups[k]), rank[k]))
        out.append(groups[k].pop())
        prev, remaining = k, remaining - 1
    return out


def _stratum(t: dict) -> str:
    if t.get("disagree"):
        return "disagree"
    if t.get("abandon"):
        return "abandon"
    if t.get("connect_hard"):
        return "connect_hard"
    if t.get("high_risk"):
        return "high_risk"
    return "baseline"


def enrich_sample(turns, *, n: int, test_frac: float = 0.33, seed: int = 0,
                  hr_frac: float = 0.2, dis_frac: float = 0.5, ch_frac: float = 0.2,
                  max_per_dialogue: int | None = None,
                  max_per_model: int | None = None) -> list[dict]:
    """Pick up to `n` turns, enriched. Each returned item is a shallow copy of the input
    turn plus `stratum` and `split` ('dev'/'test'). Order of selection: disagree up to
    `dis_frac`*n -> ALL abandon (scarce) -> high-risk up to `hr_frac`*n -> baseline fill
    -> relax every quota if a pool ran out, so `n` is always reached when enough turns
    exist. dev/test split is stratified (each stratum split by `test_frac`).

    `dis_frac` exists because disagreement turns are NOT scarce once several targets are
    in the batch (measured: 873/2858 = 31%), and taking all of them makes the gold 100%
    disagreement turns. That sample can rank evaluators against each other but (a) gives
    a selected, uninterpretable estimate of any evaluator's ABSOLUTE agreement with
    clinicians, and (b) structurally cannot contain a turn where all evaluators agree and
    are all wrong — precisely the shared-leniency failure this validation is meant to
    catch. Keeping every stratum represented also makes the population estimate
    recoverable: per-stratum agreement reweighted by the stratum's share of the pool.

    `max_per_dialogue` caps how many turns may come from one dialogue (keyed by each
    turn's `dlg`). Exchanges inside one dialogue share persona/target/context and are
    strongly correlated: without a cap, n=300 can come from a few dozen dialogues, the
    effective sample size is far below 300, and the resulting kappa's precision is
    overstated. The cap is HARD — when it binds, fewer than `n` turns are returned
    rather than silently exceeding it (the caller is expected to widen the batch).

    `ch_frac` quotas the `connect_hard` stratum — turns whose reply is hard to judge on the
    保持连接 dimension by its TEXT SHAPE: the dialogue's last turn, a minimal reply
    ("嗯。" / "……"), a bracketed stage direction ("（屏幕暗下）"), or a sign-off. Two reasons
    it earns its own stratum, both measured on the 2026-08 gold round:

    1. **The abandon dimension was structurally unmeasurable without it.** In the baseline
       and high_risk strata — 64% of the pool — both clinicians and machine judged
       "abandoned" 0–4% of the time. Zero variance means kappa is 0 there no matter how
       good the evaluator is, so two thirds of the clinicians' time bought nothing for
       this dimension. Reweighted population kappa came to 0.53, the only dimension under
       the 0.6 gate.
    2. **It is evaluator-independent.** `disagree` and `abandon` are defined from the
       candidate evaluators' own output, and the evaluator under test is one of them — so
       "the machine flags abandon on 100% of the abandon stratum" is partly true by
       construction, which inflates pooled kappa. A stratum defined purely on the reply's
       text carries no such circularity.

    `max_per_model` caps how many turns may come from one target model (keyed by each
    turn's `model`). Disagreement rate is strongly target-dependent — measured on
    valbatch_v2: companion-style 65.5% vs claude 13.6% — so "take ALL disagree" lets a
    single target supply a third of the gold, and the evaluator selected on it is merely
    the one best at judging THAT target, not the six on the leaderboard. Also HARD."""
    rng = random.Random(seed)
    pools: dict[str, list] = {s: [] for s in _PRIORITY}
    for t in turns:
        s = _stratum(t)
        pools[s].append(dict(t, stratum=s))
    for s in pools:
        rng.shuffle(pools[s])

    sel: list[dict] = []
    caps = [(f, cap) for f, cap in (("dlg", max_per_dialogue), ("model", max_per_model))
            if cap is not None]
    used: dict[tuple, int] = {}

    def _blocked(item):                     # any cap already exhausted for this item's keys
        return any(item.get(f) is not None and used.get((f, item[f]), 0) >= cap
                   for f, cap in caps)

    def _charge(item):
        for f, _cap in caps:
            if item.get(f) is not None:
                used[(f, item[f])] = used.get((f, item[f]), 0) + 1

    def _take(pool, k):
        k = max(0, min(len(pool), k))
        if not caps:
            chunk, pool[:] = pool[:k], pool[k:]
            sel.extend(chunk)
            return
        chunk, rest = [], []
        for item in pool:
            if len(chunk) < k and not _blocked(item):
                chunk.append(item)
                _charge(item)
            else:
                rest.append(item)
        pool[:] = rest
        sel.extend(chunk)

    _take(pools["disagree"], min(round(dis_frac * n), n - len(sel)))   # soft quota (see docstring)
    _take(pools["abandon"], n - len(sel))
    _take(pools["connect_hard"], min(round(ch_frac * n), n - len(sel)))  # soft quota
    _take(pools["high_risk"], min(round(hr_frac * n), n - len(sel)))   # soft quota
    _take(pools["baseline"], n - len(sel))                 # fill
    for s in ("high_risk", "baseline", "connect_hard", "abandon", "disagree"):   # relax if short
        if len(sel) >= n:
            break
        _take(pools[s], n - len(sel))
    sel = sel[:n]

    by_str: dict[str, list] = {}
    for item in sel:
        by_str.setdefault(item["stratum"], []).append(item)
    out: list[dict] = []
    for items in by_str.values():                          # items keep their shuffled order
        n_test = round(len(items) * test_frac)
        for i, item in enumerate(items):
            item["split"] = "test" if i < n_test else "dev"
            out.append(item)
    return out
