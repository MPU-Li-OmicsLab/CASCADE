"""Per-turn decay curves + C trajectory with bootstrap CI (Plan 4.5 §6). Pure stdlib
only (random/statistics), deterministic under a fixed seed. Ragged dialogues are aligned
by turn index; each turn averages only the dialogues that reached it (sample count decays,
CI widens — honest)."""
from __future__ import annotations
import random
import statistics
from cascade.scoring.leaderboard import leaderboard_a, leaderboard_b


def _fully_scored(d) -> bool:
    return len(d.turns) > 0 and all(t.score is not None for t in d.turns)


def _columns(per_dialogue: list[list[float]]) -> dict[int, list[float]]:
    """turn index (1-based) -> list of values across dialogues reaching that turn."""
    cols: dict[int, list[float]] = {}
    for series in per_dialogue:
        for i, v in enumerate(series, start=1):
            cols.setdefault(i, []).append(v)
    return cols


def _boot_ci(vals: list[float], rng: random.Random, n_boot: int) -> tuple[float, float]:
    if len(vals) < 2:
        v = vals[0] if vals else 0.0
        return (v, v)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))


def _curve(per_dialogue: list[list[float]], seed: int, n_boot: int) -> dict:
    cols = _columns(per_dialogue)
    rng = random.Random(seed)
    turns = sorted(cols)
    mean, lo, hi = [], [], []
    for t in turns:
        vals = cols[t]
        mean.append(round(statistics.fmean(vals), 4))
        l, h = _boot_ci(vals, rng, n_boot)
        lo.append(l); hi.append(h)
    return {"turns": turns, "mean": mean, "lo": lo, "hi": hi}


def decay_curve(records, leaderboard: str = "A", seed: int = 0, n_boot: int = 200) -> dict:
    lb = leaderboard_a if leaderboard == "A" else leaderboard_b
    per = [[lb(t.score) for t in d.turns] for d in records if _fully_scored(d)]
    return _curve(per, seed, n_boot)


def c_trajectory(records, seed: int = 0, n_boot: int = 200) -> dict:
    per = [[t.c for t in d.turns] for d in records if len(d.turns) > 0]
    return _curve(per, seed, n_boot)
