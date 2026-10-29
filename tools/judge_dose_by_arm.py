"""判官观测失败率按臂 × 风险段直接计算（本地模型），人设聚类自助 CI。零模型调用。

    python tools/judge_dose_by_arm.py --out reports/judge_dose_by_arm

为什么要有（2026-10-26）：Fig. 6a 的 responsive 臂原先由 tools/judge_dose 的「两臂合并」与「fixed 臂」按轮数回推，
没有 CI。本工具直接从语料按臂计算（与 judge_dose 同一语料、同一行筛选 corpus_rows、同一失败定义；只报观测、不做误判校正），
每个臂各段比例与「高 − 低」差都给人设聚类自助 95% CI。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.judge_correction import group_of  # noqa: E402
from tools.judge_dose import ANY, BANDS, OUTCOMES, corpus_rows  # noqa: E402

ARMS = ("fixed", "responsive")


def tally(crows, group="本地"):
    """→ {(臂, 段): {人设: {结局: [失败, 轮数]}}}。"""
    t = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for m, arm, per, band, _length, fails in crows:
        if group_of(m) != group or arm not in ARMS:
            continue
        for o in OUTCOMES:
            v = fails.get(o)
            if v is not None:
                c = t[(arm, band)][per][o]
                c[0] += v
                c[1] += 1
    return t


def rate(by_p, personas, o):
    num = den = 0
    for p in personas:
        c = by_p.get(p, {}).get(o)
        if c:
            num += c[0]
            den += c[1]
    return num / den if den else None


def _ci(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if len(xs) < 50 else (xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))])


def analyse(crows, *, n_boot=2000, seed=0) -> dict:
    t = tally(crows)
    pers = sorted({p for by_p in t.values() for p in by_p})
    rng = random.Random(seed)
    draws = [[rng.choice(pers) for _ in pers] for _ in range(n_boot)]
    out = {}
    for arm in ARMS:
        for o in OUTCOMES:
            pt = {b: rate(t[(arm, b)], pers, o) for b in BANDS}
            n = {b: sum(c[o][1] for c in t[(arm, b)].values()) for b in BANDS}
            boot = {b: [rate(t[(arm, b)], d, o) for d in draws] for b in BANDS}
            diff = [None if h is None or l is None else h - l for h, l in zip(boot["高"], boot["低"])]
            out[f"{arm}|{o}"] = {"point": pt, "n_turns": n, "ci": {b: _ci(boot[b]) for b in BANDS},
                                 "diff": None if pt["高"] is None or pt["低"] is None else pt["高"] - pt["低"],
                                 "diff_ci": _ci(diff)}
    return {"personas": len(pers), "cells": out}


def render(r, info) -> str:
    L = ["# 判官观测失败率：按臂 × 风险段（本地模型，Qwen 判官）", "",
         f"- 生成 {info['generated_at']}；人设聚类自助 {info['n_boot']} 次；只报观测（未做误判校正）。", "",
         "| 臂 | 结局 | 低 [CI]（轮数） | 中 [CI]（轮数） | 高 [CI]（轮数） | 高 − 低 [CI] |", "|---|---|---|---|---|---|"]
    f = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    c = lambda t: "—" if t is None else f"[{100 * t[0]:.1f}, {100 * t[1]:.1f}]"
    for k, v in r["cells"].items():
        arm, o = k.split("|")
        cols = [f"{f(v['point'][b])} {c(v['ci'][b])}（{v['n_turns'][b]}）" for b in BANDS]
        d = "—" if v["diff"] is None else f"{100 * v['diff']:+.1f} {c(v['diff_ci'])}"
        L.append(f"| {arm} | {o} | " + " | ".join(cols) + f" | {d} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官观测失败率按臂 × 风险段")
    ap.add_argument("--corpus", default="runs/v2/v7_all_qwen38_fill.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = analyse(corpus_rows(REPO / a.corpus), n_boot=a.n_boot, seed=a.seed)
    info = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_boot": a.n_boot, "corpus": a.corpus}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_dose_by_arm.json").write_text(json.dumps({"meta": info, **r}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "judge_dose_by_arm.md").write_text(render(r, info), encoding="utf-8")
    v = r["cells"][f"responsive|{ANY}"]
    print("responsive any:", {b: round(100 * v["point"][b], 1) for b in BANDS}, "diff", round(100 * v["diff"], 1), v["diff_ci"])
    v = r["cells"][f"fixed|{ANY}"]
    print("fixed any:", {b: round(100 * v["point"][b], 1) for b in BANDS}, "diff", round(100 * v["diff"], 1), v["diff_ci"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
