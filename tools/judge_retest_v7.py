"""判官 v7 重测信度：同一判官、同一题面，金标 180 段独立抽取两次（各自的信号缓存）比一致性。零 GPU。

    python tools/judge_retest_v7.py \
        --first runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --first runs/v2/v7_test_qwen38.jsonl.signals.jsonl \
        --second runs/v2/a5_retest_gold.jsonl.signals.jsonl --out reports/judge_retest_v7

报告：① 逐信号（31 个「有没有」）两次判定翻转率；② 定版规则 r6 下各维分数两次的精确一致与加权 κ、跨 0 一致与 κ；
③ 两次各自对医师金标的指标（d1–d4 跨 0 κ、d5 加权 κ、总分）—— 判官效度结论随重跑能动多少。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.goldeval.agreement import cohen_kappa, weighted_kappa  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, evaluate, load_cache, load_replies, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402


def signal_flips(a: dict, b: dict, ids) -> dict:
    """{信号: (翻转数, 比较数)} —— present 不同即翻转。"""
    out = Counter(), Counter()
    for i in ids:
        sa, sb = a[i]["signals"], b[i]["signals"]
        for k in set(sa) & set(sb):
            out[1][k] += 1
            out[0][k] += bool(sa[k]["present"]) != bool(sb[k]["present"])
    return {k: (out[0][k], out[1][k]) for k in sorted(out[1])}


def score_agreement(pa: dict, pb: dict, ids) -> dict:
    res = {}
    for d in DIMS5 + ("d6",):
        pairs = [(pa[i][d], pb[i][d]) for i in ids if pa[i].get(d) is not None and pb[i].get(d) is not None]
        res[d] = {"n": len(pairs), "exact": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
                  "wkappa": weighted_kappa(pairs, [0, 1, 2]),
                  "zero_agree": round(sum((x == 0) == (y == 0) for x, y in pairs) / len(pairs), 4) if pairs else None,
                  "zero_kappa": cohen_kappa([(int(x == 0), int(y == 0)) for x, y in pairs])}
    return res


def _d6(cache, ids):
    return {i: cache[i]["d6"]["score"] for i in ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7 重测信度（金标 180 段）")
    ap.add_argument("--gold", action="append", default=["gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl"])
    ap.add_argument("--first", action="append", required=True)
    ap.add_argument("--second", action="append", required=True)
    ap.add_argument("--dialogues", default="runs/v2/dialogues_gold180.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [g for p in a.gold for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    c1, c2 = load_cache(a.first), load_cache(a.second)
    both = [i for i in ids if i in c1 and i in c2]
    replies = load_replies(a.dialogues, set(both))
    p1 = score_table(c1, replies, both, [FROZEN])[FROZEN]
    p2 = score_table(c2, replies, both, [FROZEN])[FROZEN]
    d6a, d6b = _d6(c1, both), _d6(c2, both)
    for i in both:
        p1[i]["d6"], p2[i]["d6"] = d6a[i], d6b[i]
    flips = signal_flips(c1, c2, both)
    agree = score_agreement(p1, p2, both)
    ev1, ev2 = evaluate(p1, gold, both), evaluate(p2, gold, both)
    tf, tn = sum(f for f, _ in flips.values()), sum(n for _, n in flips.values())
    L = ["# 判官 v7 重测信度（金标 180 段，独立抽取两次）", "",
         f"- 生成 {time.strftime('%Y-%m-%d %H:%M')}；两次都有的段 {len(both)}；规则 {FROZEN}。",
         f"- 第一次 = {', '.join(a.first)}；第二次 = {', '.join(a.second)}。", "",
         f"## 逐信号翻转（合计 {tf}/{tn} = {100 * tf / tn:.2f}%）", "",
         "| 信号 | 翻转 / 比较 |", "|---|---|"]
    L += [f"| {k} | {f}/{n} |" for k, (f, n) in flips.items() if f]
    L += ["", "（未列出的信号两次完全一致。）", "", "## 各维分数两次一致性", "",
          "| 维 | n | 精确一致 | 加权 κ | 跨 0 一致 | 跨 0 κ |", "|---|---|---|---|---|---|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for d, v in agree.items():
        L.append(f"| {d} | {v['n']} | {fmt(v['exact'])} | {fmt(v['wkappa'])} | {fmt(v['zero_agree'])} | {fmt(v['zero_kappa'])} |")
    L += ["", "## 两次各自对医师金标（d1–d4 跨 0 κ、d5 加权 κ）", "",
          "| 维 | 第一次 | 第二次 |", "|---|---|---|"]
    for d in DIMS5:
        L.append(f"| {d} | {fmt(ev1['dims'][d]['metric'])} | {fmt(ev2['dims'][d]['metric'])} |")
    L.append(f"| **总分** | **{fmt(ev1['total'])}** | **{fmt(ev2['total'])}** |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retest.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "retest.json").write_text(json.dumps({"n": len(both), "flips": flips, "agreement": agree,
                                                 "gold_first": ev1, "gold_second": ev2}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"报告 {out / 'retest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
