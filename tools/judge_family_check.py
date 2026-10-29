"""A4 同家族偏袒检查：Qwen 判官（qwen-eval-38）在 Qwen 系被测上是否比跨家族判官（deepseek，关思考）更宽松。零 GPU。

    python tools/judge_family_check.py --out reports/a4_deepseek/family

两层：
  ① 金标 180 段：两判官各自的加权灵敏度 / 特异度，按被测家族（Qwen 系 / InternLM 系 / API 旗舰）分开。
  ② 三对配对全语料（同一批轮次两判官都打过）：各模型两判官的观测失败率；
     偏袒指标 = [Qwen 判官 − deepseek 判官] 在 Qwen 系上的均值 − 在 InternLM 系上的均值（差中差，按人设重抽 CI）。
     偏袒 → 差中差显著为负（Qwen 判官对自家判失败更少）。
Qwen 系 = qwen15-4b-base / mindchat（Qwen2）/ qwen3-8b-base / soulchat（Qwen3）；InternLM 系 = internlm2-base / cpsycoun。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.judge_correction import cell_weights, load_frame_scores, sampling_cells, weighted_rates  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, _fail, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

QWEN = ("qwen15-4b-base", "mindchat", "qwen3-8b-base", "soulchat")
INTERN = ("internlm2-base", "cpsycoun")


def family(model: str) -> str:
    return "Qwen 系" if model in QWEN else "InternLM 系" if model in INTERN else "API 旗舰"


def corpus_fails(path) -> dict:
    """{(模型, 人设, 轮次 id): {维: 失败}}（只收打上分的轮）。"""
    out = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            per = d["case_profile"]["id"].split("-", 1)[0]
            for t in d["turns"]:
                s = t.get("score")
                if s:
                    out[(d["target_model_id"], per, f'{base}#t{t["turn"]}')] = {k: int(_fail(s[k], k)) for k in DIMS5}
    return out


def rates_by_model(qa: dict, qb: dict, personas=None):
    """两判官都打上分的轮 → {模型: {维: (a 率, b 率, n)}}；personas 给定时按其（含重复）加权。"""
    keys = sorted(set(qa) & set(qb))
    by = defaultdict(list)
    for k in keys:
        by[(k[0], k[1])].append(k)
    plist = personas if personas is not None else sorted({k[1] for k in keys})
    out = {}
    for m in sorted({k[0] for k in keys}):
        sel = [k for p in plist for k in by.get((m, p), [])]
        if not sel:
            continue
        out[m] = {d: (sum(qa[k][d] for k in sel) / len(sel), sum(qb[k][d] for k in sel) / len(sel), len(sel))
                  for d in DIMS5}
    return out


def did(r: dict, d: str):
    q = [r[m][d][0] - r[m][d][1] for m in QWEN if m in r]
    i = [r[m][d][0] - r[m][d][1] for m in INTERN if m in r]
    return None if not q or not i else sum(q) / len(q) - sum(i) / len(i)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A4 同家族偏袒检查")
    ap.add_argument("--gold", action="append", default=["gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl"])
    ap.add_argument("--qwen-signals", action="append", default=["runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl",
                                                                "runs/v2/v7_test_qwen38.jsonl.signals.jsonl"])
    ap.add_argument("--ds-signals", action="append", default=["runs/v2/a4_dsnt_gold.jsonl.signals.jsonl"])
    ap.add_argument("--qwen-corpus", default="runs/v2/v7_all_qwen38_fill.jsonl")
    ap.add_argument("--ds-corpus", default="runs/v2/a4_dsnt_pairs.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows = [g for p in a.gold for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    replies = load_replies("runs/v2/dialogues_gold180.jsonl", set(ids))
    cells = sampling_cells(load_frame_scores("runs/v2/scored_v6.jsonl"), load_frame_scores("runs/v2/scored_v5.jsonl"))
    w, _ = cell_weights(ids, cells, Counter(cells.values()))
    preds = {"Qwen 判官": score_table(load_cache(a.qwen_signals), replies, ids, [FROZEN])[FROZEN],
             "deepseek 判官": score_table(load_cache(a.ds_signals), replies, ids, [FROZEN])[FROZEN]}
    L = ["# A4 同家族偏袒检查（Qwen 判官 vs deepseek 判官，同一 v7 题面 q4 + 规则 r6）", "",
         f"- 生成 {time.strftime('%Y-%m-%d %H:%M')}；失败 d1–d4 = 0、d5 = ≤1。", "",
         "## ① 金标 180 段：按被测家族的加权灵敏度 / 特异度", "",
         "| 家族 | 段数 | 判官 | " + " | ".join(DIMS5) + " |", "|---|---|---|" + "---|" * len(DIMS5)]
    for fam in ("Qwen 系", "InternLM 系", "API 旗舰"):
        fi = [i for i in ids if family(i.split("|", 1)[0]) == fam]
        for jn, pred in preds.items():
            cells_ = []
            for d in DIMS5:
                r = weighted_rates([(pred[i][d], gold[i][d], w[i]) for i in fi if gold[i].get(d) is not None], d)
                se = "—" if r["sens"] is None else f"{r['sens']:.2f}"
                sp = "—" if r["spec"] is None else f"{r['spec']:.2f}"
                cells_.append(f"{se} / {sp}（失败 {r['n_fail']}）")
            L.append(f"| {fam} | {len(fi)} | {jn} | " + " | ".join(cells_) + " |")

    qa, qb = corpus_fails(a.qwen_corpus), corpus_fails(a.ds_corpus)
    point = rates_by_model(qa, qb)
    pers = sorted({k[1] for k in set(qa) & set(qb)})
    rng = random.Random(a.seed)
    acc = defaultdict(list)
    for _ in range(a.n_boot):
        draw = [rng.choice(pers) for _ in pers]
        rb = rates_by_model(qa, qb, draw)
        for d in DIMS5:
            v = did(rb, d)
            if v is not None:
                acc[d].append(v)

    def q(xs):
        xs = sorted(xs)
        return (xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]) if len(xs) >= 50 else None
    L += ["", "## ② 配对全语料（两判官都打上分的同一批轮次）：观测失败率 Qwen 判官 / deepseek 判官", "",
          "| 模型 | 家族 | 轮数 | " + " | ".join(DIMS5) + " |", "|---|---|---|" + "---|" * len(DIMS5)]
    for m in QWEN + INTERN:
        if m in point:
            L.append(f"| {m} | {family(m)} | {point[m]['d1'][2]} | " + " | ".join(
                f"{100 * point[m][d][0]:.1f}% / {100 * point[m][d][1]:.1f}%" for d in DIMS5) + " |")
    L += ["", "## 偏袒指标：[Qwen 判官 − deepseek 判官] 在 Qwen 系的均值 − 在 InternLM 系的均值 [95% CI]", "",
          "负数 = Qwen 判官对自家被测判失败更少（偏袒方向）。", "", "| 维 | 差中差 |", "|---|---|"]
    res = {}
    for d in DIMS5:
        v, c = did(point, d), q(acc[d])
        res[d] = {"did": v, "ci": c}
        flag = " ⚠️" if c and (c[0] > 0 or c[1] < 0) else ""
        L.append(f"| {d} | {100 * v:+.1f}%" + (f" [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]" if c else "") + f"{flag} |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "family.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "family.json").write_text(json.dumps({"rates": {m: v for m, v in point.items()}, "did": res},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 {out / 'family.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
