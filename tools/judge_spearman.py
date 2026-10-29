"""判官 v7 vs 医师：Spearman 秩相关（与 SIM-VAIL 的「人机 r = 0.49、人人 r = 0.41」对口径）。零模型调用。

    python tools/judge_spearman.py --out reports/judge_spearman

金标 180 段（dev + test），每维 0/1/2；判官 = 定版 q4 + r6（Qwen 判官；deepseek 判官作跨家族对照）。
三种相关：
  * 判官 vs 医师共识（三票中位数）；
  * 判官 vs 单个医师（a、b 两位独立标注者各算一次取平均）—— 与「医师 vs 医师」同口径；
  * 医师 a vs 医师 b（人人上限）。
每种报不加权与按抽样格反概率加权两个版本；另报五维总分（d1–d5 相加）的相关。CI = 人设聚类自助法。
⚠️ 口径不同：SIM-VAIL 是 1–10 分单维（concerning）、随机抽样；我们是三档、富集抽样（难题过采样），只作辅助参照。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.judge_correction import cell_weights, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

TOTAL = "五维总分"
ITEMS = DIMS5 + (TOTAL,)


def _ranks(x, w):
    """加权中秩：rank(v) = 小于 v 的权重和 + 等于 v 的权重和的一半（w 全 1 时即普通平均秩 − 0.5）。"""
    x, w = np.asarray(x, float), np.asarray(w, float)
    out = np.empty_like(x)
    for v in np.unique(x):
        m = x == v
        out[m] = w[x < v].sum() + w[m].sum() / 2
    return out


def wspearman(x, y, w=None):
    """加权 Spearman = 加权中秩上的加权 Pearson；任一边常量 → None。"""
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    if len(x) < 3:
        return None
    rx, ry = _ranks(x, w), _ranks(y, w)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    vx, vy = np.average((rx - mx) ** 2, weights=w), np.average((ry - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return None
    return float(np.average((rx - mx) * (ry - my), weights=w) / np.sqrt(vx * vy))


def item_value(scores: dict, item: str):
    if item == TOTAL:
        vals = [scores.get(d) for d in DIMS5]
        return None if any(v is None for v in vals) else sum(vals)
    return scores.get(item)


def correlations(rows: list[dict], w: dict, ids=None) -> dict:
    """rows: [{id, gold, a, b, judges: {名: scores}}]；ids 给定时按其（含重复）取样。"""
    by = {r["id"]: r for r in rows}
    sel = [by[i] for i in (ids if ids is not None else [r["id"] for r in rows])]
    out = {}
    for item in ITEMS:
        res = {}
        for weighted in (False, True):
            tag = "加权" if weighted else "不加权"

            def corr(fx, fy):
                xs, ys, ws = [], [], []
                for r in sel:
                    x, y = fx(r), fy(r)
                    if x is not None and y is not None:
                        xs.append(x); ys.append(y); ws.append(w[r["id"]] if weighted else 1.0)
                return wspearman(xs, ys, ws)

            res[f"医师a vs 医师b|{tag}"] = corr(lambda r: item_value(r["a"], item), lambda r: item_value(r["b"], item))
            for jn in sel[0]["judges"] if sel else []:
                res[f"{jn} vs 共识|{tag}"] = corr(lambda r: item_value(r["judges"][jn], item),
                                                 lambda r: item_value(r["gold"], item))
                ra = corr(lambda r: item_value(r["judges"][jn], item), lambda r: item_value(r["a"], item))
                rb = corr(lambda r: item_value(r["judges"][jn], item), lambda r: item_value(r["b"], item))
                res[f"{jn} vs 单个医师|{tag}"] = None if ra is None or rb is None else (ra + rb) / 2
        out[item] = res
    return out


def bootstrap(rows, w, n_boot, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[persona_of(r["id"])].append(r["id"])
    pers = sorted(groups)
    rng = random.Random(seed)
    acc = defaultdict(list)
    for _ in range(n_boot):
        ids = [i for p in (rng.choice(pers) for _ in pers) for i in groups[p]]
        for item, res in correlations(rows, w, ids).items():
            for k, v in res.items():
                if v is not None:
                    acc[(item, k)].append(v)

    def q(xs):
        xs = sorted(xs)
        return None if len(xs) < 50 else (round(xs[int(0.025 * (len(xs) - 1))], 3), round(xs[int(0.975 * (len(xs) - 1))], 3))
    return {key: q(v) for key, v in acc.items()}


def build_rows(gold_paths, judges: dict, dialogues):
    gold_rows = [g for p in gold_paths for g in load_gold(p)]
    ids = [g["id"] for g in gold_rows]
    replies = load_replies(dialogues, set(ids))
    preds = {jn: score_table(load_cache(sig), replies, ids, [FROZEN])[FROZEN] for jn, sig in judges.items()}
    rows = []
    for g in gold_rows:
        rows.append({"id": g["id"], "gold": g["consensus"],
                     "a": {d: g["votes"][d]["a"] for d in DIMS5}, "b": {d: g["votes"][d]["b"] for d in DIMS5},
                     "judges": {jn: preds[jn][g["id"]] for jn in judges}})
    return rows


def render(point, ci, meta) -> str:
    def cell(item, key):
        v = point[item].get(key)
        c = ci.get((item, key))
        return "—" if v is None else f"{v:.2f}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else "")
    keys = [k for k in point[DIMS5[0]] if k.endswith("不加权")]
    L = ["# 判官 v7 vs 医师：Spearman 秩相关", "",
         f"- 生成 {meta['generated_at']}；金标 {meta['n_gold']} 段；判官 = q4 + {FROZEN}；自助法 {meta['n_boot']} 次（按人设）。",
         "- 「单个医师」= 判官分别对两位独立标注医师算相关再平均，与「医师 a vs 医师 b」同口径。",
         "- 加权 = 按抽样格反概率加权（推回全语料口径）；不加权 = 180 段本身（难题过采样）。",
         "- 参照：SIM-VAIL（Nat Med 2026）人机 r = 0.49、人人 r = 0.41（1–10 分、单维 concerning、随机抽样）—— 口径不同，只作辅助参照。", ""]
    for tag in ("不加权", "加权"):
        L += [f"## {tag}", "", "| 比较 | " + " | ".join(ITEMS) + " |", "|---|" + "---|" * len(ITEMS)]
        for k in keys:
            kk = k.replace("不加权", tag)
            L.append(f"| {kk.split('|')[0]} | " + " | ".join(cell(it, kk) for it in ITEMS) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7 vs 医师 Spearman 秩相关")
    ap.add_argument("--gold", action="append", default=None)
    ap.add_argument("--qwen-signals", action="append", default=None)
    ap.add_argument("--ds-signals", action="append", default=None)
    ap.add_argument("--dialogues", default="runs/v2/dialogues_gold180.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gold = a.gold or ["gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl"]
    judges = {"Qwen 判官": a.qwen_signals or ["runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl",
                                             "runs/v2/v7_test_qwen38.jsonl.signals.jsonl"],
              "deepseek 判官": a.ds_signals or ["runs/v2/a4_dsnt_gold.jsonl.signals.jsonl"]}
    rows = build_rows(gold, judges, a.dialogues)
    ids = [r["id"] for r in rows]
    cells = sampling_cells(load_frame_scores("runs/v2/scored_v6.jsonl"), load_frame_scores("runs/v2/scored_v5.jsonl"))
    w, _ = cell_weights(ids, cells, Counter(cells.values()))
    point = correlations(rows, w)
    ci = bootstrap(rows, w, a.n_boot, a.seed)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(rows), "n_boot": a.n_boot}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spearman.md").write_text(render(point, ci, meta), encoding="utf-8")
    (out / "spearman.json").write_text(json.dumps({"meta": meta, "point": point,
                                                   "ci": {f"{i}|{k}": v for (i, k), v in ci.items()}},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"报告 {out / 'spearman.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
