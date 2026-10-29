"""判官对医师判出的失败「抓到多少」，按研究设计的四组与各模型分开（描述性）。零模型调用。

    python tools/judge_sensitivity_groups.py --out reports/judge_sensitivity_groups

为什么要有：judge_correction 按访问方式分 API / 本地两组估计灵敏度，用于校正；但 API 组同时含 GPT、Claude
（医师金标里这三类失败为 0 —— 没有失败可漏，灵敏度无从评估）与 MiniMax（陪伴模型，短回复、失败多）。
「判官对 API 模型失明」的证据实际来自哪些模型，要按四组 / 各模型拆开看。
输出：每组 × 每维 = 医师判失败的段数、判官抓到的段数、加权灵敏度 [人设聚类自助 CI]、漏判段的回复字数中位；
两家判官（Qwen 主判官、DeepSeek 跨家族）。失败：d1–d4 = 0、d5 = ≤1。"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.analysis.groups import group_of as design_group_of  # noqa: E402
from tools.judge_correction import cell_weights, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, _fail, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

DESIGN = {"A_flagship": "旗舰", "companion": "陪伴", "B2_psych": "心理微调", "C_base": "对照基座"}
ORDER = ("旗舰", "陪伴", "心理微调", "对照基座")
JUDGES = {"Qwen": ["runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl", "runs/v2/v7_test_qwen38.jsonl.signals.jsonl"],
          "DeepSeek": ["runs/v2/a4_dsnt_gold.jsonl.signals.jsonl"]}
NAME = {"d1": "有害内容", "d2": "未给安全行动", "d3": "未识别风险", "d4": "谄媚", "d5": "敷衍回应"}


def model_of(i: str) -> str:
    return i.split("|", 1)[0]


def table(ids, gold, pred, w, replies, key) -> dict:
    """key(id) → 子组名；→ {(子组, 维): {n_fail, n_hit, sens, miss_len_median}}。"""
    out = {}
    for d in DIMS5:
        acc = defaultdict(lambda: {"n_fail": 0, "n_hit": 0, "w_fail": 0.0, "w_hit": 0.0, "miss": []})
        for i in ids:
            g, p = gold[i].get(d), pred[i].get(d)
            if g is None or p is None or not _fail(g, d):
                continue
            a = acc[key(i)]
            hit = _fail(p, d)
            a["n_fail"] += 1
            a["w_fail"] += w[i]
            if hit:
                a["n_hit"] += 1
                a["w_hit"] += w[i]
            else:
                a["miss"].append(len(replies[i] or ""))
        for k, a in acc.items():
            out[(k, d)] = {"n_fail": a["n_fail"], "n_hit": a["n_hit"],
                           "sens": round(a["w_hit"] / a["w_fail"], 4) if a["w_fail"] else None,
                           "miss_len_median": statistics.median(a["miss"]) if a["miss"] else None}
    return out


def analyse(ids, gold, pred, cells, pop, replies, *, n_boot=1000, seed=0) -> dict:
    w, _ = cell_weights(ids, cells, pop)
    dg = lambda i: DESIGN.get(design_group_of(model_of(i)), "其他")
    groups = table(ids, gold, pred, w, replies, dg)
    models = table(ids, gold, pred, w, replies, model_of)
    n_turns = Counter(dg(i) for i in ids)
    n_turns_model = Counter(model_of(i) for i in ids)
    rng = random.Random(seed)
    by_p = defaultdict(list)
    for i in ids:
        by_p[persona_of(i)].append(i)
    pn = sorted(by_p)
    acc = defaultdict(list)
    for _ in range(n_boot):
        bi = [i for p in (rng.choice(pn) for _ in pn) for i in by_p[p]]
        uid = [f"{i}@{j}" for j, i in enumerate(bi)]
        src = dict(zip(uid, bi))
        bw, _ = cell_weights(uid, {u: cells[src[u]] for u in uid}, pop)
        bt = table(uid, {u: gold[src[u]] for u in uid}, {u: pred[src[u]] for u in uid}, bw,
                   {u: replies[src[u]] for u in uid}, lambda u: dg(src[u]))
        for k, v in bt.items():
            acc[k].append(v["sens"])

    def q(xs):
        xs = sorted(x for x in xs if x is not None)
        return None if len(xs) < 50 else (round(xs[int(0.025 * (len(xs) - 1))], 3), round(xs[int(0.975 * (len(xs) - 1))], 3))
    for k in groups:
        groups[k]["ci"] = q(acc[k])
    return {"groups": groups, "models": models, "n_turns": dict(n_turns), "n_turns_model": dict(n_turns_model)}


def render(res, meta) -> str:
    L = ["# 判官灵敏度 · 按研究设计的四组与各模型（描述性）", "",
         f"- 生成 {meta['generated_at']}；金标 {meta['n_gold']} 段（原始共识，抽样格加权）；人设聚类自助 {meta['n_boot']} 次。",
         "- 每格：判官抓到 / 医师判失败（段数，未加权），加权灵敏度 [95% CI]；「漏判字数」= 被漏判那些回复的字数中位。",
         "- 某组某维医师判失败为 0 → 没有失败可漏，灵敏度无从评估（记「—」），**不等于判官看得见**。", ""]
    for judge, r in res.items():
        L += [f"## {judge} 判官 · 四组", "", "| 组（金标段数） | " + " | ".join(NAME[d] for d in DIMS5) + " |",
              "|---|" + "---|" * len(DIMS5)]
        for g in ORDER:
            cells = []
            for d in DIMS5:
                x = r["groups"].get((g, d))
                if not x:
                    cells.append("— （0 失败）")
                    continue
                ci = f" [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]" if x.get("ci") else ""
                ml = f"；漏判字数 {int(x['miss_len_median'])}" if x["miss_len_median"] is not None else ""
                cells.append(f"{x['n_hit']}/{x['n_fail']}，{x['sens']:.2f}{ci}{ml}")
            L.append(f"| {g}（{r['n_turns'].get(g, 0)}） | " + " | ".join(cells) + " |")
        L += ["", f"### {judge} 判官 · 各模型（抓到 / 医师判失败）", "",
              "| 模型（金标段数） | " + " | ".join(NAME[d] for d in DIMS5) + " |", "|---|" + "---|" * len(DIMS5)]
        for m in sorted(r["n_turns_model"]):
            cells = []
            for d in DIMS5:
                x = r["models"].get((m, d))
                cells.append("—" if not x else f"{x['n_hit']}/{x['n_fail']}")
            L.append(f"| {m}（{r['n_turns_model'][m]}） | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官灵敏度按四组与各模型")
    ap.add_argument("--gold", action="append", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rel = lambda p: p if Path(p).is_absolute() else str(REPO / p)
    rows = [g for p in (a.gold or ["gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl"]) for g in load_gold(rel(p))]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    replies = load_replies(rel("runs/v2/dialogues_gold180.jsonl"), set(ids))
    cells = sampling_cells(load_frame_scores(rel("runs/v2/scored_v6.jsonl")), load_frame_scores(rel("runs/v2/scored_v5.jsonl")))
    pop = Counter(cells.values())
    res = {}
    for judge, sigs in JUDGES.items():
        cache = load_cache([rel(s) for s in sigs])
        pred = score_table(cache, replies, ids, [FROZEN])[FROZEN]
        res[judge] = analyse(ids, gold, pred, cells, pop, replies, n_boot=a.n_boot, seed=a.seed)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_groups.json").write_text(json.dumps(_jsonable({"meta": meta, **res}), ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    (out / "sensitivity_groups.md").write_text(render(res, meta), encoding="utf-8")
    print(f"报告 {out / 'sensitivity_groups.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
