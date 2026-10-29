"""判官 vs 医师：直接检验两者效应之差 D = Δ判官 − Δ医师（同一次人设重抽，保留两者的相关）。零模型调用。

    python tools/judge_vs_gold_joint.py --out reports/judge_vs_gold_joint

为什么要有：tools/judge_vs_gold 的分类只看「判官点估计是否落在医师 95% CI 内」，
判官与医师各用各的自助重抽，没有对两者之差做检验，也没有计入两者共享同一批人设带来的相关。
本工具每次重抽 16 个人设，同一批人设同时用于：
  判官一侧 —— 全语料观测比例（tools/judge_vs_gold.judge_counts 的计数）；
  医师一侧 —— 金标按同一流程重算抽样格权重与 raking（tools/gold_prevalence_raked.all_weights）后的加权比例；
于是得到 Δ判官、Δ医师和 D 的联合自助分布。比较项、结局、失败定义、子组全部沿用 tools/judge_vs_gold。

新分类（对判官显著的单元格；显著 = 95% CI 不含 0）：
  确认     = 医师同向显著，且 D 的 95% CI 含 0
  方向相反 = 医师反向显著
  幅度不符 = D 的 95% CI 不含 0（且不属于方向相反）；再分「医师同向显著」与「医师不显著」两种
  无法判断 = D 的 95% CI 含 0，且医师不显著
判官不显著的单元格：医师显著 → 判官漏掉；否则 → 都不显著。
「未获支持」= 方向相反 + 幅度不符。另报 D 的自助 p（百分位反转，+1 修正）与每家判官内对判官显著单元格的 Holm 校正。
原权重与 raking 各分类一次（共用同一批重抽）。"""
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

import tools.gold_prevalence as GP  # noqa: E402
import tools.gold_prevalence_raked as GPR  # noqa: E402
import tools.judge_vs_gold as JG  # noqa: E402
from tools.judge_correction import load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import persona_of  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

SCHEMES = ("orig", "raked")
CATS = ("确认", "方向相反", "幅度不符（医师同向显著）", "幅度不符（医师不显著）", "无法判断", "判官漏掉", "都不显著")
UNSUPPORTED = ("方向相反", "幅度不符（医师同向显著）", "幅度不符（医师不显著）")


def _sig(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def classify(j, jci, g, gci, dci) -> str:
    """j / g = 判官 / 医师点估计；jci / gci / dci = 判官、医师、D 的 95% CI。"""
    js, gs, ds = _sig(jci), _sig(gci), _sig(dci)
    if js:
        if gs and (j > 0) != (g > 0):
            return "方向相反"
        if ds:
            return "幅度不符（医师同向显著）" if gs else "幅度不符（医师不显著）"
        return "确认" if gs else "无法判断"
    return "判官漏掉" if gs else "都不显著"


def _ci(xs):
    xs = np.sort(xs[~np.isnan(xs)])
    if len(xs) < 50:
        return None
    return (float(xs[int(0.025 * (len(xs) - 1))]), float(xs[int(0.975 * (len(xs) - 1))]))


def _p(xs):
    xs = xs[~np.isnan(xs)]
    n = len(xs)
    if n < 50:
        return None
    lo = (1 + np.sum(xs <= 0)) / (n + 1)
    hi = (1 + np.sum(xs >= 0)) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(ps: dict) -> dict:
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    m, out, run = len(items), {}, 0.0
    for r, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        out[k] = run
    return out


def judge_rate(counts, personas, g, o):
    num = den = 0
    for p in personas:
        c = counts.get(p, {}).get(g, {}).get(o)
        if c:
            num += c[0]
            den += c[1]
    return num / den if den else np.nan


def analyse(ids, gold, cells, meta, frame_ids, judges: dict, *, n_boot=2000, seed=0, log=None) -> dict:
    """judges = {名: (计数, 比较项列表)}。→ 每家判官的逐格结果。"""
    pop_cells = Counter(cells[i] for i in frame_ids)
    pop_model = Counter(meta[i]["model"] for i in frame_ids)
    pop_ab = Counter((meta[i]["arm"], meta[i]["band"]) for i in frame_ids)
    model_of = {i: meta[i]["model"] for i in ids}
    ab_of = {i: (meta[i]["arm"], meta[i]["band"]) for i in ids}
    groups, G, F = GPR._design(ids, gold, meta)
    gidx = {g: k for k, g in enumerate(groups)}
    oidx = {o: k for k, o in enumerate(GP.OUTCOMES)}
    W, _ = GPR.all_weights(ids, cells, model_of, ab_of, pop_cells, pop_model, pop_ab)
    gold_pt = {s: GPR._rates(G, F, W[s]) for s in SCHEMES}

    by_p = defaultdict(list)
    for k, i in enumerate(ids):
        by_p[persona_of(i)].append(k)
    pn = sorted(by_p)
    for name, (counts, _) in judges.items():
        missing = set(pn) - set(counts)
        if missing:
            raise ValueError(f"{name} 判官缺少人设 {sorted(missing)}")

    cells_all = {name: [(a, b, lab, o) for a, b, lab in cons for o in JG.OUTCOMES] for name, (_, cons) in judges.items()}
    jb = {name: np.full((n_boot, len(cl)), np.nan) for name, cl in cells_all.items()}
    gb = {s: np.full((n_boot, len(groups), len(GP.OUTCOMES)), np.nan) for s in SCHEMES}
    rng = random.Random(seed)
    t0 = time.time()
    for b in range(n_boot):
        draw = [rng.choice(pn) for _ in pn]
        bi = [k for p in draw for k in by_p[p]]
        uid = [f"{ids[k]}@{j}" for j, k in enumerate(bi)]
        bW, _ = GPR.all_weights(uid, {u: cells[ids[k]] for u, k in zip(uid, bi)},
                                {u: model_of[ids[k]] for u, k in zip(uid, bi)},
                                {u: ab_of[ids[k]] for u, k in zip(uid, bi)}, pop_cells, pop_model, pop_ab)
        for s in SCHEMES:
            gb[s][b] = GPR._rates(G[bi], F[bi], bW[s])
        for name, (counts, _) in judges.items():
            for c, (a, bb, _, o) in enumerate(cells_all[name]):
                jb[name][b, c] = judge_rate(counts, draw, a, o) - judge_rate(counts, draw, bb, o)
        if log and (b + 1) % 200 == 0:
            log(f"  自助 {b + 1}/{n_boot}（{time.time() - t0:.0f}s）")

    out = {}
    for name, (counts, _) in judges.items():
        rows = []
        for c, (a, bb, lab, o) in enumerate(cells_all[name]):
            if a not in gidx or bb not in gidx:
                continue
            j = judge_rate(counts, pn, a, o) - judge_rate(counts, pn, bb, o)
            jd = jb[name][:, c]
            row = {"contrast": lab, "outcome": o, "judge": j, "judge_ci": _ci(jd)}
            for s in SCHEMES:
                ga, gbb = gidx[a], gidx[bb]
                g = gold_pt[s][ga, oidx[o]] - gold_pt[s][gbb, oidx[o]]
                gd = gb[s][:, ga, oidx[o]] - gb[s][:, gbb, oidx[o]]
                dd = jd - gd
                row[s] = {"gold": None if np.isnan(g) else float(g), "gold_ci": _ci(gd),
                          "D": None if np.isnan(g) else float(j - g), "D_ci": _ci(dd), "D_p": _p(dd)}
            if row["orig"]["gold"] is None:
                continue
            for s in SCHEMES:
                r = row[s]
                r["cat"] = classify(j, row["judge_ci"], r["gold"], r["gold_ci"], r["D_ci"])
            rows.append(row)
        for s in SCHEMES:
            hp = holm({k: r[s]["D_p"] for k, r in enumerate(rows) if _sig(r["judge_ci"])})
            for k, r in enumerate(rows):
                r[s]["D_p_holm"] = hp.get(k)
        out[name] = {"rows": rows, "summary": {s: summary(rows, s) for s in SCHEMES}}
    return out


def summary(rows, s) -> dict:
    c = Counter(r[s]["cat"] for r in rows)
    sig = [r for r in rows if _sig(r["judge_ci"])]
    both = [r for r in sig if r["orig"]["cat"] in UNSUPPORTED and r["raked"]["cat"] in UNSUPPORTED]
    holm_sig = [r for r in sig if r[s]["cat"] in UNSUPPORTED and (r[s]["D_p_holm"] or 1) < 0.05]
    return {"counts": {k: c.get(k, 0) for k in CATS}, "judge_sig": len(sig),
            "confirmed": sum(r[s]["cat"] == "确认" for r in sig),
            "unsupported": sum(r[s]["cat"] in UNSUPPORTED for r in sig),
            "reversed": sum(r[s]["cat"] == "方向相反" for r in sig),
            "undecidable": sum(r[s]["cat"] == "无法判断" for r in sig),
            "unsupported_both_weightings": len(both),
            "unsupported_holm": len(holm_sig)}


# ---------------------------------------------------------------- 报告
def _pc(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def _cs(ci):
    return "—" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def render(res, info) -> str:
    L = ["# 判官 vs 医师：直接检验效应之差 D（联合自助）", "",
         f"- 生成 {info['generated_at']}；金标 {info['n_gold']} 段（原始共识）；人设聚类联合自助 {info['n_boot']} 次"
         "（判官与医师共用同一批人设重抽；医师每次重算抽样格权重与 raking）。",
         "- 分类：确认 = 医师同向显著且 D 的 CI 含 0；方向相反 = 医师反向显著；幅度不符 = D 的 CI 不含 0；"
         "无法判断 = D 的 CI 含 0 且医师不显著。「未获支持」= 方向相反 + 幅度不符。", ""]
    for name, r in res.items():
        L += [f"## {name} 判官", "", "| 权重 | 判官显著 | 确认 | 未获支持 | 其中方向相反 | 无法判断 | 两套权重都未获支持 | 未获支持且 D 经 Holm 校正 p<0.05 |",
              "|---|---|---|---|---|---|---|---|"]
        for s in SCHEMES:
            m = r["summary"][s]
            L.append(f"| {s} | {m['judge_sig']} | {m['confirmed']} | {m['unsupported']} | {m['reversed']} | "
                     f"{m['undecidable']} | {m['unsupported_both_weightings']} | {m['unsupported_holm']} |")
        L += ["", "| 比较 | 结局 | 判官 Δ [CI] | 医师 Δ（原）[CI] | D（原）[CI] | 分类（原） | 医师 Δ（raking）[CI] | D（raking）[CI] | 分类（raking） |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            o, k = x["orig"], x["raked"]
            L.append(f"| {x['contrast']} | {x['outcome']} | {_pc(x['judge'])} {_cs(x['judge_ci'])} | {_pc(o['gold'])} {_cs(o['gold_ci'])} | "
                     f"{_pc(o['D'])} {_cs(o['D_ci'])} | {o['cat']} | {_pc(k['gold'])} {_cs(k['gold_ci'])} | {_pc(k['D'])} {_cs(k['D_ci'])} | {k['cat']} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 vs 医师：效应之差的联合自助检验")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rel = lambda p: str(REPO / p)
    rows = [g for p in ("gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl") for g in load_gold(rel(p))]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cells = sampling_cells(load_frame_scores(rel("runs/v2/scored_v6.jsonl")), load_frame_scores(rel("runs/v2/scored_v5.jsonl")))
    meta = GP.turn_meta(rel("runs/v2/dialogues_both_arms.jsonl"))
    frame_ids = [i for i in cells if i in meta]
    judges = {}
    for name, path in JG.JUDGES.items():
        cons = JG.CONTRASTS_ALL if name == "Qwen" else [c for c in JG.CONTRASTS_ALL if c[2] in JG.LOCAL_ONLY]
        judges[name] = (JG.judge_counts(rel(path), meta), cons)
    res = analyse(ids, gold, cells, meta, frame_ids, judges, n_boot=a.n_boot, seed=a.seed, log=print)
    info = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_vs_gold_joint.json").write_text(json.dumps({"meta": info, **res}, ensure_ascii=False, indent=2,
                                                             default=lambda x: str(x)), encoding="utf-8")
    (out / "judge_vs_gold_joint.md").write_text(render(res, info), encoding="utf-8")
    for name, r in res.items():
        print(name, {s: r["summary"][s] for s in SCHEMES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
