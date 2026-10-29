"""医师金标直接估计的 raking（迭代比例拟合）校准版：让加权总量同时还原全语料的抽样格 / 模型 / 臂 × 风险段分布。

    python tools/gold_prevalence_raked.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl \
        --out reports/gold_prevalence_raked

背景：tools/gold_prevalence 的反概率权重只按抽样格（v6 最稀有层 × v5/v6 跨 0 分歧）算，而 180 段抽样另有按模型 / 按对话
的上限与必选段，加权总量还原不出全语料已知分布（如高风险段 0.70、陪伴 0.75、SoulChat 1.78、fixed 臂 0.92）。
本工具以抽样格权重为初值做 raking，三个边际的目标都取自抽样框（runs/v2/scored_v6.jsonl 的全部轮，9,491 轮）：
  ① 抽样格：目标 = cell_weights 给每个金标格代表的全语料轮数（空格已并入同层另一格，与原工具一致）；
  ② 模型（10 个）：全语料各模型轮数；
  ③ 臂 × 风险段（2 × 3）：全语料各格轮数。
收敛：一整轮三个边际调整后，所有类别 |加权和 / 目标 − 1| < 1e-10；最多 2000 轮。
截断版（稳健性）：带上限的 raking —— 每次边际调整后把权重截在「初始（抽样格）权重中位数 × 5」（上限固定不随迭代移动，
避免「截 → rake 又涨 → 再截」追着中位数跑），直到边际满足且无权重超过上限。
自助重抽里人设重复 / 缺失会让约束互不相容（单段抽样格被复制或丢失后与模型边际冲突，IPF 不收敛；实测完整格约 30% 重抽如此）：
  raked 先用完整抽样格边际，不收敛则退到 v6 层边际（12 层，不分跨 0 分歧标记）；仍不收敛照用最后一轮权重（偏差写进报告）。
  截断版上限按 5 / 7.5 / 10 / 15 / 20 × 初始中位数逐级放宽，全不相容就不截（= raked）。退路次数全部写进报告。
  点估计（原样本）三个边际都用完整抽样格、上限 5 倍，均精确收敛。
CI：人设聚类自助法，每次重抽人设后按同一流程重算抽样格权重 → raking → 截断 raking（三套权重共用同一批重抽）。
重抽里某个模型 / 臂 × 风险段类别没有样本时，该类别从约束中去掉，其余类别按比例放大到同一总数（次数写进报告）。
子组估计、失败定义、对比项全部复用 tools/gold_prevalence（不改它）。0 失败的格照原工具给三分法则上界 3/n。"""
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

from tools.gold_prevalence import (ANY, CONTRASTS, DESIGN_ORDER, LABEL, OUTCOMES, _cell, _gname, _jsonable,  # noqa: E402
                                   _pc, failed, subgroups, turn_meta)
from tools.judge_correction import cell_weights, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import persona_of  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

TOL = 1e-10
MAX_ITER = 2000
CAP_MULT = 5.0
SCHEMES = ("orig", "raked", "trim")
SCHEME_LABEL = {"orig": "原权重（抽样格）", "raked": "raked", "trim": "截断 raked（≤5×初始中位数）"}


# ---------------------------------------------------------------- raking 核心
def build_margin(values, targets: dict, total: float):
    """values：每个样本在该边际上的类别；targets：{类别: 全语料轮数}。
    → (codes, target 数组, 被去掉的类别)。样本里没有的类别去掉，其余按比例放大到 total（保证各边际总数一致）。"""
    present = sorted(set(values), key=str)
    missing_in_targets = [c for c in present if c not in targets]
    if missing_in_targets:
        raise ValueError(f"样本类别不在目标里：{missing_in_targets[:3]}")
    dropped = sorted((c for c in targets if c not in set(values) and targets[c] > 0), key=str)
    t = np.array([float(targets[c]) for c in present])
    t *= total / t.sum()
    idx = {c: k for k, c in enumerate(present)}
    return np.array([idx[v] for v in values], dtype=np.int64), t, dropped


def rake(w0, margins, *, tol=TOL, max_iter=MAX_ITER):
    """迭代比例拟合。margins = [(codes, targets)]（各边际 targets 总和须相同）。→ (权重, 信息)。"""
    w = np.asarray(w0, dtype=float).copy()
    if np.any(w <= 0):
        raise ValueError("初始权重须为正")
    dev = np.inf
    for it in range(1, max_iter + 1):
        for codes, t in margins:
            s = np.bincount(codes, weights=w, minlength=len(t))
            w *= (t / s)[codes]
        dev = max(float(np.max(np.abs(np.bincount(c, weights=w, minlength=len(t)) / t - 1))) for c, t in margins)
        if dev < tol:
            return w, {"converged": True, "iterations": it, "max_rel_dev": dev}
    return w, {"converged": False, "iterations": max_iter, "max_rel_dev": dev}


def trim_rake(w0, margins, *, cap=None, cap_mult=CAP_MULT, tol=TOL, max_iter=MAX_ITER):
    """带上限的 raking：每次边际调整后把权重截在 cap（默认 cap_mult × 初始权重中位数，固定不动），
    直到所有边际满足且无权重超过 cap。上限与边际不相容时不收敛，返回最后一轮并记偏差。"""
    w = np.asarray(w0, dtype=float).copy()
    if np.any(w <= 0):
        raise ValueError("初始权重须为正")
    cap = float(cap_mult * np.median(w)) if cap is None else float(cap)
    dev = np.inf
    for it in range(1, max_iter + 1):
        for codes, t in margins:
            s = np.bincount(codes, weights=w, minlength=len(t))
            w *= (t / s)[codes]
            np.minimum(w, cap, out=w)
        dev = max(float(np.max(np.abs(np.bincount(c, weights=w, minlength=len(t)) / t - 1))) for c, t in margins)
        if dev < tol:
            break
    return w, {"converged": dev < tol, "iterations": it, "max_rel_dev": dev, "cap": round(cap, 3),
               "max_over_median": float(np.max(w) / np.median(w)), "cap_met": bool(np.max(w) <= cap * (1 + 1e-9))}


def weight_diag(w) -> dict:
    w = np.asarray(w, dtype=float)
    return {"n": int(len(w)), "sum": round(float(w.sum()), 2), "kish_n_eff": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
            "max": round(float(w.max()), 2), "min": round(float(w.min()), 3), "median": round(float(np.median(w)), 2),
            "max_over_min": round(float(w.max() / w.min()), 1), "max_over_median": round(float(w.max() / np.median(w)), 2),
            "cv": round(float(w.std() / w.mean()), 3)}


# ---------------------------------------------------------------- 一次完整的权重计算（点估计与每次自助共用）
CELL_LEVELS = ("cell", "layer")          # raking 的抽样格边际：先用完整抽样格；不相容时退到 v6 层（不分跨 0 分歧标记）
CAP_LADDER = (5.0, 7.5, 10.0, 15.0, 20.0)  # 截断版：上限不相容时逐级放宽；全不行就不截（= raked）


def _cell_key(c, level):
    return c if level == "cell" else c[0]


def all_weights(uids, cell_of, model_of, ab_of, pop_cells, pop_model, pop_ab):
    """→ ({方案: 权重数组（按 uids 顺序）}, 信息)。cell_of/model_of/ab_of：{uid: 类别}。
    初值 = 抽样格权重（tools/judge_correction.cell_weights）。抽样格边际先用完整格，若与模型边际不相容（IPF 不收敛，
    多见于自助重抽里单段格被重复 / 缺失）退到 v6 层；截断版用同一组边际，上限按 CAP_LADDER 逐级放宽。"""
    w0d, cdiag = cell_weights(uids, cell_of, pop_cells)
    w0 = np.array([w0d[u] for u in uids])
    total = float(sum(pop_model.values()))
    med0 = float(np.median(w0))
    dropped = {}
    other = []
    for name, vals, tg in (("model", [model_of[u] for u in uids], pop_model),
                           ("arm_band", [ab_of[u] for u in uids], pop_ab)):
        codes, t, dr = build_margin(vals, tg, total)
        other.append((codes, t))
        dropped[name] = dr
    for level in CELL_LEVELS:
        mass = defaultdict(float)
        for u in uids:
            mass[_cell_key(cell_of[u], level)] += w0d[u]
        codes, t, _ = build_margin([_cell_key(cell_of[u], level) for u in uids], mass, total)
        margins = [(codes, t)] + other
        wr, ir = rake(w0, margins)
        if ir["converged"]:
            break
    ir["cell_level"] = level
    for mult in CAP_LADDER:
        wt, it = trim_rake(w0, margins, cap=mult * med0)
        if it["converged"]:
            it["cap_mult"] = mult
            break
    else:
        wt, it = wr.copy(), {**ir, "cap": None, "max_over_median": float(np.max(wr) / np.median(wr)),
                             "cap_met": False, "cap_mult": None}
    return {"orig": w0, "raked": wr, "trim": wt}, {"raked": ir, "trim": it, "dropped": dropped,
                                                     "uncovered_turns": cdiag["uncovered_turns"]}


# ---------------------------------------------------------------- 向量化子组估计
def _design(ids, gold, meta):
    groups = sorted(subgroups(meta, ids), key=str)
    sg = subgroups(meta, ids)
    pos = {i: k for k, i in enumerate(ids)}
    G = np.zeros((len(ids), len(groups)))
    for g_idx, g in enumerate(groups):
        for i in sg[g]:
            G[pos[i], g_idx] = 1.0
    F = np.full((len(ids), len(OUTCOMES)), np.nan)
    for k, i in enumerate(ids):
        for j, o in enumerate(OUTCOMES):
            f = failed(gold[i], o)
            if f is not None:
                F[k, j] = float(f)
    return groups, G, F


def _rates(G, F, w):
    """→ (比例矩阵 [组 × 结局]，组内无样本处 nan)。"""
    valid = ~np.isnan(F)
    fw = np.where(valid, F, 0.0) * w[:, None]
    vw = valid * w[:, None]
    num, den = G.T @ fw, G.T @ vw
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _q(xs):
    xs = np.sort(np.asarray([x for x in xs if not np.isnan(x)]))
    if len(xs) < 50:
        return None
    return (round(float(xs[int(0.025 * (len(xs) - 1))]), 4), round(float(xs[int(0.975 * (len(xs) - 1))]), 4))


def analyse(ids, gold, cells, meta, frame_ids, *, n_boot=2000, seed=0, log=None) -> dict:
    pop_cells = Counter(cells[i] for i in frame_ids)
    pop_model = Counter(meta[i]["model"] for i in frame_ids)
    pop_ab = Counter((meta[i]["arm"], meta[i]["band"]) for i in frame_ids)
    model_of = {i: meta[i]["model"] for i in ids}
    ab_of = {i: (meta[i]["arm"], meta[i]["band"]) for i in ids}
    W, info = all_weights(ids, cells, model_of, ab_of, pop_cells, pop_model, pop_ab)
    groups, G, F = _design(ids, gold, meta)
    gidx = {g: k for k, g in enumerate(groups)}
    point_rates = {s: _rates(G, F, W[s]) for s in SCHEMES}

    # 边际还原（加权 / 全语料）
    def recon(s):
        w = W[s]
        out = {}
        for name, key, popc in (("model", model_of, pop_model), ("arm_band", ab_of, pop_ab)):
            tot = Counter()
            for i, wi in zip(ids, w):
                tot[key[i]] += wi
            out[name] = {c: round(tot[c] / popc[c], 3) for c in sorted(popc, key=str)}
        dg_pop = Counter(meta[i]["dgroup"] for i in frame_ids)
        dg = Counter()
        for i, wi in zip(ids, w):
            dg[meta[i]["dgroup"]] += wi
        out["dgroup"] = {c: round(dg[c] / dg_pop[c], 3) for c in DESIGN_ORDER}
        bp = Counter(meta[i]["band"] for i in frame_ids)
        bs = Counter()
        for i, wi in zip(ids, w):
            bs[meta[i]["band"]] += wi
        out["band"] = {c: round(bs[c] / bp[c], 3) for c in sorted(bp)}
        return out

    # 自助
    rng = random.Random(seed)
    by_p = defaultdict(list)
    for k, i in enumerate(ids):
        by_p[persona_of(i)].append(k)
    pn = sorted(by_p)
    boot = {s: [] for s in SCHEMES}
    devs = {"raked": [], "trim": []}
    bstat = {"raked_nonconv": 0, "trim_nonconv": 0, "trim_cap_unmet": 0, "dropped_any": 0,
             "raked_layer_fallback": 0, "trim_cap_mult": Counter(),
             "dropped": Counter(), "uncovered_reps": 0}
    t0 = time.time()
    for b in range(n_boot):
        bi = [k for p in (rng.choice(pn) for _ in pn) for k in by_p[p]]
        uid = [f"{ids[k]}@{j}" for j, k in enumerate(bi)]
        bW, binfo = all_weights(uid, {u: cells[ids[k]] for u, k in zip(uid, bi)},
                                {u: model_of[ids[k]] for u, k in zip(uid, bi)},
                                {u: ab_of[ids[k]] for u, k in zip(uid, bi)}, pop_cells, pop_model, pop_ab)
        bstat["raked_nonconv"] += not binfo["raked"]["converged"]
        bstat["trim_nonconv"] += not binfo["trim"]["converged"]
        bstat["trim_cap_unmet"] += not binfo["trim"]["cap_met"]
        bstat["raked_layer_fallback"] += binfo["raked"]["cell_level"] == "layer"
        bstat["trim_cap_mult"][str(binfo["trim"]["cap_mult"])] += 1
        bstat["uncovered_reps"] += binfo["uncovered_turns"] > 0
        for s in ("raked", "trim"):
            if not binfo[s]["converged"]:
                devs[s].append(binfo[s]["max_rel_dev"])
        dr = [f"{m}:{c}" for m, cs in binfo["dropped"].items() if m != "cell" for c in cs]
        bstat["dropped_any"] += bool(dr)
        bstat["dropped"].update(dr)
        Gb, Fb = G[bi], F[bi]
        for s in SCHEMES:
            boot[s].append(_rates(Gb, Fb, bW[s]))
        if log and (b + 1) % 200 == 0:
            log(f"  自助 {b + 1}/{n_boot}（{time.time() - t0:.0f}s）")
    boot = {s: np.stack(v) for s, v in boot.items()}          # [B × 组 × 结局]

    res = {}
    for s in SCHEMES:
        P = point_rates[s]
        pt, ci = {}, {}
        for g in groups:
            pt[g], ci[g] = {}, {}
            for j, o in enumerate(OUTCOMES):
                col = F[:, j]
                mem = G[:, gidx[g]] > 0
                ok = mem & ~np.isnan(col)
                p = P[gidx[g], j]
                pt[g][o] = {"p": None if np.isnan(p) else round(float(p), 4), "n": int(ok.sum()),
                            "n_fail": int(col[ok].sum())}
                ci[g][o] = _q(boot[s][:, gidx[g], j])
        diffs = {}
        for a, b_ in CONTRASTS:
            if a in gidx and b_ in gidx:
                for j, o in enumerate(OUTCOMES):
                    pa, pb = P[gidx[a], j], P[gidx[b_], j]
                    d = None if np.isnan(pa) or np.isnan(pb) else round(float(pa - pb), 4)
                    diffs[(a, b_, o)] = (d, _q(boot[s][:, gidx[a], j] - boot[s][:, gidx[b_], j]))
        res[s] = {"point": pt, "ci": ci, "diffs": diffs, "weights": weight_diag(W[s]), "recon": recon(s)}
    bstat["dropped"] = dict(bstat["dropped"].most_common())
    bstat["trim_cap_mult"] = dict(sorted(bstat["trim_cap_mult"].items()))
    for s in ("raked", "trim"):
        v = np.sort(devs[s])
        bstat[f"{s}_dev_q"] = ("—", "—", "—") if not len(v) else tuple(
            f"{x:.3f}" for x in (np.median(v), np.quantile(v, 0.95), v[-1]))
    return {"schemes": res, "rake_info": {"raked": info["raked"], "trim": info["trim"],
                                          "dropped": info["dropped"]},
            "boot": bstat, "targets": {"pop_turns": len(frame_ids), "n_cells_pop": len(pop_cells),
                                       "n_cells_gold": len(set(cells[i] for i in ids)),
                                       "models": len(pop_model), "arm_band": len(pop_ab)}}


# ---------------------------------------------------------------- 报告
def _flag(d0, c0, d1, c1):
    if d0 is None or d1 is None:
        return ""
    fl = []
    if d0 * d1 < 0:
        fl.append("⚠️ 方向反转")
    x0 = None if not c0 else (c0[0] > 0 or c0[1] < 0)
    x1 = None if not c1 else (c1[0] > 0 or c1[1] < 0)
    if x0 is not None and x1 is not None and x0 != x1:
        fl.append("⚠️ 不再显著" if x0 else "⚠️ 变为显著")
    return "；".join(fl)


def _dcell(v, c):
    return ("—" if v is None else f"{100 * v:+.1f}%") + ("" if not c else f" [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]")


def render(r, info) -> str:
    S = r["schemes"]
    L = ["# 医师金标直接估计 · raking 校准版（不经判官）", "",
         f"- 生成 {info['generated_at']}；金标 {info['n_gold']} 段（原始共识）；抽样框 {r['targets']['pop_turns']} 轮；"
         f"人设聚类自助法 {info['n_boot']} 次（每次重抽后重算抽样格权重并重做 raking / 截断 raking）。",
         f"- raking 边际：抽样格（金标覆盖 {r['targets']['n_cells_gold']} 格，空格按原工具并入同层另一格）+ 模型 "
         f"{r['targets']['models']} 个 + 臂 × 风险段 {r['targets']['arm_band']} 格；目标全部取自抽样框。",
         f"- 收敛判据：一整轮后所有类别 |加权和 / 目标 − 1| < {TOL:g}，最多 {MAX_ITER} 轮。截断版：带上限 raking，"
         f"每次边际调整后截在 {CAP_MULT:g} × 初始（抽样格）权重中位数，上限固定。",
         "- 自助退路：完整抽样格边际与模型边际不相容时退到 v6 层边际；截断上限不相容时按 5 / 7.5 / 10 / 15 / 20 倍逐级放宽，"
         "全不行就不截；仍不收敛的照用最后一轮权重。次数见下。",
         "- 子组 = 加权比率估计；失败定义、子组、对比项与 tools/gold_prevalence 完全相同。"
         "原权重一列的 CI 是本工具同一批自助重抽算的（与 reports/gold_prevalence 的 CI 因随机数不同略有出入，点估计相同）。", ""]
    ri, bs = r["rake_info"], r["boot"]
    L += ["## 权重诊断", "", "| 指标 | " + " | ".join(SCHEME_LABEL[s] for s in SCHEMES) + " |", "|---|---|---|---|"]
    for k, lab in (("kish_n_eff", "Kish 有效样本量"), ("max", "最大权重"), ("min", "最小权重"), ("median", "中位数"),
                   ("max_over_min", "最大 / 最小"), ("max_over_median", "最大 / 中位数"), ("cv", "变异系数 CV")):
        L.append(f"| {lab} | " + " | ".join(str(S[s]["weights"][k]) for s in SCHEMES) + " |")
    L += ["", f"- raking：{'收敛' if ri['raked']['converged'] else '⚠️ 未收敛'}，{ri['raked']['iterations']} 轮，"
          f"最大相对偏差 {ri['raked']['max_rel_dev']:.1e}。截断版：上限 {ri['trim']['cap']}，"
          f"{'收敛' if ri['trim']['converged'] else '⚠️ 未收敛'}（{ri['trim']['iterations']} 轮），最终 max / 中位数 {ri['trim']['max_over_median']:.2f}。",
          f"- 自助中不收敛重抽的最大相对边际偏差：raked 中位数 {bs['raked_dev_q'][0]}、95 分位 {bs['raked_dev_q'][1]}、最大 {bs['raked_dev_q'][2]}；"
          f"截断 {bs['trim_dev_q'][0]} / {bs['trim_dev_q'][1]} / {bs['trim_dev_q'][2]}（—= 全部收敛）。",
          f"- 自助 {info['n_boot']} 次：raking 未收敛 {bs['raked_nonconv']} 次；截断版未收敛 {bs['trim_nonconv']} 次、"
          f"截断版所用上限倍数分布 {bs['trim_cap_mult']}（None = 不截）；raked 退到 v6 层边际 {bs['raked_layer_fallback']} 次；"
          f"截断上限未达 {bs['trim_cap_unmet']} 次；有类别因无样本被去掉的重抽 {bs['dropped_any']} 次"
          + (f"（{', '.join(f'{k} ×{v}' for k, v in list(bs['dropped'].items())[:8])}）" if bs["dropped"] else "")
          + f"；抽样格有整层未覆盖的重抽 {bs['uncovered_reps']} 次（该层轮数按比例分给其余格）。", ""]
    L += ["## 边际还原（加权总量 / 全语料轮数；1 = 还原）", "",
          "| 边际 | 类别 | " + " | ".join(SCHEME_LABEL[s] for s in SCHEMES) + " |", "|---|---|---|---|---|"]
    for m, lab in (("band", "风险段"), ("arm_band", "臂 × 风险段"), ("dgroup", "设计组"), ("model", "模型")):
        for c in S["orig"]["recon"][m]:
            cn = " · ".join(c) if isinstance(c, tuple) else c
            L.append(f"| {lab} | {cn} | " + " | ".join(f"{S[s]['recon'][m][c]:.2f}" for s in SCHEMES) + " |")
    order = [("全部",)] + [("设计组", g) for g in DESIGN_ORDER] + [("组", "API"), ("组", "本地")] + \
        [("风险段", b) for b in ("低", "中", "高")] + [("臂", a) for a in ("fixed", "responsive")] + \
        [("臂×风险段", a, b) for a in ("fixed", "responsive") for b in ("低", "中", "高")] + \
        sorted(g for g in S["orig"]["point"] if g[0] == "模型")
    L += ["", "## 差值并排（加权 [95% CI]，百分点）", "",
          "> 「⚠️」= 相对原权重的结论变化：方向反转 / CI 由不跨 0 变为跨 0（不再显著）/ 反之。", "",
          "| 比较 | 结局 | " + " | ".join(SCHEME_LABEL[s] for s in SCHEMES) + " | raked 变化 | 截断变化 |",
          "|---|---|---|---|---|---|---|"]
    for a, b in CONTRASTS:
        for o in OUTCOMES:
            if (a, b, o) not in S["orig"]["diffs"]:
                continue
            vals = [S[s]["diffs"][(a, b, o)] for s in SCHEMES]
            L.append(f"| {_gname(a)} − {_gname(b)} | {LABEL[o]} | " + " | ".join(_dcell(*v) for v in vals) +
                     f" | {_flag(*vals[0], *vals[1])} | {_flag(*vals[0], *vals[2])} |")
    for o in OUTCOMES:
        L += ["", f"## 各子组：{LABEL[o]}（加权 [95% CI]（金标失败段数））", "",
              "| 子组 | 金标段数 | " + " | ".join(SCHEME_LABEL[s] for s in SCHEMES) + " |", "|---|---|---|---|---|"]
        for g in order:
            if g not in S["orig"]["point"]:
                continue
            L.append(f"| {_gname(g)} | {S['orig']['point'][g][o]['n']} | " +
                     " | ".join(_cell(S[s]["point"][g][o], S[s]["ci"][g][o]) for s in SCHEMES) + " |")
    L += ["", "> 0 失败的格自助 CI 退化，照原工具给三分法则上界 ≈ 3/n（未加权，近似）。"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="医师金标直接估计 · raking 校准版")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--dialogues", default=str(REPO / "runs/v2/dialogues_both_arms.jsonl"))
    ap.add_argument("--frame-v6", default=str(REPO / "runs/v2/scored_v6.jsonl"))
    ap.add_argument("--frame-v5", default=str(REPO / "runs/v2/scored_v5.jsonl"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [g for p in a.gold for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cells = sampling_cells(load_frame_scores(a.frame_v6), load_frame_scores(a.frame_v5))
    meta = turn_meta(a.dialogues)
    miss = [i for i in ids if i not in cells or i not in meta]
    if miss:
        raise SystemExit(f"{len(miss)} 段金标不在抽样框 / 对话里（如 {miss[:3]}）")
    frame_ids = [i for i in cells if i in meta]
    if len(frame_ids) != len(cells):
        raise SystemExit(f"{len(cells) - len(frame_ids)} 轮抽样框不在对话里")
    t0 = time.time()
    r = analyse(ids, gold, cells, meta, frame_ids, n_boot=a.n_boot, seed=a.seed, log=print)
    info = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot,
            "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prevalence_raked.json").write_text(json.dumps(_jsonable({"meta": info, **r}), ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    (out / "prevalence_raked.md").write_text(render(r, info), encoding="utf-8")
    print(f"报告 {out / 'prevalence_raked.md'}（{info['seconds']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
