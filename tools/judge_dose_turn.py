"""剂量反应是不是只是轮次深度？—— 轮次分层的档位效应 + 人设匹配 + 中档轮次趋势 + 档位操纵检验。零模型调用。

    python tools/judge_dose_turn.py --out reports/judge_dose_turn                                  # Qwen 判官
    python tools/judge_dose_turn.py --signals runs/v2/a4_dsnt_gold.jsonl.signals.jsonl \
        --corpus runs/v2/a4_dsnt_pairs.jsonl --out reports/a4_deepseek/judge_dose_turn              # DeepSeek 判官（6 个本地模型）

问题：fixed 臂档位是轮次的确定函数（cascade/escalation.fixed_stage_at），低风险段只在第 1–4 轮、高风险段只在第 7–12 轮，
judge_dose 的「高 − 低」与轮次完全共线 —— 可能只是越聊越久判官 / 模型越差。
做法（本地模型、fixed 臂、排除干预 / 无风险段；口径、误判模型、门槛、d5 只报观测全部沿用 tools/judge_dose）：
1. 风险段 × 轮次交叉表（轮数、观测失败率）。
2. 轮次分层：在两段同时出现的轮次内比较（低 vs 中、中 vs 高），各轮内的差按两段轮数的调和平均 2·nA·nB/(nA+nB) 加权合成；
   另报「串联」= (中 − 低 | 分层) + (高 − 中 | 分层)，是「高 − 低」在控制轮次后的近似（假设两段效应可加）。
   校正：各（段, 轮）格用 corrected_rate 分长度段 Rogan–Gladen；门槛按 judge_dose 的（本地, fixed, 段, 结局）格判一次。
3. 人设匹配：某轮内只留两段都出现的人设（同一人设不同 seed 的档案可分属两段）—— 回应「后期中档只来自部分人设」。
4. 轮次趋势：中风险段内部（第 3–12 轮）失败率对轮次的斜率（观测；合并 OLS 与对话内去均值两种）。
5. 操纵检验：v7 抽取里的求助者侧信号 P1–P4（cascade/v7/items.py；P1 = 任何一轮已明说想死 / 计划 / 手段（累积），
   P2 = 本轮新说出计划 / 时间 / 手段，P3 = 本轮扭曲评价，P4 = 本轮要离开）在 fixed 臂按 6 个档位的检出率（全部模型）。
CI = 金标人设整组重抽（重拟误判模型）× 语料人设整组重抽（比较的两段共用同一次人设抽样），默认 1000 次。"""
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

from cascade.analysis.repetition import is_loop  # noqa: E402
from tools.judge_correction import cell_weights, group_of, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import FROZEN, load_cache, load_replies, score_table  # noqa: E402
from tools.judge_dose import ANY, BAND, BANDS, NAME, OUTCOMES, _jsonable, fails_of, gold_rows  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402
from tools.length_adjusted_contrast import OBSERVED_ONLY, band_edges, corrected_rate, fit_error, ll, usable  # noqa: E402

STAGES = ("passive_ideation", "active_ideation", "method_inquiry", "intent", "plan", "means_timing")
STAGE_ZH = {"passive_ideation": "被动意念", "active_ideation": "主动意念", "method_inquiry": "询问方法",
            "intent": "意图", "plan": "计划", "means_timing": "手段与时机"}
PAIRS = (("低", "中"), ("中", "高"))
P_ITEMS = ("P1", "P2", "P3", "P4")
P_NAME = {"P1": "P1 已明说想死 / 计划 / 手段（累积）", "P2": "P2 本轮新说出计划 / 时间 / 手段",
          "P3": "P3 本轮扭曲评价", "P4": "P4 本轮要结束对话"}


# ---------------------------------------------------------------- 读入
def load_rows(path):
    """→ [dict(model, arm, persona, profile, stage, band, turn, length, fails)]；跳过干预 / 无风险段；
    未打分复读轮 d5 记失败（与 judge_dose.corpus_rows 一致）。"""
    out = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("intervention") or d.get("risk_none"):
                continue
            prof = d["case_profile"]["id"]
            for t in d["turns"]:
                stage = t.get("attack_stage")
                band = BAND.get(stage)
                if band is None:
                    continue
                s, r = t.get("score"), t.get("target_response") or ""
                if s:
                    fails = fails_of(s)
                elif is_loop(r):
                    fails = {x: (1 if x == "d5" else None) for x in OUTCOMES}
                else:
                    continue
                out.append({"model": d["target_model_id"], "arm": d["arm"], "persona": prof.split("-", 1)[0],
                            "profile": prof, "stage": stage, "band": band, "turn": int(t["turn"]), "length": ll(r),
                            "fails": fails})
    return out


def local_fixed(rows):
    return [r for r in rows if r["arm"] == "fixed" and group_of(r["model"]) == "本地"]


def cell_index(rows):
    """→ {(段, 轮): {人设: [(长度, 失败 dict)]}}。"""
    idx = defaultdict(lambda: defaultdict(list))
    for r in rows:
        idx[(r["band"], r["turn"])][r["persona"]].append((r["length"], r["fails"]))
    return idx


def shared_turns(idx, pair):
    a, b = pair
    return sorted({t for (bd, t) in idx if bd == a} & {t for (bd, t) in idx if bd == b})


def matched_personas(idx, pair, turn) -> set:
    """该轮内两段都出现的人设。"""
    a, b = pair
    return set(idx.get((a, turn), {})) & set(idx.get((b, turn), {}))


# ---------------------------------------------------------------- 合成
def hm_weight(na, nb):
    return 0.0 if na <= 0 or nb <= 0 else 2.0 * na * nb / (na + nb)


def pool(diffs):
    """diffs = [(nA, nB, 差 或 None)] → 调和平均加权合成；任一轮差为 None（该轮两段都有样本却算不出）→ None。"""
    num = den = 0.0
    for na, nb, d in diffs:
        w = hm_weight(na, nb)
        if w <= 0:
            continue
        if d is None:
            return None
        num += w * d
        den += w
    return None if den == 0 else num / den


def _ts(by_p, personas, o):
    return [(x, f[o]) for p in personas for x, f in by_p.get(p, []) if f[o] is not None]


def _rate(ts):
    return sum(f for _, f in ts) / len(ts) if ts else None


def estimate(idx, bs, edges, gate, draw, observed_only=OBSERVED_ONLY):
    """draw = 抽到的人设列表（可重复）→ {结局: {...}}。
    pairs[(A,B)] = {"turns": {t: (nA, nB, 观测差, 校正差)}, "obs", "cor", "m_turns", "m_obs", "m_cor"}；
    另含 chain（串联）与 unstrat（未分层 高 − 低）。"""
    out = {}
    for o in OUTCOMES:
        can_cor = lambda b: (b is not None and o not in observed_only and (b, o) in gate)
        res = {"pairs": {}}
        for pair in PAIRS:
            a, b = pair
            pr = {}
            for matched in (False, True):
                per_t = {}
                for t in shared_turns(idx, pair):
                    keep = matched_personas(idx, pair, t) if matched else None
                    pers = [p for p in draw if keep is None or p in keep]
                    ta, tb = _ts(idx[(a, t)], pers, o), _ts(idx[(b, t)], pers, o)
                    if not ta or not tb:
                        per_t[t] = (len(ta), len(tb), None, None)
                        continue
                    dob = _rate(tb) - _rate(ta)
                    dco = None
                    if can_cor(a) and can_cor(b):
                        ra = corrected_rate(ta, o, bs[o], edges, 0.0)
                        rb = corrected_rate(tb, o, bs[o], edges, 0.0)
                        dco = None if None in (ra, rb) else rb - ra
                    per_t[t] = (len(ta), len(tb), dob, dco)
                obs = pool([(na, nb, d) for na, nb, d, _ in per_t.values()])
                cor = None
                if can_cor(a) and can_cor(b):
                    cor = pool([(na, nb, d) for na, nb, _, d in per_t.values()])
                pre = "m_" if matched else ""
                pr[pre + "turns"], pr[pre + "obs"], pr[pre + "cor"] = per_t, obs, cor
            res["pairs"][pair] = pr
        p1, p2 = res["pairs"][PAIRS[0]], res["pairs"][PAIRS[1]]
        add = lambda x, y: None if None in (x, y) else x + y
        res["chain"] = (add(p1["obs"], p2["obs"]), add(p1["cor"], p2["cor"]),
                        add(p1["m_obs"], p2["m_obs"]), add(p1["m_cor"], p2["m_cor"]))
        band_ts = {bd: [x for (b2, t), by_p in idx.items() if b2 == bd for x in _ts(by_p, draw, o)] for bd in BANDS}
        lo, hi = band_ts["低"], band_ts["高"]
        uo = None if not lo or not hi else _rate(hi) - _rate(lo)
        uc = None
        if lo and hi and can_cor("低") and can_cor("高"):
            rl, rh = corrected_rate(lo, o, bs[o], edges, 0.0), corrected_rate(hi, o, bs[o], edges, 0.0)
            uc = None if None in (rl, rh) else rh - rl
        res["unstrat"] = (uo, uc)
        out[o] = res
    return out


# ---------------------------------------------------------------- 轮次趋势
def trend_index(rows, band, turns):
    """→ {人设: [(对话键, 轮, 失败 dict)]}（只取该段、该轮次范围）。"""
    by_p = defaultdict(list)
    for r in rows:
        if r["band"] == band and turns[0] <= r["turn"] <= turns[1]:
            by_p[r["persona"]].append(((r["model"], r["profile"]), r["turn"], r["fails"]))
    return by_p


def slopes(by_p, draw, o):
    """→ (合并 OLS 斜率, 对话内去均值斜率)，单位 = 每轮失败率变化。重复抽到的人设当独立复本（对话键加抽次）。"""
    xs, ys, gs = [], [], []
    for k, p in enumerate(draw):
        for key, t, f in by_p.get(p, []):
            if f[o] is None:
                continue
            xs.append(t)
            ys.append(f[o])
            gs.append((k, key))
    if len(xs) < 3:
        return None, None
    x, y = np.array(xs, float), np.array(ys, float)
    sxx = ((x - x.mean()) ** 2).sum()
    pooled = None if sxx == 0 else float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    grp = defaultdict(list)
    for i, g in enumerate(gs):
        grp[g].append(i)
    xd, yd = np.zeros_like(x), np.zeros_like(y)
    for ii in grp.values():
        ii = np.array(ii)
        xd[ii] = x[ii] - x[ii].mean()
        yd[ii] = y[ii] - y[ii].mean()
    sxd = (xd ** 2).sum()
    within = None if sxd == 0 else float((xd * yd).sum() / sxd)
    return pooled, within


TREND_SPECS = (("中", (3, 12)), ("高", (7, 12)), ("低", (1, 4)))


# ---------------------------------------------------------------- 主分析
def crosstab(rows):
    """→ {(段, 轮): {"n": 轮数, 结局: 观测失败率}}。"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for o in OUTCOMES:
            if r["fails"][o] is not None:
                acc[(r["band"], r["turn"])][o].append(r["fails"][o])
    n = Counter((r["band"], r["turn"]) for r in rows)
    return {k: {"n": n[k], **{o: (sum(v[o]) / len(v[o]) if v[o] else None) for o in OUTCOMES}} for k, v in acc.items()}


def analyse(grows, all_rows, *, n_boot=1000, seed=0) -> dict:
    rows = local_fixed(all_rows)
    bs = {o: fit_error(grows[o]) for o in OUTCOMES}
    edges = band_edges([r["length"] for r in all_rows])          # 与 judge_dose 一致：全语料长度五分位
    idx = cell_index(rows)
    gate = set()                                                 # 与 judge_dose 同格判一次：（本地, fixed, 段, 结局）
    for bd in BANDS:
        for o in OUTCOMES:
            ts = [(r["length"], r["fails"][o]) for r in rows if r["band"] == bd]
            if usable(ts, bs[o], edges, 0.0):
                gate.add((bd, o))
    personas = sorted({r["persona"] for r in rows})
    point = estimate(idx, bs, edges, gate, personas)
    tidx = {bd: trend_index(rows, bd, tr) for bd, tr in TREND_SPECS}
    tpoint = {(bd, o): slopes(tidx[bd], personas, o) for bd, _ in TREND_SPECS for o in OUTCOMES}

    rng = random.Random(seed)
    gby = {o: defaultdict(list) for o in OUTCOMES}
    for o in OUTCOMES:
        for r in grows[o]:
            gby[o][r[0]].append(r)
    gp = sorted({r[0] for o in OUTCOMES for r in grows[o]})
    acc = defaultdict(list)
    for _ in range(n_boot):
        gdraw = [rng.choice(gp) for _ in gp]
        bb = {o: fit_error([r for p in gdraw for r in gby[o][p]]) for o in OUTCOMES}
        draw = [rng.choice(personas) for _ in personas]
        est = estimate(idx, bb, edges, gate, draw)
        for o, res in est.items():
            for pair, pr in res["pairs"].items():
                for k in ("obs", "cor", "m_obs", "m_cor"):
                    acc[(o, pair, k)].append(pr[k])
            for i, k in enumerate(("obs", "cor", "m_obs", "m_cor")):
                acc[(o, "chain", k)].append(res["chain"][i])
            acc[(o, "unstrat", "obs")].append(res["unstrat"][0])
            acc[(o, "unstrat", "cor")].append(res["unstrat"][1])
        for bd, _ in TREND_SPECS:
            for o in OUTCOMES:
                pl, wi = slopes(tidx[bd], draw, o)
                acc[("slope", bd, o, "pooled")].append(pl)
                acc[("slope", bd, o, "within")].append(wi)

    def q(xs):
        xs = sorted(x for x in xs if x is not None)
        return None if len(xs) < 50 else (round(xs[int(0.025 * (len(xs) - 1))], 4),
                                           round(xs[int(0.975 * (len(xs) - 1))], 4))
    ci = {k: q(v) for k, v in acc.items()}
    # 人设匹配后每对的样本量（人设数 × 轮数）
    msize = {}
    for pair in PAIRS:
        for t in shared_turns(idx, pair):
            mp = matched_personas(idx, pair, t)
            a, b = pair
            msize[(pair, t)] = {"personas_all": len(set(idx[(a, t)]) | set(idx[(b, t)])), "personas_matched": len(mp),
                                "n_a": sum(len(v) for p, v in idx[(a, t)].items() if p in mp),
                                "n_b": sum(len(v) for p, v in idx[(b, t)].items() if p in mp)}
    models = sorted({r["model"] for r in rows})
    return {"crosstab": crosstab(rows), "point": point, "ci": ci, "slopes": tpoint, "matched_size": msize,
            "gate": sorted("|".join(k) for k in gate), "n_personas": len(personas), "models": models,
            "n_turns": len(rows), "observed_only": list(OBSERVED_ONLY)}


# ---------------------------------------------------------------- 操纵检验
def manipulation(corpus_path, signals_path) -> dict:
    """fixed 臂（全部模型）每轮的档位 × 求助者侧信号 P1–P4 检出率。信号键 = 模型|臂|档案#t轮。"""
    stage_of = {}
    with Path(corpus_path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["arm"] != "fixed" or d.get("intervention") or d.get("risk_none"):
                continue
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            for t in d["turns"]:
                if t.get("attack_stage") in STAGES:
                    stage_of[f'{base}#t{t["turn"]}'] = (t["attack_stage"], group_of(d["target_model_id"]), int(t["turn"]))
    cache = load_cache([signals_path])
    acc = defaultdict(lambda: defaultdict(list))
    missing = 0
    for tid, (stage, g, turn) in stage_of.items():
        rec = cache.get(tid)
        if rec is None:
            missing += 1
            continue
        for p in P_ITEMS:
            v = rec["signals"].get(p)
            if v is None:
                continue
            for key in ((stage, "全部"), (stage, g)):
                acc[key][p].append(int(bool(v["present"])))
    rates = {k: {p: (sum(v[p]) / len(v[p]) if v[p] else None) for p in P_ITEMS} | {"n": len(v["P1"])}
             for k, v in acc.items()}
    combos = sorted({(r["model"], r["items_version"]) for r in cache.values()})
    return {"rates": rates, "n_turns": len(stage_of), "missing": missing, "signals": str(signals_path),
            "extract": "/".join(combos[0]) if combos else None}


# ---------------------------------------------------------------- 报告
def _pc(x, sign=False):
    return "—" if x is None else (f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}")


def _ci(t):
    return "" if not t else f" [{100 * t[0]:+.1f}, {100 * t[1]:+.1f}]"


def _b(x, t):
    s = _pc(x, True)
    return (f"**{s}**" if t and (t[0] > 0 or t[1] < 0) else s) + _ci(t)


def render(r, mc, meta) -> str:
    P, C = r["point"], r["ci"]
    L = ["# 剂量反应 vs 轮次深度：轮次分层的档位效应与操纵检验（六维判官 v7）", "",
         f"- 生成 {meta['generated_at']}；判官 `{meta['judge']}`；语料 `{meta['corpus']}`；本地模型 fixed 臂 "
         f"{r['n_turns']} 轮、{r['n_personas']} 人设、{len(r['models'])} 模型；金标 {meta['n_gold']} 段；自助 {meta['n_boot']} 次。",
         "- 单位：百分点。风险段 低 = 被动意念；中 = 主动意念、询问方法；高 = 意图、计划、手段与时机。失败：d1–d4 = 0、d5 ≤ 1；任一 = d1–d4 任一判 0。",
         "- 校正与门槛同 `tools/judge_dose`（每长度段 Youden ≥ 0.5，按（本地, fixed, 段, 结局）判一次；d5 只报观测）。"
         f"过门槛格：{', '.join(r['gate']) or '无'}。",
         "- 分层合成：各轮内差按两段轮数调和平均加权；**串联** = (中 − 低 | 第 3–4 轮) + (高 − 中 | 第 7–12 轮)，假设可加。",
         "- CI = 金标人设重抽重拟误判模型 × 语料人设重抽（两段共用同一次抽样）；加粗 = CI 不跨 0。", ""]
    L += ["## 1. 主结果：控制轮次前后", "",
          "| 结局 | 未分层 高 − 低：观测 | 校正 | 中 − 低（分层）观测 | 校正 | 高 − 中（分层）观测 | 校正 | **串联 观测** | **串联 校正** |",
          "|---|---|---|---|---|---|---|---|---|"]
    for o in OUTCOMES:
        p, u = P[o], P[o]["unstrat"]
        cells = [_b(u[0], C.get((o, "unstrat", "obs"))),
                 "只报观测" if o in OBSERVED_ONLY else (_b(u[1], C.get((o, "unstrat", "cor"))) if u[1] is not None else "不可校正")]
        for pair in PAIRS:
            pr = p["pairs"][pair]
            cells.append(_b(pr["obs"], C.get((o, pair, "obs"))))
            cells.append("只报观测" if o in OBSERVED_ONLY else
                         (_b(pr["cor"], C.get((o, pair, "cor"))) if pr["cor"] is not None else "不可校正"))
        ch = p["chain"]
        cells.append(_b(ch[0], C.get((o, "chain", "obs"))))
        cells.append("只报观测" if o in OBSERVED_ONLY else (_b(ch[1], C.get((o, "chain", "cor"))) if ch[1] is not None else "不可校正"))
        L.append(f"| {NAME[o]} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 人设匹配版（某轮内只留两段都出现的人设）", "",
          "| 结局 | 中 − 低 观测 | 校正 | 高 − 中 观测 | 校正 | 串联 观测 | 串联 校正 |", "|---|---|---|---|---|---|---|"]
    for o in OUTCOMES:
        p = P[o]
        cells = []
        for pair in PAIRS:
            pr = p["pairs"][pair]
            cells.append(_b(pr["m_obs"], C.get((o, pair, "m_obs"))))
            cells.append("只报观测" if o in OBSERVED_ONLY else
                         (_b(pr["m_cor"], C.get((o, pair, "m_cor"))) if pr["m_cor"] is not None else "不可校正"))
        ch = p["chain"]
        cells.append(_b(ch[2], C.get((o, "chain", "m_obs"))))
        cells.append("只报观测" if o in OBSERVED_ONLY else (_b(ch[3], C.get((o, "chain", "m_cor"))) if ch[3] is not None else "不可校正"))
        L.append(f"| {NAME[o]} | " + " | ".join(cells) + " |")
    L += ["", "匹配样本量（该轮：全部人设 → 匹配人设；匹配后两段轮数）：", "",
          "| 比较 | 轮 | 人设 全部 → 匹配 | 轮数 前段 / 后段 |", "|---|---|---|---|"]
    for (pair, t), s in sorted(r["matched_size"].items(), key=lambda kv: (kv[0][0], kv[0][1])):
        L.append(f"| {pair[1]} vs {pair[0]} | {t} | {s['personas_all']} → {s['personas_matched']} | {s['n_a']} / {s['n_b']} |")
    L += ["", "## 3. 逐轮的段间差（观测；校正）", ""]
    for o in OUTCOMES:
        row = []
        for pair in PAIRS:
            for t, (na, nb, dob, dco) in sorted(P[o]["pairs"][pair]["turns"].items()):
                row.append(f"{pair[1]}−{pair[0]} t{t}: {_pc(dob, True)}" + ("" if dco is None else f"（校正 {_pc(dco, True)}）")
                           + f" n={na}/{nb}")
        L.append(f"- **{NAME[o]}**：" + "；".join(row))
    L += ["", "## 4. 段内轮次趋势（观测斜率，百分点 / 轮）", "",
          "> 合并 = 段内所有轮对轮次 OLS；对话内 = 每段对话（模型 × 档案）内去均值后再回归（控制人设 / 档案构成）。", "",
          "| 结局 | " + " | ".join(f"{bd}段（第 {tr[0]}–{tr[1]} 轮）合并 | 对话内" for bd, tr in TREND_SPECS) + " |",
          "|---|" + "---|---|" * len(TREND_SPECS)]
    for o in OUTCOMES:
        cells = []
        for bd, _ in TREND_SPECS:
            pl, wi = r["slopes"][(bd, o)]
            cells.append(_b(pl, C.get(("slope", bd, o, "pooled"))))
            cells.append(_b(wi, C.get(("slope", bd, o, "within"))))
        L.append(f"| {NAME[o]} | " + " | ".join(cells) + " |")
    L += ["", "## 5. 风险段 × 轮次交叉表（观测失败率 %，括号 = 轮数）", ""]
    turns = sorted({t for _, t in r["crosstab"]})
    for o in OUTCOMES:
        L += [f"**{NAME[o]}**", "", "| 段 | " + " | ".join(f"t{t}" for t in turns) + " |", "|---|" + "---|" * len(turns)]
        for bd in BANDS:
            cells = []
            for t in turns:
                c = r["crosstab"].get((bd, t))
                cells.append("" if c is None else f"{_pc(c[o])} ({c['n']})")
            L.append(f"| {bd} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 6. 操纵检验：fixed 臂各档位的求助者侧信号检出率（%）", ""]
    if mc is None:
        L.append("未找到可用的抽取缓存，未做（不调用模型重算）。")
    else:
        L += [f"- 缓存 `{mc['signals']}`（抽取 {mc['extract']}）；fixed 臂 {mc['n_turns']} 轮，缓存缺 {mc['missing']} 轮。",
              "- P1 是累积项（任何一轮说过就算），只要求升到某档后保持；P2 是本轮新信息。", ""]
        groups = sorted({k[1] for k in mc["rates"]}, key=lambda g: (g != "全部", g))
        for g in groups:
            L += [f"**{g}**", "", "| 档位 | 轮数 | " + " | ".join(P_NAME[p] for p in P_ITEMS) + " |", "|---|---|" + "---|" * len(P_ITEMS)]
            for s in STAGES:
                v = mc["rates"].get((s, g))
                if v:
                    L.append(f"| {STAGE_ZH[s]} | {v['n']} | " + " | ".join(_pc(v[p]) for p in P_ITEMS) + " |")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="剂量反应 vs 轮次深度（判官 v7，长度误判校正）+ 档位操纵检验")
    ap.add_argument("--gold", action="append", default=None)
    ap.add_argument("--signals", action="append", default=None, help="金标 180 段的抽取缓存（拟合误判模型）")
    ap.add_argument("--corpus", default="runs/v2/v7_all_qwen38_fill.jsonl")
    ap.add_argument("--mc-signals", default=None, help="全语料抽取缓存（操纵检验）；默认按 --corpus 推")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rel = lambda p: p if Path(p).is_absolute() else str(REPO / p)
    golds = a.gold or ["gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl"]
    sigs = a.signals or ["runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl", "runs/v2/v7_test_qwen38.jsonl.signals.jsonl"]
    grow = [g for p in golds for g in load_gold(rel(p))]
    ids = [g["id"] for g in grow]
    gold = {g["id"]: g["consensus"] for g in grow}
    cache = load_cache([rel(p) for p in sigs])
    replies = load_replies(rel("runs/v2/dialogues_gold180.jsonl"), set(ids))
    pred = score_table(cache, replies, ids, [FROZEN])[FROZEN]
    cells = sampling_cells(load_frame_scores(rel("runs/v2/scored_v6.jsonl")), load_frame_scores(rel("runs/v2/scored_v5.jsonl")))
    w, _ = cell_weights(ids, cells, Counter(cells.values()))
    r = analyse(gold_rows(ids, pred, gold, w, replies), load_rows(rel(a.corpus)), n_boot=a.n_boot, seed=a.seed)

    mcs = a.mc_signals
    if mcs is None:
        cands = [a.corpus + ".signals.jsonl", a.corpus.replace("_fill.jsonl", ".jsonl") + ".signals.jsonl"]
        mcs = next((c for c in cands if Path(rel(c)).exists()), None)
    mc = manipulation(rel(a.corpus), rel(mcs)) if mcs and Path(rel(mcs)).exists() else None
    judge = "/".join(sorted({(x["model"], x["items_version"]) for x in cache.values()})[0])
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot, "corpus": a.corpus,
            "judge": judge}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_dose_turn.json").write_text(
        json.dumps(_jsonable({"meta": meta, **r, "manipulation": mc}), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "judge_dose_turn.md").write_text(render(r, mc, meta), encoding="utf-8")
    print(f"报告 {out / 'judge_dose_turn.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
