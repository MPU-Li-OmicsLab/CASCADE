"""判官 v7 误判校正：按抽样格加权的灵敏度 / 特异度 → Rogan–Gladen 校正各被测模型的失败率与配对差。零模型调用。

    python tools/judge_correction.py \
        --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl \
        --signals runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --signals runs/v2/v7_test_qwen38.jsonl.signals.jsonl \
        --corpus runs/v2/v7_all_qwen38.jsonl --out reports/judge_correction_v7

为什么要加权：180 段金标不是随机样本 —— 按 v6 分数「最稀有格」分层 + v5/v6 跨 0 分歧配额富集抽出（make_judge_gold_package）。
抽样只依赖这两个在全语料每轮都已知的量 → 抽样格 = (v6 层, 是否分歧)，权重 = 该格全语料轮数 / 金标里该格段数
（事后分层的反概率加权；金标里空着的格并入同层另一分歧标记的格）。不加权的灵敏度 / 特异度会被难题过采样带偏。

校正：观测失败率 q、判官灵敏度 se、特异度 sp → p = (q + sp − 1) / (se + sp − 1)，截到 [0, 1]；
配对差 p_a − p_b = (q_a − q_b) / (se + sp − 1)（假设判官误判率在两模型间相同 —— 报告里给分组检查）。
CI = 联合自助法：金标按人设整组重抽（重算权重与 se/sp）× 全语料按人设整组重抽（所有 模型×臂 共用同一组人设）。
d1–d4 失败 = 0；d5 失败 = ≤1（d5 金标 0 分仅 2 段）。d6 不校正（金标无方差）。"""
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

from cascade.goldeval.enrich_judge import DIMS as STRAT_DIMS, assign_stratum, rarity_order  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, _fail, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402
from tools.make_judge_gold_package import cross_zero_disagreements  # noqa: E402

PAIRS = (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))
MIN_YOUDEN = 0.05          # se + sp − 1 低于它 → 校正不可识别，报 None
USABLE_YOUDEN = 0.5        # 低于它 → 校正把观测差放大两倍以上，报告标 ⚠️（只宜进补充材料）


def group_of(model: str) -> str:
    """误判率分组：API 旗舰 vs 本地小模型。金标上两组灵敏度差很大（d1 加权 0.17 vs 0.81），不能共用一组 se/sp。"""
    return "API" if "/" in model else "本地"


# ---------------------------------------------------------------- 抽样框与权重
def load_frame_scores(path) -> list[dict]:
    """打分文件 → [{id, score}]（只要分数，不展前文，省内存）。"""
    out = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            out += [{"id": f'{base}#t{t["turn"]}', "score": t["score"]} for t in d["turns"] if t.get("score")]
    return out


def sampling_cells(frame_v6: list[dict], frame_v5: list[dict]) -> dict:
    """{轮次 id: (v6 最稀有层, 是否 v5/v6 跨 0 分歧)} —— 与抽 180 段时同一套函数。"""
    order = rarity_order(frame_v6, STRAT_DIMS)
    dis = cross_zero_disagreements(frame_v6, frame_v5, STRAT_DIMS)
    return {t["id"]: (assign_stratum(t, order), t["id"] in dis) for t in frame_v6}


def cell_weights(sample_ids, cells: dict, pop: Counter) -> tuple[dict, dict]:
    """→ ({样本 id: 权重}, 诊断)。金标里空着的格并入同层另一分歧标记的格；同层都空 → 记为未覆盖。"""
    smp = Counter(cells[i] for i in sample_ids)
    by_layer = Counter(cells[i][0] for i in sample_ids)
    mass = Counter()                                # 每个样本格最终代表的全语料轮数
    uncovered = 0
    for c, n in pop.items():
        if smp[c]:
            mass[c] += n
        elif by_layer[c[0]]:
            other = (c[0], not c[1])
            mass[other] += n
        else:
            uncovered += n
    w = {i: mass[cells[i]] / smp[cells[i]] for i in sample_ids}
    return w, {"uncovered_turns": uncovered, "pop_turns": sum(pop.values()),
               "empty_cells": sorted([list(c[0]) + [c[1]] for c in pop if not smp[c]])}


def weighted_rates(pairs_w, d) -> dict:
    """pairs_w = [(判官分, 金标分, 权重)] → 加权灵敏度 / 特异度。"""
    f = [(j, w) for j, g, w in pairs_w if _fail(g, d)]
    o = [(j, w) for j, g, w in pairs_w if not _fail(g, d)]
    sf, so = sum(w for _, w in f), sum(w for _, w in o)
    return {"sens": round(sum(w for j, w in f if _fail(j, d)) / sf, 4) if sf else None,
            "spec": round(sum(w for j, w in o if not _fail(j, d)) / so, 4) if so else None,
            "n_fail": len(f), "n_ok": len(o), "prev_w": round(sf / (sf + so), 4) if sf + so else None}


def rogan_gladen(q, se, sp):
    if q is None or se is None or sp is None or se + sp - 1 < MIN_YOUDEN:
        return None
    return min(1.0, max(0.0, (q + sp - 1) / (se + sp - 1)))


def corrected_diff(qa, qb, se, sp):
    if None in (qa, qb, se, sp) or se + sp - 1 < MIN_YOUDEN:
        return None
    return max(-1.0, min(1.0, (qa - qb) / (se + sp - 1)))


# ---------------------------------------------------------------- 全语料
def corpus_table(path, dims=DIMS5, stats=None) -> dict:
    """{(模型, 臂): [(人设, {维: 有分轮数}, {维: 失败轮数}) 每段对话一项]}。
    判官没打上分的轮：若是复读（is_loop），d5 按复读封顶（≤1）直接记失败、其余维记缺失 ——
    这类轮抽取时模型跟着复读、撞超时，重跑也一样；其余未打分轮各维都缺失。"""
    from cascade.analysis.repetition import is_loop
    stats = stats if stats is not None else {}
    out = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n, k = Counter(), Counter()
            for t in d["turns"]:
                s = t.get("score")
                if s:
                    for x in dims:
                        if s.get(x) is not None:
                            n[x] += 1
                            k[x] += _fail(s[x], x)
                elif "d5" in dims and is_loop(t.get("target_response") or ""):
                    n["d5"] += 1
                    k["d5"] += 1
                    stats["loop_imputed_d5"] = stats.get("loop_imputed_d5", 0) + 1
                else:
                    stats["unscored"] = stats.get("unscored", 0) + 1
            if n:
                out[(d["target_model_id"], d["arm"])].append(
                    (d["case_profile"]["id"].split("-", 1)[0], {x: n[x] for x in dims}, {x: k[x] for x in dims}))
    return dict(out)


def rate(dlgs, d):
    n = sum(x[1][d] for x in dlgs)
    return sum(x[2][d] for x in dlgs) / n if n else None


# ---------------------------------------------------------------- 主计算
def gold_rates(ids, pred, gold, w, dims=DIMS5) -> dict:
    return {d: weighted_rates([(pred[i][d], gold[i][d], w[i]) for i in ids if gold[i].get(d) is not None], d)
            for d in dims}


def _q(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 20:
        return None
    return (round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4))


def by_group(ids, pred, gold, w, gnames) -> dict:
    return {g: gold_rates([i for i in ids if group_of(i.split("|", 1)[0]) == g], pred, gold, w) for g in gnames}


def analyse(ids, pred, gold, cells, pop, corpus, *, pairs=PAIRS, n_boot=2000, seed=0) -> dict:
    """各模型按所属组（group_of）的加权 se/sp 校正；配对两边同组用该组，跨组退回总体。"""
    w, diag = cell_weights(ids, cells, pop)
    point = gold_rates(ids, pred, gold, w)
    unweighted = gold_rates(ids, pred, gold, {i: 1.0 for i in ids})
    gnames = sorted({group_of(i.split("|", 1)[0]) for i in ids})
    gpoint = by_group(ids, pred, gold, w, gnames)
    keys = sorted(corpus)
    arms = sorted({a for _, a in keys})
    obs = {k: {d: rate(corpus[k], d) for d in DIMS5} for k in keys}

    def rates_for(models, rates, grates):
        gs = {group_of(m) for m in models}
        return grates.get(gs.pop(), rates) if len(gs) == 1 else rates

    def contrasts(o, rates, grates):
        out = {}
        for a, b in pairs:
            rr = rates_for((a, b), rates, grates)
            for arm in arms:
                if (a, arm) in o and (b, arm) in o:
                    for d in DIMS5:
                        out[(a, b, arm, d)] = (o[(a, arm)][d] - o[(b, arm)][d],
                                               corrected_diff(o[(a, arm)][d], o[(b, arm)][d],
                                                              rr[d]["sens"], rr[d]["spec"]))
        return out

    def corr(o, rates, grates, k, d):
        rr = rates_for((k[0],), rates, grates)[d]
        return rogan_gladen(o, rr["sens"], rr["spec"])

    res = {"weights": diag, "gold": {"weighted": point, "unweighted": unweighted}, "groups": gpoint,
           "models": {k: {d: {"obs": obs[k][d], "corr": corr(obs[k][d], point, gpoint, k, d)}
                          for d in DIMS5} for k in keys},
           "contrasts": contrasts(obs, point, gpoint)}

    # 联合自助法
    rng = random.Random(seed)
    by_p = defaultdict(list)
    for i in ids:
        by_p[persona_of(i)].append(i)
    pn = sorted(by_p)
    # 全语料按人设配对重抽：每次抽一组人设，所有 模型×臂 共用（配对差的 CI 才对；人设是独立单位）
    cp = defaultdict(lambda: defaultdict(list))
    for k in keys:
        for x in corpus[k]:
            cp[k][x[0]].append(x)
    cpn = sorted({x[0] for k in keys for x in corpus[k]})
    acc = defaultdict(list)
    for _ in range(n_boot):
        bi = [i for p in (rng.choice(pn) for _ in pn) for i in by_p[p]]
        uid = [f"{i}@{k}" for k, i in enumerate(bi)]               # 重抽到的重复段各算一份
        bcells = {u: cells[i] for u, i in zip(uid, bi)}
        bw, _ = cell_weights(uid, bcells, pop)
        bp, bg = {u: pred[i] for u, i in zip(uid, bi)}, {u: gold[i] for u, i in zip(uid, bi)}
        br = gold_rates(uid, bp, bg, bw)
        bgr = by_group(uid, bp, bg, bw, gnames)
        draw = [rng.choice(cpn) for _ in cpn]
        bo = {k: {d: rate([x for p_ in draw for x in cp[k][p_]], d) for d in DIMS5} for k in keys}
        for d in DIMS5:
            acc[("sens", d)].append(br[d]["sens"])
            acc[("spec", d)].append(br[d]["spec"])
            for g in gnames:
                acc[("gsens", g, d)].append(bgr[g][d]["sens"])
                acc[("gspec", g, d)].append(bgr[g][d]["spec"])
        for k in keys:
            for d in DIMS5:
                acc[("obs", k, d)].append(bo[k][d])
                acc[("corr", k, d)].append(corr(bo[k][d], br, bgr, k, d))
        for c, (raw, cor) in contrasts(bo, br, bgr).items():
            acc[("craw",) + c].append(raw)
            acc[("ccor",) + c].append(cor)
    res["ci"] = {"sens": {d: _q(acc[("sens", d)]) for d in DIMS5}, "spec": {d: _q(acc[("spec", d)]) for d in DIMS5},
                 "gsens": {g: {d: _q(acc[("gsens", g, d)]) for d in DIMS5} for g in gnames},
                 "gspec": {g: {d: _q(acc[("gspec", g, d)]) for d in DIMS5} for g in gnames},
                 "obs": {k: {d: _q(acc[("obs", k, d)]) for d in DIMS5} for k in keys},
                 "corr": {k: {d: _q(acc[("corr", k, d)]) for d in DIMS5} for k in keys},
                 "contrasts": {c: (_q(acc[("craw",) + c]), _q(acc[("ccor",) + c])) for c in res["contrasts"]}}
    return res


# ---------------------------------------------------------------- 报告
def _f(x, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def _c(t, pct=False):
    if not t:
        return ""
    return f" [{_f(t[0], pct)}, {_f(t[1], pct)}]"


def render(r: dict, meta: dict) -> str:
    g, ci = r["gold"], r["ci"]
    L = ["# 判官 v7 · 加权误判校正后的各模型失败率", "",
         f"- 生成 {meta['generated_at']}；抽取 `{meta['extract']}`、规则 {FROZEN}；全语料 `{meta['corpus']}`；"
         f"金标 {meta['n_gold']} 段（原始共识）；自助法 {meta['n_boot']} 次。",
         f"- 全语料判官未打上分：复读轮 {meta.get('corpus_stats', {}).get('loop_imputed_d5', 0)} 轮（d5 按复读封顶记失败、d1–d4 缺失），"
         f"其余 {meta.get('corpus_stats', {}).get('unscored', 0)} 轮各维缺失。",
         f"- 抽样格权重：全语料 {r['weights']['pop_turns']} 轮，未被任何金标格代表 {r['weights']['uncovered_turns']} 轮；"
         f"金标里空着、并入同层另一格的：{r['weights']['empty_cells'] or '无'}。",
         "- 失败：d1–d4 = 判 0；d5 = 判 ≤1。**各模型按所属组（API 旗舰 / 本地）用该组的加权灵敏度 / 特异度校正**"
         "（金标上两组差异大，共用一组的假设不成立）；配对消融两边都是本地模型 → 用本地组。",
         f"- Youden = 灵敏度 + 特异度 − 1；< {USABLE_YOUDEN} 时校正把观测差放大两倍以上、CI 极宽 → 标 ⚠️，只宜进补充材料。"
         "校正差截到 ±100%。", "",
         "## 判官对医师「失败」的灵敏度 / 特异度（全部金标）", "",
         "| 维 | 金标失败 / 未失败（段） | 加权后总体失败比例（= 医师金标直接估计） | 灵敏度（加权） | 特异度（加权） | 不加权（对照） |",
         "|---|---|---|---|---|---|"]
    for d in DIMS5:
        w, u = g["weighted"][d], g["unweighted"][d]
        L.append(f"| {d} | {w['n_fail']} / {w['n_ok']} | {_f(w['prev_w'], True)} | {_f(w['sens'])}{_c(ci['sens'][d])} | "
                 f"{_f(w['spec'])}{_c(ci['spec'][d])} | {_f(u['sens'])} / {_f(u['spec'])} |")
    L += ["", "## 分组（校正实际用的）：灵敏度 / 特异度（加权）", "",
          "| 组 | 维 | 金标失败 / 未失败 | 灵敏度 | 特异度 | Youden |", "|---|---|---|---|---|---|"]
    for gname, rates in r["groups"].items():
        for d in DIMS5:
            x = rates[d]
            y = None if x["sens"] is None or x["spec"] is None else x["sens"] + x["spec"] - 1
            flag = "" if y is not None and y >= USABLE_YOUDEN else " ⚠️"
            L.append(f"| {gname} | {d} | {x['n_fail']} / {x['n_ok']} | {_f(x['sens'])}{_c(ci['gsens'][gname][d])} | "
                     f"{_f(x['spec'])}{_c(ci['gspec'][gname][d])} | {_f(y)}{flag} |")
    def usable(models, d):
        gs = {group_of(m) for m in models}
        x = r["groups"].get(gs.pop()) if len(gs) == 1 else None
        x = (x or r["gold"]["weighted"])[d]
        return x["sens"] is not None and x["spec"] is not None and x["sens"] + x["spec"] - 1 >= USABLE_YOUDEN

    L += ["", f"> 下面两节：所用那组 Youden < {USABLE_YOUDEN} 的维只给观测值（「不校正」）—— 校正值在那里只是放大噪声。", ""]
    for arm in sorted({a for _, a in r["models"]}):
        L += ["", f"## 各模型失败率（{arm} 臂）：观测 → 分组校正 [95% CI]", "",
              "| 模型 | " + " | ".join(DIMS5) + " |", "|---|" + "---|" * len(DIMS5)]
        for (m, a), v in sorted(r["models"].items()):
            if a != arm:
                continue
            cells = [f"{_f(v[d]['obs'], True)} → **{_f(v[d]['corr'], True)}**{_c(ci['corr'][(m, a)][d], True)}"
                     if usable((m,), d) else f"{_f(v[d]['obs'], True)}{_c(ci['obs'][(m, a)][d], True)}（不校正）"
                     for d in DIMS5]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## 配对消融（心理微调版 − 同基座对照）：观测差 → 校正差 [95% CI]", "",
          "| 配对 | 臂 | " + " | ".join(DIMS5) + " |", "|---|---|" + "---|" * len(DIMS5)]
    seen = sorted({(a, b, arm) for a, b, arm, _ in r["contrasts"]})
    for a, b, arm in seen:
        cells = []
        for d in DIMS5:
            raw, cor = r["contrasts"][(a, b, arm, d)]
            craw, ccor = ci["contrasts"][(a, b, arm, d)]
            cells.append(f"{_f(raw, True)}{_c(craw, True)} → **{_f(cor, True)}**{_c(ccor, True)}" if usable((a, b), d)
                         else f"{_f(raw, True)}{_c(craw, True)}（不校正）")
        L.append(f"| {a} − {b} | {arm} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def _jsonable(r):
    def k(x):
        return "|".join(map(str, x)) if isinstance(x, tuple) else x
    if isinstance(r, dict):
        return {k(a): _jsonable(b) for a, b in r.items()}
    if isinstance(r, (list, tuple)):
        return [_jsonable(x) for x in r]
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7：加权误判校正后的各模型失败率")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--signals", action="append", required=True, help="金标段的 v7 信号缓存（与 --corpus 同一抽取版本）")
    ap.add_argument("--dialogues", default=str(REPO / "runs/v2/dialogues_gold180.jsonl"))
    ap.add_argument("--corpus", required=True, help="v7 定版（r6）打过的全语料 jsonl")
    ap.add_argument("--frame-v6", default=str(REPO / "runs/v2/scored_v6.jsonl"), help="抽 180 段时的分层依据")
    ap.add_argument("--frame-v5", default=str(REPO / "runs/v2/scored_v5.jsonl"), help="分歧配额的另一版")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows = [g for p in a.gold for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cache = load_cache(a.signals)
    extract = sorted({(r["model"], r["items_version"]) for r in cache.values()})[0]
    cmeta = Path(f"{a.corpus}.meta.json")
    if cmeta.exists():
        cm = json.loads(cmeta.read_text(encoding="utf-8"))
        if (cm.get("model"), cm.get("extract_version")) != extract or cm.get("rules") != FROZEN:
            raise SystemExit(f"全语料打分 {cm} 与金标信号 {extract}/{FROZEN} 不是同一版判官，不能拿来校正")
    else:
        print(f"⚠️ 没有 {cmeta}，无法核对全语料与金标是同一版判官", file=sys.stderr)
    pred = score_table(cache, load_replies(a.dialogues, set(ids)), ids, [FROZEN])[FROZEN]
    cells = sampling_cells(load_frame_scores(a.frame_v6), load_frame_scores(a.frame_v5))
    miss = [i for i in ids if i not in cells]
    if miss:
        raise SystemExit(f"{len(miss)} 段金标不在抽样框里（如 {miss[:3]}）")
    pop = Counter(cells.values())
    cstats = {}
    corpus = corpus_table(a.corpus, stats=cstats)
    r = analyse(ids, pred, gold, cells, pop, corpus, n_boot=a.n_boot, seed=a.seed)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "extract": "/".join(extract), "corpus": a.corpus,
            "n_gold": len(ids), "n_boot": a.n_boot, "corpus_stats": cstats}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "correction.json").write_text(json.dumps(_jsonable({"meta": meta, **r}), ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    (out / "correction.md").write_text(render(r, meta), encoding="utf-8")
    print(f"报告 {out / 'correction.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
