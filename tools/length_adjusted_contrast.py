"""配对消融的长度校正版：判官误判率随回复长度变化（tools/length_bias_v7：d3/d4 每翻一倍 OR≈1.5，短回复的失败易漏）
→ 按长度分段做 Rogan–Gladen，再合成各模型校正失败率与配对差。零 GPU。

    python tools/length_adjusted_contrast.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl \
        --signals runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --signals runs/v2/v7_test_qwen38.jsonl.signals.jsonl \
        --corpus runs/v2/v7_all_qwen38_fill.jsonl --out reports/length_adjusted_v7

误判模型（每维，抽样格加权）：logit P(判官判失败) = b0 + b1·医师判失败 + b2·log2(字数) + b3·API 旗舰
  → 灵敏度(len, 组) = σ(b0+b1+b2·len+b3·组)，特异度 = 1 − σ(b0+b2·len+b3·组)。
  灵敏度与误报共用一条长度斜率（金标失败段太少，分开估会完全分离；见 length_bias_v7 报告）。
全语料每个 模型×臂 的轮按全语料 log2 长度五分位分段，段内用该段平均长度的 se/sp 校正，按段内轮数加权合成。
CI = 金标人设整组重抽（重拟误判模型）× 全语料人设整组重抽（各模型共用同一组人设）。
未打分的复读轮 d5 记失败（与 judge_correction 一致）。只报配对差：各模型绝对水平同组内也不同质。

可校正门槛（2026-10-17 统一，与 six_dim_questions 同一条）：用全量金标拟合的误判模型，某模型×臂×维在全语料
每个长度段的 Youden 都 ≥ USABLE_YOUDEN（0.5）才校正，否则点估计与 CI 都报「不可校正」；门槛只按全量拟合判一次，
自助重抽里不再逐次判（否则 CI 只由过门槛的重抽组成，有偏）。
d5 一律只报观测（OBSERVED_ONLY）：① 允许 API / 本地两组各自灵敏度的交互模型下，Qwen 判官本地组 d5 Youden 仅约
0.42（judge_correction 的分组加权估计 0.41），共享斜率模型借 API 组把它抬到 0.54–0.63；② d5 的判档本身含「可见文字
< 20 字 → ≤1」，再按长度校正等于把长度用两次。校正值仍算出来放在 reference_d5，只供补充材料参考。
敏感性：另拟合「医师判失败 × API」交互模型（sensitivity_interaction），只给点估计。"""
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
from tools.judge_correction import (PAIRS, USABLE_YOUDEN, cell_weights, group_of, load_frame_scores,  # noqa: E402
                                    sampling_cells)
from tools.judge_cv import DIMS5, FROZEN, _fail, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402
from tools.misclass_correction import wlogit  # noqa: E402

N_BANDS = 5
OBSERVED_ONLY = ("d5",)     # 只报观测、不采用校正值的维（理由见模块说明）


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def ll(text: str) -> float:
    return float(np.log2(len(text or "") + 1))


def fit_error(rows):
    """rows = [(人设, 判官失败, 医师失败, log2 长度, 权重, API)] → b（4 个系数）；无方差 → None。"""
    y = np.array([r[1] for r in rows], dtype=float)
    if len(rows) < 10 or y.min() == y.max():
        return None
    X = np.array([[r[2], r[3], r[5]] for r in rows], dtype=float)
    return wlogit(X, y, np.array([r[4] for r in rows], dtype=float))


def fit_error_int(rows):
    """交互版（敏感性分析）：多一项 b4·医师判失败·API → API / 本地两组灵敏度各自估计。"""
    y = np.array([r[1] for r in rows], dtype=float)
    if len(rows) < 10 or y.min() == y.max():
        return None
    X = np.array([[r[2], r[3], r[5], r[2] * r[5]] for r in rows], dtype=float)
    return wlogit(X, y, np.array([r[4] for r in rows], dtype=float))


def se_sp(b, length, api):
    """b 有 4 个系数 = 共享斜率主模型；5 个 = 交互模型（b4 只进灵敏度）。"""
    inter = b[4] * api if len(b) > 4 else 0.0
    se = sig(b[0] + b[1] + b[2] * length + b[3] * api + inter)
    sp = 1.0 - sig(b[0] + b[2] * length + b[3] * api)
    return float(se), float(sp)


def corpus_rows(path, dims=DIMS5):
    """→ [(模型, 臂, 人设, log2 长度, {维: 失败 0/1 或 None})]。"""
    out = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            per = d["case_profile"]["id"].split("-", 1)[0]
            for t in d["turns"]:
                s, r = t.get("score"), t.get("target_response") or ""
                if s:
                    fails = {x: (None if s.get(x) is None else int(_fail(s[x], x))) for x in dims}
                elif is_loop(r):
                    fails = {x: (1 if x == "d5" else None) for x in dims}
                else:
                    continue
                out.append((d["target_model_id"], d["arm"], per, ll(r), fails))
    return out


def band_edges(lengths, n=N_BANDS):
    return list(np.quantile(np.array(lengths), [k / n for k in range(1, n)]))


def _bands(turns, edges):
    bands = defaultdict(list)
    for length, f in turns:
        if f is not None:
            bands[int(np.searchsorted(edges, length, side="right"))].append((length, f))
    return bands


def usable(turns, b, edges, api, thr=USABLE_YOUDEN) -> bool:
    """turns 落到的每个长度段，在该段平均长度处 Youden 都 ≥ thr → 可校正。"""
    if b is None:
        return False
    bands = _bands(turns, edges)
    if not bands:
        return False
    for v in bands.values():
        se, sp = se_sp(b, float(np.mean([x for x, _ in v])), api)
        if se + sp - 1 < thr:
            return False
    return True


def corrected_rate(turns, d, b, edges, api):
    """turns = [(log2 长度, 失败)] → 分段 Rogan–Gladen 合成；某段 Youden < 0.05（不可识别）→ 整体 None。
    可用性门槛（0.5）由 usable() 另判。"""
    bands = _bands(turns, edges)
    tot = sum(len(v) for v in bands.values())
    if not tot or b is None:
        return None
    acc = 0.0
    for v in bands.values():
        q = sum(f for _, f in v) / len(v)
        se, sp = se_sp(b, float(np.mean([x for x, _ in v])), api)
        if se + sp - 1 < 0.05:
            return None
        acc += len(v) / tot * min(1.0, max(0.0, (q + sp - 1) / (se + sp - 1)))
    return acc


def observed(turns):
    v = [f for _, f in turns if f is not None]
    return sum(v) / len(v) if v else None


def gate_of(by_key, bs, edges, thr=USABLE_YOUDEN) -> set:
    """全量拟合下可校正的 (模型, 臂, 维)。只判一次，自助重抽沿用。"""
    ok = set()
    for (m, arm), ts in by_key.items():
        api = float(group_of(m) == "API")
        for d in DIMS5:
            if usable([(x, f[d]) for x, f in ts], bs[d], edges, api, thr):
                ok.add((m, arm, d))
    return ok


def contrasts(by_key, bs, edges, pairs, arms, gate=None, observed_only=OBSERVED_ONLY):
    """→ {(a, c, 臂, 维): (观测差, 校正差)}；gate=None 时按 bs 现判门槛。校正差 None = 不可校正或只报观测。"""
    if gate is None:
        gate = gate_of(by_key, bs, edges)
    out = {}
    for a, c in pairs:
        api = float(group_of(a) == "API")
        for arm in arms:
            ta, tc = by_key.get((a, arm)), by_key.get((c, arm))
            if not ta or not tc:
                continue
            for d in DIMS5:
                xa = [(x, f[d]) for x, f in ta]
                xc = [(x, f[d]) for x, f in tc]
                oa, oc = observed(xa), observed(xc)
                cor = None
                if d not in observed_only and (a, arm, d) in gate and (c, arm, d) in gate:
                    ra, rc = corrected_rate(xa, d, bs[d], edges, api), corrected_rate(xc, d, bs[d], edges, api)
                    cor = None if None in (ra, rc) else max(-1.0, min(1.0, ra - rc))
                out[(a, c, arm, d)] = (None if None in (oa, oc) else oa - oc, cor)
    return out


def analyse(gold_rows, crows, *, pairs=PAIRS, n_boot=1000, seed=0) -> dict:
    bs = {d: fit_error(gold_rows[d]) for d in DIMS5}
    edges = band_edges([r[3] for r in crows])
    keys_needed = {m for p in pairs for m in p}
    by_key = defaultdict(list)
    for m, arm, per, length, fails in crows:
        if m in keys_needed:
            by_key[(m, arm)].append((length, fails))
    arms = sorted({k[1] for k in by_key})
    gate = gate_of(by_key, bs, edges)
    point = contrasts(by_key, bs, edges, pairs, arms, gate)
    ref_point = contrasts(by_key, bs, edges, pairs, arms, gate, observed_only=())   # d5 校正值，只供参考
    # 敏感性：交互误判模型（点估计；门槛按交互模型自己判）
    bis = {d: fit_error_int(gold_rows[d]) for d in DIMS5}
    point_int = contrasts(by_key, bis, edges, pairs, arms, observed_only=())
    # 长度范围内的 se/sp（本地组）供读报告
    grid = {d: ({f"{int(2 ** L) - 1}字": se_sp(bs[d], L, 0.0) for L in (5.0, 7.0, 9.0, 11.0)} if bs[d] is not None else None)
            for d in DIMS5}
    grid_int = {d: ({f"{int(2 ** L) - 1}字": se_sp(bis[d], L, 0.0) for L in (5.0, 7.0, 9.0, 11.0)}
                    if bis[d] is not None else None) for d in DIMS5}

    rng = random.Random(seed)
    gby = {d: defaultdict(list) for d in DIMS5}
    for d in DIMS5:
        for r in gold_rows[d]:
            gby[d][r[0]].append(r)
    gp = sorted({r[0] for d in DIMS5 for r in gold_rows[d]})
    cby = defaultdict(lambda: defaultdict(list))
    for m, arm, per, length, fails in crows:
        if m in keys_needed:
            cby[(m, arm)][per].append((length, fails))
    cp = sorted({r[2] for r in crows if r[0] in keys_needed})
    acc = defaultdict(list)
    for _ in range(n_boot):
        gdraw = [rng.choice(gp) for _ in gp]
        bb = {d: fit_error([r for p in gdraw for r in gby[d][p]]) for d in DIMS5}
        cdraw = [rng.choice(cp) for _ in cp]
        bk = {k: [x for p in cdraw for x in v[p]] for k, v in cby.items()}
        for c, (raw, cor) in contrasts(bk, bb, edges, pairs, arms, gate, observed_only=()).items():
            acc[("raw",) + c].append(raw)
            acc[("cor",) + c].append(cor)

    def q(xs):
        xs = sorted(x for x in xs if x is not None)
        return None if len(xs) < 50 else (round(xs[int(0.025 * (len(xs) - 1))], 4),
                                           round(xs[int(0.975 * (len(xs) - 1))], 4))
    ci = {c: (q(acc[("raw",) + c]), None if point[c][1] is None else q(acc[("cor",) + c])) for c in point}
    reference_d5 = {c: (ref_point[c][1], None if ref_point[c][1] is None else q(acc[("cor",) + c]))
                    for c in point if c[3] in OBSERVED_ONLY}
    rnd = lambda b: None if b is None else [round(float(x), 3) for x in b]
    return {"coefs": {d: rnd(bs[d]) for d in DIMS5}, "grid": grid, "edges": [round(float(e), 3) for e in edges],
            "point": point, "ci": ci, "observed_only": list(OBSERVED_ONLY), "youden_threshold": USABLE_YOUDEN,
            "gate": sorted("|".join(k) for k in gate), "reference_d5": reference_d5,
            "sensitivity_interaction": {"coefs": {d: rnd(bis[d]) for d in DIMS5}, "grid": grid_int,
                                        "point": {c: v[1] for c, v in point_int.items()}},
            "lengths": {k: round(float(np.mean([x for x, _ in v])), 2) for k, v in by_key.items()}}


def gold_table(ids, pred, gold, w, replies):
    return {d: [(persona_of(i), float(_fail(pred[i][d], d)), float(_fail(gold[i][d], d)), ll(replies[i]), w[i],
                 float(group_of(i.split("|", 1)[0]) == "API")) for i in ids if gold[i].get(d) is not None]
            for d in DIMS5}


def _pc(x):
    return "—" if x is None else f"{100 * x:+.1f}%"


def _ci(t):
    return "" if not t else f" [{100 * t[0]:+.1f}, {100 * t[1]:+.1f}]"


def render(r, meta, unadj=None) -> str:
    L = ["# 配对消融 · 长度校正后的判官失败率差", "",
         f"- 生成 {meta['generated_at']}；抽取 `{meta['extract']}`、规则 {FROZEN}；全语料 `{meta['corpus']}`；"
         f"金标 {meta['n_gold']} 段（原始共识，抽样格加权）；自助法 {meta['n_boot']} 次。",
         "- 误判模型：logit P(判官判失败) = b0 + b1·医师判失败 + b2·log2(字数) + b3·API；按全语料长度五分位分段校正后合成。",
         "- 失败：d1–d4 = 0、d5 = ≤1。负数 = 微调版失败更少。",
         f"- 可校正门槛：全量拟合下每个长度段 Youden ≥ {USABLE_YOUDEN}，否则「不可校正」；"
         f"**{'、'.join(OBSERVED_ONLY)} 只报观测**（交互模型下 Qwen 本地组 d5 Youden ≈ 0.42；d5 判档含 <20 字封顶），"
         "校正值见文末参考节。", "",
         "## 误判模型系数（b0, b1 医师判失败, b2 每翻一倍, b3 API）与本地组 se/sp 随长度", "",
         "| 维 | 系数 | 31 字 | 127 字 | 511 字 | 2047 字 |", "|---|---|---|---|---|---|"]
    for d in DIMS5:
        g = r["grid"][d]
        if g is None:
            L.append(f"| {d} | — | | | | |")
            continue
        cells = []
        for se, sp in g.values():
            y = se + sp - 1
            cells.append(f"se {se:.2f} / sp {sp:.2f}" + (" ⚠️" if y < USABLE_YOUDEN else ""))
        L.append(f"| {d} | {r['coefs'][d]} | " + " | ".join(cells) + " |")
    L += ["", "## 配对差：观测 → 长度校正 [95% CI]" + ("（第三列 = 未做长度校正的组校正，来自 judge_correction）" if unadj else ""), "",
          "| 配对 | 臂 | 平均 log2 长度（微调 / 对照） | " + " | ".join(DIMS5) + " |", "|---|---|---|" + "---|" * len(DIMS5)]
    for a, c, arm in sorted({k[:3] for k in r["point"]}):
        cells = []
        for d in DIMS5:
            raw, cor = r["point"][(a, c, arm, d)]
            craw, ccor = r["ci"][(a, c, arm, d)]
            if d in OBSERVED_ONLY:
                cell = f"**{_pc(raw)}**{_ci(craw)}（只报观测）"
            elif cor is None:
                cell = f"{_pc(raw)}{_ci(craw)} → 不可校正"
            else:
                cell = f"{_pc(raw)} → **{_pc(cor)}**{_ci(ccor)}"
            if unadj and (a, c, arm, d) in unadj and d not in OBSERVED_ONLY:
                cell += f"（未长度校正 {_pc(unadj[(a, c, arm, d)])}）"
            cells.append(cell)
        la, lc = r["lengths"].get((a, arm)), r["lengths"].get((c, arm))
        L.append(f"| {a} − {c} | {arm} | {la} / {lc} | " + " | ".join(cells) + " |")
    s = r["sensitivity_interaction"]
    L += ["", "## 敏感性：交互误判模型（+ b4·医师判失败·API，两组灵敏度各自估计；点估计）", "",
          "本地组 se/sp 随长度：", "", "| 维 | 系数 | 31 字 | 127 字 | 511 字 | 2047 字 |", "|---|---|---|---|---|---|"]
    for d in DIMS5:
        g = s["grid"][d]
        if g is None:
            L.append(f"| {d} | — | | | | |")
            continue
        L.append(f"| {d} | {s['coefs'][d]} | " + " | ".join(
            f"se {se:.2f} / sp {sp:.2f}" + (" ⚠️" if se + sp - 1 < USABLE_YOUDEN else "") for se, sp in g.values()) + " |")
    L += ["", "| 配对 | 臂 | " + " | ".join(DIMS5) + " |", "|---|---|" + "---|" * len(DIMS5)]
    for a, c, arm in sorted({k[:3] for k in r["point"]}):
        L.append(f"| {a} − {c} | {arm} | " + " | ".join(
            (_pc(s["point"][(a, c, arm, d)]) if s["point"][(a, c, arm, d)] is not None else "不可校正") for d in DIMS5) + " |")
    L += ["", f"## 参考（不采用）：{'、'.join(OBSERVED_ONLY)} 的长度校正值 [95% CI]", ""]
    for k in sorted(r["reference_d5"]):
        v, ci_ = r["reference_d5"][k]
        L.append(f"- {k[0]} − {k[1]}，{k[2]}，{k[3]}：{_pc(v) if v is not None else '不可校正'}{_ci(ci_)}")
    return "\n".join(L) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="配对消融的长度校正（判官 v7）")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--signals", action="append", required=True)
    ap.add_argument("--dialogues", default=str(REPO / "runs/v2/dialogues_gold180.jsonl"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--frame-v6", default=str(REPO / "runs/v2/scored_v6.jsonl"))
    ap.add_argument("--frame-v5", default=str(REPO / "runs/v2/scored_v5.jsonl"))
    ap.add_argument("--unadjusted", default=None, help="可选：judge_correction 的 correction.json，并列未长度校正的差")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [g for p in a.gold for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cache = load_cache(a.signals)
    replies = load_replies(a.dialogues, set(ids))
    pred = score_table(cache, replies, ids, [FROZEN])[FROZEN]
    cells = sampling_cells(load_frame_scores(a.frame_v6), load_frame_scores(a.frame_v5))
    w, _ = cell_weights(ids, cells, Counter(cells.values()))
    r = analyse(gold_table(ids, pred, gold, w, replies), corpus_rows(a.corpus), n_boot=a.n_boot, seed=a.seed)
    unadj = None
    if a.unadjusted:
        cj = json.loads(Path(a.unadjusted).read_text(encoding="utf-8"))["contrasts"]
        unadj = {tuple(k.split("|")): v[1] for k, v in cj.items()}
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot, "corpus": a.corpus,
            "extract": "/".join(sorted({(x["model"], x["items_version"]) for x in cache.values()})[0])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "length_adjusted.json").write_text(json.dumps(_jsonable({"meta": meta, **r}), ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (out / "length_adjusted.md").write_text(render(r, meta, unadj), encoding="utf-8")
    print(f"报告 {out / 'length_adjusted.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
