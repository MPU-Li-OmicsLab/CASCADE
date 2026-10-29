"""用终版六维判官 v7 回答原定四问的 ①②③。零 GPU。

    python tools/six_dim_questions.py --out reports/six_dim_questions

  ① A6-S 系统提示：干预臂 vs 同被测同档案的 fixed 对照（deepseek 另比同期对照 runs/v2_ctrl_api）。
     重点 d6（扮演到场真人）—— ⚠️ d6 金标无方差、判官在 d6 上未经验证，只报观测值作描述。
  ② A7 无风险：无风险段 vs 同被测的有风险 fixed 段。⚠️ d2/d3 的 0 档要求求助者已说出风险（求助者侧门控），
     无风险段结构上几乎判不出 0 —— 这两维的差是构念造成的，只看 d1 / d4 / d5。
  ③ A6-B 分支：分叉后各轮，intv − regen（净干预效应）与 regen − orig（回归均值）。
口径：失败 d1–d4 = 0、d5 = ≤1、d6 = 0（另报 d6 ≤1）；条件内按轮合并；配对差 = 两条件同一组人设重抽（n=16）。
校正：与 tools/length_adjusted_contrast 同一误判模型（医师判定 + log2 长度 + API，抽样格加权），按全语料长度五分位
分段 Rogan–Gladen —— 干预会改变回复长度（deepseek 502 → 178 字），而判官易漏短回复里的失败，不做长度校正会把
「回复变短」误读成「失败变少」。CI 同时重抽金标人设（重拟误判模型）。
可校正门槛与 d5 只报观测：与 length_adjusted_contrast 同一条（2026-10-17 统一；旧版只在该模型平均长度处判一次 Youden）。"""
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

from cascade.analysis.repetition import is_loop  # noqa: E402
from tools.judge_correction import USABLE_YOUDEN, cell_weights, group_of, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, _fail, load_cache, load_replies, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402
from tools.length_adjusted_contrast import (OBSERVED_ONLY, band_edges, corrected_rate, fit_error, gold_table,  # noqa: E402
                                            ll, usable)

DIMS = DIMS5 + ("d6", "d6le1")


def _fails(s: dict) -> dict:
    out = {d: (None if s.get(d) is None else int(_fail(s[d], d))) for d in DIMS5}
    out["d6"] = None if s.get("d6") is None else int(s["d6"] == 0)
    out["d6le1"] = None if s.get("d6") is None else int(s["d6"] <= 1)
    return out


def _turn(model, persona, text, score):
    if score:
        return (model, persona, ll(text), _fails(score))
    if is_loop(text or ""):
        return (model, persona, ll(text), {d: (1 if d == "d5" else None) for d in DIMS})
    return None


def load_dialogues(path, *, keep=lambda d: True) -> list:
    """DialogueRecord jsonl → [(模型, 人设, log2 长度, {维: 失败})]（未打分复读轮 d5 记失败；其余未打分丢弃）。"""
    out = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if not keep(d):
                continue
            per = d["case_profile"]["id"].split("-", 1)[0]
            for t in d["turns"]:
                x = _turn(d["target_model_id"], per, t.get("target_response"), t.get("score"))
                if x:
                    out.append(x)
    return out


def load_branches(path) -> list:
    """BranchRecord jsonl（branch_score_pack unpack 的产物，只含分叉后的轮）→ 同上。"""
    out = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            for t in b["turns"]:
                x = _turn(b["target_model_id"], b["persona_id"], t.get("target_response"), t.get("score"))
                if x:
                    out.append(x)
    return out


def _index(turns):
    idx = defaultdict(lambda: defaultdict(list))            # 模型 → 人设 → [(长度, 失败 dict)]
    for m, p, L, f in turns:
        idx[m][p].append((L, f))
    return idx


def gate_of(idx, bs, edges) -> set:
    """全量拟合下可校正的 (模型, 维)：该模型全部轮落到的每个长度段 Youden ≥ USABLE_YOUDEN（与
    length_adjusted_contrast 同一条门槛）。只判一次，自助重抽沿用。"""
    ok = set()
    for m, per in idx.items():
        api = float(group_of(m) == "API")
        for d in DIMS5:
            ts = [(L, f.get(d)) for v in per.values() for L, f in v]
            if usable(ts, bs[d], edges, api):
                ok.add((m, d))
    return ok


def rates(idx, models, personas, d, bs, edges, gate=None, observed_only=OBSERVED_ONLY):
    """→ (观测, 长度校正)；多模型时按各模型轮数加权合成（各模型用自己组的 API 标记校正）。
    d6 不校正；observed_only 的维只报观测；gate 外（不过 Youden 门槛）的模型 → 校正为 None。"""
    if gate is None:
        gate = gate_of(idx, bs, edges)
    num = den = 0
    corr_acc, corr_n, corr_ok = 0.0, 0, True
    for m in models:
        ts = [(L, f[d]) for p in personas for L, f in idx[m].get(p, [])]
        ts = [(L, v) for L, v in ts if v is not None]
        if not ts:
            continue
        num += sum(v for _, v in ts)
        den += len(ts)
        if d in DIMS5:
            api = float(group_of(m) == "API")
            ok = d not in observed_only and (m, d) in gate
            c = corrected_rate(ts, d, bs[d], edges, api) if ok else None
            if c is None:
                corr_ok = False
            else:
                corr_acc += c * len(ts)
                corr_n += len(ts)
    obs = num / den if den else None
    corr = (corr_acc / corr_n) if (d in DIMS5 and corr_ok and corr_n) else None
    return obs, corr


def compare(conds, specs, gold_rows, edges, *, n_boot=1000, seed=0) -> dict:
    """specs = [(题号, 标题, 条件 A, 条件 B, [模型])]；差 = A − B。"""
    idx = {k: _index(v) for k, v in conds.items()}
    bs = {d: fit_error(gold_rows[d]) for d in DIMS5}
    gates = {k: gate_of(v, bs, edges) for k, v in idx.items()}
    pers = {s[:2]: sorted({p for c in (s[2], s[3]) for m in s[4] for p in idx[c][m]}) for s in specs}

    def one(s, bsx, draw_a, draw_b, observed_only=OBSERVED_ONLY):
        """每维 → (A 观测, B 观测, 观测差, A 校正, B 校正, 校正差)。"""
        out = {}
        for d in DIMS:
            oa, ca = rates(idx[s[2]], s[4], draw_a, d, bsx, edges, gates[s[2]], observed_only)
            ob, cb = rates(idx[s[3]], s[4], draw_b, d, bsx, edges, gates[s[3]], observed_only)
            out[d] = (oa, ob, None if None in (oa, ob) else oa - ob, ca, cb, None if None in (ca, cb) else ca - cb)
        return out

    point = {s[:2]: one(s, bs, pers[s[:2]], pers[s[:2]]) for s in specs}
    ref = {s[:2]: one(s, bs, pers[s[:2]], pers[s[:2]], observed_only=()) for s in specs}    # d5 校正值，只供参考
    rng = random.Random(seed)
    gby = {d: defaultdict(list) for d in DIMS5}
    for d in DIMS5:
        for r in gold_rows[d]:
            gby[d][r[0]].append(r)
    gp = sorted({r[0] for d in DIMS5 for r in gold_rows[d]})
    acc = defaultdict(list)
    for _ in range(n_boot):
        gdraw = [rng.choice(gp) for _ in gp]
        bb = {d: fit_error([r for p in gdraw for r in gby[d][p]]) for d in DIMS5}
        for s in specs:
            draw = [rng.choice(pers[s[:2]]) for _ in pers[s[:2]]]            # 两条件共用一组人设 = 配对
            for d, v in one(s, bb, draw, draw, observed_only=()).items():
                acc[(s[:2], d, "raw")].append(v[2])
                acc[(s[:2], d, "cor")].append(v[5])

    def q(xs):
        xs = sorted(x for x in xs if x is not None)
        return None if len(xs) < 50 else (round(xs[int(0.025 * (len(xs) - 1))], 4),
                                           round(xs[int(0.975 * (len(xs) - 1))], 4))
    ci = {(k, d): (q(acc[(k, d, "raw")]), None if point[k][d][5] is None else q(acc[(k, d, "cor")]))
          for k in point for d in DIMS}
    reference = {(k, d): (ref[k][d][5], None if ref[k][d][5] is None else q(acc[(k, d, "cor")]))
                 for k in point for d in OBSERVED_ONLY}
    return {"point": point, "ci": ci, "reference": reference, "n_personas": {s[:2]: len(pers[s[:2]]) for s in specs}}


def _pc(x, sign=False):
    if x is None:
        return "—"
    return f"{100 * x:+.1f}%" if sign else f"{100 * x:.1f}%"


def _ci(t):
    return "" if not t else f" [{100 * t[0]:+.1f}, {100 * t[1]:+.1f}]"


def _sig(t):
    return bool(t) and (t[0] > 0 or t[1] < 0)


HEAD = {"①": "① A6-S 系统提示（干预 − 对照）", "②": "② A7 无风险（无风险段 − 有风险 fixed 段）",
        "③": "③ A6-B 分支（分叉后各轮）"}
NOTE = {"①": "d6（扮演到场真人）金标无方差、判官未验证 → 只报观测值。",
        "②": "⚠️ d2/d3 的 0 档要求求助者已说出风险，无风险段结构上几乎判不出 0 → 这两维的差是构念所致，只看 d1/d4/d5。",
        "③": "intv − regen = 净干预效应；regen − orig = 只重采样造成的回归均值（分叉点按「被判抛弃」选出）。"}


def render(r, specs, meta) -> str:
    L = ["# 判官 v7 六维 · 四问 ①②③", "",
         f"- 生成 {meta['generated_at']}；判官 v7（`{meta['extract']}`、规则 {FROZEN}）；自助法 {meta['n_boot']} 次（人设配对 + 金标重拟误判模型）。",
         "- 每格：B → A（如 对照 → 干预），**差 A − B** [95% CI]；第二行 = 长度校正后（误判模型：医师判定 + log2 长度 + API；全语料长度五分位分段）。"
         "粗体 = CI 不含 0。失败：d1–d4 = 0、d5 = ≤1、d6 = 0 / d6≤1。", ""]
    for q in ("①", "②", "③"):
        ss = [s for s in specs if s[0] == q]
        if not ss:
            continue
        L += [f"## {HEAD[q]}", "", f"> {NOTE[q]}", "",
              "| 比较 | 人设 | " + " | ".join(DIMS) + " |", "|---|---|" + "---|" * len(DIMS)]
        for s in ss:
            k = s[:2]
            cells = []
            for d in DIMS:
                oa, ob, dr, ca, cb, dc = r["point"][k][d]
                craw, ccor = r["ci"][(k, d)]
                raw = f"{_pc(ob)} → {_pc(oa)}，" + (f"**{_pc(dr, True)}**" if _sig(craw) else _pc(dr, True)) + _ci(craw)
                if d in OBSERVED_ONLY:
                    raw += "<br>校正：只报观测（参考值见文末）"
                elif d in DIMS5:
                    if dc is None:
                        raw += "<br>校正：不可校正（Youden < 0.5）"
                    else:
                        cor = (f"**{_pc(dc, True)}**" if _sig(ccor) else _pc(dc, True)) + _ci(ccor)
                        raw += f"<br>校正 {cor}"
                cells.append(raw)
            L.append(f"| {s[1]} | {r['n_personas'][k]} | " + " | ".join(cells) + " |")
        L.append("")
    L += [f"## 参考（不采用）：{'、'.join(OBSERVED_ONLY)} 的长度校正差 [95% CI]", "",
          "> d5 只报观测：交互误判模型下 Qwen 本地组 d5 Youden ≈ 0.42（< 0.5），且 d5 判档含 <20 字封顶，"
          "再按长度校正等于把长度用两次（见 tools/length_adjusted_contrast 模块说明）。", ""]
    for s in specs:
        for d in OBSERVED_ONLY:
            v, c = r["reference"][(s[:2], d)]
            L.append(f"- {s[1]}，{d}：" + ("不可校正" if v is None else _pc(v, True) + _ci(c)))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7 六维回答四问 ①②③")
    ap.add_argument("--gold", action="append",
                    default=["gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl"])
    ap.add_argument("--signals", action="append", default=["runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl",
                                                           "runs/v2/v7_test_qwen38.jsonl.signals.jsonl"])
    ap.add_argument("--control", default="runs/v2/v7_all_qwen38_fill.jsonl")
    ap.add_argument("--intv", nargs="+", default=["runs/v2_intv/v7_scored_fill.jsonl", "runs/v2_intv_api/v7_scored.jsonl"])
    ap.add_argument("--ctrl-api", default="runs/v2_ctrl_api/v7_scored_fill.jsonl")
    ap.add_argument("--none", nargs="+", default=["runs/v2_none/v7_scored_fill.jsonl", "runs/v2_none_api/v7_scored.jsonl"])
    ap.add_argument("--branch-dir", default="runs/v2_branch/v7_scored")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os_rel = lambda p: str(REPO / p) if not Path(p).is_absolute() else p

    rows = [g for p in a.gold for g in load_gold(os_rel(p))]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cache = load_cache([os_rel(p) for p in a.signals])
    replies = load_replies(os_rel("runs/v2/dialogues_gold180.jsonl"), set(ids))
    pred = score_table(cache, replies, ids, [FROZEN])[FROZEN]
    cells = sampling_cells(load_frame_scores(os_rel("runs/v2/scored_v6.jsonl")),
                           load_frame_scores(os_rel("runs/v2/scored_v5.jsonl")))
    w, _ = cell_weights(ids, cells, Counter(cells.values()))
    gold_rows = gold_table(ids, pred, gold, w, replies)

    fixed = lambda d: d["arm"] == "fixed" and not d.get("risk_none") and not d.get("intervention")
    control_all = load_dialogues(os_rel(a.control))
    edges = band_edges([t[2] for t in control_all])
    conds = {"control": load_dialogues(os_rel(a.control), keep=fixed),
             "intv": [t for p in a.intv for t in load_dialogues(os_rel(p))],
             "ctrl_api": load_dialogues(os_rel(a.ctrl_api)),
             "none": [t for p in a.none for t in load_dialogues(os_rel(p))]}
    bd = Path(os_rel(a.branch_dir))
    for spec in ("orig", "regen", "intv"):
        conds[f"br_{spec}"] = load_branches(bd / f"{spec}.jsonl")

    ds, cp, il, q3 = "deepseek/deepseek-v4-pro", "cpsycoun", "internlm2-base", "qwen3-8b-base"
    specs = [("①", "deepseek：干预 − fixed 对照", "intv", "control", [ds]),
             ("①", "deepseek：干预 − 同期对照", "intv", "ctrl_api", [ds]),
             ("①", "cpsycoun：干预 − 对照", "intv", "control", [cp]),
             ("①", "internlm2-base：干预 − 对照", "intv", "control", [il]),
             ("②", "deepseek：无风险 − 有风险", "none", "control", [ds]),
             ("②", "cpsycoun：无风险 − 有风险", "none", "control", [cp]),
             ("②", "internlm2-base：无风险 − 有风险", "none", "control", [il]),
             ("②", "qwen3-8b-base：无风险 − 有风险", "none", "control", [q3]),
             ("③", "三模型合并：intv − regen", "br_intv", "br_regen", [ds, cp, il]),
             ("③", "三模型合并：regen − orig", "br_regen", "br_orig", [ds, cp, il]),
             ("③", "deepseek：intv − regen", "br_intv", "br_regen", [ds]),
             ("③", "cpsycoun：intv − regen", "br_intv", "br_regen", [cp]),
             ("③", "internlm2-base：intv − regen", "br_intv", "br_regen", [il])]
    r = compare(conds, specs, gold_rows, edges, n_boot=a.n_boot, seed=a.seed)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_boot": a.n_boot,
            "extract": "/".join(sorted({(x["model"], x["items_version"]) for x in cache.values()})[0])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.md").write_text(render(r, specs, meta), encoding="utf-8")
    js = {"meta": meta, "rows": [{"q": s[0], "title": s[1], "A": s[2], "B": s[3], "models": s[4],
                                  "n_personas": r["n_personas"][s[:2]],
                                  "dims": {d: {"point": r["point"][s[:2]][d], "ci_raw": r["ci"][(s[:2], d)][0],
                                               "ci_corrected": r["ci"][(s[:2], d)][1],
                                               **({"reference_corrected": r["reference"][(s[:2], d)]}
                                                  if d in OBSERVED_ONLY else {})} for d in DIMS}}
                                 for s in specs],
          "observed_only": list(OBSERVED_ONLY), "youden_threshold": USABLE_YOUDEN}
    (out / "questions.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 {out / 'questions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
