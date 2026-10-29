"""新主线主假设的自助法 p 值 + 按新家族做 Holm 校正（全部用医师金标；原权重与 raking 权重各一套）。零模型调用。

    python tools/primary_tests.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl \
        --out reports/primary_tests

主假设（2026-10-19 按新主线「医师金标会推翻自动评估的结论」确定；⚠️ 不是事先注册的，论文方法须如实写）：
  PA「模型比较」：旗舰（设计组）− 本地（组），「任一」。
  PB「心理健康微调」：心理微调组 − 对照基座组的「任一」、d1、d4；三对「微调 − 对照」的「任一」
      （cpsycoun − internlm2-base、mindchat − qwen15-4b-base、soulchat − qwen3-8b-base）。
子组键、失败定义、子组比率估计全部复用 tools/gold_prevalence；权重复用 tools/gold_prevalence_raked.all_weights
（它同时返回原抽样格权重 orig = judge_correction.cell_weights、raked、截断 trim）—— 两个模块只 import 不改。
自助：人设聚类重抽 B 次（默认 2000），每次重抽后按同一流程重算抽样格权重 → raking（含退路），三套权重共用同一批重抽；
保存每个比较差值的自助分布（json 里 boot_dist）。
p 值：自助百分位反转的双侧 p = min(1, 2·min(P(d* ≤ 0), P(d* ≥ 0)))，其中 P(d* ≤ 0) = (1 + #{d* ≤ 0}) / (B' + 1)
（B' = 两组都有样本的重抽数；+1 修正避免 p = 0，最小可达 p = 2/(B'+1)）。
对照：旧方法（tools/multiplicity.p_from_ci：近 0 一侧半宽当 1.96 SE 的正态反推），CI 偏斜 / 截断时不准。
退化（不进检验）：点估计任一组金标有效段数为 0，或自助差值分布全为 0，或有效重抽 < 50。
校正：PA、PB 各自 Holm；另报 PA + PB 合并一族的 Holm（严格版）。每套权重各一套。"""
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

from tools.gold_prevalence import ANY, OUTCOMES, turn_meta  # noqa: E402
from tools.gold_prevalence_raked import _design, _q, _rates, all_weights  # noqa: E402
from tools.judge_correction import load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import persona_of  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402
from tools.multiplicity import holm, p_from_ci  # noqa: E402

SCHEMES = ("orig", "raked", "trim")
SCHEME_LABEL = {"orig": "原权重（抽样格）", "raked": "raking", "trim": "截断 raking（稳健性）"}
REPORT_SCHEMES = ("orig", "raked")          # 校正与主表只用这两套；trim 只在附表
PA, PB = "PA 模型比较", "PB 心理健康微调"
FAMILIES = (PA, PB)
MERGED = "PA+PB 合并"
FLAG, LOCAL = ("设计组", "旗舰"), ("组", "本地")
PSY, BASE = ("设计组", "心理微调"), ("设计组", "对照基座")
PAIRS = (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))
SHORT = {"cpsycoun": "CPsyCounX", "internlm2-base": "InternLM2", "mindchat": "MindChat", "qwen15-4b-base": "Qwen1.5",
         "soulchat": "SoulChat", "qwen3-8b-base": "Qwen3-8B"}

# (id, 家族, 组 a, 组 b, 结局, 标签)
HYPOTHESES = (
    ("PA1", PA, FLAG, LOCAL, ANY, "旗舰 − 本地，任一"),
    ("PB1", PB, PSY, BASE, ANY, "心理微调 − 对照基座，任一"),
    ("PB2", PB, PSY, BASE, "d1", "心理微调 − 对照基座，d1 有害内容"),
    ("PB3", PB, PSY, BASE, "d4", "心理微调 − 对照基座，d4 谄媚附和"),
) + tuple((f"PB{4 + k}", PB, ("模型", a), ("模型", b), ANY, f"{SHORT[a]} − {SHORT[b]}，任一")
          for k, (a, b) in enumerate(PAIRS))
MIN_BOOT = 50


# ---------------------------------------------------------------- 统计
def boot_p(dist) -> dict:
    """自助百分位反转双侧 p（+1 修正）。dist 里的 None / nan 视为该次重抽某组无样本，剔除。"""
    d = np.asarray([x for x in dist if x is not None and not np.isnan(x)], dtype=float)
    B = len(d)
    if B == 0:
        return {"p": None, "B": 0, "le0": None, "ge0": None}
    le = (1 + int(np.sum(d <= 0))) / (B + 1)
    ge = (1 + int(np.sum(d >= 0))) / (B + 1)
    return {"p": min(1.0, 2 * min(le, ge)), "B": B, "le0": round(le, 6), "ge0": round(ge, 6)}


def degenerate(n_a: int, n_b: int, dist) -> str | None:
    """退化原因；None = 可检验。"""
    if n_a == 0 or n_b == 0:
        return "某组金标有效段数为 0"
    d = np.asarray([x for x in dist if x is not None and not np.isnan(x)], dtype=float)
    if len(d) < MIN_BOOT:
        return f"有效重抽 < {MIN_BOOT}"
    if np.all(d == 0):
        return "自助差值分布全为 0"
    return None


def family_holm(rows: list[dict], key: str, family_of) -> dict:
    """rows：[{id, p, degenerate}]；family_of(row) → 家族名。→ {(家族, id): 校正 p}，退化的不进。"""
    fam = defaultdict(list)
    for r in rows:
        if r["degenerate"] is None and r[key] is not None:
            fam[family_of(r)].append(r)
    out = {}
    for f, rs in fam.items():
        for r, a in zip(rs, holm([r[key] for r in rs])):
            out[(f, r["id"])] = a
    return out


# ---------------------------------------------------------------- 主流程
def analyse(ids, gold, cells, meta, frame_ids, *, n_boot=2000, seed=0, log=None) -> dict:
    pop_cells = Counter(cells[i] for i in frame_ids)
    pop_model = Counter(meta[i]["model"] for i in frame_ids)
    pop_ab = Counter((meta[i]["arm"], meta[i]["band"]) for i in frame_ids)
    model_of = {i: meta[i]["model"] for i in ids}
    ab_of = {i: (meta[i]["arm"], meta[i]["band"]) for i in ids}
    W, _ = all_weights(ids, cells, model_of, ab_of, pop_cells, pop_model, pop_ab)
    groups, G, F = _design(ids, gold, meta)
    gidx = {g: k for k, g in enumerate(groups)}
    oidx = {o: j for j, o in enumerate(OUTCOMES)}
    for h in HYPOTHESES:
        for g in (h[2], h[3]):
            if g not in gidx:
                raise SystemExit(f"子组 {g} 在金标里不存在")
    point = {s: _rates(G, F, W[s]) for s in SCHEMES}

    rng = random.Random(seed)
    by_p = defaultdict(list)
    for k, i in enumerate(ids):
        by_p[persona_of(i)].append(k)
    pn = sorted(by_p)
    dist = {s: {h[0]: [] for h in HYPOTHESES} for s in SCHEMES}
    fallback = Counter()
    t0 = time.time()
    for b in range(n_boot):
        bi = [k for p in (rng.choice(pn) for _ in pn) for k in by_p[p]]
        uid = [f"{ids[k]}@{j}" for j, k in enumerate(bi)]
        bW, binfo = all_weights(uid, {u: cells[ids[k]] for u, k in zip(uid, bi)},
                                {u: model_of[ids[k]] for u, k in zip(uid, bi)},
                                {u: ab_of[ids[k]] for u, k in zip(uid, bi)}, pop_cells, pop_model, pop_ab)
        fallback["raked_nonconv"] += not binfo["raked"]["converged"]
        fallback["raked_layer"] += binfo["raked"]["cell_level"] == "layer"
        fallback["trim_nonconv"] += not binfo["trim"]["converged"]
        Gb, Fb = G[bi], F[bi]
        for s in SCHEMES:
            R = _rates(Gb, Fb, bW[s])
            for hid, _, a, bb, o, _ in HYPOTHESES:
                v = R[gidx[a], oidx[o]] - R[gidx[bb], oidx[o]]
                dist[s][hid].append(None if np.isnan(v) else float(v))
        if log and (b + 1) % 200 == 0:
            log(f"  自助 {b + 1}/{n_boot}（{time.time() - t0:.0f}s）")

    def counts(g, o):
        col = F[:, oidx[o]]
        ok = (G[:, gidx[g]] > 0) & ~np.isnan(col)
        return int(ok.sum()), int(col[ok].sum())

    res = {}
    for s in SCHEMES:
        rows = []
        for hid, fam, a, bb, o, lab in HYPOTHESES:
            na, fa = counts(a, o)
            nb, fb = counts(bb, o)
            pa, pb = point[s][gidx[a], oidx[o]], point[s][gidx[bb], oidx[o]]
            est = None if np.isnan(pa) or np.isnan(pb) else float(pa - pb)
            ds = dist[s][hid]
            ci = _q([np.nan if x is None else x for x in ds])
            deg = degenerate(na, nb, ds)
            bp = boot_p(ds)
            p_old = None if (deg or est is None or ci is None) else p_from_ci(est, ci)
            rows.append({"id": hid, "family": fam, "label": lab, "outcome": o, "a": list(a), "b": list(bb),
                         "n_a": na, "fail_a": fa, "n_b": nb, "fail_b": fb,
                         "rate_a": None if np.isnan(pa) else round(float(pa), 4),
                         "rate_b": None if np.isnan(pb) else round(float(pb), 4),
                         "est": None if est is None else round(est, 4), "ci": ci, "degenerate": deg,
                         "p_boot": None if deg else bp["p"], "B_valid": bp["B"], "P_le0": bp["le0"],
                         "P_ge0": bp["ge0"], "p_old": p_old})
        hf = family_holm(rows, "p_boot", lambda r: r["family"])
        hm = family_holm(rows, "p_boot", lambda r: MERGED)
        ho = family_holm(rows, "p_old", lambda r: r["family"])
        hom = family_holm(rows, "p_old", lambda r: MERGED)
        for r in rows:
            r["holm_family"] = hf.get((r["family"], r["id"]))
            r["holm_merged"] = hm.get((MERGED, r["id"]))
            r["holm_family_old"] = ho.get((r["family"], r["id"]))
            r["holm_merged_old"] = hom.get((MERGED, r["id"]))
        res[s] = rows
    return {"results": res, "boot_dist": dist, "boot_info": dict(fallback),
            "n_personas": len(pn), "n_gold": len(ids)}


# ---------------------------------------------------------------- 报告
def _pp(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def _p(x):
    if x is None:
        return "—"
    return f"{x:.4f}" if x >= 0.001 else f"{x:.1e}"


def _sig(x):
    return "" if x is None else (" ✅" if x < 0.05 else " ✗")


def render(r, info) -> str:
    R = r["results"]
    L = ["# 新主线主假设：自助法 p 值与多重比较校正（医师金标）", "",
         f"- 生成 {info['generated_at']}；金标 {r['n_gold']} 段（医师共识，gold_dev + gold_test；不含 *_key）；"
         f"人设聚类自助 {info['n_boot']} 次（{r['n_personas']} 个人设），seed {info['seed']}。",
         "- ⚠️ **这些主假设与家族是 2026-10-19 按新主线（医师金标会推翻自动评估的结论）确定的，不是事先注册的**；"
         "数据（金标、全语料）在确定之前已经看过。论文方法须如实说明。",
         "- 权重：原 = 抽样格反概率权重（tools/gold_prevalence）；raking = 按抽样格 / 模型 / 臂 × 风险段校准"
         "（tools/gold_prevalence_raked.all_weights，自助时每次重抽后重做 raking，含退路）。两套权重共用同一批重抽。",
         "- p 值：自助百分位反转，p = min(1, 2·min(P(d*≤0), P(d*≥0)))，P = (1 + 计数)/(B′ + 1)，"
         f"B′ = 两组都有样本的重抽数；最小可达 p ≈ {2 / (info['n_boot'] + 1):.4f}。"
         "对照列「旧 p」= tools/multiplicity 的 CI 正态反推（CI 偏斜 / 截断时不准）。",
         "- 校正：PA、PB 各自 Holm；「合并」= PA + PB 一族 Holm（严格版）。退化（某组无样本 / 自助差值全 0 / 有效重抽 < 50）不进检验。",
         f"- 自助退路次数（{info['n_boot']} 次中）：raking 退到 v6 层边际 {r['boot_info'].get('raked_layer', 0)}、"
         f"raking 仍未收敛 {r['boot_info'].get('raked_nonconv', 0)}、截断版未收敛 {r['boot_info'].get('trim_nonconv', 0)}。",
         "- 差值单位：百分点；CI = 自助 2.5 / 97.5 百分位。", ""]
    base = R["orig"]
    L += ["## 金标段数与失败段数（未加权；同一结局下两组）", "",
          "| 编号 | 家族 | 比较 | 组 a 段数 / 失败 | 组 b 段数 / 失败 |", "|---|---|---|---|---|"]
    for x in base:
        L.append(f"| {x['id']} | {x['family']} | {x['label']} | {x['n_a']} / {x['fail_a']} | {x['n_b']} / {x['fail_b']} |")
    for s in REPORT_SCHEMES:
        L += ["", f"## {SCHEME_LABEL[s]}", "",
              "| 编号 | 比较 | 组 a / 组 b 比例 | 差 [95% CI] | 自助 p | 旧 p | 族内 Holm | 合并 Holm | 旧 p 族内 / 合并 Holm |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in R[s]:
            if x["degenerate"]:
                L.append(f"| {x['id']} | {x['label']} | {_pp(x['rate_a'])} / {_pp(x['rate_b'])} | {_pp(x['est'])} | "
                         f"退化，不进检验（{x['degenerate']}） | | | | |")
                continue
            ci = "" if not x["ci"] else f" [{_pp(x['ci'][0])}, {_pp(x['ci'][1])}]"
            L.append(f"| {x['id']} | {x['label']} | {100 * x['rate_a']:.1f} / {100 * x['rate_b']:.1f} | "
                     f"{_pp(x['est'])}{ci} | {_p(x['p_boot'])} | {_p(x['p_old'])} | "
                     f"{_p(x['holm_family'])}{_sig(x['holm_family'])} | {_p(x['holm_merged'])}{_sig(x['holm_merged'])} | "
                     f"{_p(x['holm_family_old'])} / {_p(x['holm_merged_old'])} |")
    L += ["", "## 结论汇总（自助 p，Holm 后 < 0.05）", "",
          "| 编号 | 比较 | 原权重 族内 | 原权重 合并 | raking 族内 | raking 合并 | 两套都成立（族内） |", "|---|---|---|---|---|---|---|"]
    for k, x in enumerate(base):
        y = R["raked"][k]

        def ok(z, key):
            return None if z["degenerate"] else (z[key] is not None and z[key] < 0.05)

        cells = [ok(x, "holm_family"), ok(x, "holm_merged"), ok(y, "holm_family"), ok(y, "holm_merged")]
        txt = ["退化" if c is None else ("成立" if c else "不成立") for c in cells]
        both = "—" if None in cells[::2] else ("🟢 是" if cells[0] and cells[2] else ("🟡 一套" if cells[0] or cells[2] else "否"))
        L.append(f"| {x['id']} | {x['label']} | " + " | ".join(txt) + f" | {both} |")
    L += ["", f"## 附：{SCHEME_LABEL['trim']}（不参与结论）", "",
          "| 编号 | 比较 | 差 [95% CI] | 自助 p | 族内 Holm | 合并 Holm |", "|---|---|---|---|---|---|"]
    for x in R["trim"]:
        if x["degenerate"]:
            L.append(f"| {x['id']} | {x['label']} | {_pp(x['est'])} | 退化（{x['degenerate']}） | | |")
            continue
        ci = "" if not x["ci"] else f" [{_pp(x['ci'][0])}, {_pp(x['ci'][1])}]"
        L.append(f"| {x['id']} | {x['label']} | {_pp(x['est'])}{ci} | {_p(x['p_boot'])} | "
                 f"{_p(x['holm_family'])} | {_p(x['holm_merged'])} |")
    L += ["", "> 每个比较差值的完整自助分布在 `primary_tests.json` 的 `boot_dist`（None = 该次重抽某组无样本）。"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="新主线主假设：自助法 p 值 + Holm 校正（医师金标）")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--dialogues", default=str(REPO / "runs/v2/dialogues_both_arms.jsonl"))
    ap.add_argument("--frame-v6", default=str(REPO / "runs/v2/scored_v6.jsonl"))
    ap.add_argument("--frame-v5", default=str(REPO / "runs/v2/scored_v5.jsonl"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if any(p.endswith("_key.jsonl") for p in a.gold):
        raise SystemExit("*_key.jsonl 不是医师判定，不能当金标")
    rows = [g for p in a.gold for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cells = sampling_cells(load_frame_scores(a.frame_v6), load_frame_scores(a.frame_v5))
    meta = turn_meta(a.dialogues)
    miss = [i for i in ids if i not in cells or i not in meta]
    if miss:
        raise SystemExit(f"{len(miss)} 段金标不在抽样框 / 对话里（如 {miss[:3]}）")
    frame_ids = [i for i in cells if i in meta]
    t0 = time.time()
    r = analyse(ids, gold, cells, meta, frame_ids, n_boot=a.n_boot, seed=a.seed, log=print)
    info = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_boot": a.n_boot, "seed": a.seed,
            "seconds": round(time.time() - t0, 1),
            "hypotheses_fixed_on": "2026-10-19（按新主线确定，非事先注册）"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "primary_tests.json").write_text(json.dumps({"meta": info, **r}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    (out / "primary_tests.md").write_text(render(r, info), encoding="utf-8")
    print(f"报告 {out / 'primary_tests.md'}（{info['seconds']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
