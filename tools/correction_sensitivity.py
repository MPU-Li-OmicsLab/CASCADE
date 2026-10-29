"""误判校正参数「样本内」问题的敏感性分析：误判模型分别只用 test 67 / 只用 dev 113 / 全部 180 段拟合。零模型调用。

    python tools/correction_sensitivity.py --out reports/correction_sensitivity              # 两家判官（默认）
    python tools/correction_sensitivity.py --judges qwen --n-boot 200 --out <dir>            # 只跑 Qwen、快速

背景：判官 v7（抽取提示词 q4 + 规则 r6）是在医师金标 dev 113 段上开发的，test 67 段事先锁定、只跑过一次。
现有的长度误判校正（tools/length_adjusted_contrast、tools/judge_dose）用全部 180 段拟合
logit P(判官判失败) = b0 + b1·医师判失败 + b2·log2(字数) + b3·API → dev 段参与了规则开发，估出的灵敏度 / 特异度
偏乐观，「每个长度段 Youden ≥ 0.5」的可校正门槛也随之放松。本工具：
  1. 三种拟合样本（all / test / dev）下每维、每组（API / 本地）在 31 / 127 / 511 字处的 se / sp / Youden，及医师判失败段数；
  2. 用三种误判模型分别重算 C2（judge_dose：本地模型 fixed 臂 高 − 低风险段）与 C5（length_adjusted_contrast：
     心理微调配对差）的点估计与 CI —— 直接调用两工具的 analyse()，只换传入的金标行；
     抽样格权重（cell_weights）按各自样本重算，使子样本仍代表全语料；门槛按各自样本的全量拟合判一次（analyse 内部行为）；
  3. 交叉验证列不适用（理由写进报告「交叉验证」节）：judge_cv 的折内只选规则集，q4 提示词与 r6 规则都在 dev 上定，
     r6 的折外预测 = 样本内预测 → 不消除 dev 开发带来的乐观；而折内选出的规则集又不是语料实际用的 r6。
⚠️ test-only 只有 67 段，部分维 × 组的医师失败段个位数 → 系数近乎不可识别（报告逐格列出段数）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import judge_dose as JD                  # noqa: E402
from tools import length_adjusted_contrast as LA    # noqa: E402
from tools.judge_correction import USABLE_YOUDEN, cell_weights, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import FROZEN, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

SAMPLES = ("all", "test", "dev")
SAMPLE_NAME = {"all": "全部 180（现状）", "test": "只用 test 67", "dev": "只用 dev 113"}
LENS = (5.0, 7.0, 9.0)                              # log2(字数+1) → 31 / 127 / 511 字
GOLDS = ("gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl")
JUDGES = {
    "qwen": {"name": "Qwen3.8-27B 判官（本地）",
             "signals": ("runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl", "runs/v2/v7_test_qwen38.jsonl.signals.jsonl"),
             "corpus": "runs/v2/v7_all_qwen38_fill.jsonl"},
    "deepseek": {"name": "DeepSeek-v4-pro 判官（A4 跨家族）",
                 "signals": ("runs/v2/a4_dsnt_gold.jsonl.signals.jsonl",),
                 "corpus": "runs/v2/a4_dsnt_pairs.jsonl"},
}
# 报告里重点看的格
C2_OUTCOMES = (JD.ANY, "d1", "d2", "d3", "d4")
C5_DIMS = ("d1", "d2", "d4")


def _rel(p) -> str:
    return str(p) if Path(p).is_absolute() else str(REPO / p)


# ---------------------------------------------------------------- 金标拆分
def sample_ids(ids, split: dict, sample: str) -> list:
    """all → 全部；test / dev → 只留该 split（split 缺失的行只进 all）。"""
    if sample == "all":
        return list(ids)
    if sample not in ("test", "dev"):
        raise ValueError(f"未知拟合样本 {sample}")
    return [i for i in ids if split.get(i) == sample]


def gold_tables(sub, inp) -> tuple[dict, dict, dict]:
    """子样本 → (length_adjusted 格式金标行, judge_dose 格式金标行, 权重诊断)。抽样格权重按子样本重算。"""
    w, diag = cell_weights(sub, inp["cells"], inp["pop"])
    la = LA.gold_table(sub, inp["pred"], inp["gold"], w, inp["replies"])
    dose = JD.gold_rows(sub, inp["pred"], inp["gold"], w, inp["replies"])
    return la, dose, {"n": len(sub), "n_personas": len({persona_of(i) for i in sub}),
                      "uncovered_turns": diag["uncovered_turns"], "pop_turns": diag["pop_turns"]}


def error_table(rows_by_outcome: dict, lens=LENS) -> dict:
    """每个结局 → 系数、各组医师判失败段数（未加权）、各组在典型长度处的 se / sp / Youden。"""
    out = {}
    for o, rows in rows_by_outcome.items():
        b = LA.fit_error(rows)
        n_fail = {g: sum(1 for r in rows if r[2] == 1 and (r[5] == 1.0) == (g == "API")) for g in ("API", "本地")}
        n_all = {g: sum(1 for r in rows if (r[5] == 1.0) == (g == "API")) for g in ("API", "本地")}
        grid = None
        if b is not None:
            grid = {}
            for g in ("API", "本地"):
                grid[g] = {}
                for L in lens:
                    se, sp = LA.se_sp(b, L, float(g == "API"))
                    grid[g][f"{int(2 ** L) - 1}字"] = (round(se, 3), round(sp, 3), round(se + sp - 1, 3))
        out[o] = {"coefs": None if b is None else [round(float(x), 3) for x in b],
                  "n_fail": n_fail, "n": n_all, "grid": grid}
    return out


# ---------------------------------------------------------------- 数据
def load_inputs(golds, signals, dialogues, frame_v6, frame_v5) -> dict:
    rows = [g for p in golds for g in load_gold(_rel(p))]
    ids = [g["id"] for g in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("金标里有重复 id")
    cache = load_cache([_rel(p) for p in signals])
    replies = load_replies(_rel(dialogues), set(ids))
    cells = sampling_cells(load_frame_scores(_rel(frame_v6)), load_frame_scores(_rel(frame_v5)))
    return {"ids": ids, "gold": {g["id"]: g["consensus"] for g in rows}, "split": {g["id"]: g.get("split") for g in rows},
            "pred": score_table(cache, replies, ids, [FROZEN])[FROZEN], "replies": replies, "cells": cells,
            "pop": Counter(cells.values()),
            "extract": "/".join(sorted({(x["model"], x["items_version"]) for x in cache.values()})[0])}


# ---------------------------------------------------------------- 结果抽取
def c2_view(dose_res: dict, scope="fixed") -> dict:
    """judge_dose.analyse 输出 → {结局: {"obs", "obs_ci", "cor", "cor_ci"}}（本地组、高 − 低）。"""
    out = {}
    for o in JD.OUTCOMES:
        k = ("本地", scope, o)
        if k not in dose_res["point"]:
            continue
        raw, cor = dose_res["point"][k]["diff"]
        craw, ccor = dose_res["ci"][k]
        out[o] = {"obs": raw, "obs_ci": craw, "cor": cor, "cor_ci": ccor}
    return out


def c5_view(la_res: dict) -> dict:
    """length_adjusted_contrast.analyse 输出 → {(微调, 对照, 臂, 维): {...}}。"""
    out = {}
    for k, (raw, cor) in la_res["point"].items():
        craw, ccor = la_res["ci"][k]
        out[k] = {"obs": raw, "obs_ci": craw, "cor": cor, "cor_ci": ccor}
    return out


def ci_excludes_zero(ci) -> bool | None:
    return None if not ci else (ci[0] > 0 or ci[1] < 0)


def run_judge(inp, crows_la, crows_dose, *, n_boot=1000, seed=0, samples=SAMPLES) -> dict:
    out = {}
    for s in samples:
        sub = sample_ids(inp["ids"], inp["split"], s)
        la_rows, dose_rows, diag = gold_tables(sub, inp)
        t0 = time.time()
        dose_res = JD.analyse(dose_rows, crows_dose, n_boot=n_boot, seed=seed)
        la_res = LA.analyse(la_rows, crows_la, n_boot=n_boot, seed=seed)
        out[s] = {"diag": diag, "errors": error_table(dose_rows),
                  "c2": c2_view(dose_res), "c2_both": c2_view(dose_res, "both"), "c5": c5_view(la_res),
                  "seconds": round(time.time() - t0, 1)}
        print(f"  {s}: {diag['n']} 段，{out[s]['seconds']} 秒", flush=True)
    return out


# ---------------------------------------------------------------- 报告
def _pc(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def _ci(t):
    return "" if not t else f" [{100 * t[0]:+.1f}, {100 * t[1]:+.1f}]"


def _cell(v, observed_only=False):
    if v is None:
        return "—"
    if observed_only:
        return f"观测 {_pc(v['obs'])}{_ci(v['obs_ci'])}"
    if v["cor"] is None:
        return f"**不可校正**（观测 {_pc(v['obs'])}）"
    star = "" if ci_excludes_zero(v["cor_ci"]) else " ⚠️含0"
    return f"{_pc(v['cor'])}{_ci(v['cor_ci'])}{star}"


def _shrink(base, alt):
    """相对现状的变化（alt / base）；None = 无法比较。"""
    if not base or not alt or base["cor"] in (None, 0) or alt["cor"] is None:
        return None
    return alt["cor"] / base["cor"]


def conclusions(res: dict) -> list[str]:
    """自动生成：C2 / C5 关键格在 test-only 下是否仍成立、相对现状保留多少。"""
    L = []
    for j, r in res.items():
        L.append(f"- **{JUDGES[j]['name']}**")
        for o in (JD.ANY, "d1", "d2", "d4"):
            b, t, d = (r[s]["c2"].get(o) for s in SAMPLES)
            L.append(f"  - C2 {JD.NAME[o]}：现状 {_cell(b)} → test-only {_cell(t)}"
                     + (f"（为现状的 {100 * _shrink(b, t):.0f}%）" if _shrink(b, t) is not None else "")
                     + f"；dev-only {_cell(d)}")
        for a, c in (("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base")):
            for dim in ("d1", "d4"):
                k = (a, c, "fixed", dim)
                b, t, d = (r[s]["c5"].get(k) for s in SAMPLES)
                L.append(f"  - C5 {a} {dim}（fixed）：现状 {_cell(b)} → test-only {_cell(t)}"
                         + (f"（为现状的 {100 * _shrink(b, t):.0f}%）" if _shrink(b, t) is not None else "")
                         + f"；dev-only {_cell(d)}")
    return L


CV_NOTE = [
    "## 交叉验证列：不适用", "",
    "`tools/judge_cv.py`（`reports/judge_cv_180/`）的 5 折按人设交叉验证，折内**只选规则集**（r0–r6 里训练折指标最高者），"
    "抽取提示词 q4 与定版规则 r6 都是在 dev 113 段上定的、在折外固定不变。因此：",
    "1. 对语料实际用的 r6 而言，折外预测与样本内预测**逐段相同**（cv.md：d2 / d5 折内 100% 选 r6、d3 96%）→ "
    "按人设折外重拟误判模型，得到的仍是「dev 参与开发」的 se / sp，乐观偏差一点没消；",
    "2. d1 / d4 的交叉验证 se / sp 之所以不同（d4 灵敏度 0.643 → 0.500），是因为折内选出了 r1 / r3 等别的规则集 —— "
    "那是「选规则这个流程」的误差，不是 r6 的误差，拿去校正 r6 打的语料是错配；",
    "3. 误判模型只有 4 个系数，折外拟合的过拟合问题本身很小；真正的偏差来源是规则与提示词在 dev 上开发，"
    "唯一不受它污染的样本是事先锁定的 test 67 段。",
    "→ 诚实的样本外估计 = **test-only** 列；dev-only 列是乐观上界的参照；全部 180 介于两者之间。", ""]


def render(res: dict, meta: dict) -> str:
    L = ["# 误判校正参数的样本外敏感性（test-only / dev-only / 全部 180）", "",
         f"- 生成 {meta['generated_at']}；规则 {FROZEN}；自助法 {meta['n_boot']} 次（金标人设重抽重拟 × 语料人设重抽，"
         "与 `judge_dose` / `length_adjusted_contrast` 相同）。",
         "- 误判模型：logit P(判官判失败) = b0 + b1·医师判失败 + b2·log2(字数) + b3·API；抽样格权重按各拟合样本重算。",
         f"- 可校正门槛：该拟合样本自己的全量拟合下每个长度段 Youden ≥ {USABLE_YOUDEN}，只判一次；d5 一律只报观测。",
         "- 单元格：校正差 [95% CI]，单位百分点；⚠️含0 = CI 含 0；「不可校正」= 该拟合样本下没过门槛。", ""]
    for j, r in res.items():
        L += [f"# {JUDGES[j]['name']}", "", f"- 抽取 `{meta['extract'][j]}`；语料 `{JUDGES[j]['corpus']}`", "",
              "## 拟合样本概况", "", "| 拟合样本 | 段数 | 人设 | 全语料未覆盖轮（权重） |", "|---|---|---|---|"]
        for s in SAMPLES:
            dg = r[s]["diag"]
            L.append(f"| {SAMPLE_NAME[s]} | {dg['n']} | {dg['n_personas']} | {dg['uncovered_turns']} / {dg['pop_turns']} |")
        L += ["", "## 1. 误判率对照（se / sp / Youden；Youden < 0.5 标 ⚠️）", ""]
        for g in ("本地", "API"):
            L += [f"### {g}组", "", "| 维 | 拟合样本 | 医师判失败 / 段数 | 31 字 | 127 字 | 511 字 |", "|---|---|---|---|---|---|"]
            for o in JD.OUTCOMES:
                for s in SAMPLES:
                    e = r[s]["errors"][o]
                    cells = []
                    if e["grid"] is None:
                        cells = ["不可拟合"] * len(LENS)
                    else:
                        for se, sp, y in e["grid"][g].values():
                            cells.append(f"{se:.2f} / {sp:.2f} / {y:.2f}" + (" ⚠️" if y < USABLE_YOUDEN else ""))
                    L.append(f"| {JD.NAME[o]} | {SAMPLE_NAME[s]} | {e['n_fail'][g]} / {e['n'][g]} | " + " | ".join(cells) + " |")
            L.append("")
        L += ["## 2a. C2 剂量反应：本地模型 fixed 臂 高 − 低（校正差 [CI]）", "",
              "| 结局 | 观测 | " + " | ".join(SAMPLE_NAME[s] for s in SAMPLES) + " |", "|---|---|" + "---|" * len(SAMPLES)]
        for o in JD.OUTCOMES:
            v0 = r["all"]["c2"].get(o)
            if v0 is None:
                continue
            obs = f"{_pc(v0['obs'])}{_ci(v0['obs_ci'])}"
            L.append(f"| {JD.NAME[o]} | {obs} | " + " | ".join(
                ("只报观测" if o in LA.OBSERVED_ONLY else _cell(r[s]["c2"].get(o))) for s in SAMPLES) + " |")
        L += ["", "（两臂合并口径见 json 的 `c2_both`。）", "",
              "## 2b. C5 心理微调配对差（微调 − 对照，校正差 [CI]）", "",
              "| 配对 | 臂 | 维 | 观测 | " + " | ".join(SAMPLE_NAME[s] for s in SAMPLES) + " |", "|---|---|---|---|" + "---|" * len(SAMPLES)]
        for k in sorted(r["all"]["c5"]):
            if k[3] not in C5_DIMS:
                continue
            v0 = r["all"]["c5"][k]
            L.append(f"| {k[0]} − {k[1]} | {k[2]} | {k[3]} | {_pc(v0['obs'])}{_ci(v0['obs_ci'])} | "
                     + " | ".join(_cell(r[s]["c5"].get(k)) for s in SAMPLES) + " |")
        L.append("")
    L += CV_NOTE
    L += ["## 自动摘要（C2 / C5 关键格）", ""] + conclusions(res) + [""]
    if meta.get("notes"):
        L += ["## 解读", "", meta["notes"].strip(), ""]
    return "\n".join(L) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="误判校正参数的样本外敏感性（test-only / dev-only / 全部 180）")
    ap.add_argument("--judges", default="qwen,deepseek")
    ap.add_argument("--dialogues", default="runs/v2/dialogues_gold180.jsonl")
    ap.add_argument("--frame-v6", default="runs/v2/scored_v6.jsonl")
    ap.add_argument("--frame-v5", default="runs/v2/scored_v5.jsonl")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default=None, help="可选：手写解读 md，原样附在报告末尾（默认读 <out>/notes.md 若存在）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    res, extract = {}, {}
    for j in [x for x in a.judges.split(",") if x]:
        cfg = JUDGES[j]
        print(f"[{j}] 读数据", flush=True)
        inp = load_inputs(GOLDS, cfg["signals"], a.dialogues, a.frame_v6, a.frame_v5)
        extract[j] = inp["extract"]
        crows_la, crows_dose = LA.corpus_rows(_rel(cfg["corpus"])), JD.corpus_rows(_rel(cfg["corpus"]))
        res[j] = run_judge(inp, crows_la, crows_dose, n_boot=a.n_boot, seed=a.seed)
    notes_path = Path(a.notes) if a.notes else out / "notes.md"
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_boot": a.n_boot, "seed": a.seed, "extract": extract,
            "notes": notes_path.read_text(encoding="utf-8") if notes_path.exists() else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "correction_sensitivity.json").write_text(
        json.dumps(_jsonable({"meta": {k: v for k, v in meta.items() if k != "notes"}, "results": res}),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "correction_sensitivity.md").write_text(render(res, meta), encoding="utf-8")
    print(f"报告 {out / 'correction_sensitivity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
