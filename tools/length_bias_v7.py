"""判官 v7 的金标锚定长度偏倚检查（六维；tools/length_bias_check 的 A 层只接 C 评估器金标）。零 GPU。

    python tools/length_bias_v7.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl \
        --signals runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --signals runs/v2/v7_test_qwen38.jsonl.signals.jsonl \
        --out reports/length_bias_v7

问：**在医师判定相同的前提下**，回复越长，判官 v7（定版 r6）越容易 / 越不容易判失败吗？
  * 主模型（每维）：加权 logit(判官判失败) ~ 医师判失败 + log2(回复字数)；log2 系数 = 长度每翻一倍的对数几率变化。
  * 拆开看：医师判失败的段里 ~ log2(len)（灵敏度随长度）；医师判未失败的段里 ~ log2(len)（误报随长度）。
  * 稳健性：去掉本轮复读段（cpsycoun 的长回复多半是复读）；加「API 旗舰」协变量（旗舰回复普遍更长，长度与模型组纠缠）。
  * 长度三分位的加权灵敏度 / 特异度表。
权重 = 抽样格权重（与 judge_correction 同一套）；CI = 人设聚类自助法（权重固定）。失败：d1–d4 = 0、d5 = ≤1。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.analysis.repetition import is_loop  # noqa: E402
from tools.judge_correction import cell_weights, group_of, load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_cv import DIMS5, FROZEN, _fail, load_cache, load_replies, persona_of, score_table  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402
from tools.misclass_correction import wlogit  # noqa: E402


def fit_coef(X, y, w, col):
    """加权 logit 的第 col 个自变量系数；y 无方差 → None。"""
    if len(y) < 5 or y.min() == y.max():
        return None
    return float(wlogit(X, y, w)[col + 1])


def boot(rows, build, col, n_boot, seed):
    """rows = [(人设, ...)]；build(rows) → (X, y, w)。→ 点估计 + 人设聚类自助 95% CI + 每翻一倍的 OR。"""
    X, y, w = build(rows)
    est = fit_coef(X, y, w, col)
    if est is None:
        return None
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    names = sorted(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = [r for j in rng.integers(0, len(names), len(names)) for r in by[names[j]]]
        c = fit_coef(*build(pick), col)
        if c is not None:
            out.append(c)
    lo, hi = (np.percentile(out, [2.5, 97.5]) if len(out) >= 50 else (float("nan"), float("nan")))
    return {"coef": round(est, 3), "or": round(float(np.exp(est)), 3),
            "ci_or": (round(float(np.exp(lo)), 3), round(float(np.exp(hi)), 3)), "n": len(rows),
            "n_boot_ok": len(out)}


def table_rows(ids, pred, gold, w, replies):
    """→ {维: [(人设, 判官失败, 医师失败, log2 长度, 权重, 是否复读, 是否 API)]}。"""
    out = {}
    for d in DIMS5:
        out[d] = [(persona_of(i), float(_fail(pred[i][d], d)), float(_fail(gold[i][d], d)),
                   float(np.log2(len(replies[i]) + 1)), w[i], is_loop(replies[i]),
                   float(group_of(i.split("|", 1)[0]) == "API"))
                  for i in ids if gold[i].get(d) is not None]
    return out


def _arr(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], dtype=float).reshape(len(rows), len(cols))


def analyse(rows_by_dim, *, n_boot=1000, seed=0) -> dict:
    res = {}
    for d, rows in rows_by_dim.items():
        main = lambda rs: (_arr(rs, (2, 3)), _arr(rs, (1,))[:, 0], _arr(rs, (4,))[:, 0])          # 医师失败 + 长度
        only_len = lambda rs: (_arr(rs, (3,)), _arr(rs, (1,))[:, 0], _arr(rs, (4,))[:, 0])
        with_grp = lambda rs: (_arr(rs, (2, 3, 6)), _arr(rs, (1,))[:, 0], _arr(rs, (4,))[:, 0])
        fail = [r for r in rows if r[2] == 1]
        ok = [r for r in rows if r[2] == 0]
        noloop = [r for r in rows if not r[5]]
        lens = np.array([r[3] for r in rows])
        cut = np.quantile(lens, [1 / 3, 2 / 3])
        terc = {}
        for k, (lo, hi) in enumerate(((-np.inf, cut[0]), (cut[0], cut[1]), (cut[1], np.inf))):
            band = [r for r in rows if lo < r[3] <= hi] if k else [r for r in rows if r[3] <= hi]
            f = [r for r in band if r[2] == 1]
            o = [r for r in band if r[2] == 0]
            sf, so = sum(r[4] for r in f), sum(r[4] for r in o)
            terc[k] = {"chars": (int(2 ** (lo if np.isfinite(lo) else lens.min()) - 1),
                                 int(2 ** (hi if np.isfinite(hi) else lens.max()) - 1)),
                       "n_fail": len(f), "n_ok": len(o),
                       "sens": round(sum(r[4] for r in f if r[1]) / sf, 3) if sf else None,
                       "spec": round(sum(r[4] for r in o if not r[1]) / so, 3) if so else None}
        res[d] = {"main": boot(rows, main, 1, n_boot, seed),
                  "sens_slope": boot(fail, only_len, 0, n_boot, seed),
                  "fp_slope": boot(ok, only_len, 0, n_boot, seed),
                  "no_loop": boot(noloop, main, 1, n_boot, seed),
                  "with_group": boot(rows, with_grp, 1, n_boot, seed),
                  "tertiles": terc, "n_loop": len(rows) - len(noloop)}
    return res


def _or(x):
    if not x:
        return "—"
    lo, hi = x["ci_or"]
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi > 50:
        return f"{x['or']:.2f}（不稳定：样本太少、完全分离）"
    flag = " ⚠️" if (lo > 1 or hi < 1) else ""
    return f"{x['or']:.2f} [{lo:.2f}, {hi:.2f}]{flag}"


def render(res, meta) -> str:
    L = ["# 判官 v7 · 金标锚定的长度偏倚（六维）", "",
         f"- 生成 {meta['generated_at']}；抽取 `{meta['extract']}`、规则 {FROZEN}；金标 {meta['n_gold']} 段（原始共识），"
         f"抽样格加权；人设聚类自助法 {meta['n_boot']} 次。",
         "- 表中是 **回复长度每翻一倍、判官判「失败」的几率比（OR）**；1 = 无长度偏倚，<1 = 越长越不判失败，>1 = 越长越判失败。"
         "⚠️ = 95% CI 不含 1。失败：d1–d4 = 0，d5 = ≤1。", "",
         "## 控制医师判定后的长度效应", "",
         "| 维 | 主模型（控制医师判定） | 只看医师判失败的段（灵敏度随长度） | 只看医师判未失败的段（误报随长度） | 去掉本轮复读段 | 再控制「API 旗舰」 |",
         "|---|---|---|---|---|---|"]
    for d in DIMS5:
        r = res[d]
        L.append(f"| {d} | {_or(r['main'])} | {_or(r['sens_slope'])}（n={r['sens_slope']['n'] if r['sens_slope'] else '—'}） | "
                 f"{_or(r['fp_slope'])} | {_or(r['no_loop'])}（去 {r['n_loop']} 段） | {_or(r['with_group'])} |")
    L += ["", "## 长度三分位的加权灵敏度 / 特异度", "",
          "| 维 | 短（字数） | 中 | 长 |", "|---|---|---|---|"]
    for d in DIMS5:
        cells = []
        for k in range(3):
            t = res[d]["tertiles"][k]
            cells.append(f"{t['chars'][0]}–{t['chars'][1]} 字：灵敏度 {t['sens'] if t['sens'] is not None else '—'}"
                         f"（失败 {t['n_fail']} 段）/ 特异度 {t['spec'] if t['spec'] is not None else '—'}")
        L.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7 金标锚定长度偏倚")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--signals", action="append", required=True)
    ap.add_argument("--dialogues", default=str(REPO / "runs/v2/dialogues_gold180.jsonl"))
    ap.add_argument("--frame-v6", default=str(REPO / "runs/v2/scored_v6.jsonl"))
    ap.add_argument("--frame-v5", default=str(REPO / "runs/v2/scored_v5.jsonl"))
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
    res = analyse(table_rows(ids, pred, gold, w, replies), n_boot=a.n_boot, seed=a.seed)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot,
            "extract": "/".join(sorted({(r["model"], r["items_version"]) for r in cache.values()})[0])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "length_bias.json").write_text(json.dumps({"meta": meta, "dims": res}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out / "length_bias.md").write_text(render(res, meta), encoding="utf-8")
    print(f"报告 {out / 'length_bias.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
