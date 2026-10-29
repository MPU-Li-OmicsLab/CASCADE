#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一轮金标回收的完整体检：标注者质量 → IAA → 裁决合成 → 人机效度。

和 `tools/reweight_kappa.py` 的分工：那个只算重加权 κ，且金标取「两人一致的格子」
（分歧格直接丢掉 —— 丢掉的恰恰是难题，剩下的是选择性偏样）。本脚本直接读 Web 工具的
SQLite，把**裁决轮**并进来做三票多数，所以每个单元三维都有定论，没有选择性丢弃。

产出一份 markdown，数字全部可复算：

    python tools/gold_round_report.py --out reports/gold_r2_validity.md

⛔ 只读打开数据库（`mode=ro`），不会碰医生正在标的库。
"""
from __future__ import annotations
import argparse
import collections
import itertools
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.io import read_jsonl                                    # noqa: E402
from cascade.goldeval.enrich import _stratum                         # noqa: E402

# 人标列 → 机器字段。h_connect 是唯一取反点：人标 1=保持连接=好，机器 abandon=1=被抛弃=差。
DIMS = (("h_empathy", "empathy", False),
        ("h_deesc", "deesc", False),
        ("h_connect", "abandon", True))
CN = {"empathy": "共情", "deesc": "有效降险", "abandon": "保持连接"}
STRATA = ("disagree", "connect_hard", "abandon", "high_risk", "baseline")
CAND_FIELD = {"empathy": "emp", "deesc": "de", "abandon": "ab"}


# ---------------------------------------------------------------- 纯函数

def kappa_cm(cm) -> float | None:
    """Cohen's κ，输入是 {(人, 机): 计数}。任一侧零方差时 pe==1 → κ 无定义，返回 None
    而不是 0：那两件事在解读上完全不同（见分层明细里的「零方差」行）。"""
    n = sum(cm.values())
    if not n:
        return None
    po = (cm[(0, 0)] + cm[(1, 1)]) / n
    ph = (cm[(1, 0)] + cm[(1, 1)]) / n
    pm = (cm[(0, 1)] + cm[(1, 1)]) / n
    pe = ph * pm + (1 - ph) * (1 - pm)
    return None if abs(pe - 1) < 1e-12 else (po - pe) / (1 - pe)


def merge_gold(rows, adj_row=None):
    """两位标注者 + 可选裁决 → 三维金标（转成机器侧极性）。

    分歧维取裁决者的值 = 三票多数（另两票必然一比一）。没有裁决就留 None，绝不猜。"""
    r1, r2 = rows
    out = {}
    for hcol, mfield, invert in DIMS:
        if r1[hcol] == r2[hcol]:
            v = r1[hcol]
        elif adj_row is not None:
            v = adj_row[hcol]
        else:
            v = None
        out[mfield] = None if v is None else (1 - v if invert else v)
    return out


def reweight(cells, weights):
    """分层重加权（等价于逆概率加权）→ (κ, 覆盖到的总体比例)。

    合并 κ 会高估，两个原因：① 跨层对比撑大边际方差；② `disagree`/`abandon` 两层是用
    候选评估器自己的输出定义的，而被验的评估器就是候选之一。所以报数一律报这个。

    抽样里没出现的层从权重里剔除 —— 覆盖不到 100% 的 κ 不能当总体估计用。"""
    present = {s: weights[s] for s in cells if s in weights}
    total = sum(present.values())
    if not total:
        return None, 0.0
    wcm = collections.Counter()
    for s, cm in cells.items():
        if s not in present:
            continue
        n = sum(cm.values())
        for cell, v in cm.items():
            wcm[cell] += v / n * (present[s] / total) * 1000
    return kappa_cm(wcm), total


def cells_by_stratum(gold, machine, strata, field):
    """→ {层: {(人, 机): 计数}}，跳过任一侧缺值的单元。"""
    out = {}
    for u, g in gold.items():
        m = machine.get(u) or {}
        if g.get(field) is None or m.get(field) is None:
            continue
        out.setdefault(strata[u], collections.Counter())[(g[field], m[field])] += 1
    return out


def longest_streak(triples):
    """最长连发同一组合。上一轮抓住 ann1 退化靠的就是这个（34 题连发）—— 中位耗时和
    `<15s` 两个既有指标当时都没报警。"""
    if not triples:
        return 0
    longest = run = 1
    for i in range(1, len(triples)):
        run = run + 1 if triples[i] == triples[i - 1] else 1
        longest = max(longest, run)
    return longest


# ---------------------------------------------------------------- 读取

def load_answers(con, dataset):
    """→ {annotator: {unit_id: row}}，按 xlsx 行序（seq）。"""
    out = collections.defaultdict(dict)
    for r in con.execute(
            "SELECT a.*, u.seq FROM answers a JOIN units u ON u.dataset=a.dataset "
            "AND u.annotator=a.annotator AND u.unit_id=a.unit_id "
            "WHERE a.dataset=? ORDER BY u.seq", (dataset,)):
        out[r["annotator"]][r["unit_id"]] = r
    return out


def load_key(path):
    return {o["unit_id"]: o for o in
            (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip())}


def pool_weights(pool_files, judgments, lang):
    """全池分层占比。复用抽样工具本身的 flatten，保证权重和抽样用的是同一套定义。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mce", Path(__file__).resolve().parent / "make_ceval_enriched.py")
    mce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mce)
    turns = [t for f in pool_files
             for t in mce.flatten(read_jsonl(f), mce.load_judgments(judgments), 6)
             if t["lang"] == lang]
    cnt = collections.Counter(_stratum(t) for t in turns)
    n = sum(cnt.values())
    if not n:
        raise SystemExit("全池为空：检查 --pool / --lang")
    return {s: c / n for s, c in cnt.items()}, n


# ---------------------------------------------------------------- 报告

def build_report(con, key, ans, adj, weights, pool_n, args, calib=None, calib_key=None):
    L = []
    P = L.append
    strata = {u: o["stratum"] for u, o in key.items()}
    splits = {u: o["split"] for u, o in key.items()}
    machine = {u: o["machine"] for u, o in key.items()}

    by_unit = collections.defaultdict(dict)
    for a, m in ans.items():
        for u, r in m.items():
            by_unit[u][a] = r
    gold = {u: merge_gold([per[k] for k in sorted(per)], adj.get(u))
            for u, per in by_unit.items()}
    names = sorted(ans)

    P(f"# {args.dataset} 效度体检报告\n")
    P("> 由 `tools/gold_round_report.py` 生成，数字可复算：\n>\n> ```bash\n"
      f"> python tools/gold_round_report.py --dataset {args.dataset} "
      f"--key {args.key} --out <此文件>\n> ```\n")
    P(f"标注包 `{Path(args.key).parent}`，{len(by_unit)} 个单元 × 2 人 + 裁决轮；"
      f"全池权重来自 `{args.pool[0]}`（N={pool_n}）。\n")

    # ---- A
    P("## A. 标注者作答模式\n")
    rows_by_a = {a: list(ans[a].values()) for a in names}
    trip = {a: [(r["h_empathy"], r["h_deesc"], r["h_connect"]) for r in rows_by_a[a]]
            for a in names}
    P("| | " + " | ".join(names) + " |")
    P("|---|" + "---:|" * len(names))
    P("| 题量 | " + " | ".join(str(len(rows_by_a[a])) for a in names) + " |")
    P("| 三维全填 (1,1,1) | " + " | ".join(
        f"{sum(t == (1, 1, 1) for t in trip[a]) / len(trip[a]):.0%}" for a in names) + " |")
    P("| 最长连发同一组合 | " + " | ".join(str(longest_streak(trip[a])) for a in names) + " |")
    secs = {a: sorted(r["seconds"] for r in rows_by_a[a]) for a in names}
    P("| 中位耗时 | " + " | ".join(f"{secs[a][len(secs[a]) // 2]:.0f}s" for a in names) + " |")
    for i, nm in enumerate(("前 1/4", "第 2/4", "第 3/4", "末 1/4")):
        cells = []
        for a in names:
            q = max(1, len(trip[a]) // 4)
            seg = trip[a][i * q:(i + 1) * q]
            s = sorted(r["seconds"] for r in rows_by_a[a][i * q:(i + 1) * q]) or [0]
            cells.append(f"{sum(t == (1, 1, 1) for t in seg) / max(len(seg), 1):.0%}"
                         f" / {s[len(s) // 2]:.0f}s")
        P(f"| {nm}：全1率 / 中位耗时 | " + " | ".join(cells) + " |")
    P("")
    P("各维给 1 的比例：\n")
    P("| | " + " | ".join(CN[mf] for _, mf, _ in DIMS) + " |")
    P("|---|" + "---:|" * 3)
    for a in names:
        P(f"| {a} | " + " | ".join(
            f"{sum(r[h] for r in rows_by_a[a]) / len(rows_by_a[a]):.0%}"
            for h, _, _ in DIMS) + " |")
    P("")

    # ---- B
    P("## B. 两两一致性（IAA）\n")
    P("| 配对 | 重叠 | " + " | ".join(CN[mf] for _, mf, _ in DIMS) + " |")
    P("|---|---:|" + "---|" * 3)
    pair_agree = []
    for a, b in itertools.combinations(names, 2):
        both = sorted(set(ans[a]) & set(ans[b]))
        cells = []
        for h, _, _ in DIMS:
            cm = collections.Counter((ans[a][u][h], ans[b][u][h]) for u in both)
            k = kappa_cm(cm)
            agr = sum(ans[a][u][h] == ans[b][u][h] for u in both) / len(both)
            pair_agree.append(agr)
            cells.append(f"κ={k:.2f}（一致 {agr:.0%}）" if k is not None
                         else f"零方差（一致 {agr:.0%}）")
        P(f"| {a}–{b} | {len(both)} | " + " | ".join(cells) + " |")
    P("")

    # ---- C
    P("## C. 裁决轮\n")
    ndis, missing = collections.Counter(), []
    for u, per in by_unit.items():
        r1, r2 = [per[k] for k in sorted(per)]
        d = [h for h, _, _ in DIMS if r1[h] != r2[h]]
        if d:
            ndis[len(d)] += 1
            if u not in adj:
                missing.append(u)
    done = sum(1 for g in gold.values() if all(v is not None for v in g.values()))
    P(f"- 两人有分歧的单元：**{sum(ndis.values())}/{len(by_unit)}"
      f"（{sum(ndis.values()) / len(by_unit):.0%}）**，分歧维数分布 {dict(sorted(ndis.items()))}")
    P(f"- 已裁决 {sum(ndis.values()) - len(missing)}，缺裁决 {len(missing)}"
      + (f" {missing[:8]}" if missing else ""))
    P(f"- 三维全部有定论的单元：**{done}/{len(by_unit)}**\n")

    # ---- D
    P("## D. 人机效度（重加权总体 κ）\n")
    P(f"全池分层占比（N={pool_n}）：" + "、".join(
        f"{s} {weights[s]:.1%}" for s in STRATA if s in weights) + "\n")
    rng = random.Random(args.seed)
    units = sorted(gold)
    P(f"| 维度 | 合并 κ | **重加权 κ** | 95% CI | 覆盖 | 闸门 {args.threshold} |")
    P("|---|---:|---:|---|---:|---|")
    for _, mf, _ in DIMS:
        cells = cells_by_stratum(gold, machine, strata, mf)
        pooled = collections.Counter()
        for c in cells.values():
            pooled.update(c)
        k_rw, cov = reweight(cells, weights)
        ks = []
        for _ in range(args.boot):
            samp = [units[rng.randrange(len(units))] for _ in units]
            cc = {}
            for u in samp:
                if gold[u][mf] is None:
                    continue
                cc.setdefault(strata[u], collections.Counter())[(gold[u][mf], machine[u][mf])] += 1
            k, _c = reweight(cc, weights)
            if k is not None:
                ks.append(k)
        ks.sort()
        ci = f"[{ks[int(.025 * len(ks))]:.3f}, {ks[int(.975 * len(ks))]:.3f}]" if ks else "n/a"
        P(f"| {CN[mf]} | {kappa_cm(pooled):.3f} | **{k_rw:.3f}** | {ci} | {cov:.0%} | "
          f"{'✅' if k_rw >= args.threshold else '❌'} |")
    P("")
    for _, mf, _ in DIMS:
        cells = cells_by_stratum(gold, machine, strata, mf)
        P(f"**{CN[mf]}** 分层明细：\n")
        P("| 层 | n | 一致率 | 人正例 | 机正例 | 层内 κ |")
        P("|---|---:|---:|---:|---:|---:|")
        for s in STRATA:
            if s not in cells:
                continue
            c = cells[s]
            n = sum(c.values())
            k = kappa_cm(c)
            P(f"| {s} | {n} | {(c[(0, 0)] + c[(1, 1)]) / n:.0%} | "
              f"{(c[(1, 0)] + c[(1, 1)]) / n:.0%} | {(c[(0, 1)] + c[(1, 1)]) / n:.0%} | "
              + (f"{k:.2f} |" if k is not None else "零方差 |"))
        P("")

    P("### dev / test 分别\n")
    P("| 维度 | dev | test |")
    P("|---|---:|---:|")
    for _, mf, _ in DIMS:
        vals = []
        for sp in ("dev", "test"):
            sub = {u: g for u, g in gold.items() if splits[u] == sp}
            k, _c = reweight(cells_by_stratum(sub, machine, strata, mf), weights)
            vals.append(f"{k:.3f}（n={len(sub)}）" if k is not None else "n/a")
        P(f"| {CN[mf]} | " + " | ".join(vals) + " |")
    P("")

    # ---- E
    cand_ids = sorted({c for o in key.values() for c in (o.get("candidates") or {})})
    if cand_ids:
        P("## E. 换评估器能不能救\n")
        P("| 评估器 | " + " | ".join(CN[mf] for _, mf, _ in DIMS) + " |")
        P("|---|" + "---:|" * 3)
        P("| in-loop appraisal | " + " | ".join(
            f"{reweight(cells_by_stratum(gold, machine, strata, mf), weights)[0]:.3f}"
            for _, mf, _ in DIMS) + " |")
        for cid in cand_ids:
            vals = []
            for _, mf, _ in DIMS:
                mm = {u: {mf: (o["candidates"].get(cid) or {}).get(CAND_FIELD[mf])}
                      for u, o in key.items()}
                k, _c = reweight(cells_by_stratum(gold, mm, strata, mf), weights)
                vals.append(f"{k:.3f}" if k is not None else "n/a")
            P(f"| {cid} | " + " | ".join(vals) + " |")
        P("")

    # ---- F
    P("## F. 错误方向\n")
    P("| 维度 | 医生 1 / 机器 0 | 医生 0 / 机器 1 | 读法 |")
    P("|---|---:|---:|---|")
    for _, mf, invert in DIMS:
        lo = sum(1 for u, g in gold.items() if g[mf] == 1 and machine[u][mf] == 0)
        hi = sum(1 for u, g in gold.items() if g[mf] == 0 and machine[u][mf] == 1)
        # 取反维（保持连接）这里按机器极性计数：1 = 被抛弃。标签写明，免得读成「少给保持连接」。
        what = "判抛弃" if invert else "给"
        note = (f"机器系统性**少{what}**" if lo > 2 * hi else
                f"机器系统性**多{what}**" if hi > 2 * lo else "**双向都错**，不是阈值问题")
        label = f"{CN[mf]}（按被抛弃计：1 = 抛弃）" if invert else CN[mf]
        P(f"| {label} | {lo} | {hi} | {note} |")
    P("")

    # ---- G
    if calib:
        cu = collections.defaultdict(dict)
        for a, m in calib.items():
            for u, r in m.items():
                cu[u][a] = r
        if cu:
            same = sum(len({tuple(per[a][h] for h, _, _ in DIMS) for a in per}) == 1
                       for per in cu.values())
            cs = sorted(r["seconds"] for per in cu.values() for r in per.values())
            m_agr = sum(pair_agree) / len(pair_agree)
            ncell = len(cu) * 3
            P("## G. 校准集独立性\n")
            P(f"- 三人三维完全相同的题：**{same}/{len(cu)}**，中位耗时 {cs[len(cs) // 2]:.0f}s")
            P(f"- 正式集实测两两一致率均值 {m_agr:.2f} → 若真独立，{ncell} 个格子三人全同的"
              f"概率约 **{m_agr ** ncell:.2%}**")
            if same == len(cu):
                P("- ⛔ 校准轮**没有独立标注**，不能当作对齐证据"
                  "（正式集不受影响：分歧率见 §C）")
            P("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="金标回收体检报告")
    ap.add_argument("--db", default="webapp/data/annotations.sqlite3")
    ap.add_argument("--dataset", default="gold_r2")
    ap.add_argument("--adjudicate", default="adjudicate")
    ap.add_argument("--key", default="gold_ceval_r2/key.jsonl")
    ap.add_argument("--calib", default="calib_r2", help="留空则跳过 §G")
    ap.add_argument("--calib-key", default="gold_ceval_calib_r2/key.jsonl")
    ap.add_argument("--pool", nargs="+", default=["runs/valbatch_v2/dialogues.jsonl"])
    ap.add_argument("--judgments", default="cmp_eval_v2/judgments.csv")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default=None,
                    help="一份手写的解读 md，原样附在报告末尾。分开放是为了让报告本身"
                         "随时可以重新生成而不冲掉结论。")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    key = load_key(args.key)
    ans = load_answers(con, args.dataset)
    if not ans:
        raise SystemExit(f"{args.db} 里没有 dataset={args.dataset} 的标注")
    adj = {u: r for m in load_answers(con, args.adjudicate).values() for u, r in m.items()}
    calib = load_answers(con, args.calib) if args.calib else None
    ckey = load_key(args.calib_key) if args.calib and Path(args.calib_key).exists() else None
    weights, pool_n = pool_weights(args.pool, args.judgments, args.lang)

    text = build_report(con, key, ans, adj, weights, pool_n, args, calib, ckey)
    if args.notes:
        text += ("\n---\n\n"
                 + Path(args.notes).read_text(encoding="utf-8").rstrip() + "\n")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"报告 → {args.out}（{len(text)} 字符）")
    else:
        print(text)


if __name__ == "__main__":
    main()
