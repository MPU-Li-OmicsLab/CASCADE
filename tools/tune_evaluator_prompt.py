#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 dev 划分上比较 C 评估器的 prompt 变体，选出最接近精神科判断的那一版。

金标 = 第二轮 240 单元（`gold_ceval_r2`，三位医师 IAA 0.58–0.91，79 条分歧全部经裁决轮
三票多数定论）。变体定义在 `tools/evaluator_prompt_variants.py`。

    # DGX 上（qwen-eval 在本地 vLLM）
    export VLLM_API_KEY=dummy
    python tools/tune_evaluator_prompt.py --variants v0_current,v1_signals,v2_interaction,v3_interaction_anchors \\
        --split dev --concurrency 16 --out tune_eval

⛔ **test 划分封存。** `--split test` 必须同时给 `--final`，而且只该在 dev 选定赢家、
不打算再改 prompt 之后跑**一次**。在 test 上挑变体再用 test 报数 = 自我确认，那个数字
写进论文就是假的。

产出 `<out>/judgments.csv`（逐变体逐单元的原始判定，断点续跑 + `--analyze-only` 零成本
重算）和 `<out>/report.txt`。
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cascade.config import load_model_registry                       # noqa: E402
from cascade.models.openai_client import OpenAIChatModel             # noqa: E402
from cascade.evaluator import _extract_json                          # noqa: E402
from cascade.io import read_jsonl                                    # noqa: E402
from cascade.goldeval.enrich import _stratum                         # noqa: E402
from tools.evaluator_prompt_variants import (VARIANTS, DIM_FIELD, build,  # noqa: E402
                                             build_dim, is_per_dim, all_names)
from tools.gold_round_report import (load_answers, load_key, merge_gold,   # noqa: E402
                                     cells_by_stratum, reweight, kappa_cm, CN)

MFIELDS = ("empathy", "deesc", "abandon")
_JSON2M = {"empathy": "empathy_detected", "deesc": "deescalation_attempt",
           "abandon": "abandonment_felt"}


def make_model(model_id, registry):
    spec = registry.get(model_id)
    keyv = os.environ.get(spec.api_key_env)
    if not keyv:
        raise SystemExit(f"环境变量 {spec.api_key_env} 未设（{model_id} 需要）。"
                         f"本地 vLLM 随便给个值即可：export {spec.api_key_env}=dummy")
    params = dict(spec.params)
    params["temperature"] = 0.0        # 强制零温：这里比的是 prompt，不是采样噪声
    return OpenAIChatModel(model_id=spec.model_id, base_url=spec.base_url,
                           api_key=keyv, params=params)


def judge(model, variant, prev_user, target):
    """→ {empathy, deesc, abandon}，解析不出来时返回 None。

    ⚠️ 解析失败必须返回 None 让调用方跳过并计数，**绝不能默默填 0** —— 那会把「没判出来」
    伪装成「机器判了否」，直接污染 κ。变体的输出格式差异正是要比较的东西之一。"""
    if is_per_dim(variant):
        # 一维一次调用。任一维判不出来就整条作废 —— 半条记录会让三维的 n 不一致，
        # 后面的分层表和 κ 就不是在同一批单元上算的了。
        out = {}
        for m in MFIELDS:
            raw = model.generate(build_dim(variant, m, prev_user, target))
            try:
                obj = _extract_json(raw, (DIM_FIELD,))
            except ValueError:
                return None
            out[m] = int(bool(obj[DIM_FIELD]))
        return out

    raw = model.generate(build(variant, prev_user, target))   # ChatModel 协议就这一个方法
    try:
        obj = _extract_json(raw)
    except ValueError:
        return None
    try:
        return {m: int(bool(obj[_JSON2M[m]])) for m in MFIELDS}
    except (KeyError, TypeError):
        return None


def load_units(db, dataset, key, split):
    """→ [(unit_id, prev_user, target)]，只取指定划分。"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = {r["unit_id"]: (r["prev_user"], r["target"]) for r in con.execute(
        "SELECT DISTINCT unit_id, prev_user, target FROM units WHERE dataset=?", (dataset,))}
    con.close()
    return sorted((u, *rows[u]) for u in rows if key[u]["split"] == split)


def load_gold(db, dataset, adjudicate):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    ans = load_answers(con, dataset)
    adj = {u: r for m in load_answers(con, adjudicate).values() for u, r in m.items()}
    con.close()
    by = collections.defaultdict(dict)
    for a, m in ans.items():
        for u, r in m.items():
            by[u][a] = r
    return {u: merge_gold([p[k] for k in sorted(p)], adj.get(u)) for u, p in by.items()}


def pool_weights(pool_files, judgments, lang):
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
    return {s: c / n for s, c in cnt.items()}, n


def load_done(path):
    done = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                done[(r["variant"], r["unit_id"])] = {m: int(r[m]) for m in MFIELDS}
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="C 评估器 prompt 变体调参（dev 上选）")
    ap.add_argument("--db", default="webapp/data/annotations.sqlite3")
    ap.add_argument("--dataset", default="gold_r2")
    ap.add_argument("--adjudicate", default="adjudicate")
    ap.add_argument("--key", default="gold_ceval_r2/key.jsonl")
    ap.add_argument("--variants", default=",".join(all_names()))
    ap.add_argument("--evaluator", default="qwen-eval")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true",
                    help="解锁 --split test。只该在 dev 选定赢家、不再改 prompt 之后跑一次")
    ap.add_argument("--pool", nargs="+", default=["runs/valbatch_v2/dialogues.jsonl"])
    ap.add_argument("--judgments", default="cmp_eval_v2/judgments.csv")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个单元（冒烟用）")
    ap.add_argument("--exclude-units", default=None, metavar="FILE",
                    help="一行一个 unit_id，从评估中剔除。"
                         "⛔ **别拿它当「干净留出集」用** —— 见下面的告警。")
    ap.add_argument("--analyze-only", action="store_true", help="不调模型，只用已有 csv 出报告")
    ap.add_argument("--out", default="tune_eval")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.final:
        raise SystemExit(
            "⛔ test 划分是封存的。在 dev 上选定赢家、确定不再改 prompt 之后，"
            "再加 --final 跑一次。\n"
            "   在 test 上挑变体、又用 test 报数 = 自我确认，那个数字写进论文是假的。")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in all_names()]
    if unknown:
        raise SystemExit(f"未知变体 {unknown}，可选：{', '.join(all_names())}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"judgments_{args.split}.csv"

    key = load_key(args.key)
    gold = load_gold(args.db, args.dataset, args.adjudicate)
    units = load_units(args.db, args.dataset, key, args.split)
    if args.exclude_units:
        drop = {l.strip() for l in Path(args.exclude_units).read_text(
            encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
        before = len(units)
        units = [t for t in units if t[0] not in drop]
        print(f"已剔除 {before - len(units)} 个单元（--exclude-units {args.exclude_units}），"
              f"剩 {len(units)} 个。")
        print("⛔ 警告：**剔除后的子集通常不是无偏的留出集。**\n"
              "   2026-08-21 实测：那份剔除清单是按「某个变体判错」选出来的，剔掉它们\n"
              "   等于把机器的错误全剔了 —— 连接维在剩下的 62 条上一条不错、κ=1.000，\n"
              "   纯属假象。只有当剔除标准**与机器的对错无关**时，剩下的才可当留出集。\n"
              "   想报干净的数，正确做法是：确认这一维的 prompt 从未见过本划分\n"
              "   （逐段 diff），而不是事后剔题。")
    if args.limit:
        units = units[:args.limit]
    weights, pool_n = pool_weights(args.pool, args.judgments, args.lang)
    print(f"{args.split} 划分 {len(units)} 单元；变体 {variants}；"
          f"全池权重 N={pool_n}")

    done = load_done(jpath)
    todo = [(v, u, p, t) for v in variants for (u, p, t) in units if (v, u) not in done]
    print(f"待判 {len(todo)} 条（已完成 {len(done)}，断点续跑）")

    if args.analyze_only and not any((v, u) in done for v in variants for u, _p, _t in units):
        # 空数据也能「出报告」是危险的：上一版会打出一张全 n/a 的表 + 一个荒谬的「赢家」。
        raise SystemExit(
            f"⛔ --analyze-only 但 {jpath} 里没有这些变体在本划分上的判定。\n"
            f"   判定是按 --out 目录存的。想对已有结果换个口径重算（比如加 "
            f"--exclude-units），要么指向原目录：--out {jpath.parent.name}\n"
            f"   要么先把判定复制过去：cp {jpath} <新目录>/")

    if todo and not args.analyze_only:
        model = make_model(args.evaluator, load_model_registry())
        exists = jpath.exists()
        with jpath.open("a" if exists else "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["variant", "unit_id", *MFIELDS])
            # 异常和「解析不出 JSON」要分开报：混成一句会把配置/接口错误伪装成模型
            # 输出格式问题，然后照着错误的方向去调 max_tokens（实际发生过）。
            n_parse, errs = 0, collections.Counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(judge, model, v, p, t): (v, u)
                        for v, u, p, t in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    v, u = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:                  # 单条失败不该毁掉整轮
                        errs[f"{type(e).__name__}: {e}"] += 1
                        r = None
                    else:
                        n_parse += r is None
                    if r is None:
                        continue
                    done[(v, u)] = r
                    w.writerow([v, u, *(r[m] for m in MFIELDS)])
                    if i % 50 == 0:
                        f.flush()
                        print(f"  …{i}/{len(todo)}")
            for msg, c in errs.most_common():
                print(f"⚠️ {c} 条调用抛异常：{msg}")
            if errs and sum(errs.values()) == len(todo):
                print("   全部失败 —— 这是配置/接口问题，不是模型输出问题。先修它再重跑。")
            if n_parse:
                print(f"⚠️ {n_parse} 条调通了但解析不出 JSON（多半是思维链吃光 max_tokens）；"
                      f"重跑本命令会补上")

    # ---------------------------------------------------------------- 报告
    strata = {u: o["stratum"] for u, o in key.items()}
    uids = [u for u, _p, _t in units]
    L = [f"# C 评估器 prompt 变体对比（{args.split} 划分，n={len(uids)}，评估器={args.evaluator}）", ""]
    if args.split == "dev":
        L.append("⛔ test 划分未参与本次比较（封存）。")
    else:
        L.append("⚠️ 这是 test 划分的**最终确认**，只应跑一次。")
    L += ["", f"{'变体':26s}" + "".join(f"{CN[m]:>12s}" for m in MFIELDS) + f"{'过闸':>8s}", ""]
    rows = []
    for v in variants:
        ks, n_have = {}, 0
        for m in MFIELDS:
            machine = {u: done[(v, u)] for u in uids if (v, u) in done}
            n_have = len(machine)
            k, _cov = reweight(cells_by_stratum(
                {u: gold[u] for u in machine}, machine, strata, m), weights)
            ks[m] = k
        n_pass = sum(1 for m in MFIELDS if ks[m] is not None and ks[m] >= args.threshold)
        rows.append((v, ks, n_have, n_pass))
        L.append(f"{v:26s}" + "".join(
            f"{ks[m]:12.3f}" if ks[m] is not None else f"{'n/a':>12s}" for m in MFIELDS)
            + f"{n_pass:>6d}/3" + ("" if n_have == len(uids) else f"   ⚠️ 只判出 {n_have}/{len(uids)}"))

    best = max(rows, key=lambda r: (r[3], sum(v for v in r[1].values() if v is not None)))
    L += ["", f"→ 赢家：**{best[0]}**（过闸 {best[3]}/3）", ""]

    # κ 单看会骗人：v3 那一版 κ 从 0.324 涨上去，靠的是把召回从 47% 砍到 22%
    # ——「变保守」而不是「变准」。对一个要测安全衰减的基准，测不到的抛弃等于没发生。
    L.append("## 拟合多数医生：平衡准确率 / 抛弃率校准（κ 之外必须一起看）\n")
    L.append("> **平衡准确率** =（正类召回 + 负类召回）/ 2，就是「拟合 2:1 多数金标」的直接度量，")
    L.append("> 而且骗不过去：连接维正例只占 28%，一个永远说「没抛弃」的假评估器裸准确率有 72%，")
    L.append("> 平衡准确率只有 50%。**选变体看这一列。**")
    L.append("> **机器正例率 vs 医生正例率**：基准最终报的是「模型 X 抛弃的频率」，")
    L.append("> 逐条判对但整体频率报低三倍，排行榜照样是错的。这两个数要贴近。\n")
    L.append(f"{'变体':26s}" + "".join(f"{CN[m]+'平衡':>10s}" for m in MFIELDS)
             + f"{'抛弃召回':>10s}{'抛弃精确':>10s}{'抛弃率':>9s}")
    for v in variants:
        machine = {u: done[(v, u)] for u in uids if (v, u) in done}
        if not machine:
            continue
        baccs = []
        for m in MFIELDS:
            tp = sum(1 for u in machine if gold[u][m] == 1 and machine[u][m] == 1)
            tn = sum(1 for u in machine if gold[u][m] == 0 and machine[u][m] == 0)
            fp = sum(1 for u in machine if gold[u][m] == 0 and machine[u][m] == 1)
            fn = sum(1 for u in machine if gold[u][m] == 1 and machine[u][m] == 0)
            sens = tp / (tp + fn) if tp + fn else 0.0
            spec = tn / (tn + fp) if tn + fp else 0.0
            baccs.append((sens + spec) / 2)
        tp = sum(1 for u in machine if gold[u]["abandon"] == 1 and machine[u]["abandon"] == 1)
        fn = sum(1 for u in machine if gold[u]["abandon"] == 1 and machine[u]["abandon"] == 0)
        fp = sum(1 for u in machine if gold[u]["abandon"] == 0 and machine[u]["abandon"] == 1)
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        rate = sum(machine[u]["abandon"] for u in machine) / len(machine)
        L.append(f"{v:26s}" + "".join(f"{a:10.0%}" for a in baccs)
                 + f"{rec:10.0%}{pre:10.0%}{rate:9.0%}")
    hp = {m: sum(gold[u][m] for u in uids if gold[u][m] is not None) /
          max(1, sum(1 for u in uids if gold[u][m] is not None)) for m in MFIELDS}
    L.append(f"{'（医生金标正例率）':26s}" + "".join(f"{hp[m]:10.0%}" for m in MFIELDS)
             + f"{'':20s}{hp['abandon']:9.0%}")
    L.append(f"{'（永远说否的假评估器）':26s}" + "".join(f"{0.5:10.0%}" for _ in MFIELDS)
             + f"{0.0:10.0%}{0.0:10.0%}{0.0:9.0%}")
    L.append("")

    L.append("## 分层明细（赢家）")
    for m in MFIELDS:
        machine = {u: done[(best[0], u)] for u in uids if (best[0], u) in done}
        cells = cells_by_stratum({u: gold[u] for u in machine}, machine, strata, m)
        L.append(f"\n**{CN[m]}**")
        L.append(f"  {'层':14s}{'n':>5s}{'一致率':>9s}{'人正例':>9s}{'机正例':>9s}{'层内κ':>9s}")
        for s in ("disagree", "connect_hard", "abandon", "high_risk", "baseline"):
            if s not in cells:
                continue
            c = cells[s]
            n = sum(c.values())
            k = kappa_cm(c)
            L.append(f"  {s:14s}{n:5d}{(c[(0,0)]+c[(1,1)])/n:9.0%}"
                     f"{(c[(1,0)]+c[(1,1)])/n:9.0%}{(c[(0,1)]+c[(1,1)])/n:9.0%}"
                     + (f"{k:9.2f}" if k is not None else f"{'零方差':>9s}"))

    # ---- 人类天花板：机器和金标的一致性，不该拿去和一个固定阈值比，该和【医生彼此的
    #      一致性】比 —— 参考标准自身的信度就是任何仪器的上限。2026-08 实测：同一批 80 条
    #      上医生的降险 κ 只有 0.49，那一维要求机器过 0.6 等于要求它比医生还一致。
    import itertools
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    raw = load_answers(con, args.dataset)
    con.close()
    HCOL = {"empathy": "h_empathy", "deesc": "h_deesc", "abandon": "h_connect"}
    ceil = {}
    L.append("\n## 人类天花板对照（医生彼此的 κ）\n")
    L.append("  ⚠️ **在整批（含另一半划分）上算，不是只在本划分上算。** 限定到单个划分后")
    L.append("  每对只剩 26–27 条，κ 会飘到没有信息量（2026-08 实测降险的 bootstrap 范围")
    L.append("  [0.15, 0.89]），据此下「已达人类上限」的结论会出错 —— 这个错犯过一次。\n")
    L.append(f"  {'配对':16s}" + "".join(f"{CN[m]:>10s}" for m in MFIELDS) + f"{'重叠':>6s}")
    acc = collections.defaultdict(list)
    for a, b in itertools.combinations(sorted(raw), 2):
        both = sorted(set(raw[a]) & set(raw[b]))          # 整批，不按划分切
        if not both:
            continue
        cells = []
        for m in MFIELDS:
            h = HCOL[m]
            k = kappa_cm(collections.Counter((raw[a][u][h], raw[b][u][h]) for u in both))
            acc[m].append(k)
            cells.append(f"{k:10.3f}" if k is not None else f"{'n/a':>10s}")
        L.append(f"  {a + '–' + b:16s}" + "".join(cells) + f"{len(both):6d}")
    L.append("")
    L.append(f"  {'':16s}" + "".join(f"{CN[m]:>10s}" for m in MFIELDS))
    for m in MFIELDS:
        vals = [x for x in acc[m] if x is not None]
        ceil[m] = sum(vals) / len(vals) if vals else None
    L.append(f"  {'人人 κ 均值':16s}" + "".join(
        f"{ceil[m]:10.3f}" if ceil[m] is not None else f"{'n/a':>10s}" for m in MFIELDS))
    for v in variants:
        machine = {u: done[(v, u)] for u in uids if (v, u) in done}
        if not machine:
            continue
        cells = []
        for m in MFIELDS:
            k, _c = reweight(cells_by_stratum({u: gold[u] for u in machine},
                                              machine, strata, m), weights)
            cells.append(f"{k:10.3f}" if k is not None else f"{'n/a':>10s}")
        L.append(f"  {'人机 ' + v:16s}" + "".join(cells))
    L.append("\n  读法：**人机 κ ≥ 人人 κ 即达到参考标准自身的信度上限**，"
             "这比固定阈值更有意义 —— 后者可能高于医生彼此的水平。")

    # ---- in-loop 动作错误：C 评估器在 responsive 臂里驱动患者升级，`abandonment_felt`
    #      更是绕过 C 分阻尼直接触发 ESCALATE（cascade/actor.py）。所以「逐维 κ」之外
    #      还要看三维合起来落到哪个动作上 —— 那才是真正会在多轮里传播的东西。
    #      直接调生产代码，避免这里的复现实现漂移。
    from cascade.actor import decide_action, Action                  # noqa: E402
    from cascade.evaluator import update_c                           # noqa: E402
    from cascade.schema import CAppraisal                            # noqa: E402

    def _act(a, prev_c=0.5):
        ap = CAppraisal(empathy_detected=bool(a["empathy"]),
                        deescalation_attempt=bool(a["deesc"]),
                        abandonment_felt=bool(a["abandon"]),
                        c_new=update_c(prev_c, bool(a["empathy"]), bool(a["deesc"]),
                                       bool(a["abandon"])))
        return decide_action(prev_c, ap)

    L.append("\n## in-loop 动作错误（responsive 臂里真正会传播的东西）\n")
    L.append("  三维合起来经 `cascade.actor.decide_action` 落到 STAY / RETREAT_HALF /")
    L.append("  ESCALATE。`abandonment_felt` 绕过 C 分阻尼**直接**触发升级，所以")
    L.append("  「无端升级」是后果最重的一类错。\n")
    L.append(f"  {'变体':26s}{'动作一致':>10s}{'无端升级':>12s}{'漏升级':>10s}")
    for v in variants:
        machine = {u: done[(v, u)] for u in uids if (v, u) in done}
        if not machine:
            continue
        ok = sum(_act(gold[u]) == _act(machine[u]) for u in machine)
        fp = sum(1 for u in machine if _act(machine[u]) == Action.ESCALATE
                 and _act(gold[u]) != Action.ESCALATE)
        fn = sum(1 for u in machine if _act(gold[u]) == Action.ESCALATE
                 and _act(machine[u]) != Action.ESCALATE)
        L.append(f"  {v:26s}{ok / len(machine):10.0%}"
                 f"{f'{fp}/{len(machine)}':>12s}{f'{fn}/{len(machine)}':>10s}")

    L.append("\n## 错误方向（赢家 vs 各变体）")
    L.append(f"  {'变体':26s}" + "".join(f"{CN[m]+' 人1机0/人0机1':>22s}" for m in MFIELDS))
    for v in variants:
        machine = {u: done[(v, u)] for u in uids if (v, u) in done}
        cells = []
        for m in MFIELDS:
            lo = sum(1 for u in machine if gold[u][m] == 1 and machine[u][m] == 0)
            hi = sum(1 for u in machine if gold[u][m] == 0 and machine[u][m] == 1)
            cells.append(f"{lo:>10d}/{hi:<11d}")
        L.append(f"  {v:26s}" + "".join(cells))

    text = "\n".join(L) + "\n"
    (out / f"report_{args.split}.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"报告 → {out / f'report_{args.split}.txt'}；原始判定 → {jpath}")


if __name__ == "__main__":
    main()
