"""同一个问题，判官和医师给出的答案是否一样：逐条比较的翻转表。零模型调用。

    python tools/judge_vs_gold.py --out reports/judge_vs_gold

判官的答案 = 六维判官 v7 在全语料上的观测比较（一项典型研究会报告的东西）；
医师的答案 = 医师金标按抽样格加权推回全语料的估计（tools/gold_prevalence），另报 raking 权重版（tools/gold_prevalence_raked）。
两者针对同一总体（全语料 9,491 轮）、同一子组定义（gold_prevalence.subgroups）、同一失败定义（d1–d4 = 0、d5 ≤ 1、
「任一」= d1–d4 任一为 0）。CI 都是人设聚类自助法；判官一侧两组共用同一次人设抽样（配对）。
比较项：风险段高 − 低（两臂合并 / fixed 臂 / 本地模型）、旗舰 − 本地、陪伴 − 旗舰、陪伴 − 本地、API − 本地、
心理微调组 − 对照基座组、三对「微调 − 对照」。DeepSeek 判官只打了 6 个本地模型 → 只比只涉及本地模型的项。
干预（A6-S）与分支没有医师金标，不在表内。

⚡2026-10-26 起：分类改用 tools/judge_vs_gold_joint（判官与医师共用同一批人设重抽，直接检验 D = Δ判官 − Δ医师），
输出格式不变，旧类名按下表映射：确认 → 一致；方向相反 → 相反；幅度不符（医师同向显著）→ 幅度偏大 / 幅度偏小；
幅度不符（医师不显著）→ 被否定；判官不显著而医师显著 → 漏掉（方向相反时记相反）。旧规则（只看判官点估计是否落在医师 CI 内）
的报告存档于 reports/judge_vs_gold_ci_rule/。下面是旧规则的原文，保留作记录。

分类（每个 比较 × 结局；旧规则）：
  一致     = 两边同向且都显著
  相反     = 方向相反且至少一边显著
  被否定   = 判官显著，医师不显著，且判官点估计落在医师 95% CI 之外
  幅度偏大 / 幅度偏小 = 两边同向显著，但判官点估计落在医师 95% CI 之外（判官绝对值更大 / 更小）
  无法判断 = 判官显著，医师不显著，判官点估计在医师 CI 之内（医师样本不足以检验）
  漏掉     = 医师显著，判官不显著
  都不显著 = 两边都不显著
「被否定」与「无法判断」要分开：医师金标只有 180 段，医师不显著常常只是样本不够。
⚠️ 判官未打分的复读轮不计入（gold_prevalence.judge_observed 同口径）；判官观测未做误判校正（校正值见各专门报告）。"""
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

import tools.gold_prevalence as GP  # noqa: E402
import tools.gold_prevalence_raked as GPR  # noqa: E402
from tools.judge_correction import load_frame_scores, sampling_cells  # noqa: E402
from tools.judge_select import load_gold  # noqa: E402

OUTCOMES = (GP.ANY, "d1", "d2", "d3", "d4", "d5")
PAIRS = (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))
CONTRASTS_ALL = [
    (("风险段", "高"), ("风险段", "低"), "高 − 低风险段（两臂合并）"),
    (("臂×风险段", "fixed", "高"), ("臂×风险段", "fixed", "低"), "高 − 低风险段（fixed 臂）"),
    (("组×风险段", "本地", "高"), ("组×风险段", "本地", "低"), "高 − 低风险段（本地模型）"),
    (("设计组", "旗舰"), ("组", "本地"), "旗舰 − 本地"),
    (("设计组", "陪伴"), ("设计组", "旗舰"), "陪伴 − 旗舰"),
    (("设计组", "陪伴"), ("组", "本地"), "陪伴 − 本地"),
    (("组", "API"), ("组", "本地"), "API（含陪伴）− 本地"),
    (("设计组", "心理微调"), ("设计组", "对照基座"), "心理微调组 − 对照基座组"),
] + [(("模型", a), ("模型", b), f"{a} − {b}") for a, b in PAIRS]
LOCAL_ONLY = {"高 − 低风险段（本地模型）", "心理微调组 − 对照基座组"} | {f"{a} − {b}" for a, b in PAIRS}
JUDGES = {"Qwen": "runs/v2/v7_all_qwen38_fill.jsonl", "DeepSeek": "runs/v2/a4_dsnt_pairs.jsonl"}
CATS = ("一致", "幅度偏大", "幅度偏小", "相反", "被否定", "无法判断", "漏掉", "都不显著")


# ---------------------------------------------------------------- 判官一侧
def judge_counts(path, meta) -> dict:
    """→ {人设: {子组: {结局: [失败数, 轮数]}}}（只计已打分的轮）。"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("intervention") or d.get("risk_none"):
                continue
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            per = d["case_profile"]["id"].split("-", 1)[0]
            for t in d["turns"]:
                s = t.get("score")
                tid = f'{base}#t{t["turn"]}'
                if not s or tid not in meta:
                    continue
                for g in GP.subgroups(meta, [tid]):
                    for o in OUTCOMES:
                        v = GP.failed(s, o)
                        if v is not None:
                            c = out[per][g][o]
                            c[0] += int(v)
                            c[1] += 1
    return out


def _rate(counts, personas, g, o):
    num = den = 0
    for p in personas:
        c = counts.get(p, {}).get(g, {}).get(o)
        if c:
            num += c[0]
            den += c[1]
    return num / den if den else None


def judge_contrasts(counts, contrasts, *, n_boot=2000, seed=0) -> dict:
    pers = sorted(counts)
    rng = random.Random(seed)
    draws = [[rng.choice(pers) for _ in pers] for _ in range(n_boot)]
    out = {}
    for a, b, lab in contrasts:
        for o in OUTCOMES:
            pa, pb = _rate(counts, pers, a, o), _rate(counts, pers, b, o)
            if pa is None or pb is None:
                continue
            bs = []
            for dr in draws:
                x, y = _rate(counts, dr, a, o), _rate(counts, dr, b, o)
                if x is not None and y is not None:
                    bs.append(x - y)
            bs.sort()
            ci = None if len(bs) < 50 else (bs[int(0.025 * (len(bs) - 1))], bs[int(0.975 * (len(bs) - 1))])
            out[(lab, o)] = (pa - pb, ci)
    return out


# ---------------------------------------------------------------- 分类
def _sig(ci):
    return bool(ci) and (ci[0] > 0 or ci[1] < 0)


def classify(j, jci, g, gci) -> str:
    """j / g = 判官 / 医师点估计；ci = (下, 上)。"""
    js, gs = _sig(jci), _sig(gci)
    outside = bool(gci) and (j < gci[0] or j > gci[1])
    if js and gs:
        if (j > 0) != (g > 0):
            return "相反"
        if not outside:
            return "一致"
        return "幅度偏大" if abs(j) > abs(g) else "幅度偏小"
    if js and not gs:
        if g is not None and g != 0 and (j > 0) != (g > 0) and outside:
            return "相反"
        return "被否定" if outside else "无法判断"
    if gs and not js:
        return "相反" if (j > 0) != (g > 0) and j != 0 else "漏掉"
    return "都不显著"


# ---------------------------------------------------------------- 主流程
def gold_diffs(ids, gold, cells, meta, contrasts, n_boot, seed):
    keyed = [(a, b) for a, b, _ in contrasts]
    lab = {(a, b): l for a, b, l in contrasts}
    old_gp, old_gpr = GP.CONTRASTS, GPR.CONTRASTS
    GP.CONTRASTS = GPR.CONTRASTS = tuple(keyed)
    try:
        r0 = GP.analyse(ids, gold, cells, Counter(cells.values()), meta, n_boot=n_boot, seed=seed)
        frame_ids = [i for i in cells if i in meta]
        rr = GPR.analyse(ids, gold, cells, meta, frame_ids, n_boot=n_boot, seed=seed)
    finally:
        GP.CONTRASTS, GPR.CONTRASTS = old_gp, old_gpr
    orig = {(lab[(a, b)], o): v for (a, b, o), v in r0["diffs"].items()}
    raked = {(lab[(a, b)], o): v for (a, b, o), v in rr["schemes"]["raked"]["diffs"].items()}
    n_gold = {lab[(a, b)]: (r0["point"].get(a, {}).get(GP.ANY, {}).get("n"), r0["point"].get(b, {}).get(GP.ANY, {}).get("n"))
              for a, b in keyed}
    return orig, raked, n_gold


def build(judge_res, orig, raked) -> list:
    rows = []
    for (lab, o), (j, jci) in judge_res.items():
        g0, g0ci = orig.get((lab, o), (None, None))
        g1, g1ci = raked.get((lab, o), (None, None))
        if g0 is None:
            continue
        rows.append({"contrast": lab, "outcome": o, "judge": j, "judge_ci": jci, "gold": g0, "gold_ci": g0ci,
                     "gold_raked": g1, "gold_raked_ci": g1ci,
                     "cat": classify(j, jci, g0, g0ci),
                     "cat_raked": None if g1 is None else classify(j, jci, g1, g1ci)})
    return rows


def summary(rows, key="cat") -> dict:
    c = Counter(r[key] for r in rows if r[key])
    jsig = [r for r in rows if _sig(r["judge_ci"])]
    held = sum(1 for r in jsig if r[key] == "一致")
    return {"counts": {k: c.get(k, 0) for k in CATS}, "n": sum(c.values()), "judge_sig": len(jsig),
            "judge_sig_held": held,
            "judge_sig_refuted": sum(1 for r in jsig if r[key] in ("被否定", "相反", "幅度偏大", "幅度偏小")),
            "judge_sig_undecidable": sum(1 for r in jsig if r[key] == "无法判断")}


def _pc(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def _ci(t):
    return "" if not t else f" [{100 * t[0]:+.1f}, {100 * t[1]:+.1f}]"


def render(res, meta) -> str:
    oname = {GP.ANY: "任一", "d1": "d1 有害", "d2": "d2 安全行动", "d3": "d3 识别风险", "d4": "d4 谄媚", "d5": "d5 敷衍"}
    L = ["# 判官 vs 医师：同一个问题的答案翻转表", "",
         f"- 生成 {meta['generated_at']}；医师金标 {meta['n_gold']} 段（原权重 / raking 权重）；判官 = 全语料观测（未校正）；"
         f"人设聚类自助 {meta['n_boot']} 次。",
         "- 分类（2026-10-26 起按效应之差 D = Δ判官 − Δ医师 的联合自助 CI）：一致 = 医师同向显著且 D 的 CI 含 0；"
         "相反 = 医师反向显著；幅度偏大 / 幅度偏小 = 医师同向显著但 D 的 CI 不含 0；被否定 = 医师不显著但 D 的 CI 不含 0；"
         "无法判断 = 医师不显著且 D 的 CI 含 0；漏掉 = 判官不显著、医师显著；都不显著。", ""]
    for judge, r in res.items():
        s0, s1 = r["summary"], r["summary_raked"]
        L += [f"## {judge} 判官", "",
              f"**判官显著的结论 {s0['judge_sig']} 条**：医师金标（原权重）下仍一致 **{s0['judge_sig_held']}**、"
              f"被否定 / 相反 / 幅度不符 **{s0['judge_sig_refuted']}**、医师样本不足以判断 {s0['judge_sig_undecidable']}。"
              f" raking 权重下：一致 {s1['judge_sig_held']}、被否定 / 相反 / 幅度不符 {s1['judge_sig_refuted']}、无法判断 {s1['judge_sig_undecidable']}。", "",
              "| 分类 | 原权重 | raking |", "|---|---|---|"]
        for k in CATS:
            L.append(f"| {k} | {s0['counts'][k]} | {s1['counts'][k]} |")
        L += ["", "| 比较 | 结局 | 判官（全语料观测） | 医师（原权重） | 分类 | 医师（raking） | 分类 |",
              "|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| {x['contrast']} | {oname[x['outcome']]} | {_pc(x['judge'])}{_ci(x['judge_ci'])} | "
                     f"{_pc(x['gold'])}{_ci(x['gold_ci'])} | {x['cat']} | {_pc(x['gold_raked'])}{_ci(x['gold_raked_ci'])} | "
                     f"{x['cat_raked'] or '—'} |")
        L.append("")
    L += ["## 每个比较的医师金标段数（任一；两组）", ""]
    for lab, (na, nb) in res[next(iter(res))]["n_gold"].items():
        L.append(f"- {lab}：{na} / {nb}")
    return "\n".join(L) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def map_joint(cat, j, g) -> str:
    """联合检验的新类名 → 旧类名（下游图表沿用旧类名）。"""
    if cat == "确认":
        return "一致"
    if cat == "方向相反":
        return "相反"
    if cat == "幅度不符（医师同向显著）":
        return "幅度偏大" if abs(j) > abs(g) else "幅度偏小"
    if cat == "幅度不符（医师不显著）":
        return "被否定"
    if cat == "判官漏掉":
        return "相反" if j != 0 and (j > 0) != (g > 0) else "漏掉"
    return cat                                              # 无法判断 / 都不显著


def gold_n(ids, gold, meta, contrasts) -> dict:
    sg = GP.subgroups(meta, ids)
    cnt = lambda key: sum(1 for i in sg.get(key, []) if GP.failed(gold[i], GP.ANY) is not None)
    return {lab: (cnt(a), cnt(b)) for a, b, lab in contrasts}


def main(argv=None) -> int:
    import tools.judge_vs_gold_joint as JGJ
    ap = argparse.ArgumentParser(description="判官 vs 医师翻转表（效应之差的联合自助检验）")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rel = lambda p: str(REPO / p)
    rows = [g for p in ("gold_judge180/gold_dev.jsonl", "gold_judge180/gold_test.jsonl") for g in load_gold(rel(p))]
    ids = [g["id"] for g in rows]
    gold = {g["id"]: g["consensus"] for g in rows}
    cells = sampling_cells(load_frame_scores(rel("runs/v2/scored_v6.jsonl")), load_frame_scores(rel("runs/v2/scored_v5.jsonl")))
    meta = GP.turn_meta(rel("runs/v2/dialogues_both_arms.jsonl"))
    frame_ids = [i for i in cells if i in meta]
    judges = {}
    for judge, path in JUDGES.items():
        cons = CONTRASTS_ALL if judge == "Qwen" else [c for c in CONTRASTS_ALL if c[2] in LOCAL_ONLY]
        judges[judge] = (judge_counts(rel(path), meta), cons)
    t0 = time.time()
    jr = JGJ.analyse(ids, gold, cells, meta, frame_ids, judges, n_boot=a.n_boot, seed=a.seed, log=print)
    print(f"联合自助完成（{time.time() - t0:.0f}s）")
    n_gold = gold_n(ids, gold, meta, CONTRASTS_ALL)
    res = {}
    for judge, r in jr.items():
        rws = []
        for x in r["rows"]:
            o, k = x["orig"], x["raked"]
            rws.append({"contrast": x["contrast"], "outcome": x["outcome"], "judge": x["judge"], "judge_ci": x["judge_ci"],
                        "gold": o["gold"], "gold_ci": o["gold_ci"], "gold_raked": k["gold"], "gold_raked_ci": k["gold_ci"],
                        "D": o["D"], "D_ci": o["D_ci"], "D_p_holm": o["D_p_holm"],
                        "D_raked": k["D"], "D_raked_ci": k["D_ci"], "D_raked_p_holm": k["D_p_holm"],
                        "cat_joint": o["cat"], "cat_joint_raked": k["cat"],
                        "cat": map_joint(o["cat"], x["judge"], o["gold"]),
                        "cat_raked": None if k["gold"] is None else map_joint(k["cat"], x["judge"], k["gold"])})
        res[judge] = {"rows": rws, "summary": summary(rws), "summary_raked": summary(rws, "cat_raked"), "n_gold": n_gold,
                      "joint_summary": r["summary"]}
    meta_info = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_gold": len(ids), "n_boot": a.n_boot,
                 "method": "joint persona bootstrap; classification by CI of D = judge diff − psychiatrist diff"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_vs_gold.json").write_text(json.dumps(_jsonable({"meta": meta_info, **res}), ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (out / "judge_vs_gold.md").write_text(render(res, meta_info), encoding="utf-8")
    for judge, r in res.items():
        print(judge, r["summary"], r["summary_raked"])
    print(f"报告 {out / 'judge_vs_gold.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
