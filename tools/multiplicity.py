"""多重比较校正：把各报告里已出的效应量 + CI 收成检验家族，主假设族 Holm、次要族 BH-FDR，另做全体 BH 作敏感性。零模型调用。

    python tools/multiplicity.py --out reports/multiplicity

p 值来源（逐条注明）：
  * 报告里已有 p 的（配对 Wilcoxon，按人设 n=16）→ 直接用；
  * 只有自助法 CI 的 → 由 CI 反推（Altman & Bland 2011）：SE = 靠近 0 那一侧的半宽 / 1.96，z = 估计 / SE，
    双侧正态 p。用近 0 一侧保证「CI 不含 0」⇔「p < 0.05」，与原报告的显著判定一致。
  * C 评估器剂量反应（最高档 − 最低档）原报告只有各档单独 CI → 本工具按人设配对重抽补算差值 CI。
家族划分（写进论文方法；不是事先注册的，须如实说明是写作前按「摘要 / 主文核心结论」划定）：
  主假设按研究问题分三族（各自 Holm）：P1 剂量反应 / P2 心理微调 / P3 旗舰 vs 本地；
  另给「三族合并成一族 Holm」的严格版本作敏感性（合并后 P1 不过，须如实报告）；
  ⚠️ 2026-10-17 改（用户定，方法须如实写这次调整）：
    ① 原 P1 = 医师金标两臂合并「高 − 低」+ C 评估器旗舰抛弃率剂量反应。前者混了臂的构成（低段金标多来自 responsive 臂，
       而 responsive 臂的档位是模型行为的后果；fixed 臂金标只有 13 个低段轮次 → +11.8 [−26.5, 50.6]），后者长度误判校正后跨 0。
    ② 现 P1 = 六维判官（Qwen）本地模型 fixed 臂「任一严重失职」高 − 低，长度误判校正（tools/judge_dose.py）。
       医师金标两臂合并 / fixed 臂 / responsive 臂进 S1 作描述；C 评估器观测与校正进 S2；其余判官剂量反应（含 DeepSeek 复现）进 S8。
    ③ P3 改为按研究设计的组：旗舰（3 个，不含陪伴模型 MiniMax）− 本地（6 个）。原「API − 本地」按访问方式分，
       API 组同时含 GPT / Claude（金标 0 失败）与 MiniMax（金标失败最多），没有临床意义 → 降到 S1；陪伴 − 旗舰等进 S1。
  次要族（各自 BH）= 分维、子组、复现、干预、无风险；最后再全体 BH 作敏感性。
六维判官：用长度校正后的差；不可校正的格（全语料任一长度段 Youden < 0.5）不进检验，只作描述。
d5（敷衍回应）一律用观测差（2026-10-17 定；理由见 tools/length_adjusted_contrast 模块说明）。"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DIMS5 = ("d1", "d2", "d3", "d4", "d5")
PAIRS = (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))
SHORT = {"cpsycoun": "CPsyCounX", "internlm2-base": "InternLM2", "mindchat": "MindChat", "qwen15-4b-base": "Qwen1.5",
         "soulchat": "SoulChat", "qwen3-8b-base": "Qwen3-8B", "deepseek/deepseek-v4-pro": "deepseek"}
DIM_NAME = {"d1": "d1 有害内容", "d2": "d2 未给可用帮助", "d3": "d3 未识别风险", "d4": "d4 扭曲附和",
            "d5": "d5 敷衍回应", "any": "任一严重失职"}
P1, P2, P3 = "P1 剂量反应（fixed 臂）", "P2 心理微调配对消融", "P3 旗舰（3 个）vs 本地"
PRIMARY = (P1, P2, P3)
FAMILIES = [(P1, "holm"), (P2, "holm"), (P3, "holm"), ("S1 医师金标：分维与子组", "bh"), ("S2 C 评估器剂量反应（观测 / 误判校正）", "bh"),
            ("S3 配对消融（Qwen 判官）：其余维与 responsive 臂", "bh"), ("S4 配对消融跨家族复现（deepseek 判官）", "bh"),
            ("S5 A6-S 系统提示干预", "bh"), ("S6 A6-B 反事实分支", "bh"), ("S7 A7 无风险对照", "bh"),
            ("S8 六维判官剂量反应旁证（本地模型，长度校正）", "bh")]


# ---------------------------------------------------------------- 统计
def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def p_from_ci(est: float, ci) -> float | None:
    """CI 反推双侧 p（靠近 0 一侧的半宽当 1.96 SE）；退化（半宽 0 且估计非 0）→ None。"""
    lo, hi = ci
    if est == 0:
        return 1.0
    half = est - lo if est > 0 else hi - est
    if half <= 0:
        return None
    z = abs(est) / (half / 1.96)
    return max(0.0, min(1.0, 2 * (1 - _phi(z))))


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def bh(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adj, run = [0.0] * m, 1.0
    for k, i in enumerate(order):
        run = min(run, ps[i] * m / (m - k))
        adj[i] = run
    return adj


def adjust(ps, method):
    return holm(ps) if method == "holm" else bh(ps)


# ---------------------------------------------------------------- 读报告
def _j(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test(family, label, est, ci, *, p=None, source, note=""):
    return {"family": family, "label": label, "est": est, "ci": list(ci) if ci else None, "p_reported": p,
            "source": source, "note": note}


def gold_tests(path):
    d = _j(path)["diffs"]
    src = path
    out = []

    def key(a, b, k):
        return f"{a}|{b}|{k}"
    hi_lo = ("('风险段', '高')", "('风险段', '低')")
    api_loc = ("('组', 'API')", "('组', '本地')")
    flag_loc = ("('设计组', '旗舰')", "('组', '本地')")
    e, c = d[key(*hi_lo, "any")]
    out.append(test(FAMILIES[3][0], "医师金标：高 − 低风险段，任一严重失职（两臂合并）", e, c, source=src,
                    note="混了臂的构成，不作剂量反应证据"))
    for arm in ("fixed", "responsive"):
        k = key(f"('臂×风险段', '{arm}', '高')", f"('臂×风险段', '{arm}', '低')", "any")
        if k in d and d[k][1]:
            out.append(test(FAMILIES[3][0], f"医师金标：{arm} 臂内 高 − 低风险段，任一严重失职", *d[k], source=src,
                            note="fixed 臂低段仅 13 轮" if arm == "fixed" else "responsive 臂档位是后果（对撞因子）"))
    e, c = d[key(*flag_loc, "any")]
    out.append(test(P3, "医师金标：旗舰（3 个）− 本地（6 个），任一严重失职", e, c, source=src,
                    note="旗舰的失败全部来自 DeepSeek；GPT、Claude 金标 0 失败"))
    fam = FAMILIES[3][0]
    e, c = d[key(*api_loc, "any")]
    out.append(test(fam, "医师金标：API（含陪伴）− 本地，任一严重失职", e, c, source=src, note="按访问方式分，只作描述"))
    for a, b, lab in ((("('设计组', '陪伴')", "('设计组', '旗舰')"), None, "陪伴 − 旗舰"),
                      (("('设计组', '陪伴')", "('组', '本地')"), None, "陪伴 − 本地"),
                      (("('设计组', '心理微调')", "('设计组', '对照基座')"), None, "心理微调组 − 对照基座组")):
        k = key(*a, "any")
        if k in d and d[k][1]:
            out.append(test(fam, f"医师金标：{lab}，任一严重失职", *d[k], source=src,
                            note="陪伴模型金标仅 12 段" if "陪伴" in lab else "组间合并，配对差见 P2"))
    for dim in DIMS5:
        out.append(test(fam, f"高 − 低风险段，{DIM_NAME[dim]}", *d[key(*hi_lo, dim)], source=src))
        out.append(test(fam, f"旗舰 − 本地，{DIM_NAME[dim]}", *d[key(*flag_loc, dim)], source=src))
    for g, gn in (("本地", "本地"), ("API", "旗舰")):
        a, b = f"('组×风险段', '{g}', '高')", f"('组×风险段', '{g}', '低')"
        for dim in ("any",) + DIMS5:
            k = key(a, b, dim)
            if k in d and d[k][1]:
                out.append(test(fam, f"{gn}内 高 − 低风险段，{DIM_NAME[dim]}", *d[k], source=src,
                                note="子组小，只看方向" if g == "API" else ""))
    return out


def ceval_dose(path, n_boot=2000, seed=0):
    """fixed 臂各组：手段与时机档 − 被动意念档的轮级率差，人设配对重抽（两档用同一次人设抽样）。"""
    from cascade.analysis.groups import group_of
    from cascade.io import read_jsonl
    lo_st, hi_st = "passive_ideation", "means_timing"
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))     # (组, 指标) → 人设 → 档 → 值
    for r in read_jsonl(path):
        if r.arm.value != "fixed":
            continue
        g = group_of(r.target_model_id)
        per = r.case_profile.persona_id or r.case_profile.id.split("-")[0]
        for t in r.turns:
            st = t.attack_stage.value
            if st in (lo_st, hi_st):
                for gg in (g, "all"):
                    cells[(gg, "abandon")][per][st].append(float(t.appraisal.abandonment_felt))
                    cells[(gg, "empathy")][per][st].append(float(t.appraisal.empathy_detected))

    def diff(by_p, plist):
        hi = [x for p in plist for x in by_p[p][hi_st]]
        lo = [x for p in plist for x in by_p[p][lo_st]]
        return None if not hi or not lo else sum(hi) / len(hi) - sum(lo) / len(lo)
    out = {}
    rng = random.Random(seed)
    for k, by_p in cells.items():
        pers = sorted(by_p)
        est = diff(by_p, pers)
        bs = sorted(v for v in (diff(by_p, [rng.choice(pers) for _ in pers]) for _ in range(n_boot)) if v is not None)
        out[k] = (est, (bs[int(0.025 * (len(bs) - 1))], bs[int(0.975 * (len(bs) - 1))]))
    return out


def ceval_dose_tests(path, n_boot):
    res = ceval_dose(path, n_boot)
    name = {"A_flagship": "旗舰", "B2_psych": "心理微调", "C_base": "对照基座", "companion": "陪伴", "all": "全部 10 个模型"}
    out = []
    for (g, m), (e, c) in sorted(res.items()):
        lab = f"C 评估器 fixed 臂 {name.get(g, g)}：{'抛弃率' if m == 'abandon' else '共情率'} 手段与时机档 − 被动意念档"
        out.append(test(FAMILIES[4][0], lab + "（观测）", e, c, source=f"{path}（本工具按人设配对重抽）",
                        note="观测值，未做误判校正"))
    return out


def misclass_dose_tests(path):
    """C 评估器剂量反应的误判校正值（只取抛弃维；共情 / 降险落在低 Youden 带，只在报告里作敏感性）。"""
    name = {"A_flagship": "旗舰", "B2_psych": "心理微调", "C_base": "对照基座", "companion": "陪伴", "all": "全部 10 个模型"}
    out = []
    for g, dd in _j(path).get("dose", {}).items():
        c = dd["abandon"]["corrected"]
        out.append(test(FAMILIES[4][0], f"C 评估器 fixed 臂 {name.get(g, g)}：抛弃率 手段与时机档 − 被动意念档（误判校正）",
                        c["diff"], c["ci"], source=path, note=f"低 Youden 占比 {c['low_youden_share']}"))
    return out


def judge_dose_tests(path, judge):
    """六维判官剂量反应旁证：本地模型、fixed 臂、高 − 低；可校正的维用长度校正差，d5 用观测差，不可校正的不进检验。"""
    if not Path(REPO / path).exists():
        return []
    d = _j(path)
    out = []
    for o in ("any", "d1", "d2", "d3", "d4", "d5"):
        k = f"本地|fixed|{o}"
        if k not in d["point"]:
            continue
        raw, cor = d["point"][k]["diff"]
        craw, ccor = d["ci"][k]
        est, ci, how = (raw, craw, "观测") if o == "d5" else (cor, ccor, "长度校正")
        if est is None or not ci:
            continue
        fam = P1 if (judge, o) == ("Qwen", "any") else FAMILIES[10][0]
        out.append(test(fam, f"六维判官（{judge}）本地模型 fixed 臂 高 − 低风险段，{DIM_NAME[o]}（{how}）",
                        est, ci, source=path, note=f"观测 {100 * raw:+.1f}%"))
    return out


def misclass_tests(path):
    d = _j(path)["contrasts"]
    out = []
    for k, v in d.items():
        c = v["abandon"]["corrected"]
        if k.startswith("配对"):
            a, b = k.replace("配对 ", "").split(" − ")
            out.append(test(P2, f"C 评估器抛弃率（误判校正）：{SHORT[a]} − {SHORT[b]}", c["diff"], c["ci"], source=path))
        else:
            m = k.split(" ", 1)[1].split(":")[0]
            out.append(test(FAMILIES[7][0], f"C 评估器抛弃率（误判校正）：{SHORT.get(m, m)} 干预 − 对照", c["diff"], c["ci"],
                            source=path))
    return out


def length_adj_tests(path, judge):
    d = _j(path)
    out = []
    for a, b in PAIRS:
        for arm in ("fixed", "responsive"):
            for dim in DIMS5:
                k = f"{a}|{b}|{arm}|{dim}"
                raw, cor = d["point"][k]
                craw, ccor = d["ci"][k]
                if dim == "d5":                                  # 只报观测
                    est, ci, how = raw, craw, "观测"
                else:
                    est, ci, how = cor, ccor, "长度校正"
                if est is None or not ci:
                    continue
                if judge == "qwen":
                    fam = P2 if arm == "fixed" and dim in ("d1", "d4") else FAMILIES[5][0]
                else:
                    if arm != "fixed":
                        continue
                    fam = FAMILIES[6][0]
                out.append(test(fam, f"六维 {DIM_NAME[dim]}（{how}，{judge} 判官，{arm}）：{SHORT[a]} − {SHORT[b]}",
                                est, ci, source=path, note=f"观测 {100 * raw:+.1f}% [{100 * craw[0]:+.1f}, {100 * craw[1]:+.1f}]"))
    return out


def six_dim_tests(path):
    rows = _j(path)["rows"]
    fam = {"①": FAMILIES[7][0], "③": FAMILIES[8][0], "②": FAMILIES[9][0]}
    out, skipped = [], []
    for r in rows:
        if r["q"] == "③" and r["A"] == "br_regen":       # regen − orig 是机制说明，另由 C 评估器那条检验
            continue
        dims = ("d1", "d4", "d5") if r["q"] == "②" else DIMS5
        for dim in dims:
            x = r["dims"][dim]
            if dim == "d5":                                      # 只报观测
                dc, ccor, how = x["point"][2], x["ci_raw"], "观测"
            else:
                dc, ccor, how = x["point"][5], x["ci_corrected"], "长度校正"
            if dc is None or not ccor:
                skipped.append(f"{r['title']} {dim}")
                continue
            out.append(test(fam[r["q"]], f"六维 {DIM_NAME[dim]}（{how}）：{r['title']}", dc, ccor, source=path))
    return out, skipped


def branch_tests(path):
    d = _j(path)["paired"]
    out = []
    for m, v in d.items():
        x = v["intv_minus_regen"]["abandon"]
        out.append(test(FAMILIES[8][0], f"C 评估器抛弃率：{SHORT.get(m, m)} 分支 干预 − 只重生", x["mean_diff"], x["ci"],
                        p=x["p"], source=path))
    src = "reports/intervention_v2/diagnostics.md §3（三模型合并，按人设配对 Wilcoxon）"
    out.append(test(FAMILIES[8][0], "C 评估器抛弃率：三模型合并 分支 干预 − 只重生（净效应）", -0.070, (-0.12, -0.01), p=0.044, source=src))
    out.append(test(FAMILIES[8][0], "C 评估器抛弃率：三模型合并 只重生 − 原回复（回归均值）", -0.143, (-0.20, -0.09), p=0.001, source=src))
    return out


# ---------------------------------------------------------------- 汇总
def run(tests: list[dict]) -> dict:
    for t in tests:
        t["p_from_ci"] = None if not t["ci"] else p_from_ci(t["est"], t["ci"])
        t["p"] = t["p_reported"] if t["p_reported"] is not None else t["p_from_ci"]
        t["p_source"] = "报告" if t["p_reported"] is not None else "CI 反推"
        # 与原报告同一判定：有 CI 的按「CI 严格不含 0」，否则按 p < 0.05
        t["sig_raw"] = (t["ci"][0] > 0 or t["ci"][1] < 0) if t["ci"] else (t["p"] is not None and t["p"] < 0.05)
    usable = [t for t in tests if t["p"] is not None]
    for name, method in FAMILIES:
        fam = [t for t in usable if t["family"] == name]
        for t, q in zip(fam, adjust([t["p"] for t in fam], method)):
            t["p_adj"], t["method"] = q, method
    for t, q in zip(usable, bh([t["p"] for t in usable])):
        t["q_global"] = q
    prim = [t for t in usable if t["family"] in PRIMARY]
    for t, q in zip(prim, holm([t["p"] for t in prim])):
        t["p_strict"] = q
    return {"tests": tests, "n_usable": len(usable)}


def _pc(x):
    return f"{100 * x:+.1f}%"


def _verdict(t):
    if t["p"] is None:
        return "p 无法计算"
    if not t["sig_raw"]:
        return "—（原本就不显著）"
    return "✅ 成立" if t["p_adj"] < 0.05 else "❌ 校正后不成立"


def render(res, meta, skipped) -> str:
    T = res["tests"]
    prim = [t for t in T if t["family"] in PRIMARY and t["p"] is not None]
    L = ["# 多重比较校正", "",
         f"- 生成 {meta['generated_at']}；共 {len(T)} 个检验，可算 p 的 {res['n_usable']} 个。",
         "- 主假设按研究问题分三族（P1 剂量反应 / P2 心理微调 / P3 旗舰 vs 本地），各自 Holm；次要族各自 BH-FDR。",
         "- 两个敏感性版本：「主假设合并 Holm」= 三族 12 个检验并成一族；「全体 BH」= 全部检验合并做 BH。",
         "- p 值：报告里已有的（配对 Wilcoxon，人设 n=16）直接用；只有自助法 CI 的按 CI 反推（Altman & Bland 2011，靠近 0 一侧半宽 = 1.96 SE）。",
         "- 「原本显著」= 与原报告同一判定（CI 严格不含 0）。",
         "- ⚠️ 家族划分不是事先注册的，是写作前按研究问题划定的，论文须如实写明，并同时报告严格版本。", ""]
    lost = [t for t in T if t["p"] is not None and t["sig_raw"] and t["p_adj"] >= 0.05]
    L += ["## 总览", "", f"- 原本显著 {sum(t['sig_raw'] for t in T)} 个 → 族内校正后仍成立 "
          f"{sum(t['sig_raw'] and t['p_adj'] < 0.05 for t in T if t['p'] is not None)} 个；全体 BH 后仍成立 "
          f"{sum(t['sig_raw'] and t.get('q_global', 1) < 0.05 for t in T)} 个。",
          f"- 主假设：原本显著 {sum(t['sig_raw'] for t in prim)} 个 → 分族 Holm 后 "
          f"{sum(t['sig_raw'] and t['p_adj'] < 0.05 for t in prim)} 个、合并 Holm 后 "
          f"{sum(t['sig_raw'] and t['p_strict'] < 0.05 for t in prim)} 个。", ""]
    strict_lost = [t for t in prim if t["sig_raw"] and t["p_adj"] < 0.05 <= t["p_strict"]]
    if strict_lost:
        L += ["**分族成立、但主假设合并 Holm 后不成立：**", ""] + [
            f"- [{t['family']}] {t['label']}：{_pc(t['est'])}，p = {t['p']:.3f} → 分族 {t['p_adj']:.3f} / 合并 {t['p_strict']:.3f}"
            for t in strict_lost] + [""]
    if lost:
        L += ["**族内校正后不再成立的结论：**", ""] + [
            f"- [{t['family']}] {t['label']}：{_pc(t['est'])}，p = {t['p']:.3f} → 校正后 {t['p_adj']:.3f}" for t in lost] + [""]
    for name, method in FAMILIES:
        fam = [t for t in T if t["family"] == name]
        if not fam:
            continue
        is_p = name in PRIMARY
        L += [f"## {name}（{'Holm' if method == 'holm' else 'BH-FDR'}，{len(fam)} 个检验）", "",
              "| 结论 | 估计 [95% CI] | p（来源） | 族内校正后 | " + ("主假设合并 Holm | " if is_p else "")
              + "全体 BH | 判定 | 备注 |", "|---|---|---|---|" + ("---|" if is_p else "") + "---|---|---|"]
        for t in fam:
            ci = f" [{_pc(t['ci'][0])}, {_pc(t['ci'][1])}]" if t["ci"] else ""
            ok = t["p"] is not None
            p = f"{t['p']:.4f}（{t['p_source']}）" if ok else "—"
            pa = f"{t['p_adj']:.4f}" if ok else "—"
            ps = (f"{t['p_strict']:.4f} | " if ok else "— | ") if is_p else ""
            qg = f"{t['q_global']:.4f}" if ok else "—"
            L.append(f"| {t['label']} | {_pc(t['est'])}{ci} | {p} | {pa} | {ps}{qg} | {_verdict(t)} | {t['note']} |")
        L.append("")
    if skipped:
        L += ["## 不进检验的格（不可校正，只作描述）", "", "、".join(skipped), ""]
    L += ["## 来源", ""] + [f"- {s}" for s in sorted({t['source'] for t in T})]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多重比较校正")
    ap.add_argument("--dialogues", default="runs/v2/dialogues_both_arms.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    six, skipped = six_dim_tests("reports/six_dim_questions/questions.json")
    tests = (gold_tests("reports/gold_prevalence/prevalence.json")
             + ceval_dose_tests(a.dialogues, a.n_boot)
             + misclass_dose_tests("reports/misclass_correction/results.json")
             + judge_dose_tests("reports/judge_dose_v7/judge_dose.json", "Qwen")
             + judge_dose_tests("reports/a4_deepseek/judge_dose/judge_dose.json", "DeepSeek")
             + misclass_tests("reports/misclass_correction/results.json")
             + length_adj_tests("reports/length_adjusted_v7/length_adjusted.json", "qwen")
             + length_adj_tests("reports/a4_deepseek/length_adjusted/length_adjusted.json", "deepseek")
             + six + branch_tests("reports/branch_v2/report.json"))
    order = {n: i for i, (n, _) in enumerate(FAMILIES)}
    tests.sort(key=lambda t: order[t["family"]])
    res = run(tests)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_boot": a.n_boot}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "multiplicity.md").write_text(render(res, meta, skipped), encoding="utf-8")
    (out / "multiplicity.json").write_text(json.dumps({"meta": meta, **res, "skipped": skipped}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"报告 {out / 'multiplicity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
