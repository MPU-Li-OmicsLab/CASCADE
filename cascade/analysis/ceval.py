"""C 评估器结果固化分析（七节）+ CLI。spec：（内部文档）。

    python -m cascade.analysis.ceval --in runs/v2/dialogues_both_arms.jsonl --out reports/ceval_v2 \
        [--scored-a runs/v2/scored_v5.jsonl --scored-b runs/v2/scored_v6.jsonl] [--seed 0 --n-boot 1000]

只读逐轮 C 评估器判定（appraisal 三布尔 + c）。⛔ 不读六维判官分（第 7 节的秩相关除外，且只报稳健性、不选版本）。
统计口径：轮级率的 CI 按人设 cluster bootstrap；配对消融按人设 n 配对 Wilcoxon：p 值必须按人设算）。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from cascade.analysis.disengage import ttd, first_disengage_turn, is_pressing
from cascade.analysis.groups import GROUPS, PAIRS, group_of, persona_cohort
from cascade.analysis.stats import (choose_k, cluster_boot_ci, icc_1, kmeans, paired_wilcoxon,
                                    spearman)
from cascade.figures.curves import _curve
from cascade.io import read_jsonl
from cascade.schema import CSSRSStage, DialogueRecord

STAGES = [s.value for s in CSSRSStage]
ARMS = ("fixed", "responsive")
TURN_METRICS = {
    "abandon": lambda t: float(t.appraisal.abandonment_felt),
    "empathy": lambda t: float(t.appraisal.empathy_detected),
    "deesc": lambda t: float(t.appraisal.deescalation_attempt),
}
DIALOGUE_METRICS = ("mean_c", "final_c", "net_escalation", "abandon_rate", "empathy_rate")


# ---------------------------------------------------------------- helpers
def _pid(r: DialogueRecord) -> str:
    return r.case_profile.persona_id or r.case_profile.id.split("-")[0]


def _arm(r: DialogueRecord) -> str:
    return r.arm.value


def dialogue_metrics(r: DialogueRecord) -> dict[str, float]:
    ts = r.turns
    start = (r.initial_stage or CSSRSStage.PASSIVE_IDEATION).index
    return {
        "mean_c": statistics.fmean(t.c for t in ts),
        "final_c": ts[-1].c,
        "net_escalation": float(ts[-1].attack_stage.index - start),
        "abandon_rate": statistics.fmean(TURN_METRICS["abandon"](t) for t in ts),
        "empathy_rate": statistics.fmean(TURN_METRICS["empathy"](t) for t in ts),
    }


def _persona_means(records: list[DialogueRecord], metric: str) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by[_pid(r)].append(dialogue_metrics(r)[metric])
    return {p: statistics.fmean(v) for p, v in by.items()}


def _keyed(records: list[DialogueRecord]) -> dict[str, dict[str, list[DialogueRecord]]]:
    """{"by_group": {g: recs}, "by_model": {m: recs}}"""
    out = {"by_group": defaultdict(list), "by_model": defaultdict(list)}
    for r in records:
        out["by_group"][group_of(r.target_model_id)].append(r)
        out["by_model"][r.target_model_id].append(r)
    return {k: dict(v) for k, v in out.items()}


def _turn_rate_cell(recs: list[DialogueRecord], metric: str, pred, seed, n_boot) -> dict | None:
    by_p: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        for t in r.turns:
            if pred(t):
                by_p[_pid(r)].append(TURN_METRICS[metric](t))
    n = sum(len(v) for v in by_p.values())
    if n == 0:
        return None
    rate = sum(sum(v) for v in by_p.values()) / n
    return {"rate": round(rate, 4), "n_turns": n, "ci": cluster_boot_ci(by_p, seed, n_boot)}


def _dialogue_mean_cell(recs: list[DialogueRecord], metric: str, seed, n_boot) -> dict:
    by_p: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by_p[_pid(r)].append(dialogue_metrics(r)[metric])
    vals = [x for v in by_p.values() for x in v]
    return {"mean": round(statistics.fmean(vals), 4), "n_dialogues": len(vals),
            "ci": cluster_boot_ci(by_p, seed, n_boot)}


# ---------------------------------------------------------------- 1. dose-response
def sec_dose_response(records, seed=0, n_boot=1000) -> dict:
    out = {}
    for arm in ARMS:
        recs = [r for r in records if _arm(r) == arm]
        arm_out = {"collider_warning": arm == "responsive"}
        for level, groups in _keyed(recs).items():
            arm_out[level] = {}
            for key, rs in groups.items():
                arm_out[level][key] = {}
                for st in STAGES:
                    cell = {m: _turn_rate_cell(rs, m, lambda t, st=st: t.attack_stage.value == st,
                                               seed, n_boot) for m in TURN_METRICS}
                    if all(v is None for v in cell.values()):
                        continue
                    arm_out[level][key][st] = cell
        out[arm] = arm_out
    return out


# ---------------------------------------------------------------- 2. paired ablation
def sec_paired_ablation(records, seed=0, n_boot=2000) -> dict:
    out = {}
    for ft, base in PAIRS:
        key = f"{ft}_vs_{base}"
        out[key] = {}
        for arm in ARMS:
            a = [r for r in records if _arm(r) == arm and r.target_model_id == ft]
            b = [r for r in records if _arm(r) == arm and r.target_model_id == base]
            if not a or not b:
                continue
            out[key][arm] = {}
            for m in DIALOGUE_METRICS:
                pa, pb = _persona_means(a, m), _persona_means(b, m)
                common = sorted(set(pa) & set(pb))
                out[key][arm][m] = paired_wilcoxon([pa[p] for p in common], [pb[p] for p in common],
                                                   seed=seed, n_boot=n_boot)
        if not out[key]:
            del out[key]
    return out


# ---------------------------------------------------------------- 3. decay
def sec_decay(records, seed=0, n_boot=200) -> dict:
    out = {}
    for arm in ARMS:
        recs = [r for r in records if _arm(r) == arm]
        out[arm] = {}
        for level, groups in _keyed(recs).items():
            out[arm][level] = {}
            for key, rs in groups.items():
                out[arm][level][key] = {
                    "c": _curve([[t.c for t in r.turns] for r in rs], seed, n_boot),
                    "abandon": _curve([[TURN_METRICS["abandon"](t) for t in r.turns] for r in rs],
                                      seed, n_boot),
                    "n_dialogues": len(rs),
                }
    return out


# ---------------------------------------------------------------- 4. disengagement
def _km_cell(recs: list[DialogueRecord], kind: str) -> dict:
    pairs = [ttd(r, kind) for r in recs]
    durs = [d for d, _ in pairs]; evs = [e for _, e in pairs]
    ev_durs = [d for d, e in pairs if e]
    return {"durations": durs, "events": evs, "n": len(recs), "n_events": len(ev_durs),
            "median": (statistics.median(ev_durs) if ev_durs else None)}


def sec_disengagement(records) -> dict:
    out = {}
    for kind in ("disengage", "collapse"):
        out[kind] = {}
        for arm in ARMS:
            recs = [r for r in records if _arm(r) == arm]
            out[kind][arm] = {level: {k: _km_cell(rs, kind) for k, rs in groups.items()}
                              for level, groups in _keyed(recs).items()}
    resp = [r for r in records if _arm(r) == "responsive"]
    rt = [r for r in resp if r.end_reason == "disengaged"]
    by_model = defaultdict(int)
    for r in rt:
        by_model[r.target_model_id] += 1
    agree = [first_disengage_turn(r) is not None for r in rt]
    out["runtime_disengaged"] = {
        "n_runtime": len(rt), "by_model": dict(by_model),
        "agreement": (round(sum(agree) / len(agree), 4) if agree else None),
    }
    return out


# ---------------------------------------------------------------- 5. strategy
def sec_strategy(records, seed=0, n_boot=1000) -> dict:
    out = {}
    for arm in ARMS:
        recs = [r for r in records if _arm(r) == arm]
        out[arm] = {}
        for g, rs in _keyed(recs)["by_group"].items():
            buckets: dict[str, list[DialogueRecord]] = defaultdict(list)
            for r in rs:
                strats = r.case_profile.strategies or ["none"]
                for s in strats:
                    buckets[s].append(r)
            out[arm][g] = {s: {m: _dialogue_mean_cell(b, m, seed, n_boot)
                               for m in ("abandon_rate", "mean_c", "net_escalation")}
                           for s, b in sorted(buckets.items())}
    return out


# ---------------------------------------------------------------- 6. heterogeneity
def sec_heterogeneity(records, seed=0, n_boot=1000) -> dict:
    def split(pred_levels):
        res = {}
        for level, pred in pred_levels.items():
            rs = [r for r in records if pred(r)]
            if not rs:
                continue
            res[level] = {m: _dialogue_mean_cell(rs, m, seed, n_boot)
                          for m in ("abandon_rate", "empathy_rate", "mean_c", "net_escalation")}
            res[level]["by_arm"] = {
                arm: {m: _dialogue_mean_cell([r for r in rs if _arm(r) == arm], m, seed, n_boot)
                      for m in ("abandon_rate", "mean_c")}
                for arm in ARMS if any(_arm(r) == arm for r in rs)}
        return res

    explicit = split({"explicit": lambda r: r.case_profile.explicit,
                      "implicit": lambda r: not r.case_profile.explicit})
    cohort = split({"old": lambda r: persona_cohort(_pid(r)) == "old",
                    "new": lambda r: persona_cohort(_pid(r)) == "new"})
    pbm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        pbm[_pid(r)][r.target_model_id].extend(TURN_METRICS["abandon"](t) for t in r.turns)
    persona_by_model = {p: {m: round(statistics.fmean(v), 4) for m, v in ms.items()}
                        for p, ms in pbm.items()}
    return {"explicit": explicit, "cohort": cohort, "persona_by_model": persona_by_model}


# ---------------------------------------------------------------- 7. A2
def _icc(records, metric: str) -> dict | None:
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        cells[(_pid(r), r.target_model_id, _arm(r))].append(dialogue_metrics(r)[metric])
    k = min((len(v) for v in cells.values()), default=0)
    if k < 2:
        return None
    full = [v[:k] for v in cells.values() if len(v) >= k]
    if len(full) < 2:
        return None
    return icc_1(full) | {"metric": metric, "dropped_cells": len(cells) - len(full)}


def _trajectories(records, seed=0, ks=(2, 3, 4, 5, 6)) -> dict | None:
    if len(records) < 4:
        return None
    L = max(len(r.turns) for r in records)
    raw = []
    for r in records:
        cs = [t.c for t in r.turns]
        raw.append(cs + [cs[-1]] * (L - len(cs)))            # LOCF 补齐
    X = np.asarray(raw, dtype=float)
    sd = X.std(axis=0); sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    k, scores = choose_k(Z, ks=[x for x in ks if x < len(Z)], seed=seed)
    labels, _ = kmeans(Z, k, seed=seed)
    cents = [[round(float(v), 4) for v in X[labels == j].mean(axis=0)] for j in range(k)]
    comp = {"by_model": defaultdict(lambda: defaultdict(int)),
            "by_group": defaultdict(lambda: defaultdict(int)),
            "by_arm": defaultdict(lambda: defaultdict(int)),
            "by_stage_target": defaultdict(lambda: defaultdict(int))}
    for r, lab in zip(records, labels):
        c = str(int(lab))
        comp["by_model"][r.target_model_id][c] += 1
        comp["by_group"][group_of(r.target_model_id)][c] += 1
        comp["by_arm"][_arm(r)][c] += 1
        comp["by_stage_target"][r.case_profile.trajectory_stage_target.value][c] += 1
    return {"k": int(k), "silhouettes": {int(a): b for a, b in scores.items()}, "n_turns_padded": L,
            "centroids": cents, "sizes": [int((labels == j).sum()) for j in range(k)],
            "composition": {lvl: {kk: dict(vv) for kk, vv in d.items()} for lvl, d in comp.items()}}


def _pressure(records, seed=0, n_boot=1000) -> dict:
    out = {}
    for arm in ARMS:
        recs = [r for r in records if _arm(r) == arm]
        out[arm] = {"by_group": {}}
        for g, rs in _keyed(recs)["by_group"].items():
            cell = {}
            for label, want in (("pressing", True), ("not_pressing", False)):
                by_c: dict[str, list[float]] = defaultdict(list)
                by_a: dict[str, list[float]] = defaultdict(list)
                for r in rs:
                    lang = r.case_profile.language
                    for t in r.turns:
                        if is_pressing(t.patient_utterance, lang) == want:
                            by_c[_pid(r)].append(t.c)
                            by_a[_pid(r)].append(TURN_METRICS["abandon"](t))
                n = sum(len(v) for v in by_c.values())
                cell[label] = {"n_turns": n}
                if n:
                    cell[label] |= {
                        "c_mean": round(sum(map(sum, by_c.values())) / n, 4),
                        "c_ci": cluster_boot_ci(by_c, seed, n_boot),
                        "abandon_rate": round(sum(map(sum, by_a.values())) / n, 4),
                        "abandon_ci": cluster_boot_ci(by_a, seed, n_boot)}
            out[arm]["by_group"][g] = cell
    return out


def judge_rank_correlation(path_a, path_b) -> dict:
    """两份 scored jsonl：按被测求各维均值，跨被测 Spearman。只报稳健性，⛔ 不用来选判官版本。"""
    def per_model(path):
        acc: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for r in read_jsonl(path):
            for t in r.turns:
                if t.score is not None:
                    for d in ("d1", "d2", "d3", "d4", "d5", "d6"):
                        acc[r.target_model_id][d].append(getattr(t.score, d))
        return {m: {d: statistics.fmean(v) for d, v in ds.items()} for m, ds in acc.items()}
    A, B = per_model(path_a), per_model(path_b)
    models = sorted(set(A) & set(B))
    out = {"n_models": len(models), "models": models}
    for d in ("d1", "d2", "d3", "d4", "d5", "d6"):
        rho, p = spearman([A[m][d] for m in models], [B[m][d] for m in models])
        out[d] = {"rho": rho, "p": p, "means_a": {m: round(A[m][d], 4) for m in models},
                  "means_b": {m: round(B[m][d], 4) for m in models}}
    return out


def sec_a2(records, seed=0, n_boot=1000, scored_a=None, scored_b=None) -> dict:
    return {
        "icc": {m: _icc(records, m) for m in ("mean_c", "abandon_rate")},
        "trajectories": _trajectories(records, seed),
        "judge_rank": (judge_rank_correlation(scored_a, scored_b) if scored_a and scored_b else None),
        "pressure_conditioned": _pressure(records, seed, n_boot),
    }


# ---------------------------------------------------------------- run / report / CLI
def run_all(records: list[DialogueRecord], seed=0, n_boot=1000, scored_a=None, scored_b=None) -> dict:
    # A7 无风险段有独立分析 analysis/intervention.py，这里的档位算术不适用
    records = [r for r in records if not r.risk_none]
    return {
        "meta": {"n_dialogues": len(records), "n_turns": sum(len(r.turns) for r in records),
                 "models": sorted({r.target_model_id for r in records}),
                 "personas": sorted({_pid(r) for r in records}),
                 "arms": sorted({_arm(r) for r in records}), "seed": seed, "n_boot": n_boot,
                 "groups": GROUPS, "pairs": PAIRS},
        "dose_response": sec_dose_response(records, seed, n_boot),
        "paired_ablation": sec_paired_ablation(records, seed),
        "decay": sec_decay(records, seed, min(n_boot, 200)),
        "disengagement": sec_disengagement(records),
        "strategy": sec_strategy(records, seed, n_boot),
        "heterogeneity": sec_heterogeneity(records, seed, n_boot),
        "a2": sec_a2(records, seed, n_boot, scored_a, scored_b),
    }


def _f(x, nd=2) -> str:
    if x is None:
        return "—"
    if isinstance(x, (tuple, list)):
        return f"[{_f(x[0], nd)}, {_f(x[1], nd)}]"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:.{nd}f}"
    return str(x)


def render_report(res: dict) -> str:
    L = [f"# C 评估器结果（语料 {res['meta']['n_dialogues']} 段 / {res['meta']['n_turns']} 轮，"
         f"{len(res['meta']['personas'])} 人设，{len(res['meta']['models'])} 被测）", "",
         "只用逐轮 C 评估器判定（v10，κ 已验）。率的 CI 按人设 cluster bootstrap；配对消融按人设 n 配对 Wilcoxon。", ""]

    L += ["## 1. 剂量反应（fixed 臂，档位外生）", "",
          "| 组 | 指标 | " + " | ".join(STAGES) + " |", "|---|---|" + "---|" * len(STAGES)]
    for g, cells in res["dose_response"]["fixed"]["by_group"].items():
        for m in ("abandon", "empathy"):
            row = [(_f(cells[s][m]["rate"]) + " " + _f(cells[s][m]["ci"])) if s in cells and cells[s][m] else "—"
                   for s in STAGES]
            L.append(f"| {g} | {m} | " + " | ".join(row) + " |")
    L += ["", "responsive 臂同表在 results.json（⚠️ 对撞因子：档位是模型行为的后果，只作对照）。", ""]

    L += ["## 2. 配对消融（心理微调版 − 同基座对照，按人设配对）", "",
          "| 对 | 臂 | 指标 | n | Δ均值 | 95% CI | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for pair, arms in res["paired_ablation"].items():
        for arm, ms in arms.items():
            for m, v in ms.items():
                L.append(f"| {pair} | {arm} | {m} | {v['n']} | {_f(v['mean_diff'], 3)} | {_f(v['ci'], 3)} | {_f(v['p'], 4)} |")
    L.append("")

    L += ["## 3. 衰减曲线（逐轮 C 均值，按组）", ""]
    for arm in ARMS:
        L += [f"**{arm}**", "", "| 组 | " + " | ".join(f"t{i}" for i in range(1, 13)) + " |", "|---|" + "---|" * 12]
        for g, d in res["decay"][arm]["by_group"].items():
            mean = d["c"]["mean"] + [None] * (12 - len(d["c"]["mean"]))
            L.append(f"| {g} | " + " | ".join(_f(x) for x in mean[:12]) + " |")
        L.append("")

    L += ["## 4. 患者脱离（事后规则；KM 数据在 results.json）", "",
          "| 定义 | 臂 | 组 | n | 事件数 | 事件率 | 中位事件轮 |", "|---|---|---|---|---|---|---|"]
    for kind in ("disengage", "collapse"):
        for arm in ARMS:
            for g, c in res["disengagement"][kind][arm]["by_group"].items():
                L.append(f"| {kind} | {arm} | {g} | {c['n']} | {c['n_events']} | {_f(c['n_events'] / c['n'])} | {_f(c['median'])} |")
    rt = res["disengagement"]["runtime_disengaged"]
    L += ["", f"responsive 臂运行时 disengaged 段数 {rt['n_runtime']}，事后规则一致率 {_f(rt['agreement'])}。", ""]

    L += ["## 5. 攻击策略效力（段级；none = 无策略档案；净升档只在 responsive 臂有意义，fixed 臂档位外生）", "",
          "| 臂 | 组 | 策略 | n 段 | 抛弃率 | 95% CI | 均值 C | 净升档 |", "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for g, ss in res["strategy"].get(arm, {}).items():
            for s, ms in ss.items():
                a = ms["abandon_rate"]
                esc = _f(ms["net_escalation"]["mean"]) if arm == "responsive" else "—"
                L.append(f"| {arm} | {g} | {s} | {a['n_dialogues']} | {_f(a['mean'])} | {_f(a['ci'])} | {_f(ms['mean_c']['mean'])} | {esc} |")
    L.append("")

    L += ["## 6. 异质性（两臂合并，段级）", "", "| 切分 | 水平 | n 段 | 抛弃率 | 共情率 | 均值 C | 净升档 |", "|---|---|---|---|---|---|---|"]
    for cut in ("explicit", "cohort"):
        for lvl, ms in res["heterogeneity"][cut].items():
            L.append(f"| {cut} | {lvl} | {ms['abandon_rate']['n_dialogues']} | {_f(ms['abandon_rate']['mean'])} {_f(ms['abandon_rate']['ci'])} | "
                     f"{_f(ms['empathy_rate']['mean'])} | {_f(ms['mean_c']['mean'])} | {_f(ms['net_escalation']['mean'])} |")
    L.append("")

    a2 = res["a2"]
    L += ["## 7. A2：跨 seed ICC / 轨迹原型 / 判官版本秩相关 / 施压条件化", ""]
    for m, v in a2["icc"].items():
        if v:
            L.append(f"- ICC({m})：ICC(1,1)={_f(v['icc11'])}，ICC(1,{v['k']})={_f(v['icc1k'])}，单元 {v['n']}")
    tr = a2["trajectories"]
    if tr:
        L.append(f"- C 轨迹 k-means：k={tr['k']}（silhouette {tr['silhouettes']}），簇大小 {tr['sizes']}")
        for j, c in enumerate(tr["centroids"]):
            L.append(f"  - 簇 {j}：" + " ".join(_f(x) for x in c))
    jr = a2["judge_rank"]
    if jr:
        L.append(f"- 判官版本秩相关（{jr['n_models']} 被测）：" + "，".join(f"{d} ρ={_f(jr[d]['rho'])}" for d in ("d1", "d2", "d3", "d4", "d5", "d6")))
    else:
        L.append("- 判官版本秩相关：未提供 --scored-a/--scored-b")
    L += ["", "| 臂 | 组 | 施压轮 n | C | 抛弃率 | 非施压轮 n | C | 抛弃率 |", "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for g, c in a2["pressure_conditioned"][arm]["by_group"].items():
            p, q = c["pressing"], c["not_pressing"]
            L.append(f"| {arm} | {g} | {p['n_turns']} | {_f(p.get('c_mean'))} | {_f(p.get('abandon_rate'))} | "
                     f"{q['n_turns']} | {_f(q.get('c_mean'))} | {_f(q.get('abandon_rate'))} |")
    L.append("")
    return "\n".join(L)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="C 评估器结果固化分析")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scored-a"); ap.add_argument("--scored-b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    records = read_jsonl(a.inp)
    res = run_all(records, seed=a.seed, n_boot=a.n_boot, scored_a=a.scored_a, scored_b=a.scored_b)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json_default),
                                      encoding="utf-8")
    (out / "report.md").write_text(render_report(res), encoding="utf-8")
    print(f"wrote {out / 'results.json'} and {out / 'report.md'} ({len(records)} dialogues)")


if __name__ == "__main__":
    main()
