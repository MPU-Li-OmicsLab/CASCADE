"""A6/A7 汇总分析：系统提示干预（RCT）、分支反事实（引用 branch 报告）、无风险对照。

    python -m cascade.analysis.intervention --control runs/v2/dialogues_both_arms.jsonl \
        --intv runs/v2_intv/dialogues.jsonl runs/v2_intv_api/dialogues.jsonl \
        --none runs/v2_none/dialogues.jsonl runs/v2_none_api/dialogues.jsonl \
        [--branch reports/branch_v2/report.json] --out reports/intervention_v2

只读 C 评估器逐轮判定；CI 按人设 cluster bootstrap；配对按人设均值 Wilcoxon。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cascade.analysis.ceval import _json_default
from cascade.analysis.disengage import first_collapse_turn, first_disengage_turn
from cascade.analysis.stats import cluster_boot_ci, paired_wilcoxon
from cascade.io import read_jsonl
from cascade.schema import Arm, CSSRSStage, DialogueRecord

LOW = {CSSRSStage.PASSIVE_IDEATION, CSSRSStage.ACTIVE_IDEATION}
HIGH = {CSSRSStage.INTENT, CSSRSStage.PLAN, CSSRSStage.MEANS_TIMING}
DIAL_METRICS = ("abandon_rate", "empathy_rate", "deesc_rate", "final_c", "disengaged", "collapsed",
                "disengage_turn", "collapse_turn")
_PAIRED_METRICS = DIAL_METRICS[:6]    # disengage_turn/collapse_turn 仅事件段描述性，不配对
TURN_METRICS = {"abandon_rate": lambda t: float(t.appraisal.abandonment_felt),
                "empathy_rate": lambda t: float(t.appraisal.empathy_detected)}


def _pid(r: DialogueRecord) -> str:
    return r.case_profile.persona_id or r.case_profile.id.split("-")[0]


def rec_metrics(r: DialogueRecord) -> dict:
    n = len(r.turns)
    disengage_turn = first_disengage_turn(r)
    collapse_turn = first_collapse_turn(r)
    return {"abandon_rate": sum(t.appraisal.abandonment_felt for t in r.turns) / n,
            "empathy_rate": sum(t.appraisal.empathy_detected for t in r.turns) / n,
            "deesc_rate": sum(t.appraisal.deescalation_attempt for t in r.turns) / n,
            "final_c": r.turns[-1].c,
            "disengaged": float(disengage_turn is not None),
            "collapsed": float(collapse_turn is not None),
            "disengage_turn": disengage_turn,
            "collapse_turn": collapse_turn}


def _cell(vals_by_p: dict, seed, n_boot) -> dict | None:
    vals = [v for vs in vals_by_p.values() for v in vs if v is not None]
    if not vals:
        return None
    clean = {p: [v for v in vs if v is not None] for p, vs in vals_by_p.items()}
    return {"mean": round(sum(vals) / len(vals), 4), "ci": cluster_boot_ci(clean, seed=seed, n_boot=n_boot), "n": len(vals)}


def _dialogue_cell(recs, metric, seed, n_boot):
    by_p = defaultdict(list)
    for r in recs:
        by_p[_pid(r)].append(rec_metrics(r)[metric])
    return _cell(by_p, seed, n_boot)


def _turn_cell(recs, metric, pred, seed, n_boot):
    by_p = defaultdict(list)
    for r in recs:
        for t in r.turns:
            if pred(t):
                by_p[_pid(r)].append(TURN_METRICS[metric](t))
    return _cell(by_p, seed, n_boot)


def _paired_by_persona(a_recs, b_recs, metric, seed, n_boot):
    def pm(recs):
        by_p = defaultdict(list)
        for r in recs:
            v = rec_metrics(r)[metric]
            if v is not None:
                by_p[_pid(r)].append(v)
        return {p: sum(v) / len(v) for p, v in by_p.items()}
    pa, pb = pm(a_recs), pm(b_recs)
    common = sorted(set(pa) & set(pb))
    return paired_wilcoxon([pa[p] for p in common], [pb[p] for p in common], seed=seed, n_boot=n_boot) if common else None


def sec_system_prompt(control, intv, seed=0, n_boot=1000) -> dict:
    """干预臂 vs 对照臂（同被测、同档案的 fixed 段）。paired = intv − control。"""
    ctrl_fixed = [r for r in control if r.arm is Arm.FIXED and not r.risk_none and not r.intervention]
    models = sorted({r.target_model_id for r in intv})
    names = sorted({r.intervention for r in intv if r.intervention})
    out = {"interventions": names, "by_model": {}, "by_stage": {}}
    for m in models:
        im = [r for r in intv if r.target_model_id == m]
        ids = {r.case_profile.id for r in im}
        cm = [r for r in ctrl_fixed if r.target_model_id == m and r.case_profile.id in ids]
        out["by_model"][m] = {met: {"control": _dialogue_cell(cm, met, seed, n_boot),
                                    "intv": _dialogue_cell(im, met, seed, n_boot),
                                    "paired": (_paired_by_persona(im, cm, met, seed, n_boot)
                                               if met in _PAIRED_METRICS else None)}
                              for met in DIAL_METRICS}
        out["by_stage"][m] = {}
        for name, stages in (("low", LOW), ("high", HIGH)):
            pred = lambda t, S=stages: t.attack_stage in S
            out["by_stage"][m][name] = {"control": _turn_cell(cm, "abandon_rate", pred, seed, n_boot),
                                        "intv": _turn_cell(im, "abandon_rate", pred, seed, n_boot)}
    return out


def sec_none_control(control, none_recs, seed=0, n_boot=1000) -> dict:
    """无风险段 vs 有风险 fixed 段的 passive 档轮 / intent 及以上轮（同被测）。"""
    ctrl_fixed = [r for r in control if r.arm is Arm.FIXED and not r.risk_none and not r.intervention]
    out = {}
    for m in sorted({r.target_model_id for r in none_recs}):
        nm = [r for r in none_recs if r.target_model_id == m]
        cm = [r for r in ctrl_fixed if r.target_model_id == m]
        out[m] = {"n_none_dialogues": len(nm)}
        for met in ("abandon_rate", "empathy_rate"):
            out[m][met] = {"none": _turn_cell(nm, met, lambda t: True, seed, n_boot),
                           "passive_turns": _turn_cell(cm, met, lambda t: t.attack_stage == CSSRSStage.PASSIVE_IDEATION, seed, n_boot),
                           "high_turns": _turn_cell(cm, met, lambda t: t.attack_stage in HIGH, seed, n_boot)}
        out[m]["disengage_rate"] = {"none": _dialogue_cell(nm, "disengaged", seed, n_boot)}
    return out


def run_all(control, intv, none_recs, branch_json, seed=0, n_boot=1000) -> dict:
    return {"meta": {"n_control": len(control), "n_intv": len(intv), "n_none": len(none_recs)},
            "system_prompt": sec_system_prompt(control, intv, seed, n_boot) if intv else None,
            "branch": branch_json,
            "none_control": sec_none_control(control, none_recs, seed, n_boot) if none_recs else None}


def _f(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, (tuple, list)):
        return f"[{_f(x[0], nd)}, {_f(x[1], nd)}]"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:.{nd}f}"
    return str(x)


def _c(cell):
    return f"{_f(cell['mean'])} {_f(cell['ci'])}" if cell else "—"


def render_report(res: dict) -> str:
    L = [f"# A6/A7 干预与无风险对照（对照 {res['meta']['n_control']} 段 / 干预 {res['meta']['n_intv']} / 无风险 {res['meta']['n_none']}）", ""]
    L += ["## 1. 系统提示干预（A6-S，fixed 臂，同档案配对）", ""]
    sp = res["system_prompt"]
    if not sp:
        L.append("未提供干预臂数据。")
    else:
        L.append(f"干预语：{', '.join(sp['interventions'])}"); L.append("")
        for m, d in sp["by_model"].items():
            L += [f"### {m}", "", "| 指标 | 对照 | 干预 | 干预−对照（按人设） |", "|---|---|---|---|"]
            for met in DIAL_METRICS:
                p = d[met]["paired"]
                label = met + "（仅事件段，描述性）" if met in ("disengage_turn", "collapse_turn") else met
                L.append(f"| {label} | {_c(d[met]['control'])} | {_c(d[met]['intv'])} | "
                         f"{_f(p['mean_diff']) if p else '—'} {_f(p['ci']) if p else ''} p={_f(p['p'], 3) if p else '—'} |")
            st = sp["by_stage"][m]
            L += ["", "| 档位层 | 对照抛弃率 | 干预抛弃率 |", "|---|---|---|"]
            for name in ("low", "high"):
                L.append(f"| {name} | {_c(st[name]['control'])} | {_c(st[name]['intv'])} |")
            L.append("")
    L += ["## 2. 同前缀反事实分支（A6-B）", ""]
    br = res["branch"]
    if not br:
        L.append("未提供分支报告。")
    else:
        L.append(f"分叉点 {br.get('n_branches')}；改写成功率 intv {_f(br.get('success', {}).get('intv'))} / "
                 f"regen {_f(br.get('success', {}).get('regen'))}。详见 reports/branch_v2/report.md。")
    L += ["", "## 3. 无风险对照（A7）", ""]
    nc = res["none_control"]
    if not nc:
        L.append("未提供无风险数据。")
    else:
        L += ["| 被测 | n | 抛弃率 none | passive 轮 | intent+ 轮 | 共情率 none | passive 轮 | intent+ 轮 |", "|---|---|---|---|---|---|---|---|"]
        for m, d in nc.items():
            a, e = d["abandon_rate"], d["empathy_rate"]
            L.append(f"| {m} | {d['n_none_dialogues']} | {_c(a['none'])} | {_c(a['passive_turns'])} | {_c(a['high_turns'])} | "
                     f"{_c(e['none'])} | {_c(e['passive_turns'])} | {_c(e['high_turns'])} |")
        L += ["", "读法：none 段抛弃率应接近 passive 轮、远低于 intent+ 轮 → 失误是风险特异的。降险维在无风险段语义不同，不报。"]
    return "\n".join(L)


def _read_many(paths):
    out = []
    for p in paths or []:
        out += read_jsonl(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="A6/A7 汇总")
    ap.add_argument("--control", required=True)
    ap.add_argument("--intv", nargs="*", default=[])
    ap.add_argument("--none", dest="none_paths", nargs="*", default=[])
    ap.add_argument("--branch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    branch = json.loads(Path(a.branch).read_text(encoding="utf-8")) if a.branch else None
    res = run_all(read_jsonl(a.control), _read_many(a.intv), _read_many(a.none_paths), branch, a.seed, a.n_boot)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    (out / "report.md").write_text(render_report(res), encoding="utf-8")
    print(f"wrote {out / 'results.json'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
