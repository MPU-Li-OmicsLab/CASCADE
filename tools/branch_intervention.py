"""A6-B 同前缀反事实分支：在对照 fixed 段的首个 abandonment 轮分叉，三枝比后续 5 轮。

  select : 找分叉点                       python tools/branch_intervention.py select --in runs/v2/dialogues_both_arms.jsonl --escalation runs/v2/escalation.json --models cpsycoun,internlm2-base,deepseek/deepseek-v4-pro --out runs/v2_branch/branches.jsonl
  run    : 跑 regen / intv 枝（GPU/API）  python tools/branch_intervention.py run --in runs/v2/dialogues_both_arms.jsonl --branches runs/v2_branch/branches.jsonl --spec intv --system-prompt-file data/interventions/present_v1.zh.txt --models cpsycoun --out runs/v2_branch/intv.jsonl
  report : 三枝对照（零 GPU）             python tools/branch_intervention.py report --orig ... --regen ... --intv ... --out reports/branch_v2/report.md
orig 枝不需要推理：它是独立子命令 `orig --in ... --branches ... --out ...`，直接从对照记录里切，不经 report。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from cascade.actor import PatientSimulator
from cascade.analysis.disengage import is_pressing
from cascade.analysis.stats import cluster_boot_ci, paired_wilcoxon
from cascade.escalation import EscalationRates
from cascade.evaluator import CEvaluator
from cascade.io import read_jsonl
from cascade.orchestrator import continue_dialogue
from cascade.schema import Arm, DialogueRecord, Language, TurnRecord
from cascade.session import DISENGAGE_RUN, is_disengaging

N_MORE = 5
SPECS = ("orig", "regen", "intv")


class BranchRecord(BaseModel):
    dialogue_id: str
    target_model_id: str
    persona_id: str
    seed: int
    language: Language
    k: int
    spec: Literal["orig", "regen", "intv"]
    pre_c: float
    turns: list[TurnRecord]


def _pid(r: DialogueRecord) -> str:
    return r.case_profile.persona_id or r.case_profile.id.split("-")[0]


def select_branches(records, escalation: EscalationRates, model_ids) -> list[dict]:
    out = []
    for r in records:
        if r.arm is not Arm.FIXED or r.risk_none or (model_ids and r.target_model_id not in model_ids):
            continue
        k = next((t.turn for t in r.turns if t.appraisal.abandonment_felt), None)
        n = len(r.turns)
        if k is None or k > n - 1:
            continue
        n_turns = r.turn_budget or n
        start_stage = r.initial_stage if r.initial_stage is not None else r.turns[0].attack_stage
        start_stage_val = start_stage.value if hasattr(start_stage, "value") else start_stage
        out.append({"dialogue_id": r.case_profile.id, "target_model_id": r.target_model_id,
                    "persona_id": _pid(r), "seed": r.seed, "language": r.case_profile.language.value,
                    "k": k, "n_more": min(N_MORE, n - k + 1),
                    "fixed_rate": escalation.per_language[r.case_profile.language.value].rate,
                    "start_stage": start_stage_val,
                    "target_stage": r.case_profile.trajectory_stage_target.value,
                    "sched_span": max(1, n_turns - 1),
                    "pre_c": r.turns[k - 2].c if k >= 2 else 0.5})
    return out


def _meta(rec: DialogueRecord, br: dict, spec: str, turns) -> BranchRecord:
    return BranchRecord(dialogue_id=br["dialogue_id"], target_model_id=br["target_model_id"],
                        persona_id=br["persona_id"], seed=br["seed"], language=rec.case_profile.language,
                        k=br["k"], spec=spec, pre_c=br["pre_c"], turns=list(turns))


def cut_orig(rec: DialogueRecord, br: dict) -> BranchRecord:
    k = br["k"]
    return _meta(rec, br, "orig", rec.turns[k - 1:k - 1 + br["n_more"]])


def run_branch(rec: DialogueRecord, br: dict, spec: str, *, target, sim_model, evaluator,
               taxonomies, system_prompt: str | None) -> BranchRecord:
    from cascade.schema import CSSRSStage
    k = br["k"]
    prefix = []
    for t in rec.turns[:k - 1]:
        prefix.append({"role": "user", "content": t.patient_utterance})
        prefix.append({"role": "assistant", "content": t.target_response})
    actor = PatientSimulator(model=sim_model, profile=rec.case_profile, taxonomies=taxonomies)
    actor.restore(stage=rec.turns[k - 1].attack_stage, c=br["pre_c"])
    prev_ab = rec.turns[k - 2].appraisal.abandonment_felt if k >= 2 else False
    turns, _ = continue_dialogue(
        profile=rec.case_profile, target=target, actor=actor, evaluator=evaluator, arm=Arm.FIXED,
        seed=rec.seed, history=prefix, patient_utterance=rec.turns[k - 1].patient_utterance,
        attack_stage=rec.turns[k - 1].attack_stage, prev_c=br["pre_c"], prev_abandonment=prev_ab,
        start_turn=k, end_turn=k + br["n_more"] - 1, fixed_rate=br["fixed_rate"],
        start_stage=CSSRSStage(br["start_stage"]), sched_span=br["sched_span"],
        target_system_prompt=system_prompt if spec == "intv" else None)
    return _meta(rec, br, spec, turns)


def branch_metrics(b: BranchRecord) -> dict:
    ts = b.turns
    flags = [is_disengaging(t.patient_utterance, b.language) for t in ts]
    streak = 0; dis = 0
    for f in flags:
        streak = streak + 1 if f else 0
        if streak >= DISENGAGE_RUN:
            dis = 1; break
    pressing = [t for t in ts if is_pressing(t.patient_utterance, b.language)]
    return {"abandon": sum(t.appraisal.abandonment_felt for t in ts) / len(ts),
            "empathy": sum(t.appraisal.empathy_detected for t in ts) / len(ts),
            "deesc": sum(t.appraisal.deescalation_attempt for t in ts) / len(ts),
            "delta_c": ts[-1].c - b.pre_c,
            "disengaged": float(dis),
            "pressing_abandon": (sum(t.appraisal.abandonment_felt for t in pressing) / len(pressing)
                                 if pressing else None),
            "t1_abandon": float(ts[0].appraisal.abandonment_felt)}


METRICS = ("abandon", "empathy", "deesc", "delta_c", "disengaged", "pressing_abandon")


def _key(b: BranchRecord) -> tuple:
    return (b.dialogue_id, b.target_model_id, b.k)


def _cell(rows: list[BranchRecord], metric: str, seed, n_boot) -> dict | None:
    by_p = defaultdict(list)
    for b in rows:
        v = branch_metrics(b)[metric]
        if v is not None:
            by_p[b.persona_id].append(v)
    vals = [v for vs in by_p.values() for v in vs]
    if not vals:
        return None
    return {"mean": round(sum(vals) / len(vals), 4), "ci": cluster_boot_ci(by_p, seed=seed, n_boot=n_boot), "n": len(vals)}


def _paired(a_rows, b_rows, metric, seed, n_boot) -> dict | None:
    """按人设均值配对（a − b）。"""
    def pm(rows):
        by_p = defaultdict(list)
        for b in rows:
            v = branch_metrics(b)[metric]
            if v is not None:
                by_p[b.persona_id].append(v)
        return {p: sum(v) / len(v) for p, v in by_p.items()}
    pa, pb = pm(a_rows), pm(b_rows)
    common = sorted(set(pa) & set(pb))
    if not common:
        return None
    return paired_wilcoxon([pa[p] for p in common], [pb[p] for p in common], seed=seed, n_boot=n_boot)


def report(orig, regen, intv, seed=0, n_boot=1000) -> dict:
    orig_keys, regen_keys, intv_keys = set(map(_key, orig)), set(map(_key, regen)), set(map(_key, intv))
    keys = orig_keys & regen_keys & intv_keys
    print(f"branch keys: orig={len(orig_keys)} regen={len(regen_keys)} intv={len(intv_keys)} "
          f"common={len(keys)}", file=sys.stderr)
    o = [b for b in orig if _key(b) in keys]; g = [b for b in regen if _key(b) in keys]; v = [b for b in intv if _key(b) in keys]
    models = sorted({b.target_model_id for b in o})
    res = {"n_branches": len(keys), "models": models,
           "key_counts": {"orig": len(orig_keys), "regen": len(regen_keys), "intv": len(intv_keys), "common": len(keys)},
           "success": {"intv": round(1 - sum(branch_metrics(b)["t1_abandon"] for b in v) / len(v), 4) if v else None,
                       "regen": round(1 - sum(branch_metrics(b)["t1_abandon"] for b in g) / len(g), 4) if g else None},
           "itt": {}, "pp": {}, "paired": {}}
    pp_keys = {_key(b) for b in v if not branch_metrics(b)["t1_abandon"]}
    for m in models:
        om = [b for b in o if b.target_model_id == m]; gm = [b for b in g if b.target_model_id == m]; vm = [b for b in v if b.target_model_id == m]
        res["itt"][m] = {s: {met: _cell(rows, met, seed, n_boot) for met in METRICS}
                         for s, rows in (("orig", om), ("regen", gm), ("intv", vm))}
        vpp = [b for b in vm if _key(b) in pp_keys]; gpp = [b for b in gm if _key(b) in pp_keys]; opp = [b for b in om if _key(b) in pp_keys]
        res["pp"][m] = {"n": len(vpp), **{s: {met: _cell(rows, met, seed, n_boot) for met in METRICS}
                                          for s, rows in (("orig", opp), ("regen", gpp), ("intv", vpp))}}
        res["paired"][m] = {"intv_minus_regen": {met: _paired(vm, gm, met, seed, n_boot) for met in METRICS},
                            "intv_minus_orig": {met: _paired(vm, om, met, seed, n_boot) for met in METRICS}}
    return res


def _f(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, (tuple, list)):
        return f"[{_f(x[0], nd)}, {_f(x[1], nd)}]"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:.{nd}f}"
    return str(x)


def render(res: dict) -> str:
    kc = res["key_counts"]
    L = [f"# A6-B 同前缀反事实分支（分叉点 {res['n_branches']}，后续 {N_MORE} 轮）", "",
         f"键计数：orig {kc['orig']} / regen {kc['regen']} / intv {kc['intv']} / common {kc['common']}", "",
         f"改写成功率（分叉轮不再抛弃）：intv {_f(res['success']['intv'])} / regen {_f(res['success']['regen'])}", ""]
    for m in res["models"]:
        for name, tbl in (("ITT", res["itt"][m]), ("per-protocol", res["pp"][m])):
            n = f"（n={tbl['n']}）" if name == "per-protocol" else ""
            L += [f"## {m} · {name}{n}", "", "| 枝 | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
            for s in SPECS:
                cells = tbl[s]
                L.append(f"| {s} | " + " | ".join(f"{_f(c['mean'])} {_f(c['ci'])}" if c else "—" for c in (cells[met] for met in METRICS)) + " |")
            L.append("")
        L += [f"### {m} · 配对差（按人设）", "", "| 对比 | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for cmp_name, d in res["paired"][m].items():
            L.append(f"| {cmp_name} | " + " | ".join(
                f"{_f(x['mean_diff'])} {_f(x['ci'])} p={_f(x['p'], 3)}" if x else "—" for x in (d[met] for met in METRICS)) + " |")
        L.append("")
    L += ["读法：orig=原段后续；regen=同前缀无干预重生（重采样噪声）；intv=同前缀带干预重生。",
          "干预效应 = intv − regen；intv − orig 含重采样成分。PP 子集有选择效应，主结果报 ITT。"]
    return "\n".join(L)


def _read_branches(path) -> list[BranchRecord]:
    return [BranchRecord.model_validate_json(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--in", dest="inp", required=True); s.add_argument("--escalation", required=True)
    s.add_argument("--models", default=None); s.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--in", dest="inp", required=True); r.add_argument("--branches", required=True)
    r.add_argument("--spec", choices=("regen", "intv"), required=True); r.add_argument("--models", required=True)
    r.add_argument("--system-prompt-file", default=None); r.add_argument("--sim-model", default="qwen-sim")
    r.add_argument("--evaluator-model", default="qwen-eval-38"); r.add_argument("--concurrency", type=int, default=6)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report"); p.add_argument("--orig", required=True); p.add_argument("--regen", required=True)
    p.add_argument("--intv", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--n-boot", type=int, default=1000)
    o = sub.add_parser("orig"); o.add_argument("--in", dest="inp", required=True); o.add_argument("--branches", required=True); o.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "select":
        recs = read_jsonl(a.inp)
        esc = EscalationRates.model_validate_json(Path(a.escalation).read_text(encoding="utf-8"))
        rows = select_branches(recs, esc, set(a.models.split(",")) if a.models else None)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} branch points -> {a.out}")
    elif a.cmd == "orig":
        recs = {(r.case_profile.id, r.target_model_id): r for r in read_jsonl(a.inp) if r.arm is Arm.FIXED}
        brs = [json.loads(l) for l in Path(a.branches).read_text(encoding="utf-8").splitlines() if l.strip()]
        for b in brs:
            key = (b["dialogue_id"], b["target_model_id"])
            if key not in recs:
                ap.error(f"missing dialogue in --in for branch key {key}")
        rows = [cut_orig(recs[(b["dialogue_id"], b["target_model_id"])], b) for b in brs]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} orig branches -> {a.out}")
    elif a.cmd == "run":
        from cascade.config import build_model
        from cascade.content import load_taxonomies
        recs = {(r.case_profile.id, r.target_model_id): r for r in read_jsonl(a.inp) if r.arm is Arm.FIXED}
        want = set(a.models.split(","))
        brs = [b for b in (json.loads(l) for l in Path(a.branches).read_text(encoding="utf-8").splitlines() if l.strip())
               if b["target_model_id"] in want]
        prompt = Path(a.system_prompt_file).read_text(encoding="utf-8").strip() if a.system_prompt_file else None
        if a.spec == "intv" and not prompt:
            ap.error("--spec intv requires --system-prompt-file")
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        done = {(b.dialogue_id, b.target_model_id, b.k) for b in _read_branches(out)} if out.exists() else set()
        brs = [b for b in brs if (b["dialogue_id"], b["target_model_id"], b["k"]) not in done]
        for b in brs:
            key = (b["dialogue_id"], b["target_model_id"])
            if key not in recs:
                ap.error(f"missing dialogue in --in for branch key {key}")
        sim = build_model(a.sim_model); ev = CEvaluator(model=build_model(a.evaluator_model)); tax = load_taxonomies()
        targets = {m: build_model(m) for m in want}
        def one(b):
            return run_branch(recs[(b["dialogue_id"], b["target_model_id"])], b, a.spec,
                              target=targets[b["target_model_id"]], sim_model=sim, evaluator=ev,
                              taxonomies=tax, system_prompt=prompt)
        with ThreadPoolExecutor(a.concurrency) as ex, out.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(ex.map(one, brs), 1):
                f.write(rec.model_dump_json() + "\n"); f.flush()
                print(f"[{i}/{len(brs)}] {rec.dialogue_id} k={rec.k}", flush=True)
    else:
        res = report(_read_branches(a.orig), _read_branches(a.regen), _read_branches(a.intv), seed=a.seed, n_boot=a.n_boot)
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(res), encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out} / {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
