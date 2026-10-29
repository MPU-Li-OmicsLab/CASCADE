"""Escalation-rate calibration (1.A → per-language rate) and the deterministic
fixed-arm schedule it drives (spec 3B §4–5).

Named `escalation` (C-SSRS stage climb), NOT `calibration` — that belongs to
eval_calibration.py / data/calibration.yaml (C-evaluator anti-gaming accuracy).
"""
from __future__ import annotations
from pydantic import BaseModel
from cascade.schema import CSSRSStage, DialogueRecord, Arm

_STAGES = list(CSSRSStage)


def fixed_stage_at(turn: int, rate: float, start: CSSRSStage,
                   target: CSSRSStage, max_turns: int | None = None) -> CSSRSStage:
    """Fixed arm climbs at the calibrated language `rate`, ignoring C.
    index(t) = min(start.index + int(rate*turn + 0.5), target.index), clamped to
    [start.index, target.index]. Monotonic non-decreasing in turn (rate >= 0).

    `max_turns` enables the **reach-target floor**: rate = max(rate, span/max_turns),
    so the arm is guaranteed to arrive at `target` by the last turn.
    Why (2026-07-30): the fitted rate comes from the responsive arm, and a highly
    empathic target pins C near 1.0 -> decide_action almost never escalates -> the
    fitted rate collapsed to 0.0278; int(0.0278*12 + 0.5) == 0, so the fixed arm
    never advanced a single stage and the whole batch produced ZERO high-risk turns
    (measured: simdata3 fixed = 288/288 passive_ideation). The fixed arm's definition
    is "ignore C and climb to the target on schedule", so failing to reach the target
    voids the arm. Omit max_turns to get the raw calibrated schedule (ablation)."""
    if max_turns and max_turns > 0:
        span = target.index - start.index
        if span > 0:
            rate = max(rate, span / max_turns)
    climbed = start.index + int(rate * turn + 0.5)
    idx = max(start.index, min(climbed, target.index))
    return _STAGES[idx]


class LangRate(BaseModel):
    rate: float                       # Σ(last.attack_stage.index − initial_stage.index) / Σ turns
    mean_c_trajectory: list[float]    # per-turn mean C (right-censored); report only
    stage_advance_prob: float         # advancing turns / total turns; report only
    n_dialogues: int
    n_turns: int


class EscalationRates(BaseModel):
    per_language: dict[str, LangRate]


def fit_escalation_rates(dialogues: list[DialogueRecord]) -> EscalationRates:
    """Fit per-language scalar net escalation rate from 1.A RESPONSIVE dialogues."""
    responsive = [d for d in dialogues if d.arm == Arm.RESPONSIVE]
    if not responsive:
        raise ValueError("no responsive (1.A) dialogues to calibrate from")
    by_lang: dict[str, list[DialogueRecord]] = {}
    for d in responsive:
        by_lang.setdefault(d.case_profile.language.value, []).append(d)

    per_language: dict[str, LangRate] = {}
    for lang, ds in by_lang.items():
        total_turns = total_delta = advance_turns = 0
        max_len = max(len(d.turns) for d in ds)
        c_sums = [0.0] * max_len
        c_counts = [0] * max_len
        for d in ds:
            if d.initial_stage is None:
                raise ValueError(
                    f"dialogue {d.case_profile.id!r} lacks initial_stage; "
                    "calibration needs 3B-runner output")
            prev_idx = d.initial_stage.index
            total_turns += len(d.turns)
            for t, turn in enumerate(d.turns):
                if turn.attack_stage.index > prev_idx:
                    advance_turns += 1
                prev_idx = turn.attack_stage.index
                c_sums[t] += turn.c
                c_counts[t] += 1
            if d.turns:
                total_delta += d.turns[-1].attack_stage.index - d.initial_stage.index
        rate = total_delta / total_turns if total_turns else 0.0
        mean_c = [round(c_sums[t] / c_counts[t], 4) for t in range(max_len) if c_counts[t]]
        prob = advance_turns / total_turns if total_turns else 0.0
        per_language[lang] = LangRate(
            rate=round(rate, 4), mean_c_trajectory=mean_c,
            stage_advance_prob=round(prob, 4),
            n_dialogues=len(ds), n_turns=total_turns)
    return EscalationRates(per_language=per_language)


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path
    from cascade.io import read_jsonl
    p = argparse.ArgumentParser(prog="cascade.escalation",
                                description="Fit per-language escalation rates from 1.A JSONL.")
    p.add_argument("--in", dest="inp", required=True, help="1.A dialogues.jsonl")
    p.add_argument("--out", required=True, help="escalation.json output")
    args = p.parse_args(argv)
    rates = fit_escalation_rates(read_jsonl(args.inp))
    Path(args.out).write_text(rates.model_dump_json(indent=2), encoding="utf-8")
    for lang, lr in rates.per_language.items():
        print(f"{lang}: rate={lr.rate} (n={lr.n_dialogues} dialogues, {lr.n_turns} turns)")


if __name__ == "__main__":
    main()
