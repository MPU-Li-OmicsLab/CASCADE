# cascade/goldeval/sample.py
"""Stratified turn sampler + blind annotation-package export (Plan 4.4 §4). Strata =
(language, attack_stage, arm) — generation-side facts only, NEVER machine judgments
(circular + defeats blinding). Deterministic under a fixed seed."""
from __future__ import annotations
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from cascade.goldeval.spec import ALL_FIELDS

# never allowed into annotate.csv (machine judgments + attack machinery)
_HIDDEN = ("attack_stage", "c", "arm", "expressed_risk", "appraisal", "score")
# 人标列名一律从 spec 取（h_connect 是 abandon 的正向列，见 FieldSpec.h_col），
# 别再在这里手写字符串：手写过一次就会和 spec 漂移，而漂移的后果是整批人标读不进来。
_H_COLS = [f.h_col for f in ALL_FIELDS]


@dataclass(frozen=True)
class TurnSample:
    id: str
    language: str
    attack_stage: str
    arm: str
    history: list       # list[tuple[str, str]] 全前轮 (patient, target)
    prev_user: str
    patient_utterance: str
    target_response: str
    machine: dict       # {FieldSpec.name: 机器判 | None}


def _machine(turn) -> dict:
    return {f.name: f.extract(turn) for f in ALL_FIELDS}


def _flatten(records) -> list[TurnSample]:
    """`prev_user` 恒为「被测这条回复所应答的那句患者发言」（标注模板里的『病人最新一句』）。

    exchange_v2：一条 TurnRecord 就是一次交换 → 前文就是本条的 patient_utterance。
    legacy_v1（旧批 valbatch / cmp_eval 等）：target_response 回的是**上一条**的
    patient_utterance，首轮回的是 seed —— 按旧语义还原，才不会把旧数据错标。
    """
    out: list[TurnSample] = []
    for d in records:
        if d.risk_none:
            continue    # A7 无风险段不进金标抽样（attack_stage="none" 是字符串，.value 会炸）
        cp = d.case_profile
        v2 = getattr(d, "turn_pairing", None) == "exchange_v2"
        prev_user = cp.seed_text
        history: list = []
        for t in d.turns:
            out.append(TurnSample(
                id=f"{cp.id}#t{t.turn}", language=cp.language.value,
                attack_stage=t.attack_stage.value, arm=d.arm.value,
                history=list(history),
                prev_user=t.patient_utterance if v2 else prev_user,
                patient_utterance=t.patient_utterance, target_response=t.target_response,
                machine=_machine(t)))
            history = history + [(t.patient_utterance, t.target_response)]
            prev_user = t.patient_utterance
    return out


def stratified_sample(records, *, n: int, seed: int, min_per_cell: int = 2) -> list[TurnSample]:
    """Sample n turns stratified by (language, attack_stage, arm). Phase 1 floors each
    non-empty cell with min(min_per_cell, cell size) (rare high-risk stages get a
    floor); phase 2 fills remaining slots from leftovers. Deterministic under `seed`."""
    pool = _flatten(records)
    rng = random.Random(seed)
    cells: dict = defaultdict(list)
    for s in pool:
        cells[(s.language, s.attack_stage, s.arm)].append(s)
    picked: list[TurnSample] = []
    for key in sorted(cells):
        items = cells[key][:]
        rng.shuffle(items)
        take = min(min_per_cell, len(items), max(0, n - len(picked)))
        picked.extend(items[:take])
        cells[key] = items[take:]
        if len(picked) >= n:
            return picked[:n]
    leftover = [s for key in sorted(cells) for s in cells[key]]
    rng.shuffle(leftover)
    picked.extend(leftover[:n - len(picked)])
    return picked[:n]


def _fmt_history(history) -> str:
    return " || ".join(f"[P] {p} [A] {t}" for p, t in history)


def _write_csv(path, samples) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "history", "prev_user", "patient_utterance",
                    "target_response"] + _H_COLS)
        for s in samples:
            w.writerow([s.id, s.language, _fmt_history(s.history), s.prev_user,
                        s.patient_utterance, s.target_response] + [""] * len(_H_COLS))


def export_annotation_package(samples, out_dir, *, annotators: int = 2,
                              double_frac: float = 0.3, seed: int = 0) -> None:
    """Write blind annotate.csv + hidden key.jsonl + assignment.json. Double-scored
    subset (first double_frac of a deterministic shuffle) goes to ALL annotators for
    IAA; the rest split round-robin."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = samples[:]
    rng.shuffle(order)
    n_double = round(len(order) * double_frac)
    ann_ids = [f"ann{i + 1}" for i in range(annotators)]
    assignment: dict = {}
    for k, s in enumerate(order):
        assignment[s.id] = list(ann_ids) if k < n_double else [ann_ids[k % annotators]]
    _write_csv(out / "annotate.csv", samples)
    with (out / "key.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"id": s.id, "language": s.language,
                                "attack_stage": s.attack_stage, "arm": s.arm,
                                "machine": s.machine}, ensure_ascii=False) + "\n")
    (out / "assignment.json").write_text(
        json.dumps({"annotators": ann_ids, "double_scored": [s.id for s in order[:n_double]],
                    "assignment": assignment}, ensure_ascii=False, indent=2), encoding="utf-8")
