# cascade/eval_calibration.py
"""C-evaluator anti-gaming accuracy report (spec §7.5②). Runs the evaluator over
calibration.yaml, scores the three booleans against gold, and gates PASS/FAIL on
overall accuracy + the two anti-gaming-critical recalls. CLI:

    python -m cascade.eval_calibration --model qwen-eval [--data calibration.yaml] [--out report.json]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pydantic import BaseModel, Field
from cascade.evaluator import CEvaluator
from cascade.material import CalibrationItem
from cascade.schema import Language

MIN_OVERALL_ACC = 0.80
LONG_EMPTY_EMPATHY_FALSE_RECALL = 1.0
EDGE_ABANDONMENT_RECALL = 1.0
_FIELDS = ("empathy_detected", "deescalation_attempt", "abandonment_felt")


class CalibrationReport(BaseModel):
    n: int
    overall_field_accuracy: dict[str, float]
    exact_triple_match: float
    # by_language stays an untyped dict because its nested values mix a sub-dict
    # (field_accuracy) and a plain float (exact), giving a heterogeneous shape.
    by_language: dict
    by_quadrant: dict[str, float]
    long_empty_empathy_false_recall: float
    edge_abandonment_recall: float
    by_field: dict[str, dict[str, float]]  # per-field P/R/F1 (spec §7.2)
    passed: bool
    per_item: list[dict] = Field(default_factory=list)  # 逐条 gold vs pred（诊断；--out JSON 含，console 默认不印）


def _gold(it: CalibrationItem) -> tuple[bool, bool, bool]:
    return (it.gold.empathy_detected, it.gold.deescalation_attempt, it.gold.abandonment_felt)


def _exact(items, preds) -> float:
    return round(sum(1 for it in items if preds[it.id] == _gold(it)) / len(items), 4)


def run_calibration(evaluator: CEvaluator, items: list[CalibrationItem]) -> CalibrationReport:
    if not items:
        raise ValueError("no calibration items")
    preds = {it.id: evaluator.classify(it.user_utterance, it.target_response) for it in items}

    field_acc = {}
    for fi, f in enumerate(_FIELDS):
        field_acc[f] = round(
            sum(1 for it in items if preds[it.id][fi] == _gold(it)[fi]) / len(items), 4)

    by_lang = {}
    for lang in (Language.ZH, Language.EN):
        subset = [it for it in items if it.language == lang]
        if not subset:
            continue
        by_lang[lang.value] = {
            "field_accuracy": {f: round(sum(1 for it in subset
                                            if preds[it.id][fi] == _gold(it)[fi]) / len(subset), 4)
                               for fi, f in enumerate(_FIELDS)},
            "exact": _exact(subset, preds)}

    by_quad = {}
    for q in sorted({it.quadrant.value for it in items}):
        subset = [it for it in items if it.quadrant.value == q]
        by_quad[q] = _exact(subset, preds)

    long_empty = [it for it in items if it.quadrant.value == "long_empty"]
    le_recall = (round(sum(1 for it in long_empty if not preds[it.id][0]) / len(long_empty), 4)
                 if long_empty else 1.0)
    edge_aband = [it for it in items if it.quadrant.value == "edge_abandonment"]
    ea_recall = (round(sum(1 for it in edge_aband if preds[it.id][2]) / len(edge_aband), 4)
                 if edge_aband else 1.0)

    # Per-field Precision / Recall / F1 (spec §7.2), True = positive class
    by_field: dict[str, dict[str, float]] = {}
    for fi, f in enumerate(_FIELDS):
        tp = sum(1 for it in items if preds[it.id][fi] and _gold(it)[fi])
        fp = sum(1 for it in items if preds[it.id][fi] and not _gold(it)[fi])
        fn = sum(1 for it in items if not preds[it.id][fi] and _gold(it)[fi])
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        by_field[f] = {"precision": precision, "recall": recall, "f1": f1}

    overall = sum(field_acc.values()) / len(_FIELDS)
    passed = (overall >= MIN_OVERALL_ACC
              and le_recall >= LONG_EMPTY_EMPATHY_FALSE_RECALL
              and ea_recall >= EDGE_ABANDONMENT_RECALL)
    per_item = [{"id": it.id, "language": it.language.value, "quadrant": it.quadrant.value,
                 "gold": list(_gold(it)), "pred": list(preds[it.id]),
                 "match": preds[it.id] == _gold(it)} for it in items]
    return CalibrationReport(
        n=len(items), overall_field_accuracy=field_acc, exact_triple_match=_exact(items, preds),
        by_language=by_lang, by_quadrant=by_quad,
        long_empty_empathy_false_recall=le_recall, edge_abandonment_recall=ea_recall,
        by_field=by_field, passed=passed, per_item=per_item)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cascade.eval_calibration")
    parser.add_argument("--model", required=True, help="model_id in data/models.yaml")
    parser.add_argument("--data", default=None, help="calibration.yaml path (default: packaged)")
    parser.add_argument("--out", default=None, help="write report JSON here")
    parser.add_argument("--dump-preds", action="store_true",
                        help="print per-item gold-vs-pred mismatches to console")
    args = parser.parse_args(argv)

    from cascade.config import build_model
    from cascade.content import _load_yaml
    from cascade.material import CalibrationSet, load_calibration
    cal = (CalibrationSet(**_load_yaml(Path(args.data))) if args.data else load_calibration())
    report = run_calibration(CEvaluator(model=build_model(args.model)), cal.items)

    print(json.dumps(report.model_dump(exclude={"per_item"}), ensure_ascii=False, indent=2))
    print("PASS" if report.passed else "FAIL")
    if args.dump_preds:
        mism = [r for r in report.per_item if not r["match"]]
        print(f"\n=== {len(mism)}/{report.n} mismatches (gold vs pred; [empathy,deesc,aband]) ===")
        for r in mism:
            print(f"  {r['id']:32s} [{r['quadrant']:18s}] gold={r['gold']} pred={r['pred']}")
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
