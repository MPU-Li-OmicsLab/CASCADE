# tests/test_eval_calibration.py
from cascade.eval_calibration import run_calibration, CalibrationReport
from cascade.evaluator import CEvaluator
from cascade.material import CalibrationItem
from cascade.models.fake import FakeChatModel, appraisal_scripts
from cascade.schema import Language


def _item(id_, lang, quad, gold):
    e, d, a = gold
    return CalibrationItem(
        id=id_, language=lang, quadrant=quad,
        user_utterance="u", target_response="t",
        gold={"empathy_detected": e, "deescalation_attempt": d, "abandonment_felt": a},
        rationale="r")


def _scripts(*triples):
    """每个三元组摊成 3 条 —— 评估器一维一次调用。"""
    return appraisal_scripts(*triples)


def test_perfect_predictions_pass_gate():
    items = [
        _item("zh_long_empty_01", Language.ZH, "long_empty", (False, False, False)),
        _item("en_edge_aband_01", Language.EN, "edge_abandonment", (False, True, True)),
    ]
    ev = CEvaluator(model=FakeChatModel(model_id="j",
                    scripts=_scripts((False, False, False), (False, True, True))))
    rep = run_calibration(ev, items)
    assert isinstance(rep, CalibrationReport)
    assert rep.exact_triple_match == 1.0
    assert rep.long_empty_empathy_false_recall == 1.0
    assert rep.edge_abandonment_recall == 1.0
    assert rep.passed is True


def test_long_empty_false_empathy_fails_gate():
    # evaluator fooled by a long-but-empty reply -> empathy=True on a long_empty item
    items = [_item("zh_long_empty_02", Language.ZH, "long_empty", (False, False, False))]
    ev = CEvaluator(model=FakeChatModel(model_id="j", scripts=_scripts((True, False, False))))
    rep = run_calibration(ev, items)
    assert rep.long_empty_empathy_false_recall == 0.0
    assert rep.passed is False


def test_edge_abandonment_missed_fails_gate():
    # evaluator MISSES abandonment on an edge_abandonment item -> recall 0 -> gate fails
    items = [_item("en_edge_aband_02", Language.EN, "edge_abandonment", (False, True, True))]
    ev = CEvaluator(model=FakeChatModel(model_id="j", scripts=_scripts((False, True, False))))
    rep = run_calibration(ev, items)
    assert rep.edge_abandonment_recall == 0.0
    assert rep.passed is False


def test_by_language_and_quadrant_breakdown():
    items = [
        _item("zh_short_substantive_01", Language.ZH, "short_substantive", (True, True, False)),
        _item("en_short_substantive_01", Language.EN, "short_substantive", (True, True, False)),
    ]
    ev = CEvaluator(model=FakeChatModel(model_id="j",
                    scripts=_scripts((True, True, False), (False, True, False))))
    rep = run_calibration(ev, items)
    assert rep.by_language["zh"]["exact"] == 1.0
    assert rep.by_language["en"]["exact"] == 0.0
    assert "short_substantive" in rep.by_quadrant


def test_per_field_prf1():
    # Hand-computed expected values:
    # item1: gold=(T,F,F) pred=(T,F,F) -> empathy:TP  deesc:TN  aband:TN
    # item2: gold=(T,F,F) pred=(F,T,F) -> empathy:FN  deesc:FP  aband:TN
    # item3: gold=(F,F,F) pred=(T,F,F) -> empathy:FP  deesc:TN  aband:TN
    #
    # empathy_detected: TP=1, FP=1, FN=1
    #   precision=1/2=0.5  recall=1/2=0.5  f1=2*0.5*0.5/1.0=0.5
    # deescalation_attempt: TP=0, FP=1, FN=0
    #   precision=0.0  recall=0.0 (TP+FN=0 -> 0.0)  f1=0.0
    # abandonment_felt: TP=0, FP=0, FN=0 -> all 0.0
    items = [
        _item("zh_prf_01", Language.ZH, "short_substantive", (True, False, False)),
        _item("zh_prf_02", Language.ZH, "short_substantive", (True, False, False)),
        _item("zh_prf_03", Language.ZH, "short_substantive", (False, False, False)),
    ]
    ev = CEvaluator(model=FakeChatModel(model_id="j", scripts=_scripts(
        (True, False, False),   # item1 pred: empathy TP
        (False, True, False),   # item2 pred: empathy FN, deesc FP
        (True, False, False),   # item3 pred: empathy FP
    )))
    rep = run_calibration(ev, items)
    assert set(rep.by_field.keys()) == {"empathy_detected", "deescalation_attempt", "abandonment_felt"}

    emp = rep.by_field["empathy_detected"]
    assert emp["precision"] == 0.5
    assert emp["recall"] == 0.5
    assert emp["f1"] == 0.5

    des = rep.by_field["deescalation_attempt"]
    assert des["precision"] == 0.0
    assert des["recall"] == 0.0
    assert des["f1"] == 0.0

    aband = rep.by_field["abandonment_felt"]
    assert aband["precision"] == 0.0
    assert aband["recall"] == 0.0
    assert aband["f1"] == 0.0


def test_per_item_records_gold_pred_match():
    items = [
        _item("zh_a", Language.ZH, "long_empty", (False, False, True)),
        _item("en_b", Language.EN, "short_substantive", (True, True, False)),
    ]
    ev = CEvaluator(model=FakeChatModel(model_id="j",
                    scripts=_scripts((False, False, True), (True, False, False))))
    rep = run_calibration(ev, items)
    assert len(rep.per_item) == 2
    a = next(r for r in rep.per_item if r["id"] == "zh_a")
    assert a["gold"] == [False, False, True] and a["pred"] == [False, False, True] and a["match"] is True
    b = next(r for r in rep.per_item if r["id"] == "en_b")
    assert b["gold"] == [True, True, False] and b["pred"] == [True, False, False] and b["match"] is False
