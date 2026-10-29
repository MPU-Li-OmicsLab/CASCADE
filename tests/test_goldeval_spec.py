from cascade.goldeval.spec import FieldSpec, JUDGES, ALL_FIELDS, RISK_LEVELS
from cascade.schema import (TurnRecord, CAppraisal, TurnScore, CSSRSStage)

_APPR = CAppraisal(empathy_detected=True, deescalation_attempt=False,
                   abandonment_felt=True, c_new=0.5)


def _turn(score=None, risk=None):
    return TurnRecord(turn=1, target_response="r", appraisal=_APPR, patient_utterance="p",
                      attack_stage=CSSRSStage.PASSIVE_IDEATION, c=0.5,
                      expressed_risk=risk, score=score)


def test_judges_shape():
    assert set(JUDGES) == {"c_evaluator", "risk_labeler", "rubric_judge"}
    assert [f.name for f in JUDGES["c_evaluator"]] == ["empathy", "deesc", "abandon"]
    assert [f.name for f in JUDGES["risk_labeler"]] == ["expressed_risk"]
    assert [f.name for f in JUDGES["rubric_judge"]] == [f"d{i}" for i in range(1, 7)]
    assert len(ALL_FIELDS) == 10


def test_risk_levels_are_none_floor_plus_six_cssrs():
    assert RISK_LEVELS == ("none",) + tuple(s.value for s in CSSRSStage)
    assert len(RISK_LEVELS) == 7 and RISK_LEVELS[0] == "none"     # no-ideation floor


def test_c_evaluator_extract_reads_appraisal_booleans_as_ints():
    t = _turn()
    by = {f.name: f for f in JUDGES["c_evaluator"]}
    assert by["empathy"].extract(t) == 1     # empathy_detected=True
    assert by["deesc"].extract(t) == 0       # deescalation_attempt=False
    assert by["abandon"].extract(t) == 1     # abandonment_felt=True
    assert all(f.kind == "binary" and f.levels == (0, 1) for f in JUDGES["c_evaluator"])


def test_risk_extract_is_stage_value_or_none_literal_or_none():
    rf = JUDGES["risk_labeler"][0]
    assert rf.extract(_turn(risk=CSSRSStage.PLAN)) == "plan"
    assert rf.extract(_turn(risk="none")) == "none"          # no-ideation floor (str, not a CSSRSStage)
    assert rf.extract(_turn(risk=None)) is None              # unlabeled / unparseable
    assert rf.kind == "ordinal" and rf.levels == RISK_LEVELS


def test_rubric_extract_reads_score_or_none():
    d3 = {f.name: f for f in JUDGES["rubric_judge"]}["d3"]
    assert d3.extract(_turn(score=TurnScore(d1=0, d2=1, d3=2, d4=0, d5=1, d6=2))) == 2
    assert d3.extract(_turn(score=None)) is None
    assert all(f.kind == "ordinal" and f.levels == (0, 1, 2) for f in JUDGES["rubric_judge"])


def test_empathy_aux_never_a_field():
    assert "empathy_aux" not in {f.name for f in ALL_FIELDS}
