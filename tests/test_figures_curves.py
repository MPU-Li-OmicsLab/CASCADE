from cascade.figures.curves import decay_curve, c_trajectory
from cascade.schema import (TurnRecord, DialogueRecord, CAppraisal, CaseProfile, TurnScore,
                            CSSRSStage, Arm, Language)

_APPR = CAppraisal(empathy_detected=False, deescalation_attempt=False,
                   abandonment_felt=False, c_new=0.5)
_OK = TurnScore(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)       # leaderboard_a/b = 1.0
_GATE = TurnScore(d1=2, d2=2, d3=2, d4=0, d5=2, d6=2)     # leaderboard_b = 0.0


def _dlg(pid, scores, cs=None):
    prof = CaseProfile(id=pid, seed_text="s", persona_profile="x",
                       trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                       language=Language.EN, explicit=True)
    cs = cs or [0.5] * len(scores)
    turns = [TurnRecord(turn=i + 1, target_response="r", appraisal=_APPR, patient_utterance="p",
                        attack_stage=CSSRSStage.PASSIVE_IDEATION, c=c, score=s)
             for i, (s, c) in enumerate(zip(scores, cs))]
    return DialogueRecord(case_profile=prof, arm=Arm.RESPONSIVE, turns=turns,
                          target_model_id="m", seed=1)


def test_decay_curve_per_turn_mean():
    # turn1: both 1.0 -> mean 1.0 ; turn2: 1.0 and 0.0 -> mean 0.5
    recs = [_dlg("a", [_OK, _OK]), _dlg("b", [_OK, _GATE])]
    d = decay_curve(recs, leaderboard="B")
    assert d["turns"] == [1, 2]
    assert d["mean"] == [1.0, 0.5]


def test_decay_curve_ragged_alignment():
    # turn3 only present in dlg 'a' -> mean over 1 dialogue
    recs = [_dlg("a", [_OK, _OK, _OK]), _dlg("b", [_OK, _OK])]
    d = decay_curve(recs, leaderboard="A")
    assert d["turns"] == [1, 2, 3]
    assert d["mean"][2] == 1.0        # only dlg a at turn 3


def test_decay_curve_ci_deterministic_and_brackets_mean():
    recs = [_dlg("a", [_OK, _GATE]), _dlg("b", [_OK, _OK])]
    d1 = decay_curve(recs, leaderboard="B", seed=0)
    d2 = decay_curve(recs, leaderboard="B", seed=0)
    assert d1 == d2                                   # deterministic under fixed seed
    for lo, m, hi in zip(d1["lo"], d1["mean"], d1["hi"]):
        assert lo <= m <= hi


def test_c_trajectory_mean():
    recs = [_dlg("a", [_OK, _OK], cs=[0.2, 0.4]), _dlg("b", [_OK, _OK], cs=[0.6, 0.8])]
    d = c_trajectory(recs)
    assert d["turns"] == [1, 2]
    assert d["mean"] == [0.4, 0.6]     # (0.2+0.6)/2, (0.4+0.8)/2
