import json
from cascade.analysis import intervention as I
from cascade.analysis import ceval
from cascade.schema import Arm, CAppraisal, CaseProfile, CSSRSStage, DialogueRecord, Language, TurnRecord

S = list(CSSRSStage)


def _rec(pid, model, abandon_flags, stages=None, risk_none=False, intervention=None):
    prof = CaseProfile(id=f"{pid}-c00", seed_text="s", persona_profile="p", trajectory_stage_target=CSSRSStage.PLAN,
                       language=Language.ZH, explicit=True, persona_id=pid)
    turns = [TurnRecord(turn=i + 1, target_response="r", patient_utterance="我还在说话呢",
                        attack_stage="none" if risk_none else (stages[i] if stages else S[min(i, 5)]),
                        c=0.5, appraisal=CAppraisal(empathy_detected=not a, deescalation_attempt=True,
                                                    abandonment_felt=a, c_new=0.5))
             for i, a in enumerate(abandon_flags)]
    return DialogueRecord(case_profile=prof, arm=Arm.FIXED, turns=turns, target_model_id=model, seed=1,
                          initial_stage="none" if risk_none else CSSRSStage.PASSIVE_IDEATION,
                          risk_none=risk_none, intervention=intervention)


def _ctrl():
    return [_rec(f"zh_p{i:02d}", "m", [False, False, True, True, True, True]) for i in range(1, 9)]


def _intv():
    return [_rec(f"zh_p{i:02d}", "m", [False, False, False, False, True, False], intervention="present_v1.zh")
            for i in range(1, 9)]


def test_rec_metrics():
    m = I.rec_metrics(_ctrl()[0])
    assert m["abandon_rate"] == 4 / 6 and m["empathy_rate"] == 2 / 6 and m["disengage_turn"] is None
    assert m["disengaged"] == 0.0


def test_sec_system_prompt_pairs_by_profile_and_stage():
    res = I.sec_system_prompt(_ctrl(), _intv(), seed=0, n_boot=50)
    cell = res["by_model"]["m"]["abandon_rate"]
    assert cell["control"]["mean"] > cell["intv"]["mean"]
    assert cell["paired"]["n"] == 8 and cell["paired"]["mean_diff"] < 0      # intv − control
    assert res["by_stage"]["m"]["high"]["control"]["mean"] == 1.0
    assert res["by_stage"]["m"]["high"]["intv"]["mean"] < 1.0
    assert res["by_model"]["m"]["disengage_turn"]["paired"] is None


def test_sec_none_control():
    none = [_rec(f"zh_p{i:02d}", "m", [False] * 6, risk_none=True) for i in range(1, 5)]
    res = I.sec_none_control(_ctrl(), none, seed=0, n_boot=50)
    assert res["m"]["disengage_rate"]["none"]["mean"] == 0.0
    assert res["m"]["abandon_rate"]["none"]["mean"] == 0.0
    assert res["m"]["abandon_rate"]["passive_turns"]["mean"] == 0.0
    assert res["m"]["abandon_rate"]["high_turns"]["mean"] == 1.0
    assert res["m"]["n_none_dialogues"] == 4


def test_run_all_and_render_with_missing_parts(tmp_path):
    res = I.run_all(_ctrl(), [], [], None, seed=0, n_boot=20)
    md = I.render_report(res)
    assert "系统提示" in md and "未提供" in md
    ctrl = tmp_path / "c.jsonl"; ctrl.write_text("\n".join(r.model_dump_json() for r in _ctrl()) + "\n", encoding="utf-8")
    iv = tmp_path / "i.jsonl"; iv.write_text("\n".join(r.model_dump_json() for r in _intv()) + "\n", encoding="utf-8")
    bj = tmp_path / "b.json"; bj.write_text(json.dumps({"n_branches": 3, "success": {"intv": 0.7, "regen": 0.3}}), encoding="utf-8")
    I.main(["--control", str(ctrl), "--intv", str(iv), "--branch", str(bj), "--out", str(tmp_path / "o")])
    out = (tmp_path / "o" / "report.md").read_text(encoding="utf-8")
    assert "present_v1.zh" in out and "0.70" in out
    assert (tmp_path / "o" / "results.json").exists()


def test_render_report_missing_success_key_does_not_raise():
    res = I.run_all(_ctrl(), [], [], {"n_branches": 2}, seed=0, n_boot=20)
    md = I.render_report(res)
    assert "—" in md


def test_ceval_run_all_filters_risk_none():
    recs = [_rec("zh_p01", "m", [False, False, True, True, True, True], risk_none=True),
            _rec("zh_p02", "m", [False, False, True, True, True, True])]
    res = ceval.run_all(recs, seed=0, n_boot=20)
    assert res["meta"]["n_dialogues"] == 1
