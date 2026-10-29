import csv
from cascade.goldeval.sample import stratified_sample, export_annotation_package, TurnSample
from cascade.schema import (TurnRecord, DialogueRecord, CAppraisal, CaseProfile, TurnScore,
                            CSSRSStage, Arm, Language)

_APPR = CAppraisal(empathy_detected=True, deescalation_attempt=False,
                   abandonment_felt=False, c_new=0.5)


def _dlg(pid, lang, arm, stages):
    prof = CaseProfile(id=pid, seed_text="SEED_LINE", persona_profile="x",
                       trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                       language=lang, explicit=True)
    turns = [TurnRecord(turn=i + 1, target_response=f"resp{i}", appraisal=_APPR,
                        patient_utterance=f"pt{i}", attack_stage=st, c=0.5,
                        expressed_risk=st, score=TurnScore(d1=1, d2=1, d3=1, d4=1, d5=1, d6=1))
             for i, st in enumerate(stages)]
    return DialogueRecord(case_profile=prof, arm=arm, turns=turns, target_model_id="m", seed=1)


def test_flatten_prev_user_history_and_machine_legacy_v1():
    # 旧批（turn_pairing 缺失 = legacy_v1）：target_response 回的是**上一条**患者发言，
    # 首轮回的是 seed。旧数据（valbatch / cmp_eval）必须仍按旧语义还原，否则错标。
    s = stratified_sample([_dlg("d1", Language.EN, Arm.RESPONSIVE,
                                [CSSRSStage.PASSIVE_IDEATION, CSSRSStage.PLAN])],
                          n=2, seed=0, min_per_cell=1)
    by_id = {x.id: x for x in s}
    t1, t2 = by_id["d1#t1"], by_id["d1#t2"]
    assert t1.prev_user == "SEED_LINE" and t1.history == []
    assert t2.prev_user == "pt0" and t2.history == [("pt0", "resp0")]
    assert t1.machine["empathy"] == 1 and t1.machine["expressed_risk"] == "passive_ideation"
    assert t1.machine["d3"] == 1


def test_flatten_prev_user_is_same_turn_patient_line_in_exchange_v2():
    # exchange_v2：一条 TurnRecord 就是一次交换 → 被测回复应答的正是本条的 patient_utterance。
    # 标注模板『病人最新一句』这一列必须给对，否则医生按错的前文判 C 评估器。
    d = _dlg("d2", Language.EN, Arm.RESPONSIVE,
             [CSSRSStage.PASSIVE_IDEATION, CSSRSStage.PLAN])
    d = d.model_copy(update={"turn_pairing": "exchange_v2"})
    s = stratified_sample([d], n=2, seed=0, min_per_cell=1)
    by_id = {x.id: x for x in s}
    assert by_id["d2#t1"].prev_user == "pt0"        # 本条自己的患者发言，不是 seed
    assert by_id["d2#t2"].prev_user == "pt1"
    assert by_id["d2#t2"].history == [("pt0", "resp0")]   # history 仍是此前的完整交换


def test_risk_none_records_skipped_without_error():
    # A7 无风险段的 attack_stage 是字符串 "none"，.value 会炸；必须在入口被过滤掉。
    normal = _dlg("d1", Language.EN, Arm.RESPONSIVE, [CSSRSStage.PASSIVE_IDEATION])
    none_rec = _dlg("d2", Language.EN, Arm.FIXED, [CSSRSStage.PASSIVE_IDEATION])
    none_rec = none_rec.model_copy(update={"risk_none": True})
    s = stratified_sample([normal, none_rec], n=5, seed=0, min_per_cell=1)
    assert {x.id.split("#")[0] for x in s} == {"d1"}


def test_deterministic_same_seed():
    d = [_dlg("d1", Language.EN, Arm.RESPONSIVE, [CSSRSStage.PASSIVE_IDEATION] * 5)]
    a = [x.id for x in stratified_sample(d, n=3, seed=7)]
    b = [x.id for x in stratified_sample(d, n=3, seed=7)]
    assert a == b


def test_min_per_cell_floors_rare_stage():
    # 10 passive turns + 1 rare 'plan' turn; without a floor the rare stage is easily missed
    common = _dlg("c", Language.EN, Arm.RESPONSIVE, [CSSRSStage.PASSIVE_IDEATION] * 10)
    rare = _dlg("r", Language.EN, Arm.FIXED, [CSSRSStage.PLAN])
    s = stratified_sample([common, rare], n=4, seed=0, min_per_cell=1)
    assert any(x.attack_stage == "plan" for x in s)   # floor guaranteed the rare cell a slot


def test_export_blinds_and_writes_key(tmp_path):
    s = stratified_sample([_dlg("d1", Language.ZH, Arm.RESPONSIVE, [CSSRSStage.PLAN])],
                          n=1, seed=0, min_per_cell=1)
    export_annotation_package(s, tmp_path, annotators=2, double_frac=1.0, seed=0)
    text = (tmp_path / "annotate.csv").read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    # blinding: no machine-judgment / attack-machinery column leaks
    for banned in ("attack_stage", "expressed_risk", "appraisal", "score", "arm", ",c,"):
        assert banned not in header
    assert "plan" not in text          # attack_stage value never leaks into the sheet
    assert "h_empathy" in header and "h_d6" in header
    # key.jsonl carries the hidden machine judgments
    import json
    key = json.loads((tmp_path / "key.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert key["machine"]["expressed_risk"] == "plan" and key["attack_stage"] == "plan"
    # double_frac=1.0 -> the one item is assigned to both annotators (IAA)
    assign = json.loads((tmp_path / "assignment.json").read_text(encoding="utf-8"))
    assert assign["assignment"]["d1#t1"] == ["ann1", "ann2"]
