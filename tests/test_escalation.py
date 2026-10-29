import pytest
from cascade.escalation import fixed_stage_at
from cascade.schema import CSSRSStage

P = CSSRSStage.PASSIVE_IDEATION       # index 0
MT = CSSRSStage.MEANS_TIMING          # index 5
MI = CSSRSStage.METHOD_INQUIRY        # index 2


def test_zero_rate_stays_at_start():
    assert [fixed_stage_at(t, 0.0, P, MT) for t in range(1, 6)] == [P] * 5


def test_rate_one_climbs_one_per_turn():
    # int(1.0*t + 0.5): t1->1, t2->2, t3->3
    assert [fixed_stage_at(t, 1.0, P, MT).index for t in (1, 2, 3)] == [1, 2, 3]


def test_rate_half_climbs_every_other_turn():
    # int(0.5*t + 0.5): t1->1, t2->1, t3->2, t4->2, t5->3
    assert [fixed_stage_at(t, 0.5, P, MT).index for t in range(1, 6)] == [1, 1, 2, 2, 3]


def test_caps_at_target():
    # rate 1.0 but target METHOD_INQUIRY(2) -> capped at 2 from t>=2
    assert fixed_stage_at(10, 1.0, P, MI) == MI
    assert fixed_stage_at(2, 1.0, P, MI) == MI


def test_monotonic_non_decreasing():
    seq = [fixed_stage_at(t, 0.4, P, MT).index for t in range(1, 12)]
    assert seq == sorted(seq)


# ---- reach-target floor（2026-07-30）----
# 实测事故：拟合速率来自 responsive 臂，而高共情被测把 C 顶在 ~1.0 → 几乎不升级 →
# zh rate 掉到 0.0278；int(0.0278*12+0.5)==0，fixed 臂一阶都爬不动，全批高危档为 0
# （simdata3 fixed = 288/288 passive_ideation）。fixed 臂的定义就是「无视 C 按排程到达目标」，
# 到不了目标这条臂就没有意义。给 max_turns 即启用地板：rate = max(rate, span/max_turns)。

def test_collapsed_rate_still_reaches_target_with_floor():
    # 事故复现：真实标定值 0.0278、12 轮。不给 max_turns = 原样锁死；给了就必须到达目标。
    assert fixed_stage_at(12, 0.0278, P, MT).index == 0                    # 旧行为：爬不动
    assert fixed_stage_at(12, 0.0278, P, MT, max_turns=12) == MT           # 有地板：末轮到达


def test_floor_climbs_monotonically_and_never_overshoots():
    seq = [fixed_stage_at(t, 0.0, P, MT, max_turns=12).index for t in range(1, 13)]
    assert seq == sorted(seq)                    # 单调不减
    assert seq[-1] == MT.index                   # 末轮抵达目标
    assert max(seq) <= MT.index                  # 绝不越过目标


def test_floor_never_slows_a_healthy_rate():
    # 拟合速率已经够快时，地板不得拖慢它（取 max，不是取代）
    fast = [fixed_stage_at(t, 1.0, P, MT, max_turns=12).index for t in (1, 2, 3)]
    assert fast == [1, 2, 3]


def test_floor_noop_when_target_equals_start():
    # span==0（目标就是起始档）→ 不得因除法/取整而误升
    assert fixed_stage_at(5, 0.0, P, P, max_turns=12) == P


from cascade.escalation import fit_escalation_rates, EscalationRates, LangRate
from cascade.schema import (DialogueRecord, TurnRecord, CAppraisal, CaseProfile,
                            Language, Arm)


def _turn(i, stage, c):
    appr = CAppraisal(empathy_detected=False, deescalation_attempt=False,
                      abandonment_felt=True, c_new=c)
    return TurnRecord(turn=i, target_response="r", appraisal=appr,
                      patient_utterance="u", attack_stage=stage, c=c)


def _dlg(lang, stage_c_pairs, arm=Arm.RESPONSIVE, init=CSSRSStage.PASSIVE_IDEATION):
    turns = [_turn(i + 1, s, c) for i, (s, c) in enumerate(stage_c_pairs)]
    prof = CaseProfile(id="d", seed_text="s", persona_profile="p",
                       trajectory_stage_target=MT, strategies=[],
                       language=lang, explicit=True)
    return DialogueRecord(case_profile=prof, arm=arm, turns=turns,
                          target_model_id="m", seed=1, initial_stage=init)


def test_rate_is_net_delta_over_turns():
    # PASSIVE(0) -> ACTIVE(1) -> METHOD(2): net delta 2 over 2 turns = 1.0
    d = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4), (MI, 0.2)])
    r = fit_escalation_rates([d]).per_language["en"]
    assert r.rate == 1.0
    assert r.n_dialogues == 1 and r.n_turns == 2
    assert r.stage_advance_prob == 1.0


def test_fixed_arm_dialogues_excluded():
    resp = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4)])
    fix = _dlg(Language.EN, [(MT, 0.1)], arm=Arm.FIXED)
    assert fit_escalation_rates([resp, fix]).per_language["en"].n_dialogues == 1


def test_raises_without_responsive():
    fix = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4)], arm=Arm.FIXED)
    with pytest.raises(ValueError, match="no responsive"):
        fit_escalation_rates([fix])


def test_raises_without_initial_stage():
    d = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4)])
    d.initial_stage = None
    with pytest.raises(ValueError, match="initial_stage"):
        fit_escalation_rates([d])


def test_right_censored_mean_c_trajectory():
    d1 = _dlg(Language.ZH, [(CSSRSStage.ACTIVE_IDEATION, 0.4), (MI, 0.3)])
    d2 = _dlg(Language.ZH, [(CSSRSStage.ACTIVE_IDEATION, 0.6)])
    traj = fit_escalation_rates([d1, d2]).per_language["zh"].mean_c_trajectory
    assert traj == [0.5, 0.3]   # turn1=(0.4+0.6)/2, turn2 only d1


def test_round_trips_through_json():
    d = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4)])
    rates = fit_escalation_rates([d])
    assert EscalationRates.model_validate_json(rates.model_dump_json()) == rates


def test_escalation_cli_writes_rates(tmp_path):
    from cascade.escalation import main
    from cascade.io import write_jsonl
    d = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4), (MI, 0.2)])
    inp, out = tmp_path / "d.jsonl", tmp_path / "esc.json"
    write_jsonl([d], inp)
    main(["--in", str(inp), "--out", str(out)])
    rates = EscalationRates.model_validate_json(out.read_text(encoding="utf-8"))
    assert rates.per_language["en"].rate == 1.0


def test_multi_language_partition():
    # invariant 7 (bilingual symmetry): zh and en calibrate independently in one call
    en = _dlg(Language.EN, [(CSSRSStage.ACTIVE_IDEATION, 0.4), (MI, 0.2)])  # 2 turns
    zh = _dlg(Language.ZH, [(CSSRSStage.ACTIVE_IDEATION, 0.5)])             # 1 turn
    rates = fit_escalation_rates([en, zh])
    assert set(rates.per_language) == {"en", "zh"}
    assert rates.per_language["en"].n_dialogues == 1 and rates.per_language["en"].n_turns == 2
    assert rates.per_language["zh"].n_dialogues == 1 and rates.per_language["zh"].n_turns == 1


def test_zero_turn_responsive_yields_zero_rate():
    # empty-turns edge: rate degrades to 0.0 with no crash (max_len=0 path)
    r = fit_escalation_rates([_dlg(Language.EN, [])]).per_language["en"]
    assert r.rate == 0.0
    assert r.mean_c_trajectory == []
    assert r.n_turns == 0
