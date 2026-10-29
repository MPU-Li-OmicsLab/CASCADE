from cascade.orchestrator import run_dialogue, continue_dialogue
from cascade.evaluator import CEvaluator
from cascade.actor import PatientSimulator
from cascade.schema import CaseProfile, CSSRSStage, Language, Arm, DialogueRecord
from cascade.models.fake import FakeChatModel, appraisal_scripts

_APPR = appraisal_scripts((True, True, False))


def _profile():
    return CaseProfile(id="x", seed_text="I feel empty", persona_profile="p",
                       trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                       language=Language.EN, explicit=True)


def _parts(n_turns=4):
    target = FakeChatModel("t", [f"r{i}" for i in range(1, n_turns + 1)])
    evaluator = CEvaluator(model=FakeChatModel("j", _APPR * n_turns))
    actor = PatientSimulator(model=FakeChatModel("a", [f"u{i}" for i in range(2, n_turns + 1)]),
                             profile=_profile())
    return target, evaluator, actor


def test_run_dialogue_unchanged_and_intervention_default_none():
    target, evaluator, actor = _parts()
    rec = run_dialogue(profile=_profile(), target=target, actor=actor, evaluator=evaluator,
                       max_turns=4, arm=Arm.FIXED, seed=1, fixed_rate=0.5)
    assert [t.patient_utterance for t in rec.turns] == ["I feel empty", "u2", "u3", "u4"]
    assert [t.target_response for t in rec.turns] == ["r1", "r2", "r3", "r4"]
    assert [t.turn for t in rec.turns] == [1, 2, 3, 4]
    assert rec.intervention is None and rec.risk_none is False
    # 被测没收到任何 system 消息
    assert all(m["role"] != "system" for call in target.calls for m in call)


def test_system_prompt_only_reaches_target():
    target, evaluator, actor = _parts()
    rec = run_dialogue(profile=_profile(), target=target, actor=actor, evaluator=evaluator,
                       max_turns=4, arm=Arm.FIXED, seed=1, fixed_rate=0.5,
                       target_system_prompt="STAY PRESENT")
    for call in target.calls:
        assert call[0] == {"role": "system", "content": "STAY PRESENT"}
        assert sum(m["role"] == "system" for m in call) == 1
    # 模拟器与评估器的调用里没有这句
    for m in actor.model.calls[0]:
        assert "STAY PRESENT" not in m["content"]
    for call in evaluator.model.calls:
        assert all("STAY PRESENT" not in m["content"] for m in call)
    assert rec.intervention is None            # 名字由 runner 填，orchestrator 不猜


def test_continue_dialogue_resumes_numbering_and_history():
    target, evaluator, actor = _parts(3)
    history = [{"role": "user", "content": "I feel empty"}, {"role": "assistant", "content": "r0"}]
    actor.restore(stage=CSSRSStage.ACTIVE_IDEATION, c=0.7)
    turns, end = continue_dialogue(profile=_profile(), target=target, actor=actor,
                                   evaluator=evaluator, arm=Arm.FIXED, seed=1,
                                   history=history, patient_utterance="u2",
                                   attack_stage=CSSRSStage.ACTIVE_IDEATION,
                                   prev_c=0.7, prev_abandonment=False,
                                   start_turn=2, end_turn=4, fixed_rate=0.5,
                                   start_stage=CSSRSStage.PASSIVE_IDEATION, sched_span=3)
    assert [t.turn for t in turns] == [2, 3, 4] and end == "max_turns"
    assert turns[0].patient_utterance == "u2" and turns[0].attack_stage == CSSRSStage.ACTIVE_IDEATION
    assert len(history) == 2 + 3 * 2            # 原地追加 3 次交换
    assert target.calls[0][:2] == [{"role": "user", "content": "I feel empty"},
                                   {"role": "assistant", "content": "r0"}]


def test_old_record_json_still_loads():
    raw = '{"case_profile":{"id":"x","seed_text":"s","persona_profile":"p","trajectory_stage_target":"plan","language":"zh","explicit":true},"arm":"fixed","turns":[],"target_model_id":"m","seed":1}'
    rec = DialogueRecord.model_validate_json(raw)
    assert rec.intervention is None and rec.risk_none is False
