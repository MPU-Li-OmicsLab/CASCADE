import json
from cascade.judge import RubricJudge, score_dialogue
from cascade.models.fake import FakeChatModel
from cascade.schema import (TurnScore, TurnRecord, DialogueRecord, CAppraisal,
                            CaseProfile, CSSRSStage, Arm, Language)

# 一维一次调用：每轮 7 次（d1–d6 + 辅助项）。FakeChatModel 用尽脚本后会
# 重复最后一条，所以只需给一条就能喂完任意多轮。
DIM_OK = json.dumps({"score": 2, "reason": "ok"})
PER_TURN_CALLS = 7


def _judge(*scripts):
    return RubricJudge(model=FakeChatModel(model_id="judge", scripts=list(scripts)))


def test_axis_keys_interlock_with_leaderboard_sets():
    from cascade.scoring.leaderboard import A_DIMS, B_DIMS
    from cascade.prompts import JUDGE_DIMS
    assert set(JUDGE_DIMS) == set(A_DIMS) | set(B_DIMS)


def _rec(n):
    appr = CAppraisal(empathy_detected=False, deescalation_attempt=False,
                      abandonment_felt=False, c_new=0.5)
    prof = CaseProfile(id="p", seed_text="s", persona_profile="x",
                       trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                       language=Language.EN, explicit=True)
    turns = [TurnRecord(turn=i + 1, target_response=f"r{i}", appraisal=appr,
                        patient_utterance=f"p{i}", attack_stage=CSSRSStage.PASSIVE_IDEATION, c=0.5)
             for i in range(n)]
    return DialogueRecord(case_profile=prof, arm=Arm.RESPONSIVE, turns=turns,
                          target_model_id="m", seed=1)


def test_score_dialogue_fills_every_turn():
    judge = _judge(DIM_OK)
    out = score_dialogue(judge, _rec(2))
    assert all(t.score is not None for t in out.turns)


def test_score_dialogue_idempotent_skips_scored():
    judge = _judge(DIM_OK)
    rec = _rec(2)
    pre = rec.turns[0].model_copy(update={"score": TurnScore(d1=1, d2=1, d3=1, d4=1, d5=1, d6=1)})
    rec = rec.model_copy(update={"turns": [pre, rec.turns[1]]})
    out = score_dialogue(judge, rec)
    assert out.turns[0].score.d1 == 1               # pre-filled untouched
    assert out.turns[1].score is not None           # only the empty one scored
    assert len(judge.model.calls) == PER_TURN_CALLS  # 只给空的那一轮打分


def test_score_dialogue_passes_prior_turns_as_history():
    judge = _judge(DIM_OK)
    score_dialogue(judge, _rec(2))
    turn2_first_call = judge.model.calls[PER_TURN_CALLS]   # 第 2 轮的第一次调用
    joined = " ".join(m["content"] for m in turn2_first_call)
    assert "p0" in joined and "r0" in joined         # turn 1 is turn 2's context


def test_reasons_default_to_none_so_already_scored_data_still_loads():
    """加 reason 不能是破坏性改动：runs/prod_all/scored.jsonl 里已有几千条不带理由的分。"""
    s = TurnScore(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)
    assert s.d1_reason is None and s.empathy_aux_reason is None


def test_correction_is_absent_from_the_first_call():
    """4768 条一次就过的轮次，prompt 必须和现在跑出来的那批逐字一致 —— 否则分数不可比。"""
    j = _judge(DIM_OK)
    j.score_turn(history=[], patient_utterance="p", target_response="r", language=Language.EN)
    assert len(j.model.calls[0]) == 2                     # system + user，没有第三条


def _dim_json(score, reason="r"):
    from cascade.prompts import JUDGE_DIM_SCORE_FIELD, JUDGE_DIM_REASON_FIELD
    return json.dumps({JUDGE_DIM_SCORE_FIELD: score, JUDGE_DIM_REASON_FIELD: reason})


def test_score_turn_makes_one_call_per_dimension_in_order():
    """一维一次调用：7 次（d1–d6 + 辅助项），顺序 = JUDGE_DIMS + JUDGE_AUX。

    动机是实测的跨维干扰：只改 d3/d6/empathy_aux 的锚点，**一个字没动的** d4/d5
    分数崩了 40%/56%。同一次调用里维度互相搅动，改一维就无法归因任何结果。
    """
    from cascade.prompts import JUDGE_DIMS, JUDGE_AUX
    scripts = [_dim_json(i % 3, f"reason-{d}")
               for i, d in enumerate(JUDGE_DIMS + JUDGE_AUX)]
    j = _judge(*scripts)
    s = j.score_turn(history=[], patient_utterance="p", target_response="r",
                     language=Language.EN)
    assert len(j.model.calls) == len(JUDGE_DIMS) + len(JUDGE_AUX) == 7
    # 每一次调用拿到的 system prompt 必须各不相同 —— 一维一次调用的全部意义在此
    assert len({c[0]["content"] for c in j.model.calls}) == 7
    assert (s.d1, s.d2, s.d3, s.d4, s.d5, s.d6) == (0, 1, 2, 0, 1, 2)
    assert s.d1_reason == "reason-d1" and s.d6_reason == "reason-d6"
    assert s.empathy_aux == 0 and s.empathy_aux_reason == "reason-empathy_aux"


def test_score_turn_is_none_when_a_scored_dimension_never_parses():
    j = RubricJudge(model=FakeChatModel(model_id="j", scripts=["garbage"]), retries=1)
    assert j.score_turn(history=[], patient_utterance="p", target_response="r",
                        language=Language.EN) is None


def test_auxiliary_dimension_failure_does_not_sink_the_turn():
    """empathy_aux 永不聚合，它挂了不该让六个真维度的分一起作废。"""
    from cascade.prompts import JUDGE_DIMS
    j = RubricJudge(model=FakeChatModel(
        model_id="j", scripts=[_dim_json(2) for _ in JUDGE_DIMS] + ["not json"]), retries=0)
    s = j.score_turn(history=[], patient_utterance="p", target_response="r",
                     language=Language.EN)
    assert s is not None and s.d1 == 2 and s.empathy_aux is None
