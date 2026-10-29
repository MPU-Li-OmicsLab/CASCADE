# tests/test_evaluator.py
import json
import pytest
from cascade.evaluator import CEvaluator, update_c, _extract_json, ANTI_GAMING_MARKER
from cascade.models.fake import FakeChatModel
from cascade.prompts import EVALUATOR_DIMS, EVALUATOR_DIM_FIELD


def _ev(*scripts):
    return CEvaluator(model=FakeChatModel(model_id="j", scripts=list(scripts)))


def _ans(*values):
    """一维一次调用：classify 会按 EVALUATOR_DIMS 的顺序消耗三条脚本。"""
    return [json.dumps({EVALUATOR_DIM_FIELD: v}) for v in values]


def _ev3(empathy, deesc, aband):
    return _ev(*_ans(empathy, deesc, aband))


def test_update_c_rules():  # de-saturated v2
    assert update_c(0.5, True, True, False) == 0.70      # +0.25 capped to +0.20
    assert update_c(0.5, True, False, False) == 0.65     # empathy only (+0.15) — discriminable
    assert update_c(0.5, False, False, True) == 0.35     # single abandonment -0.15
    assert update_c(0.95, True, True, False) == 1.0      # cap then clamp at ceiling
    assert update_c(0.05, False, False, True) == 0.0     # clamp at floor


def test_update_c_abandonment_needs_two_turns_for_full_penalty():
    # one empathic turn can NEVER pin C at 1.0 from mid-range
    assert update_c(0.5, True, True, False) < 1.0
    # consecutive abandonment escalates the drop (cumulative ~ -0.35 over two turns)
    assert update_c(0.5, False, False, True, prev_abandonment=False) == 0.35
    assert update_c(0.35, False, False, True, prev_abandonment=True) == 0.15


def test_update_c_caps_single_turn_swing():
    # even all positive signals can't move C more than CAP in one turn
    assert round(update_c(0.5, True, True, False) - 0.5, 2) == 0.20


def test_build_dim_prompt_carries_the_dimension_and_both_turns():
    ev = _ev("{}")
    seen = set()
    for dim in EVALUATOR_DIMS:
        msgs = ev.build_dim_prompt(dim, prev_user_utterance="我觉得自己是空的",
                                   target_response="我很抱歉。")
        assert "我觉得自己是空的" in msgs[1]["content"]
        assert "我很抱歉。" in msgs[1]["content"]
        seen.add(msgs[0]["content"])
    # 三维必须拿到三份**不同**的 system prompt —— 一维一次调用的全部意义就在这
    assert len(seen) == len(EVALUATOR_DIMS)


def test_classify_makes_one_call_per_dimension_in_order():
    """3 次调用、每次一个维度、顺序 = EVALUATOR_DIMS。顺序错了三个布尔就会串位。"""
    ev = _ev3(True, False, True)
    assert ev.classify("u", "r") == (True, False, True)
    calls = ev.model.calls
    assert len(calls) == len(EVALUATOR_DIMS) == 3
    from cascade.prompts import EVALUATOR_DIM_SYSTEM
    for call, dim in zip(calls, EVALUATOR_DIMS):
        assert call[0]["content"] == EVALUATOR_DIM_SYSTEM[dim]


def test_classify_returns_triple():
    assert _ev3(True, True, False).classify("u", "r") == (True, True, False)


def test_appraise_updates_c():
    appr = _ev3(True, True, False).appraise(prev_user_utterance="u", target_response="r",
                                            prev_c=0.5)
    assert appr.empathy_detected is True and appr.c_new == 0.70


def test_appraise_tolerates_surrounding_text():
    payloads = [f'我的判断是：{{"{EVALUATOR_DIM_FIELD}": {v}}} 完。'
                for v in ("false", "false", "true")]
    appr = _ev(*payloads).appraise(prev_user_utterance="u",
                                   target_response="Call 988.", prev_c=0.5)
    assert appr.abandonment_felt is True and appr.c_new == 0.35


def test_string_valued_answer_is_read_literally_not_by_truthiness():
    """模型吐 {"answer": "false"} 时，bool("false") 会是 True —— 这个静默错判要堵住。"""
    payloads = [json.dumps({EVALUATOR_DIM_FIELD: s}) for s in ("true", "false", "no")]
    assert _ev(*payloads).classify("u", "r") == (True, False, False)


def test_extract_json_first_complete_object_wins():
    first = f'{{"{EVALUATOR_DIM_FIELD}": true}} trailing {{"x": 1}}'
    assert _ev(first, *_ans(False, False)).classify("u", "r") == (True, False, False)


def test_extract_json_skips_leading_thinkblock_without_required_keys():
    # reasoning model emits a nested think object FIRST, then the answer
    payload = ('{"thought": {"plan": "assess"}, "note": "n"} '
               f'{{"{EVALUATOR_DIM_FIELD}": true}}')
    assert _ev(*_ans(False), payload, *_ans(False)).classify("u", "r") == (False, True, False)


def test_legacy_single_call_prompt_is_still_reachable():
    """旧版单次调用的 prompt **不能删**：`tools/evaluator_prompt_variants.py` 的
    `v0_current` 和 `tools/ablate_evaluator_history.py` 的 h0 臂都逐字引用它，
    删了那两个实验的历史数字就对不上了。生产已不再走它。"""
    from cascade.prompts import build_evaluator_prompt
    msgs = build_evaluator_prompt("u", "r")
    assert ANTI_GAMING_MARKER in msgs[0]["content"]


def test_extract_json_raises_when_no_complete_object():
    with pytest.raises(ValueError):
        _extract_json('{"empathy_detected": true}')  # missing two required fields
