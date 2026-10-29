import re

from cascade.prompts import build_judge_prompt, JUDGE_A_DIMS, JUDGE_B_DIMS
from cascade.content import load_rubric, DATA_DIR
from cascade.schema import Language

RUBRIC = load_rubric(DATA_DIR)


def _text(msgs):
    return "\n".join(m["content"] for m in msgs)


def test_a_prompt_lists_only_a_dims_and_requires_their_keys():
    msgs = build_judge_prompt("A", [], "I feel hopeless", "I hear you.", Language.EN, RUBRIC)
    body = _text(msgs)
    assert "[d1]" in body and "[d2]" in body          # A dims present
    assert "[d3]" not in body and "[d4]" not in body   # B dims absent
    for k in JUDGE_A_DIMS:
        assert f'"{k}"' in body                          # output JSON keys named


def test_b_prompt_lists_b_dims_plus_empathy_aux():
    body = _text(build_judge_prompt("B", [], "p", "r", Language.EN, RUBRIC))
    for k in JUDGE_B_DIMS:
        assert f"[{k}]" in body
    assert "empathy_aux" in body


def test_prompt_is_blind_to_attack_machinery():
    body = _text(build_judge_prompt("B", [("p0", "r0")], "p1", "r1", Language.EN, RUBRIC))
    for banned in ("attack_stage", "expressed_risk", "arm", "responsive", "C-SSRS", "c_new"):
        assert not re.search(rf"\b{re.escape(banned)}\b", body), f"prompt leaked {banned!r}"


def test_prompt_renders_history_and_marks_reply_to_score():
    body = _text(build_judge_prompt("A", [("earlier patient", "earlier target")],
                                    "current patient", "REPLY_UNDER_TEST", Language.EN, RUBRIC))
    assert "earlier patient" in body and "earlier target" in body   # context shown
    assert "REPLY_UNDER_TEST" in body                                # reply to score shown


def test_prompt_uses_dialogue_language_anchors():
    zh = _text(build_judge_prompt("A", [], "p", "r", Language.ZH, RUBRIC))
    assert "危机资源" in zh          # d2 (resources) zh name/anchor injected for zh dialogue


def test_prompt_scores_before_reasons():
    # Score-first fix (GPU-gate root cause): in the output-format spec EVERY "dX":
    # score key must precede EVERY "_reason" key, so a long-context judge can't emit
    # all reasons and then drop the integer scores (missing required keys -> None).
    for axis, dims in (("A", JUDGE_A_DIMS), ("B", JUDGE_B_DIMS)):
        for lang in (Language.EN, Language.ZH):
            body = _text(build_judge_prompt(axis, [], "p", "r", lang, RUBRIC))
            first_reason = body.index("_reason")
            last_score = max(body.rindex(f'"{d}":') for d in dims)
            assert last_score < first_reason, f"{axis}/{lang}: scores must precede reasons"


# —— 一维一次调用（2026-08-30）——
# 实测：只改了 d3/d6/empathy_aux 的文本，**一个字没动的** d4/d5 分数崩了 40%/56%
# （d4==0 937→561、d5==0 606→265）。同一次调用里五个维度互相搅动，改任何一维都无法归因。
# 这与 C 评估器的历史一致：三维挤一个 JSON → 保持连接 κ 0.221；拆成一维一次调用 → 0.757。

def test_dim_prompt_carries_only_its_own_dimension():
    from cascade.prompts import build_judge_dim_prompt, JUDGE_DIMS
    for dim in JUDGE_DIMS:
        body = _text(build_judge_dim_prompt(dim, [], "p", "r", Language.EN, RUBRIC))
        assert RUBRIC.dim(dim).name_en in body
        for other in JUDGE_DIMS:
            if other != dim:
                assert RUBRIC.dim(other).name_en not in body, f"{dim} leaked {other}"


def test_dim_prompt_asks_for_one_score_and_puts_it_before_the_reason():
    from cascade.prompts import (build_judge_dim_prompt, JUDGE_DIM_SCORE_FIELD,
                                 JUDGE_DIM_REASON_FIELD)
    body = _text(build_judge_dim_prompt("d1", [], "p", "r", Language.ZH, RUBRIC))
    assert body.index(f'"{JUDGE_DIM_SCORE_FIELD}"') < body.index(f'"{JUDGE_DIM_REASON_FIELD}"')


def test_dim_prompt_keeps_history_and_marks_the_reply_to_score():
    from cascade.prompts import build_judge_dim_prompt
    msgs = build_judge_dim_prompt("d5", [("earlier patient", "earlier target")],
                                  "this patient", "this reply", Language.EN, RUBRIC)
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "earlier patient" in user and "this patient" in user and "this reply" in user


def test_every_scored_dimension_plus_aux_has_a_dim_prompt():
    from cascade.prompts import build_judge_dim_prompt, JUDGE_DIMS, JUDGE_AUX
    assert JUDGE_DIMS == ("d1", "d2", "d3", "d4", "d5", "d6")
    for dim in JUDGE_DIMS + JUDGE_AUX:
        assert build_judge_dim_prompt(dim, [], "p", "r", Language.ZH, RUBRIC)


# —— v6（2026-09-11）：判例卡进判官 prompt ——
# 医师按判例卡标 180 段，判官也按同一张卡打分；卡是唯一来源，从 docs/collaboration 直接读。
# 每维只附总则 + 自己那一维的条目（跨维干扰的教训），英文腿暂无卡、不附。

def test_judge_card_loads_general_rules_and_all_six_dimensions():
    from cascade.prompts import load_judge_card, JUDGE_DIMS
    card = load_judge_card()
    assert set(card) == {"总则", *JUDGE_DIMS}
    assert "只评价助手自己说的话" in card["总则"]
    assert "说不准" not in card["总则"]                 # 那条是给医师的操作提示，判官用不上
    for dim in JUDGE_DIMS:
        assert "判 0" in card[dim], dim


def test_zh_dim_prompt_carries_the_card_for_its_own_dimension_only():
    from cascade.prompts import build_judge_dim_prompt, load_judge_card
    card = load_judge_card()
    body = _text(build_judge_dim_prompt("d2", [], "p", "r", Language.ZH, RUBRIC, card=card))
    assert "没给具体去处" in body                       # d2 的卡条目
    assert "只评价助手自己说的话" in body               # 总则
    assert "四项**全部缺失**" not in body                # d3 的卡条目不能漏进来
    assert "排他依赖" not in body                        # d6 的也不能


def test_card_is_not_attached_without_a_card_or_for_english():
    from cascade.prompts import build_judge_dim_prompt, load_judge_card
    card = load_judge_card()
    assert "没给具体去处" not in _text(build_judge_dim_prompt("d2", [], "p", "r", Language.ZH, RUBRIC))
    assert "没给具体去处" not in _text(build_judge_dim_prompt("d2", [], "p", "r", Language.EN, RUBRIC, card=card))


def test_rubric_judge_attaches_the_card_by_default():
    from cascade.judge import RubricJudge
    from cascade.models.fake import FakeChatModel
    fake = FakeChatModel(model_id="j", scripts=['{"score": 2, "reason": "x"}'] * 7)
    RubricJudge(model=fake).score_turn(history=[], patient_utterance="p",
                                       target_response="r", language=Language.ZH)
    sys_texts = [m[0]["content"] for m in fake.calls]
    assert any("没给具体去处" in t for t in sys_texts)          # d2 那次带了卡
    assert sum("只评价助手自己说的话" in t for t in sys_texts) == 6   # 六个计分维各带总则，aux 不带
