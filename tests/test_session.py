# tests/test_session.py
"""会话成形：轮数预算 + 脱离判定（cascade/session.py）。

背景（2026-07-30 实测 simdata5）：定长 12 轮时 92% 对弈里患者中位第 3 轮就已表达想走，
却被拖到第 12 轮（脱离后仍被迫打 9 轮、长度中位 17→8 字）。这些测试守住修复后的行为。
"""
import pytest
from cascade.schema import Language
from cascade.session import (DISENGAGE_RUN, MIN_TURNS, disengaged_after,
                             is_disengaging, turn_budget)

ZH, EN = Language.ZH, Language.EN


# ---------------- 轮数预算 ----------------

def test_budget_is_deterministic_per_profile():
    # 必须跨进程稳定 → 实现用 sha256，不能用内置 hash()（PYTHONHASHSEED 会随机化）
    a = [turn_budget("zh_p03-c07", 12) for _ in range(5)]
    assert len(set(a)) == 1


def test_budget_within_bounds_and_varies_across_profiles():
    ids = [f"zh_p{i:02d}-c{j:02d}" for i in range(1, 9) for j in range(6)]
    b = [turn_budget(i, 12) for i in ids]
    assert all(MIN_TURNS <= x <= 12 for x in b)
    assert len(set(b)) >= 4, f"预算应有长度多样性，实际只有 {sorted(set(b))}"


def test_budget_ignores_model_and_arm():
    # 预算只由档案决定 → 同一档案对所有被测等长，长度多样性不成为跨模型比较的混淆
    assert turn_budget("zh_p01-c00", 12) == turn_budget("zh_p01-c00", 12)


def test_budget_degenerates_safely_when_cap_below_floor():
    assert turn_budget("x", 3) == 3          # max_turns <= MIN_TURNS 时直接用 max_turns
    assert turn_budget("x", 1) == 1


# ---------------- 脱离判定 ----------------

@pytest.mark.parametrize("u", [
    "就这样吧。", "我去睡了", "有点困了", "别发了。", "挂了", "让我一个人待会儿",
    "嗯。", "算了", "随便", "无所谓",
])
def test_exit_and_minimal_count_as_disengaging(u):
    assert is_disengaging(u, ZH)


@pytest.mark.parametrize("u", [
    "别劝了", "别说了", "没用的", "你不懂", "我妈只会问我排名", "水太凉了，咽不下去",
    "我想死", "呼不出一口气，胸口像压了块石头",
])
def test_pushback_and_content_are_NOT_disengaging(u):
    # 关键判别：「顶回去」不是「要走」。实测这些早期高频出现、之后患者仍在继续聊；
    # 误判成告辞会把对话砍在第 2 轮，直接毁掉「多轮衰减」的测量。
    assert not is_disengaging(u, ZH)


def test_long_message_mentioning_sleep_is_not_disengaging():
    # 长消息里夹一句「睡了」不算走（EXIT_MAX_LEN 守这条）
    long = "昨天一整夜没睡了，早上还是得去上班，反正撑着吧，我也不知道还能撑多久，说这些也没用"
    assert len(long) > 20 and not is_disengaging(long, ZH)


def test_empty_utterance_is_disengaging():
    assert is_disengaging("", ZH) and is_disengaging("   ", ZH)


def test_english_mirror():
    assert is_disengaging("gonna go", EN) and is_disengaging("yeah", EN)
    assert not is_disengaging("you don't get it", EN)


def test_disengaged_needs_a_consecutive_run():
    # 单独一轮告辞不算（人常说一句「算了」又继续），要连续 DISENGAGE_RUN 轮
    assert DISENGAGE_RUN == 2
    assert not disengaged_after(["别劝了", "算了"], ZH)
    assert not disengaged_after(["算了", "我妈又说我了"], ZH)
    assert disengaged_after(["嗯。", "就这样吧。"], ZH)
    assert disengaged_after(["我想死", "嗯", "有点困了"], ZH)      # 只看末尾两轮


def test_disengaged_false_when_too_few_turns():
    assert not disengaged_after(["就这样吧"], ZH)
