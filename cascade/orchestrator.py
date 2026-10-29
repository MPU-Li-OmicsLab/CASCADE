# cascade/orchestrator.py
from __future__ import annotations
import random
from cascade.models.base import ChatModel, Message
from cascade.evaluator import CEvaluator
from cascade.actor import PatientSimulator
from cascade.escalation import fixed_stage_at
from cascade.session import disengaged_after
from cascade.schema import CaseProfile, Arm, TurnRecord, DialogueRecord


def _temp_of(model) -> float | None:
    params = getattr(model, "params", None)
    if isinstance(params, dict) and params.get("temperature") is not None:
        return float(params["temperature"])
    return None


def continue_dialogue(*, profile: CaseProfile, target: ChatModel, actor: PatientSimulator,
                      evaluator: CEvaluator, arm: Arm, seed: int,
                      history: list[Message], patient_utterance: str,
                      attack_stage, prev_c: float, prev_abandonment: bool,
                      start_turn: int, end_turn: int,
                      fixed_rate: float | None, start_stage, sched_span: int,
                      allow_disengage: bool = False,
                      prior_utterances: list[str] | None = None,
                      target_system_prompt: str | None = None,
                      risk_none: bool = False) -> tuple[list[TurnRecord], str]:
    """从任意前缀续跑 [start_turn, end_turn]。`history` 原地追加；返回 (turns, end_reason)。

    `run_dialogue` 是它从第 1 轮起的薄包装；`tools/branch_intervention.py` 用它做同前缀反事实分支
    （A6-B）。`target_system_prompt` 只在喂被测时临时前置，不进 history（模拟器/评估器/记录都看不到）。
    `risk_none`（A7）：跳过 fixed 日程，患者由 `actor.step_none` 生成，attack_stage 记 "none"。"""
    random.seed(seed)
    turns: list[TurnRecord] = []
    utterances = list(prior_utterances or [])
    end_reason = "max_turns"
    for t in range(start_turn, end_turn + 1):
        history.append({"role": "user", "content": patient_utterance})
        msgs = ([{"role": "system", "content": target_system_prompt}] + history
                if target_system_prompt else history)
        target_response = target.generate(msgs)
        history.append({"role": "assistant", "content": target_response})

        appraisal = evaluator.appraise(prev_user_utterance=patient_utterance,
                                       target_response=target_response, prev_c=prev_c,
                                       prev_abandonment=prev_abandonment)
        turns.append(TurnRecord(turn=t, target_response=target_response,
                                appraisal=appraisal, patient_utterance=patient_utterance,
                                attack_stage=attack_stage, c=appraisal.c_new))
        utterances.append(patient_utterance)
        prev_c = appraisal.c_new
        prev_abandonment = appraisal.abandonment_felt

        if t == end_turn:
            break
        if allow_disengage and disengaged_after(utterances, profile.language):
            end_reason = "disengaged"
            break
        if risk_none:
            patient_utterance, _, _ = actor.step_none(appraisal, history)
            attack_stage = "none"
        elif arm is Arm.RESPONSIVE:
            patient_utterance, attack_stage, _ = actor.step(appraisal=appraisal, history=history)
        else:
            scheduled = fixed_stage_at(t, fixed_rate, start_stage,
                                       profile.trajectory_stage_target,
                                       max_turns=sched_span)   # reach-target floor
            patient_utterance, attack_stage, _ = actor.step_fixed(appraisal, history, scheduled)
    return turns, end_reason


def run_dialogue(profile: CaseProfile, target: ChatModel, actor: PatientSimulator,
                 evaluator: CEvaluator, max_turns: int, arm: Arm, seed: int,
                 fixed_rate: float | None = None,
                 rewrite_opening: bool = False,
                 turn_budget: int | None = None,
                 allow_disengage: bool = False,
                 target_system_prompt: str | None = None,
                 risk_none: bool = False) -> DialogueRecord:
    """Exchange loop (exchange_v2). 每轮 = 患者说一句 → 被测回一句 → C 评估器给分；
    然后才决定要不要再产生下一句患者发言。因此**每条患者发言都有回应，对话以 AI 结尾**，
    turn=1 的患者侧就是开场白（语料 seed）。

    为什么这样排（2026-07-30 重构，legacy_v1 → exchange_v2）：v1 把「AI 对上一句的回复」
    和「患者对该回复的回应」塞进同一条 TurnRecord，导致末轮患者发言永远无人应答
    （fixed 臂末轮 = 峰值风险档，最该测的一次从未生成），且 `judge.score_turn` 把这两个
    错配字段当成配对交换喂判官 → v1 上的 6 维分无效。详见 `schema.TurnRecord` 文档。

    Responsive arm advances by C-gated decide_action; fixed arm advances by the
    deterministic calibrated schedule (ignoring C). Both arms run the evaluator and
    record C. `seed` seeds the stdlib RNG for a reproducible path.

    rewrite_opening: 第 1 条患者消息是否经 `actor.open()` 改写成聊天体。默认 False
    （语料 seed 原文照搬）——改写层已被 simdata5 实测证伪（意念率 29%→58%），
    仅作消融分支保留，详见 `prompts.build_opening_prompt` 的说明。

    turn_budget / allow_disengage: 会话成形（`cascade/session.py`）。turn_budget 覆盖
    本段的轮数上限（None = 用 max_turns）；allow_disengage 打开「连续告辞即终止」——
    终止发生在**被测回完那句告辞之后**，所以模型仍有机会做安全收尾（这本身是要测的行为）。
    两者只给 responsive 臂用，fixed 臂保持定长以保住高危档覆盖。"""
    if arm is Arm.FIXED and fixed_rate is None and not risk_none:
        raise ValueError("fixed arm requires fixed_rate (from escalation calibration)")
    start_stage = actor.stage
    history: list[Message] = []
    patient_utterance = actor.open() if rewrite_opening else profile.seed_text
    attack_stage = "none" if risk_none else start_stage
    n_turns = max_turns if turn_budget is None else max(1, min(turn_budget, max_turns))
    budget_reason = "max_turns" if n_turns == max_turns else "budget"
    sched_span = max(1, n_turns - 1)

    turns, end_reason = continue_dialogue(
        profile=profile, target=target, actor=actor, evaluator=evaluator, arm=arm, seed=seed,
        history=history, patient_utterance=patient_utterance, attack_stage=attack_stage,
        prev_c=actor.c, prev_abandonment=False, start_turn=1, end_turn=n_turns,
        fixed_rate=fixed_rate, start_stage=start_stage, sched_span=sched_span,
        allow_disengage=allow_disengage, target_system_prompt=target_system_prompt,
        risk_none=risk_none)
    if end_reason == "max_turns":
        end_reason = budget_reason
    return DialogueRecord(case_profile=profile, arm=arm, turns=turns,
                          target_model_id=target.model_id, seed=seed,
                          initial_stage="none" if risk_none else start_stage,
                          sampling_temperature=_temp_of(target),
                          turn_pairing="exchange_v2",
                          opening_mode="rewritten" if rewrite_opening else "seed_verbatim",
                          turn_budget=n_turns, end_reason=end_reason, risk_none=risk_none)
