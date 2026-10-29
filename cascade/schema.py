# cascade/schema.py
from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class CSSRSStage(str, Enum):
    """C-SSRS-derived expressed-risk ladder — NOT the C-SSRS ideation scale itself.

    Stages 0-4 (passive_ideation -> plan) map 1:1 to C-SSRS ideation severity L1-L5;
    stage 5 (means_timing) is the C-SSRS behavior subscale's preparatory behavior,
    appended as the ladder top. See docs/collaboration/风险梯说明-C-SSRS衍生六档.md.
    """
    PASSIVE_IDEATION = "passive_ideation"
    ACTIVE_IDEATION = "active_ideation"
    METHOD_INQUIRY = "method_inquiry"
    INTENT = "intent"
    PLAN = "plan"
    MEANS_TIMING = "means_timing"

    @property
    def index(self) -> int:
        return list(CSSRSStage).index(self)


class Language(str, Enum):
    ZH = "zh"
    EN = "en"


class Arm(str, Enum):
    RESPONSIVE = "responsive"
    FIXED = "fixed"


class CaseProfile(BaseModel):
    id: str
    seed_text: str
    persona_profile: str
    trajectory_stage_target: CSSRSStage
    strategies: list[str] = Field(default_factory=list)
    language: Language
    explicit: bool
    persona_id: str | None = None  # provenance: personas.yaml id
    seed_id: str | None = None     # provenance: SeedRecord id


class CAppraisal(BaseModel):
    empathy_detected: bool
    deescalation_attempt: bool
    abandonment_felt: bool
    c_new: float = Field(ge=0.0, le=1.0)


class TurnRecord(BaseModel):
    """一次**交换**（exchange_v2，2026-07-30 起）：患者说 `patient_utterance`，
    被测在同一条记录里用 `target_response` 回它。turn=1 的患者侧就是开场白（语料 seed）。

    ⚠️ 与 legacy_v1 的区别（这是个 bug 修复，不只是体例）：v1 里 `target_response` 回的是
    **上一条** TurnRecord 的患者发言，而 `patient_utterance` 是对本条 target_response 的
    回应 —— 每条记录横跨两次交互的半边。后果：① 末轮患者发言永远没有 AI 回应（fixed 臂
    末轮恰是峰值风险档，等于最该测的一次从未生成）；② `judge.score_turn` / `score_dialogue`
    一直把同一条记录的两个字段当成配对交换喂给判官，实际是错配 → v1 数据上算出的 6 维分
    无效；③ 显示顺序要靠工具层纠正。v2 下这三个问题同时消失。
    读旧文件请看 `DialogueRecord.turn_pairing`（None/legacy_v1 = 旧语义）。"""
    turn: int
    target_response: str
    appraisal: CAppraisal
    patient_utterance: str
    attack_stage: CSSRSStage | Literal["none"]        # 模拟器的攻击/意图档（原 `stage`，正名）；"none" = A7 无风险模式（全程无意念）
    c: float = Field(ge=0.0, le=1.0)
    expressed_risk: CSSRSStage | Literal["none"] | None = None   # 离线判定的表达风险档 + none 底档（分析唯一风险轴）
    score: "TurnScore | None" = None                  # offline 6-dim judge output (Plan 4.3)


class TurnScore(BaseModel):
    d1: int = Field(ge=0, le=2)
    d2: int = Field(ge=0, le=2)
    d3: int = Field(ge=0, le=2)
    d4: int = Field(ge=0, le=2)
    d5: int = Field(ge=0, le=2)
    d6: int = Field(ge=0, le=2)
    empathy_aux: int | None = Field(default=None, ge=0, le=2)  # auxiliary; NEVER aggregated
    # 判官自己的逐维理由。**可选**（旧的 scored.jsonl 不带理由，仍要能读）。
    # 留它们的唯一目的是诊断：某一维 κ 不达标时，判官的理由是唯一能说明「它凭什么这么判」
    # 的东西。⚠️ 理由必须和分数来自同一次生成 —— vLLM 连续批处理下同一输入在不同批次里
    # 可能吐出不同结果，事后补跑取到的理由解释的是另一次的分数。
    d1_reason: str | None = None
    d2_reason: str | None = None
    d3_reason: str | None = None
    d4_reason: str | None = None
    d5_reason: str | None = None
    d6_reason: str | None = None
    empathy_aux_reason: str | None = None


class DialogueRecord(BaseModel):
    case_profile: CaseProfile
    arm: Arm
    turns: list[TurnRecord]
    target_model_id: str
    seed: int
    initial_stage: CSSRSStage | Literal["none"] | None = None       # calibration net-Δstage anchor (avoids implicit PASSIVE)
    sampling_temperature: float | None = None      # master spec §8 "采样温度"; fills 3A §12.4 deferral
    # 配对语义标记：读旧 JSONL 必看。None/"legacy_v1" = 旧的错配语义（见 TurnRecord 文档），
    # "exchange_v2" = 一条记录就是一次完整交换。新批一律写 v2。
    turn_pairing: Literal["legacy_v1", "exchange_v2"] | None = None
    # 开场白怎么来的（v2 下开场白本身 = turns[0].patient_utterance，不再单独存以免冗余漂移）。
    # seed_verbatim = 语料原文照搬；rewritten = 经 build_opening_prompt 改写（已证伪，仅消融用）。
    opening_mode: Literal["seed_verbatim", "rewritten"] | None = None
    # 会话成形（cascade/session.py）：本段的轮数预算，以及为什么停下来。
    # end_reason="disengaged" 即患者自己走了 → 该段是**右删失**观测，
    # len(turns) 就是 TTD（time-to-disengage），可直接喂 Kaplan-Meier。
    turn_budget: int | None = None
    end_reason: Literal["max_turns", "budget", "disengaged"] | None = None
    # A6 干预实验：被测收到的系统提示名（文件名去后缀，如 present_v1.zh）；None = 无干预。
    # 提示文本本身不进 history / TurnRecord，只喂被测（orchestrator.continue_dialogue）。
    intervention: str | None = None
    # A7 无风险对照：True = 模拟器全程无自杀意念，attack_stage 全为 "none"，fixed 日程被跳过。
    risk_none: bool = False
