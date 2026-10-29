"""Offline 6-dimension rubric judge (Plan 4.3 §5). Scores one dialogue turn in
context with ONE CALL PER DIMENSION (d1-d6 + the empathy auxiliary = 7 calls).
⚡2026-08-30：此前是按榜单轴分两次调用（A=d1,d2 / B=d3-d6+aux），已废弃 ——
只改 d3/d6/aux 的锚点文本，**一个字没动的** d4/d5 分数就崩了 40%/56%，维度在同一次
调用里互相搅动，任何单维改动都无法归因。旧的 `build_judge_prompt` 保留在 prompts.py，
用于说明 runs/prod_all/scored.jsonl 与 scored_v2.jsonl 的来历。Mirrors RiskLabeler:
injected ChatModel, brace-balanced JSON parse, None on unparseable output (never
a batch-killing crash). Blind to the attack machinery. Clinical validation = P4.4.

Two-stage iron law: this is OFFLINE only — never wire it into the stage-1 loop."""
from __future__ import annotations
from cascade.models.base import ChatModel
from cascade.prompts import (build_judge_dim_prompt, load_judge_card, JUDGE_DIMS, JUDGE_AUX,
                            JUDGE_DIM_SCORE_FIELD, JUDGE_DIM_REASON_FIELD)
from cascade.evaluator import _extract_json
from cascade.content import Rubric, load_rubric
from cascade.schema import DialogueRecord, TurnScore


DEFAULT_RETRIES = 2


class RubricJudge:
    def __init__(self, model: ChatModel, rubric: Rubric | None = None,
                 retries: int = DEFAULT_RETRIES, card: dict[str, str] | None | bool = True):
        self.model = model
        self.rubric = rubric or load_rubric()
        self.retries = retries
        # v6（2026-09-11）：默认附判例卡（医师定的边界判法）。card=False/None = v5 行为（无卡）。
        self.card = load_judge_card() if card is True else (card or None)

    def _dim(self, dim_id: str, history, patient_utterance, target_response, language) -> dict:
        """一个维度一次调用，解析失败就重问这一维（别的维已成的不重问）。"""
        msgs = build_judge_dim_prompt(dim_id, history, patient_utterance, target_response,
                                      language, self.rubric, card=self.card)
        for attempt in range(self.retries + 1):
            raw = self.model.generate(msgs)
            try:
                return _extract_json(raw, required=(JUDGE_DIM_SCORE_FIELD,))
            except ValueError:
                if attempt == self.retries:
                    raise

    def score_turn(self, *, history, patient_utterance, target_response, language,
                   turn_id: str | None = None) -> TurnScore | None:
        """一维一次调用（2026-08-30 起）。六个计分维必须全部解析成功，否则整轮 None；
        `empathy_aux` 永不聚合，它挂了不拖累其余六维。
        `turn_id` 只为与 v7 ExtractJudge 同接口（score_runs 统一传），这里不用。

        ⚠️ 为什么不再一次调用出多维：实测只改 d3/d6/empathy_aux 的锚点文本，**一个字
        没动的** d4/d5 分数就崩了 40%/56%（d4==0 937→561、d5==0 606→265）。维度在
        同一次调用里互相搅动 → 任何单维改动都无法归因。C 评估器走过同一条路：三维挤
        一个 JSON 时保持连接 κ 0.221，拆开后 0.757。
        """
        fields: dict = {}
        try:
            for dim in JUDGE_DIMS:
                got = self._dim(dim, history, patient_utterance, target_response, language)
                fields[dim] = int(got[JUDGE_DIM_SCORE_FIELD])
                reason = got.get(JUDGE_DIM_REASON_FIELD)
                if reason is not None:
                    fields[f"{dim}_reason"] = str(reason)
        except (ValueError, TypeError):
            return None
        for dim in JUDGE_AUX:                      # 辅助项：失败就留空，不作废整轮
            try:
                got = self._dim(dim, history, patient_utterance, target_response, language)
                fields[dim] = int(got[JUDGE_DIM_SCORE_FIELD])
                reason = got.get(JUDGE_DIM_REASON_FIELD)
                if reason is not None:
                    fields[f"{dim}_reason"] = str(reason)
            except (ValueError, TypeError):
                pass
        try:
            return TurnScore(**fields)
        except (ValueError, TypeError):            # 分数越界 -> 与旧行为一致，整轮 None
            return None


def score_dialogue(judge: RubricJudge, record: DialogueRecord,
                   force: bool = False) -> DialogueRecord:
    """Fill each turn's .score, using prior (patient,target) turns as history.
    Idempotent: a turn already carrying a score is skipped unless force."""
    new_turns = []
    history: list[tuple[str, str]] = []
    lang = record.case_profile.language
    for turn in record.turns:
        if turn.score is None or force:
            score = judge.score_turn(history=history, patient_utterance=turn.patient_utterance,
                                     target_response=turn.target_response, language=lang)
            if score is not None:
                turn = turn.model_copy(update={"score": score})
        history = history + [(turn.patient_utterance, turn.target_response)]
        new_turns.append(turn)
    return record.model_copy(update={"turns": new_turns})
