# cascade/models/fake.py
from __future__ import annotations
import json
from cascade.models.base import Message


def appraisal_scripts(*triples: tuple[bool, bool, bool]) -> list[str]:
    """把 (empathy, deesc, abandon) 三元组摊成 C 评估器实际会消耗的脚本序列。

    ⚡2026-08-25：评估器改成**一维一次调用**后，一次 `appraise` = **3 次模型调用**，
    顺序即 `EVALUATOR_DIMS`。夹具直接写一条三字段 JSON 会只喂到第一维、后两维拿到
    同一条而解析失败。所有给假评估器排脚本的地方都该走这个函数。"""
    from cascade.prompts import EVALUATOR_DIM_FIELD      # 局部 import：避免 fake↔prompts 绕圈
    return [json.dumps({EVALUATOR_DIM_FIELD: v}) for triple in triples for v in triple]


class FakeChatModel:
    """Deterministic scripted model for tests. Repeats last script when exhausted."""

    def __init__(self, model_id: str, scripts: list[str]):
        if not scripts:
            raise ValueError("scripts must be non-empty")
        self.model_id = model_id
        self._scripts = list(scripts)
        self._i = 0
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        out = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        return out
