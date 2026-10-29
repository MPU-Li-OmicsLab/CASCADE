"""事后脱离判定：对两臂所有段套同一条规则，得到可喂 KM 的 (duration, event)。

运行时只有 responsive 臂允许患者退出（session.py），fixed 臂定长。所以「患者第几轮不想聊了」
必须事后判，两种定义互为稳健性：
  * disengage —— 套运行时同一条 `is_disengaging` 规则，连续 DISENGAGE_RUN 轮告辞，事件轮 = 该串的第一轮
  * collapse  —— 连续 2 轮患者话 ≤ COLLAPSE_MAX_CHARS 字
"""
from __future__ import annotations

from cascade.schema import DialogueRecord, Language
from cascade.session import DISENGAGE_RUN, is_disengaging

COLLAPSE_MAX_CHARS = 5
COLLAPSE_RUN = 2


def _first_run(flags: list[bool], run: int) -> int | None:
    """连续 `run` 个 True 的第一个位置（1-based），否则 None。"""
    streak = 0
    for i, f in enumerate(flags, start=1):
        streak = streak + 1 if f else 0
        if streak >= run:
            return i - run + 1
    return None


def first_disengage_turn(record: DialogueRecord) -> int | None:
    lang = record.case_profile.language
    return _first_run([is_disengaging(t.patient_utterance, lang) for t in record.turns],
                      DISENGAGE_RUN)


def first_collapse_turn(record: DialogueRecord, max_chars: int = COLLAPSE_MAX_CHARS) -> int | None:
    return _first_run([len(t.patient_utterance.strip()) <= max_chars for t in record.turns],
                      COLLAPSE_RUN)


def ttd(record: DialogueRecord, kind: str) -> tuple[int, int]:
    """(duration, event)。事件轮或删失在末轮。"""
    if kind == "disengage":
        t = first_disengage_turn(record)
    elif kind == "collapse":
        t = first_collapse_turn(record)
    else:
        raise ValueError(f"unknown ttd kind: {kind}")
    if t is None:
        return len(record.turns), 0
    return t, 1


def is_pressing(utterance: str, language: Language, max_chars: int = COLLAPSE_MAX_CHARS) -> bool:
    """患者本轮是否仍在施加真实压力：不是告辞、也不是碎片。"""
    u = (utterance or "").strip()
    return len(u) > max_chars and not is_disengaging(u, language)
