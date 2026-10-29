"""被测分组、配对消融、人设新旧。键用 data/models.yaml 的 model_id（此前只散在那里的注释里）。"""
from __future__ import annotations

GROUPS: dict[str, list[str]] = {
    "A_flagship": ["openai/gpt-5.6-sol", "anthropic/claude-opus-5", "deepseek/deepseek-v4-pro"],
    "companion": ["minimax/minimax-m2-her"],
    "B2_psych": ["cpsycoun", "mindchat", "soulchat"],
    "C_base": ["internlm2-base", "qwen15-4b-base", "qwen3-8b-base"],
}

# (心理微调版, 同基座对照)
PAIRS: list[tuple[str, str]] = [
    ("cpsycoun", "internlm2-base"),
    ("mindchat", "qwen15-4b-base"),
    ("soulchat", "qwen3-8b-base"),
]

OLD_PERSONAS: frozenset[str] = frozenset(f"zh_p{i:02d}" for i in range(1, 9))

_GROUP_OF = {m: g for g, ms in GROUPS.items() for m in ms}


def group_of(model_id: str) -> str:
    return _GROUP_OF.get(model_id, "other")


def persona_cohort(persona_id: str | None) -> str:
    if not persona_id:
        return "unknown"
    return "old" if persona_id in OLD_PERSONAS else "new"
