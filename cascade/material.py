# cascade/material.py
"""Case-material layer models: personas, calibration set, audit expressions.

Spec: （内部文档）
ZH and EN content is natively authored on each side (not translations).
Symmetry of the construction protocol (counts, facet multisets) is enforced
at the model level; cultural content differs by design.
No method-related content may ever appear in these files.
"""
from __future__ import annotations
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from pydantic import Field, model_validator
from cascade.content import DATA_DIR, ContentModel, _load_yaml
from cascade.schema import CSSRSStage, Language


def _check_id_language(record_id: str, language: Language, kind: str) -> None:
    if not (record_id.startswith(f"{language.value}_") or f"_{language.value}_" in record_id):
        raise ValueError(f"{kind} {record_id}: id must encode language '{language.value}'")

FACET_KEYS = ("age_band", "life_setting", "online_community")


class Persona(ContentModel):
    id: str
    language: Language
    archetype: str = Field(min_length=1)
    summary_en: str = Field(min_length=1)
    background: str = Field(min_length=1)
    expression_style: str = Field(min_length=1)
    facets: dict[str, str]
    implicit_examples: list[str] = Field(min_length=3)
    explicit_examples: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def _facet_keys_fixed(self) -> "Persona":
        if set(self.facets.keys()) != set(FACET_KEYS):
            raise ValueError(f"persona {self.id}: facet keys must be exactly {FACET_KEYS}")
        return self

    @model_validator(mode="after")
    def _id_prefix_matches_language(self) -> "Persona":
        expected_prefix = f"{self.language.value}_"
        if not self.id.startswith(expected_prefix):
            raise ValueError(
                f"persona {self.id}: id prefix must match language '{self.language.value}_'"
            )
        return self


class PersonaLibrary(ContentModel):
    version: int
    personas: list[Persona] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids_and_symmetry(self) -> "PersonaLibrary":
        ids = [p.id for p in self.personas]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate persona ids")
        zh = [p for p in self.personas if p.language == Language.ZH]
        en = [p for p in self.personas if p.language == Language.EN]
        if len(zh) != len(en):
            raise ValueError(f"persona counts must be symmetric: zh={len(zh)} en={len(en)}")
        for key in FACET_KEYS:
            if Counter(p.facets[key] for p in zh) != Counter(p.facets[key] for p in en):
                raise ValueError(f"facet '{key}' multisets must be symmetric across languages")
        return self


def load_personas(data_dir: Path = DATA_DIR) -> PersonaLibrary:
    return PersonaLibrary(**_load_yaml(data_dir / "personas.yaml"))


class Register(str, Enum):
    GAMING_SLANG = "gaming_slang"
    PLATFORM_EUPHEMISM = "platform_euphemism"
    COMMUNITY_JARGON = "community_jargon"
    LITERARY = "literary"
    INDIRECT_IDIOM = "indirect_idiom"


class Ambiguity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


with warnings.catch_warnings():
    # field name "register" shadows abc.ABCMeta.register (unused here, harmless)
    warnings.filterwarnings("ignore", message='Field name "register"', category=UserWarning)

    class AuditExpression(ContentModel):
        id: str
        language: Language
        text: str = Field(min_length=1)
        gloss_en: str = Field(min_length=1)
        register: Register
        stage_mapping: CSSRSStage
        ambiguity: Ambiguity

        @model_validator(mode="after")
        def _id_encodes_language(self) -> "AuditExpression":
            _check_id_language(self.id, self.language, "audit expression")
            return self


class AuditSet(ContentModel):
    version: int
    expressions: list[AuditExpression] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_and_symmetric(self) -> "AuditSet":
        ids = [e.id for e in self.expressions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate audit expression ids")
        zh = sum(1 for e in self.expressions if e.language == Language.ZH)
        en = sum(1 for e in self.expressions if e.language == Language.EN)
        if zh != en:
            raise ValueError(f"audit counts must be symmetric: zh={zh} en={en}")
        zh_exprs = [e for e in self.expressions if e.language == Language.ZH]
        en_exprs = [e for e in self.expressions if e.language == Language.EN]
        for axis, getter in (("stage_mapping", lambda e: e.stage_mapping),
                             ("ambiguity", lambda e: e.ambiguity)):
            zh_c = Counter(getter(e) for e in zh_exprs)
            en_c = Counter(getter(e) for e in en_exprs)
            if zh_c != en_c:
                raise ValueError(
                    f"audit {axis} multisets must be symmetric across languages: "
                    f"zh={dict(zh_c)} en={dict(en_c)}")
        return self


def load_audit(data_dir: Path = DATA_DIR) -> AuditSet:
    return AuditSet(**_load_yaml(data_dir / "audit_expressions.yaml"))


class Quadrant(str, Enum):
    LONG_EMPTY = "long_empty"
    SHORT_SUBSTANTIVE = "short_substantive"
    LONG_SUBSTANTIVE = "long_substantive"
    SHORT_EMPTY = "short_empty"
    EDGE_REFERRAL = "edge_referral"
    EDGE_ABANDONMENT = "edge_abandonment"


class GoldAppraisal(ContentModel):
    empathy_detected: bool
    deescalation_attempt: bool
    abandonment_felt: bool


class CalibrationItem(ContentModel):
    id: str
    language: Language
    quadrant: Quadrant
    user_utterance: str = Field(min_length=1)
    target_response: str = Field(min_length=1)
    gold: GoldAppraisal
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _id_encodes_language(self) -> "CalibrationItem":
        _check_id_language(self.id, self.language, "calibration item")
        return self


class CalibrationSet(ContentModel):
    version: int
    items: list[CalibrationItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "CalibrationSet":
        ids = [i.id for i in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate calibration item ids")
        return self


def load_calibration(data_dir: Path = DATA_DIR) -> CalibrationSet:
    return CalibrationSet(**_load_yaml(data_dir / "calibration.yaml"))


def render_persona(p: Persona) -> str:
    """Render the persona block consumed by the simulator prompt (Plan 3)."""
    implicit = "\n".join(f"- {ex}" for ex in p.implicit_examples)
    explicit = "\n".join(f"- {ex}" for ex in p.explicit_examples)
    return (
        f"[PERSONA {p.id}] {p.archetype}\n"
        f"背景/Background: {p.background}\n"
        f"表达风格/Expression style: {p.expression_style}\n"
        f"隐式意念表达示例/Implicit ideation examples:\n{implicit}\n"
        f"直接表达示例/Explicit expression examples:\n{explicit}"
    )


def render_persona_none(p: Persona) -> str:
    """A7 无风险模式的人设块：去掉意念表达示例（否则示例本身就在教模拟器说想死）。"""
    return (
        f"[PERSONA {p.id}] {p.archetype}\n"
        f"背景/Background: {p.background}\n"
        f"表达风格/Expression style: {p.expression_style}"
    )
