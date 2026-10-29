# cascade/content.py
"""Loader + pydantic models for the scoring-content layer (single source of truth).

Spec: （内部文档）
EN fields are canonical; ZH fields are reviewed derived translations.
Content files describe failure modes at meta level only — never store
actual lethal-means information in data/.
"""
from __future__ import annotations
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at document root")
    return data


class ContentModel(BaseModel):
    """Base for all content-layer models: hand-edited YAML must not contain unknown keys."""
    model_config = ConfigDict(extra="forbid")


class ProvenanceKind(str, Enum):
    DIRECT = "direct"                  # citable clause in the source
    OPERATIONALIZED = "operationalized"  # our operationalization of a source principle


class SourceType(str, Enum):
    CLINICAL_SCALE = "clinical_scale"
    GUIDELINE = "guideline"
    HOTLINE_OFFICIAL = "hotline_official"
    TEXTBOOK = "textbook"
    RED_TEAMING_LIT = "red_teaming_lit"


class Source(ContentModel):
    id: str
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    year: int
    url: str = Field(min_length=1)
    accessed: str = Field(min_length=1)  # ISO date the URL was last checked
    type: SourceType


class ProvenanceRegistry(ContentModel):
    version: int
    sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ProvenanceRegistry":
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source ids in provenance registry")
        return self

    def ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.sources)


class ProvenanceRef(ContentModel):
    source_id: str
    kind: ProvenanceKind


def load_provenance(data_dir: Path = DATA_DIR) -> ProvenanceRegistry:
    return ProvenanceRegistry(**_load_yaml(data_dir / "provenance.yaml"))


class ResourceType(str, Enum):
    HOTLINE = "hotline"
    TEXT_LINE = "text_line"
    EMERGENCY = "emergency"


class CrisisResource(ContentModel):
    id: str
    name_en: str = Field(min_length=1)
    name_zh: str = Field(min_length=1)
    identifiers: list[str] = Field(min_length=1)  # matchable strings, e.g. "12356"
    type: ResourceType
    notes_en: str = ""
    notes_zh: str = ""
    provenance_refs: list[ProvenanceRef] = Field(min_length=1)


class LocaleResources(ContentModel):
    penalty_rule_en: str = Field(min_length=1)  # wrong-locale -> d2 tier 0 (machine-readable rule text)
    penalty_rule_zh: str = Field(min_length=1)
    resources: list[CrisisResource] = Field(min_length=1)


class ResourceTable(ContentModel):
    version: int
    locales: dict[str, LocaleResources]

    @model_validator(mode="after")
    def _locales_match_language_enum(self) -> "ResourceTable":
        from cascade.schema import Language
        expected = {lang.value for lang in Language}
        if set(self.locales.keys()) != expected:
            raise ValueError(f"resource locales {set(self.locales)} != Language enum {expected}")
        return self

    @model_validator(mode="after")
    def _unique_resource_ids(self) -> "ResourceTable":
        all_ids: list[str] = []
        for lr in self.locales.values():
            all_ids.extend(r.id for r in lr.resources)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("duplicate CrisisResource ids across all locales in ResourceTable")
        return self


def load_resources(data_dir: Path = DATA_DIR) -> ResourceTable:
    return ResourceTable(**_load_yaml(data_dir / "resources.yaml"))


class LocaleExamples(ContentModel):
    en: list[str] = Field(default_factory=list)
    zh: list[str] = Field(default_factory=list)


class TrajectoryStage(ContentModel):
    id: str
    name_en: str = Field(min_length=1)
    name_zh: str = Field(min_length=1)
    definition_en: str = Field(min_length=1)
    definition_zh: str = Field(min_length=1)
    behavioral_markers_en: list[str] = Field(min_length=1)
    behavioral_markers_zh: list[str] = Field(min_length=1)
    explicit_examples: LocaleExamples = Field(default_factory=LocaleExamples)
    implicit_examples: LocaleExamples = Field(default_factory=LocaleExamples)
    provenance_refs: list[ProvenanceRef] = Field(min_length=1)


class InductionStrategy(ContentModel):
    id: str
    name_en: str = Field(min_length=1)
    name_zh: str = Field(min_length=1)
    definition_en: str = Field(min_length=1)
    definition_zh: str = Field(min_length=1)
    example_en: str = Field(min_length=1)
    example_zh: str = Field(min_length=1)
    counter_behavior_en: str = Field(min_length=1)  # judge reference for d3/d4
    counter_behavior_zh: str = Field(min_length=1)
    provenance_refs: list[ProvenanceRef] = Field(min_length=1)


class Taxonomies(ContentModel):
    version: int
    trajectory: list[TrajectoryStage] = Field(min_length=1)
    strategies: list[InductionStrategy] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Taxonomies":
        for group_name, items in (("trajectory", self.trajectory), ("strategies", self.strategies)):
            ids = [x.id for x in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate ids in taxonomies.{group_name}")
        return self

    @model_validator(mode="after")
    def _trajectory_matches_cssrs(self) -> "Taxonomies":
        from cascade.schema import CSSRSStage
        expected = [st.value for st in CSSRSStage]
        if [s.id for s in self.trajectory] != expected:
            raise ValueError(
                f"trajectory stage ids must equal CSSRSStage values in order: {expected}"
            )
        return self


def load_taxonomies(data_dir: Path = DATA_DIR) -> Taxonomies:
    return Taxonomies(**_load_yaml(data_dir / "taxonomies.yaml"))


class LeaderboardId(str, Enum):
    A = "A"
    B = "B"


class Tier(ContentModel):
    score: int = Field(ge=0, le=2)
    anchor_en: str = Field(min_length=1)
    anchor_zh: str = Field(min_length=1)
    examples: LocaleExamples = Field(default_factory=LocaleExamples)
    provenance_refs: list[ProvenanceRef] = Field(min_length=1)


class Dimension(ContentModel):
    id: str
    name_en: str = Field(min_length=1)
    name_zh: str = Field(min_length=1)
    leaderboard: LeaderboardId | None  # None => auxiliary, never aggregated
    red_line: bool
    gate: bool
    aggregated: bool
    definition_en: str = Field(min_length=1)
    definition_zh: str = Field(min_length=1)
    tiers: list[Tier]

    @model_validator(mode="after")
    def _tiers_are_0_1_2(self) -> "Dimension":
        if sorted(t.score for t in self.tiers) != [0, 1, 2]:
            raise ValueError(f"dimension {self.id}: tiers must be exactly scores 0, 1, 2")
        return self

    @model_validator(mode="after")
    def _flags_coherent(self) -> "Dimension":
        """Enforce flag-coherence rules (spec §3/§7):
        - red_line=True requires leaderboard A
        - gate=True requires leaderboard B
        - leaderboard=None (aux) must have red_line=False, gate=False, aggregated=False
        - aggregated must equal (leaderboard is not None)
        """
        if self.red_line and self.leaderboard != LeaderboardId.A:
            raise ValueError(
                f"dimension {self.id}: red_line=True requires leaderboard A"
            )
        if self.gate and self.leaderboard != LeaderboardId.B:
            raise ValueError(
                f"dimension {self.id}: gate=True requires leaderboard B"
            )
        if self.aggregated != (self.leaderboard is not None):
            raise ValueError(
                f"dimension {self.id}: aggregated must equal (leaderboard is not None)"
            )
        return self


class ScaleNote(ContentModel):
    description_en: str = Field(min_length=1)
    description_zh: str = Field(min_length=1)


class Rubric(ContentModel):
    version: int
    scale: ScaleNote
    dimensions: list[Dimension] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_dim_ids(self) -> "Rubric":
        ids = [d.id for d in self.dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate dimension ids in rubric")
        return self

    def dim(self, dim_id: str) -> Dimension:
        for d in self.dimensions:
            if d.id == dim_id:
                return d
        raise KeyError(dim_id)


def load_rubric(data_dir: Path = DATA_DIR) -> Rubric:
    return Rubric(**_load_yaml(data_dir / "rubric.yaml"))


class ContentBundle(ContentModel):
    rubric: Rubric
    resources: ResourceTable
    taxonomies: Taxonomies
    provenance: ProvenanceRegistry

    @model_validator(mode="after")
    def _all_refs_resolve(self) -> "ContentBundle":
        known = self.provenance.ids()
        for owner, refs in self._iter_refs():
            for ref in refs:
                if ref.source_id not in known:
                    raise ValueError(f"{owner}: unknown provenance source '{ref.source_id}'")
        return self

    def _iter_refs(self) -> Iterator[tuple[str, list[ProvenanceRef]]]:
        for d in self.rubric.dimensions:
            for t in d.tiers:
                yield f"rubric.{d.id}.tier{t.score}", t.provenance_refs
        for locale, lr in self.resources.locales.items():
            for r in lr.resources:
                yield f"resources.{locale}.{r.id}", r.provenance_refs
        for s in self.taxonomies.trajectory:
            yield f"taxonomies.trajectory.{s.id}", s.provenance_refs
        for s in self.taxonomies.strategies:
            yield f"taxonomies.strategies.{s.id}", s.provenance_refs


def load_content(data_dir: Path = DATA_DIR) -> ContentBundle:
    return ContentBundle(
        rubric=load_rubric(data_dir),
        resources=load_resources(data_dir),
        taxonomies=load_taxonomies(data_dir),
        provenance=load_provenance(data_dir),
    )
