# cascade/seeds.py
"""Seed pipeline: SeedRecord model, PII scrubbing, corpus adapters, CLI.

Seeds provide NATURAL ENTRY POINTS only (passive/active ideation at most);
escalation is driven by the simulator per the taxonomy (master spec 5).
Real extracted seeds are written to corpus/ (gitignored — PsyQA license;
ESConv derivatives follow the same policy). Only synthetic fixtures live in data/.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from cascade.content import DATA_DIR, ContentModel, _load_yaml
from cascade.material import _check_id_language
from cascade.schema import CSSRSStage, Language

ENTRY_STAGES = (CSSRSStage.PASSIVE_IDEATION, CSSRSStage.ACTIVE_IDEATION)
MAX_SEED_CHARS = 300
SEED_FACET_KEYS = ("age_band", "life_setting")


class SeedSource(str, Enum):
    FIXTURE = "fixture"
    ESCONV = "esconv"
    PSYQA = "psyqa"


class Severity(str, Enum):
    MODERATE = "moderate"
    SEVERE = "severe"


class SeedRecord(ContentModel):
    id: str
    language: Language
    source: SeedSource
    text: str = Field(min_length=1)
    stage_hint: CSSRSStage | Literal["none"]
    severity: Severity
    facets: dict[str, str] | None = None   # (age_band, life_setting); fixtures 必填、真实语料抽取暂留 None

    @model_validator(mode="after")
    def _entry_stage_only(self) -> "SeedRecord":
        if self.stage_hint != "none" and self.stage_hint not in ENTRY_STAGES:
            raise ValueError(
                f"seed {self.id}: stage_hint must be an entry stage "
                f"{[s.value for s in ENTRY_STAGES]} (escalation belongs to the simulator)")
        return self

    @model_validator(mode="after")
    def _id_encodes_language(self) -> "SeedRecord":
        _check_id_language(self.id, self.language, "seed")
        return self


class SeedFixtures(ContentModel):
    version: int
    seeds: list[SeedRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids_all_fixture(self) -> "SeedFixtures":
        ids = [s.id for s in self.seeds]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate seed ids")
        if any(s.source != SeedSource.FIXTURE for s in self.seeds):
            raise ValueError("seed_fixtures.yaml may only contain source=fixture records")
        return self

    @model_validator(mode="after")
    def _fixtures_have_valid_facets(self) -> "SeedFixtures":
        for s in self.seeds:
            if s.facets is None or set(s.facets.keys()) != set(SEED_FACET_KEYS):
                raise ValueError(
                    f"seed {s.id}: fixtures must set facets with keys {SEED_FACET_KEYS}")
        return self

    @model_validator(mode="after")
    def _facets_symmetric_across_languages(self) -> "SeedFixtures":
        zh = [s for s in self.seeds if s.language == Language.ZH]
        en = [s for s in self.seeds if s.language == Language.EN]
        for key in SEED_FACET_KEYS:
            if Counter(s.facets[key] for s in zh) != Counter(s.facets[key] for s in en):
                raise ValueError(
                    f"seed facet '{key}' multisets must be symmetric across languages")
        return self


def load_seed_fixtures(data_dir: Path = DATA_DIR) -> list[SeedRecord]:
    return SeedFixtures(**_load_yaml(data_dir / "seed_fixtures.yaml")).seeds


def load_seed_fixtures_none(data_dir: Path = DATA_DIR) -> list[SeedRecord]:
    """A7 无风险对照种子（合成，作者撰写）。stage_hint 全为 "none"。

    绕过 SeedFixtures 的 zh/en 对称校验器（none 夹具只有 zh），直接构造
    SeedRecord 并自行校验 facets 键。
    """
    raw = _load_yaml(data_dir / "seed_fixtures_none.yaml")["seeds"]
    seeds = [SeedRecord(**s) for s in raw]
    bad = [s.id for s in seeds if s.stage_hint != "none"]
    if bad:
        raise ValueError(f"seed_fixtures_none: non-none stage_hint in {bad}")
    bad_facets = [s.id for s in seeds
                  if s.facets is None or set(s.facets.keys()) != set(SEED_FACET_KEYS)]
    if bad_facets:
        raise ValueError(
            f"seed_fixtures_none: facets must have keys {SEED_FACET_KEYS} in {bad_facets}")
    ids = [s.id for s in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError("seed_fixtures_none: duplicate seed ids")
    return seeds


_SCRUB_PATTERNS = [
    re.compile(r"https?://\S+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"@\w+"),
    re.compile(r"\d{7,}"),  # long digit runs: phones, QQ, ids
]


def scrub(text: str) -> str:
    """Deterministic PII scrub: URLs, emails, @handles, long digit runs.

    deliberately aggressive — ``\\d{7,}`` removes any long digit run (incl. dates);
    formatted phone numbers with separators are NOT caught (acceptable: corpus text is informal prose).
    """
    for pat in _SCRUB_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int = MAX_SEED_CHARS) -> str:
    return text[:limit].strip()


# --- corpus adapters (verify field names against real corpora on arrival) ---

ESCONV_PROBLEM_TYPES = {"ongoing depression"}
ESCONV_MIN_INTENSITY = 4

PSYQA_FILTER_KEYWORDS = ("抑郁", "绝望", "想死", "轻生", "自杀", "自残", "活不下去")
PSYQA_HIGH_RISK_KEYWORDS = ("想死", "轻生", "自杀", "自残", "活不下去")


def extract_esconv(path: str | Path) -> list[SeedRecord]:
    """ESConv: keep depression-type dialogs with initial intensity >= 4;
    seed = scrubbed situation + first seeker utterance."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[SeedRecord] = []
    for i, rec in enumerate(data):
        try:
            if rec.get("problem_type") not in ESCONV_PROBLEM_TYPES:
                continue
            raw = (rec.get("survey_score") or {}).get("seeker", {}).get(
                "initial_emotion_intensity", "0")
            intensity = int(raw)
            if intensity < ESCONV_MIN_INTENSITY:
                continue
            first_seeker = next((t.get("content", "") for t in rec.get("dialog", [])
                                 if t.get("speaker") == "seeker"), "")
            text = truncate(scrub(f"{rec.get('situation', '')} {first_seeker}"))
            if not text:
                continue
            out.append(SeedRecord(
                id=f"en_esconv_{i:04d}", language=Language.EN, source=SeedSource.ESCONV,
                text=text, stage_hint=CSSRSStage.PASSIVE_IDEATION,
                severity=Severity.SEVERE if intensity >= 5 else Severity.MODERATE))
        except Exception as exc:
            raise ValueError(f"{path}: record {i} malformed: {exc}") from exc
    return out


def extract_psyqa(path: str | Path) -> list[SeedRecord]:
    """PsyQA: keep items whose question/description/keywords hit the filter set;
    high-risk keyword hits grade to active_ideation/severe."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[SeedRecord] = []
    for i, rec in enumerate(data):
        try:
            kw = rec.get("keywords", "")
            kw = " ".join(kw) if isinstance(kw, list) else str(kw)
            blob = f"{rec.get('question', '')} {rec.get('description', '')} {kw}"
            if not any(k in blob for k in PSYQA_FILTER_KEYWORDS):
                continue
            high = any(k in blob for k in PSYQA_HIGH_RISK_KEYWORDS)
            text = truncate(scrub(f"{rec.get('question', '')} {rec.get('description', '')}"))
            if not text:
                continue
            out.append(SeedRecord(
                id=f"zh_psyqa_{i:04d}", language=Language.ZH, source=SeedSource.PSYQA,
                text=text,
                stage_hint=CSSRSStage.ACTIVE_IDEATION if high else CSSRSStage.PASSIVE_IDEATION,
                severity=Severity.SEVERE if high else Severity.MODERATE))
        except Exception as exc:
            raise ValueError(f"{path}: record {i} malformed: {exc}") from exc
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cascade.seeds",
                                     description="Extract entry-point seeds from a corpus.")
    parser.add_argument("--corpus", choices=["esconv", "psyqa"], required=True)
    parser.add_argument("--path", required=True, help="path to the raw corpus json")
    parser.add_argument("--out", required=True, help="output jsonl (use corpus/seeds/, gitignored)")
    args = parser.parse_args(argv)
    extractor = extract_esconv if args.corpus == "esconv" else extract_psyqa
    seeds = extractor(args.path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(s.model_dump_json() + "\n")
    print(f"wrote {len(seeds)} seeds -> {out}")


if __name__ == "__main__":
    main()
