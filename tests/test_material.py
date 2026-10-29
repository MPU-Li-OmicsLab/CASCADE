# tests/test_material.py
from collections import Counter
import pytest
from pydantic import ValidationError
from cascade.content import DATA_DIR
from cascade.material import (
    FACET_KEYS,
    Persona,
    PersonaLibrary,
    load_personas,
    render_persona,
)
from cascade.schema import Language


def test_personas_load_and_count_16_16():   # ⚡2026-09-14 语料 v2：8 → 16
    lib = load_personas(DATA_DIR)
    by_lang = Counter(p.language for p in lib.personas)
    assert by_lang[Language.ZH] == 16
    assert by_lang[Language.EN] == 16


def test_personas_facet_multisets_symmetric():
    lib = load_personas(DATA_DIR)
    for key in FACET_KEYS:
        zh = Counter(p.facets[key] for p in lib.personas if p.language == Language.ZH)
        en = Counter(p.facets[key] for p in lib.personas if p.language == Language.EN)
        assert zh == en, f"facet {key} asymmetric: zh={zh} en={en}"


def test_persona_example_minimums():
    lib = load_personas(DATA_DIR)
    for p in lib.personas:
        assert len(p.implicit_examples) >= 3, p.id
        assert len(p.explicit_examples) >= 2, p.id
        assert p.summary_en.strip()


def test_persona_library_rejects_asymmetric_counts():
    lib = load_personas(DATA_DIR)
    data = lib.model_dump()
    data["personas"] = data["personas"][:-1]  # drop one EN persona
    with pytest.raises(ValidationError, match="symmetric"):
        PersonaLibrary(**data)


def test_persona_rejects_wrong_facet_keys():
    lib = load_personas(DATA_DIR)
    data = lib.personas[0].model_dump()
    data["facets"] = {"age_band": "teen"}  # missing two keys
    with pytest.raises(ValidationError, match="facet keys"):
        Persona(**data)


def test_render_persona_contains_voice_assets():
    lib = load_personas(DATA_DIR)
    p = lib.personas[0]
    text = render_persona(p)
    assert p.background in text
    assert p.expression_style in text
    for ex in p.implicit_examples:
        assert ex in text


from cascade.material import AuditSet, load_audit
from cascade.schema import CSSRSStage


def test_audit_set_symmetric_counts():
    audit = load_audit(DATA_DIR)
    zh = [e for e in audit.expressions if e.language == Language.ZH]
    en = [e for e in audit.expressions if e.language == Language.EN]
    assert len(zh) == len(en) == 30


def test_audit_stage_mappings_are_entry_to_plan_only():
    audit = load_audit(DATA_DIR)
    allowed = {CSSRSStage.PASSIVE_IDEATION, CSSRSStage.ACTIVE_IDEATION,
               CSSRSStage.INTENT, CSSRSStage.PLAN}
    for e in audit.expressions:
        assert e.stage_mapping in allowed, e.id


def test_audit_set_rejects_asymmetric_counts():
    audit = load_audit(DATA_DIR)
    data = audit.model_dump()
    data["expressions"] = data["expressions"][:-1]
    with pytest.raises(ValidationError, match="symmetric"):
        AuditSet(**data)


def test_audit_glosses_nonempty_and_ids_unique():
    audit = load_audit(DATA_DIR)
    ids = [e.id for e in audit.expressions]
    assert len(ids) == len(set(ids))
    for e in audit.expressions:
        assert e.gloss_en.strip(), e.id


from collections import defaultdict
from cascade.material import CalibrationSet, Quadrant, load_calibration
from cascade.schema import CAppraisal


def test_calibration_counts_and_quotas():
    cal = load_calibration(DATA_DIR)
    by_lang_quadrant = defaultdict(int)
    for item in cal.items:
        by_lang_quadrant[(item.language, item.quadrant)] += 1
    quotas = {Quadrant.LONG_EMPTY: 5, Quadrant.SHORT_SUBSTANTIVE: 5,
              Quadrant.LONG_SUBSTANTIVE: 4, Quadrant.SHORT_EMPTY: 4,
              Quadrant.EDGE_REFERRAL: 3, Quadrant.EDGE_ABANDONMENT: 4}
    for lang in (Language.ZH, Language.EN):
        for quadrant, quota in quotas.items():
            assert by_lang_quadrant[(lang, quadrant)] == quota, (lang, quadrant)


def test_calibration_gold_constructs_cappraisal():
    # interlock: gold keys == CAppraisal boolean fields
    cal = load_calibration(DATA_DIR)
    for item in cal.items:
        appr = CAppraisal(**item.gold.model_dump(), c_new=0.5)
        assert isinstance(appr.empathy_detected, bool)


def test_calibration_gold_not_uniform():
    # the set must discriminate: both True and False must occur for every gold field
    cal = load_calibration(DATA_DIR)
    for field in ("empathy_detected", "deescalation_attempt", "abandonment_felt"):
        values = {getattr(item.gold, field) for item in cal.items}
        assert values == {True, False}, field


def test_calibration_rationales_and_ids():
    cal = load_calibration(DATA_DIR)
    ids = [i.id for i in cal.items]
    assert len(ids) == len(set(ids))
    for item in cal.items:
        assert item.rationale.strip(), item.id
        assert item.user_utterance.strip() and item.target_response.strip(), item.id


# ---------- Follow-up: Persona id-language coherence (negative test) ----------

def test_persona_id_prefix_must_match_language():
    lib = load_personas(DATA_DIR)
    data = lib.personas[0].model_dump()  # zh_p01 with language=zh
    data["id"] = "en_p01"  # mismatched: id says EN, language is ZH
    with pytest.raises(ValidationError, match="id prefix"):
        Persona(**data)


# ---------- Follow-up: Two missing negative tests for PersonaLibrary ----------

def test_persona_library_rejects_duplicate_ids():
    lib = load_personas(DATA_DIR)
    data = lib.model_dump()
    # duplicate the first persona
    data["personas"] = data["personas"] + [data["personas"][0]]
    with pytest.raises(ValidationError, match="duplicate persona ids"):
        PersonaLibrary(**data)


def test_persona_library_rejects_facet_multiset_asymmetry():
    lib = load_personas(DATA_DIR)
    data = lib.model_dump()
    # swap one ZH persona's age_band from teen to senior so counts stay 8/8
    # but the multiset differs (teen count changes 2→1, senior count changes 1→2)
    for p in data["personas"]:
        if p["language"] == "zh" and p["facets"]["age_band"] == "teen":
            p["facets"]["age_band"] = "senior"
            break
    with pytest.raises(ValidationError, match="multisets must be symmetric"):
        PersonaLibrary(**data)


# ---------- Follow-up: Single-line field discipline ----------

def test_persona_str_fields_are_single_line():
    lib = load_personas(DATA_DIR)
    for p in lib.personas:
        for field in ("archetype", "summary_en", "background", "expression_style"):
            val = getattr(p, field)
            assert "\n" not in val, f"persona {p.id}.{field} contains newline"
        for ex in p.implicit_examples:
            assert "\n" not in ex, f"persona {p.id} implicit_example contains newline"
        for ex in p.explicit_examples:
            assert "\n" not in ex, f"persona {p.id} explicit_example contains newline"


# ---------- Fix A: audit difficulty multiset symmetry ----------

def test_audit_difficulty_multisets_symmetric():
    """Positive: both stage_mapping and ambiguity Counters equal across zh/en."""
    audit = load_audit(DATA_DIR)
    zh = [e for e in audit.expressions if e.language == Language.ZH]
    en = [e for e in audit.expressions if e.language == Language.EN]
    zh_stage = Counter(e.stage_mapping for e in zh)
    en_stage = Counter(e.stage_mapping for e in en)
    assert zh_stage == en_stage, f"stage asymmetric: zh={dict(zh_stage)} en={dict(en_stage)}"
    zh_ambi = Counter(e.ambiguity for e in zh)
    en_ambi = Counter(e.ambiguity for e in en)
    assert zh_ambi == en_ambi, f"ambiguity asymmetric: zh={dict(zh_ambi)} en={dict(en_ambi)}"


def test_audit_set_rejects_difficulty_multiset_asymmetry():
    """Negative: flipping one entry's ambiguity triggers must be symmetric validator."""
    audit = load_audit(DATA_DIR)
    data = audit.model_dump()
    # flip ambiguity of first ZH expression from its current value to something else
    zh_entries = [e for e in data["expressions"] if e["language"] == "zh"]
    first_zh = zh_entries[0]
    current = first_zh["ambiguity"]
    # pick a different ambiguity value
    alts = [v for v in ("high", "medium", "low") if v != current]
    first_zh["ambiguity"] = alts[0]
    with pytest.raises(ValidationError, match="must be symmetric"):
        AuditSet(**data)


# ---------- Fix C: id-language interlock negative tests ----------

from cascade.material import AuditExpression, CalibrationItem, GoldAppraisal, Quadrant
from cascade.seeds import SeedRecord, SeedSource
from cascade.schema import CSSRSStage


@pytest.mark.parametrize("kind,build_bad", [
    ("AuditExpression", lambda: AuditExpression(
        id="bad_00", language=Language.EN,
        text="test", gloss_en="test", register="gaming_slang",
        stage_mapping=CSSRSStage.PASSIVE_IDEATION, ambiguity="medium")),
    ("CalibrationItem", lambda: CalibrationItem(
        id="bad_00", language=Language.EN, quadrant="short_substantive",
        user_utterance="u", target_response="r",
        gold=GoldAppraisal(empathy_detected=True, deescalation_attempt=False, abandonment_felt=False),
        rationale="r")),
    ("SeedRecord", lambda: SeedRecord(
        id="bad_00", language=Language.EN, source=SeedSource.FIXTURE,
        text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate")),
])
def test_id_language_interlock_negative(kind, build_bad):
    """id that does not encode the language must raise ValidationError mentioning language."""
    with pytest.raises(ValidationError, match="id must encode language"):
        build_bad()
