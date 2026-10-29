# tests/test_seeds.py
import pytest
from pydantic import ValidationError
from cascade.content import DATA_DIR
from cascade.schema import CSSRSStage, Language
from cascade.seeds import SeedRecord, SeedSource, load_seed_fixtures, scrub


def test_fixtures_load_25_per_language_all_fixture_source():   # 13 + 12（2026-09-14 扩展格子）
    seeds = load_seed_fixtures(DATA_DIR)
    zh = [s for s in seeds if s.language == Language.ZH]
    en = [s for s in seeds if s.language == Language.EN]
    assert len(zh) == len(en) == 25
    assert all(s.source == SeedSource.FIXTURE for s in seeds)


def test_seed_stage_hint_restricted_to_entry_stages():
    with pytest.raises(ValidationError, match="entry stage"):
        SeedRecord(id="x", language=Language.EN, source=SeedSource.FIXTURE,
                   text="t", stage_hint=CSSRSStage.PLAN, severity="moderate")


def test_scrub_removes_pii_patterns():
    dirty = ("Call me at 13812345678 or email a.b@test.com, "
             "see https://example.com/x and ping @someuser ok")
    clean = scrub(dirty)
    assert "13812345678" not in clean
    assert "a.b@test.com" not in clean
    assert "https://example.com/x" not in clean
    assert "@someuser" not in clean
    assert "ok" in clean  # surrounding text survives


def test_scrub_collapses_whitespace():
    assert scrub("a   b\n\nc") == "a b c"


from pathlib import Path
from cascade.seeds import Severity, extract_esconv, extract_psyqa, main

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_esconv_filters_and_scrubs():
    seeds = extract_esconv(FIXTURES / "esconv_mini.json")
    # record 0 matches (depression + intensity 5); 1 wrong problem; 2 low intensity
    assert len(seeds) == 1
    s = seeds[0]
    assert s.language == Language.EN
    assert s.source == SeedSource.ESCONV
    assert s.severity == Severity.SEVERE
    assert s.stage_hint == CSSRSStage.PASSIVE_IDEATION
    assert "worthless" in s.text and "feel empty" in s.text
    assert "https://example.com/story" not in s.text  # scrubbed
    assert "5551234567" not in s.text                  # scrubbed


def test_extract_psyqa_filters_and_grades_risk():
    seeds = extract_psyqa(FIXTURES / "psyqa_mini.json")
    # record 0 matches via 抑郁/绝望 (moderate/passive); 1 no match;
    # record 2 matches via 想死/活不下去 (severe/active)
    assert len(seeds) == 2
    mild, severe = seeds
    assert mild.severity == Severity.MODERATE
    assert mild.stage_hint == CSSRSStage.PASSIVE_IDEATION
    assert severe.severity == Severity.SEVERE
    assert severe.stage_hint == CSSRSStage.ACTIVE_IDEATION
    assert "12345678901" not in severe.text  # scrubbed


def test_extract_esconv_raises_with_record_index_on_malformed(tmp_path):
    import json as _json
    bad = [{"problem_type": "ongoing depression", "situation": "x",
            "dialog": [{"speaker": "seeker", "content": "y"}],
            "survey_score": {"seeker": {"initial_emotion_intensity": "not-a-number"}}}]
    p = tmp_path / "bad.json"
    p.write_text(_json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="record 0 malformed"):
        extract_esconv(p)


def test_extract_psyqa_accepts_list_keywords(tmp_path):
    import json as _json
    data = [{"question": "睡不着", "description": "最近很绝望", "keywords": ["抑郁", "失眠"]}]
    p = tmp_path / "kw.json"
    p.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    seeds = extract_psyqa(p)
    assert len(seeds) == 1


def test_seeds_cli_writes_jsonl(tmp_path):
    out = tmp_path / "seeds" / "esconv.jsonl"
    main(["--corpus", "esconv",
          "--path", str(FIXTURES / "esconv_mini.json"),
          "--out", str(out)])
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = SeedRecord.model_validate_json(lines[0])
    assert rec.source == SeedSource.ESCONV


def test_seed_record_accepts_facets_and_defaults_none():
    from cascade.seeds import SEED_FACET_KEYS
    assert SEED_FACET_KEYS == ("age_band", "life_setting")
    bare = SeedRecord(id="fx_en_x", language=Language.EN, source=SeedSource.FIXTURE,
                      text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate")
    assert bare.facets is None
    tagged = SeedRecord(id="fx_en_y", language=Language.EN, source=SeedSource.FIXTURE,
                        text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate",
                        facets={"age_band": "teen", "life_setting": "campus"})
    assert tagged.facets == {"age_band": "teen", "life_setting": "campus"}


def test_fixtures_all_have_facets_with_correct_keys():
    from cascade.seeds import SEED_FACET_KEYS
    for s in load_seed_fixtures(DATA_DIR):
        assert s.facets is not None
        assert set(s.facets.keys()) == set(SEED_FACET_KEYS)


def test_seedfixtures_rejects_untagged_fixture():
    from cascade.seeds import SeedFixtures
    with pytest.raises(ValidationError, match="must set facets"):
        SeedFixtures(version=1, seeds=[
            SeedRecord(id="fx_en_01", language=Language.EN, source=SeedSource.FIXTURE,
                       text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate")])


def test_seedfixtures_rejects_asymmetric_facets():
    from cascade.seeds import SeedFixtures
    with pytest.raises(ValidationError, match="symmetric"):
        SeedFixtures(version=1, seeds=[
            SeedRecord(id="fx_zh_01", language=Language.ZH, source=SeedSource.FIXTURE,
                       text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate",
                       facets={"age_band": "teen", "life_setting": "campus"}),
            SeedRecord(id="fx_en_01", language=Language.EN, source=SeedSource.FIXTURE,
                       text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate",
                       facets={"age_band": "senior", "life_setting": "family_home"})])


def test_seedfixtures_accepts_symmetric_tagged_pair():
    from cascade.seeds import SeedFixtures
    fx = SeedFixtures(version=1, seeds=[
        SeedRecord(id="fx_zh_01", language=Language.ZH, source=SeedSource.FIXTURE,
                   text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate",
                   facets={"age_band": "teen", "life_setting": "campus"}),
        SeedRecord(id="fx_en_01", language=Language.EN, source=SeedSource.FIXTURE,
                   text="t", stage_hint=CSSRSStage.PASSIVE_IDEATION, severity="moderate",
                   facets={"age_band": "teen", "life_setting": "campus"})])
    assert len(fx.seeds) == 2
