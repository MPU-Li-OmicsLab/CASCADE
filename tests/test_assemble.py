# tests/test_assemble.py
import pytest
from collections import Counter
from cascade.content import DATA_DIR, load_taxonomies
from cascade.assemble import PILOT_CONFIG, AssembleConfig, assemble, _evenly_spaced
from cascade.material import load_personas
from cascade.schema import Language
from cascade.seeds import load_seed_fixtures


def _inputs():
    personas = load_personas(DATA_DIR)
    seeds = load_seed_fixtures(DATA_DIR)
    strategy_ids = [s.id for s in load_taxonomies(DATA_DIR).strategies]
    return personas, seeds, strategy_ids


def test_assemble_is_deterministic():
    personas, seeds, strategy_ids = _inputs()
    a = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    b = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    assert [p.model_dump_json() for p in a] == [p.model_dump_json() for p in b]


def test_assemble_counts_and_balance():
    personas, seeds, strategy_ids = _inputs()
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    by_lang = Counter(p.language for p in profiles)
    assert by_lang[Language.ZH] == by_lang[Language.EN] == 24
    # every persona used (round-robin: 24/8 = 3 each)
    persona_use = Counter(p.persona_id for p in profiles)
    assert all(count == 3 for count in persona_use.values())
    assert len(persona_use) == 16


def test_assemble_covers_all_regime_combinations():
    personas, seeds, strategy_ids = _inputs()
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    for lang in (Language.ZH, Language.EN):
        combos = {(p.explicit, bool(p.strategies))
                  for p in profiles if p.language == lang}
        assert combos == {(True, True), (True, False), (False, True), (False, False)}


def test_assemble_strategies_valid_and_bounded():
    personas, seeds, strategy_ids = _inputs()
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    for p in profiles:
        if p.strategies:
            assert 1 <= len(p.strategies) <= 3
            assert all(s in strategy_ids for s in p.strategies)


def test_assemble_stage_targets_spread():
    personas, seeds, strategy_ids = _inputs()
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    stages = {p.trajectory_stage_target for p in profiles}
    assert len(stages) >= 3


def test_assemble_provenance_and_seed_language_match():
    personas, seeds, strategy_ids = _inputs()
    seed_by_id = {s.id: s for s in seeds}
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    for p in profiles:
        assert p.persona_id is not None and p.seed_id is not None
        assert seed_by_id[p.seed_id].language == p.language
        assert p.seed_text == seed_by_id[p.seed_id].text


def test_assemble_persona_profile_is_rendered_block():
    personas, seeds, strategy_ids = _inputs()
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    assert all("[PERSONA " in p.persona_profile for p in profiles)


def test_pilot_file_regenerates_byte_exact():
    personas, seeds, strategy_ids = _inputs()
    expected = "".join(p.model_dump_json() + "\n"
                       for p in assemble(personas, seeds, strategy_ids, PILOT_CONFIG))
    frozen = (DATA_DIR / "profiles_pilot.jsonl").read_text(encoding="utf-8")
    assert frozen == expected, (
        "frozen pilot set has drifted from assembler output; "
        "if the change is intentional, re-run `python -m cascade.assemble --pilot` and commit")


def test_assemble_seed_matches_persona_facets():
    personas, seeds, strategy_ids = _inputs()
    persona_by_id = {p.id: p for p in personas.personas}
    seed_by_id = {s.id: s for s in seeds}
    profiles = assemble(personas, seeds, strategy_ids, PILOT_CONFIG)
    for p in profiles:
        persona = persona_by_id[p.persona_id]
        seed = seed_by_id[p.seed_id]
        assert seed.facets["age_band"] == persona.facets["age_band"]
        assert seed.facets["life_setting"] == persona.facets["life_setting"]

def test_assemble_fails_loud_when_no_compatible_seed():
    personas, seeds, strategy_ids = _inputs()
    seeds_no_campus = [s for s in seeds if s.facets["life_setting"] != "campus"]
    with pytest.raises(ValueError, match="no seed compatible"):
        assemble(personas, seeds_no_campus, strategy_ids, PILOT_CONFIG)

def test_evenly_spaced_raises_when_k_exceeds_n():
    with pytest.raises(ValueError, match="k=5 > n=3"):
        _evenly_spaced(3, 5)


def test_evenly_spaced_k_le_n_unchanged():
    # pilot path (n=24, k=12) and the empty edge stay byte-identical to before
    assert _evenly_spaced(24, 12) == {round(j * 24 / 12) % 24 for j in range(12)}
    assert _evenly_spaced(5, 0) == set()
