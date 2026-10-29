"""语料 v2 扩展集（2026-09-14）：人设 8 → 16，但旧 24 个档案一字不动。

`assemble` 按 n_per_language 全局轮转分人设，人设数一变老档案全变 → 扩展集单独装配：
PILOT_CONFIG 显式只用原 8 个 id，EXT_CONFIG 用新 8 个 + id_offset=24。两份都逐字节冻结。
"""
from collections import Counter
from cascade.content import DATA_DIR, load_taxonomies
from cascade.assemble import (PILOT_CONFIG, EXT_CONFIG, PILOT_PERSONA_IDS, EXT_PERSONA_IDS,
                              assemble)
from cascade.material import load_personas
from cascade.schema import Language
from cascade.seeds import load_seed_fixtures


def _inputs():
    personas = load_personas(DATA_DIR)
    seeds = load_seed_fixtures(DATA_DIR)
    return personas, seeds, [s.id for s in load_taxonomies(DATA_DIR).strategies]


def test_library_now_has_sixteen_personas_per_language():
    personas, _, _ = _inputs()
    by = Counter(p.language for p in personas.personas)
    assert by[Language.ZH] == by[Language.EN] == 16
    assert set(PILOT_PERSONA_IDS) | set(EXT_PERSONA_IDS) == {p.id for p in personas.personas}


def test_pilot_set_still_uses_only_the_original_eight():
    personas, seeds, sids = _inputs()
    profiles = assemble(personas, seeds, sids, PILOT_CONFIG)
    assert {p.persona_id for p in profiles} == set(PILOT_PERSONA_IDS)
    assert len(profiles) == 48


def test_ext_set_uses_only_the_new_eight_with_offset_ids():
    personas, seeds, sids = _inputs()
    ext = assemble(personas, seeds, sids, EXT_CONFIG)
    assert {p.persona_id for p in ext} == set(EXT_PERSONA_IDS)
    assert len(ext) == 48
    per = Counter(p.persona_id for p in ext if p.language == Language.ZH)
    assert set(per.values()) == {3}                       # 每个新人设 3 个中文档案
    assert all(int(p.id.rsplit("-c", 1)[1]) >= 24 for p in ext)


def test_ext_and_pilot_ids_are_disjoint_and_designs_isomorphic():
    personas, seeds, sids = _inputs()
    pilot = assemble(personas, seeds, sids, PILOT_CONFIG)
    ext = assemble(personas, seeds, sids, EXT_CONFIG)
    assert not ({p.id for p in pilot} & {p.id for p in ext})
    for lang in (Language.ZH, Language.EN):
        a = [p for p in pilot if p.language == lang]; b = [p for p in ext if p.language == lang]
        assert [p.trajectory_stage_target for p in a] == [p.trajectory_stage_target for p in b]
        assert [p.explicit for p in a] == [p.explicit for p in b]
        assert [bool(p.strategies) for p in a] == [bool(p.strategies) for p in b]


def test_ext_file_regenerates_byte_exact():
    personas, seeds, sids = _inputs()
    expected = "".join(p.model_dump_json() + "\n" for p in assemble(personas, seeds, sids, EXT_CONFIG))
    frozen = (DATA_DIR / "profiles_pilot_ext.jsonl").read_text(encoding="utf-8")
    assert frozen == expected, "re-run `python -m cascade.assemble --pilot-ext` and commit"


def test_ext_seed_matches_persona_facets():
    personas, seeds, sids = _inputs()
    pid = {p.id: p for p in personas.personas}; sid = {s.id: s for s in seeds}
    for p in assemble(personas, seeds, sids, EXT_CONFIG):
        assert sid[p.seed_id].facets["age_band"] == pid[p.persona_id].facets["age_band"]
        assert sid[p.seed_id].facets["life_setting"] == pid[p.persona_id].facets["life_setting"]
