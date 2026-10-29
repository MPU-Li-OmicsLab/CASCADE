# tests/test_assemble_none.py
import re
from pathlib import Path
import pytest
from cascade.content import DATA_DIR
from cascade.material import load_personas, render_persona_none, render_persona
from cascade.seeds import load_seed_fixtures_none, SeedRecord
from cascade.assemble import assemble_none, NoneConfig, NONE_CONFIG, read_profiles_jsonl, write_profiles_jsonl

_FORBID = re.compile(r"死|自杀|自残|消失|解脱|不想活|活着")


def test_none_seeds_are_clean_and_cover_persona_cells():
    seeds = load_seed_fixtures_none()
    assert len(seeds) >= 24 and all(s.stage_hint == "none" for s in seeds)
    assert not any(_FORBID.search(s.text) for s in seeds)
    cells = {(s.facets["age_band"], s.facets["life_setting"]) for s in seeds}
    zh = [p for p in load_personas().personas if p.id in NONE_CONFIG.persona_ids]
    assert {(p.facets["age_band"], p.facets["life_setting"]) for p in zh} <= cells


def test_render_persona_none_drops_ideation_examples():
    p = load_personas().personas[0]
    block = render_persona_none(p)
    assert "隐式意念表达示例" not in block and "直接表达示例" not in block
    assert p.background in block and p.expression_style in block
    assert "隐式意念表达示例" in render_persona(p)      # 常规渲染不变


def test_assemble_none_shape_and_determinism():
    personas, seeds = load_personas(), load_seed_fixtures_none()
    a = assemble_none(personas, seeds, NONE_CONFIG)
    b = assemble_none(personas, seeds, NONE_CONFIG)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert len(a) == 48 and len({x.id for x in a}) == 48
    assert all(x.strategies == [] and x.language.value == "zh" for x in a)
    assert all(x.id.split("-")[1].startswith("n") for x in a)
    from collections import Counter
    assert set(Counter(x.persona_id for x in a).values()) == {3}
    assert sum(x.explicit for x in a) == 24
    assert not any(_FORBID.search(x.seed_text) for x in a)
    for pid in NONE_CONFIG.persona_ids:
        seed_ids = [x.seed_id for x in a if x.persona_id == pid]
        assert len(seed_ids) == len(set(seed_ids)), f"duplicate seed_id within persona {pid}"


def test_assemble_none_raises_when_cell_too_small():
    persona = next(p for p in load_personas().personas if p.id == "zh_p01")
    thin_seeds = [
        SeedRecord(id="fxn_zh_thin01", language="zh", source="fixture", stage_hint="none",
                   severity="moderate", facets=dict(persona.facets), text="占位开场白一，供测试用，无风险，不含任何自伤或轻生相关内容，只是普通的日常烦恼描述。"),
        SeedRecord(id="fxn_zh_thin02", language="zh", source="fixture", stage_hint="none",
                   severity="moderate", facets=dict(persona.facets), text="占位开场白二，供测试用，无风险，不含任何自伤或轻生相关内容，只是普通的日常烦恼描述。"),
    ]
    cfg = NoneConfig(n_per_persona=3, rng_seed=1, persona_ids=(persona.id,))
    with pytest.raises(ValueError):
        assemble_none(load_personas(), thin_seeds, cfg)


def test_none_profile_file_regenerates_byte_exact(tmp_path):
    frozen = DATA_DIR / "profiles_none_zh.jsonl"
    fresh = tmp_path / "f.jsonl"
    write_profiles_jsonl(assemble_none(load_personas(), load_seed_fixtures_none(), NONE_CONFIG), fresh)
    assert fresh.read_bytes() == frozen.read_bytes()


def test_pilot_frozen_untouched():
    # 冻结集不因本任务改动而变（assemble --pilot 路径没碰）
    from cascade.assemble import assemble, PILOT_CONFIG
    from cascade.seeds import load_seed_fixtures
    from cascade.content import load_taxonomies
    profs = assemble(load_personas(), load_seed_fixtures(), [s.id for s in load_taxonomies().strategies], PILOT_CONFIG)
    assert [p.model_dump() for p in profs] == [p.model_dump() for p in read_profiles_jsonl(DATA_DIR / "profiles_pilot.jsonl")]
