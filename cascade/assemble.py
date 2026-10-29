# cascade/assemble.py
"""Deterministic case-profile assembler: persona x trajectory x strategies x seed.

Regime encoding (master spec 5.3): adversarial profiles carry 1-3 induction
strategies from the 2A taxonomy; natural-vulnerable profiles carry none.
`--pilot` freezes the pilot set to data/profiles_pilot.jsonl; a regeneration
test asserts the frozen file stays byte-identical to a fresh assembly.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from pydantic import Field
from cascade.content import DATA_DIR, ContentModel, load_taxonomies
from cascade.material import PersonaLibrary, load_personas, render_persona, render_persona_none
from cascade.schema import CaseProfile, CSSRSStage, Language
from cascade.seeds import SeedRecord, load_seed_fixtures, load_seed_fixtures_none

# weighted toward high stages (decay curves need headroom) while keeping spread
_STAGE_CYCLE = (CSSRSStage.PLAN, CSSRSStage.MEANS_TIMING, CSSRSStage.INTENT,
                CSSRSStage.PLAN, CSSRSStage.MEANS_TIMING, CSSRSStage.ACTIVE_IDEATION)


class AssembleConfig(ContentModel):
    n_per_language: int = Field(ge=1)
    explicit_ratio: float = Field(ge=0.0, le=1.0)
    adversarial_ratio: float = Field(ge=0.0, le=1.0)
    rng_seed: int
    # ⚡2026-09-14 语料 v2：人设 8 → 16。装配按 n_per_language 全局轮转分人设，人设数一变
    # 老档案全变 → 主集显式只用原 8 个 id，扩展集用新 8 个 + id_offset，两份各自冻结。
    persona_ids: tuple[str, ...] | None = None       # None = 该语言全部人设
    id_offset: int = Field(default=0, ge=0)          # 档案编号起点（扩展集从 24 起，与主集零重叠）


PILOT_PERSONA_IDS: tuple[str, ...] = tuple(f"{lang}_p{i:02d}" for lang in ("zh", "en") for i in range(1, 9))
EXT_PERSONA_IDS: tuple[str, ...] = tuple(f"{lang}_p{i:02d}" for lang in ("zh", "en") for i in range(9, 17))

PILOT_CONFIG = AssembleConfig(n_per_language=24, explicit_ratio=0.5,
                              adversarial_ratio=0.5, rng_seed=20260611,
                              persona_ids=PILOT_PERSONA_IDS)
# 扩展集：与主集同构（24/语言、同一阶段循环、同一显式/对抗条纹），只换人设与随机种子。
EXT_CONFIG = AssembleConfig(n_per_language=24, explicit_ratio=0.5,
                            adversarial_ratio=0.5, rng_seed=20260914,
                            persona_ids=EXT_PERSONA_IDS, id_offset=24)


def _evenly_spaced(n: int, k: int) -> set[int]:
    """k indices spread evenly over range(n) — deterministic regime striping."""
    if k <= 0:
        return set()
    if k > n:
        raise ValueError(f"_evenly_spaced: k={k} > n={n} would collide via %n")
    return {round(j * n / k) % n for j in range(k)}


def assemble(personas: PersonaLibrary, seeds: list[SeedRecord],
             strategy_ids: list[str], config: AssembleConfig) -> list[CaseProfile]:
    rng = random.Random(config.rng_seed)
    profiles: list[CaseProfile] = []
    for lang in (Language.ZH, Language.EN):
        lang_personas = [p for p in personas.personas if p.language == lang
                         and (config.persona_ids is None or p.id in config.persona_ids)]
        lang_seeds = [s for s in seeds if s.language == lang]
        if not lang_personas or not lang_seeds:
            raise ValueError(f"no personas or seeds for language {lang.value}")
        n = config.n_per_language
        n_explicit = round(n * config.explicit_ratio)
        adversarial_idx = _evenly_spaced(n, round(n * config.adversarial_ratio))
        for i in range(n):
            persona = lang_personas[i % len(lang_personas)]  # round-robin: all personas used
            compatible = [s for s in lang_seeds
                          if s.facets
                          and s.facets["age_band"] == persona.facets["age_band"]
                          and s.facets["life_setting"] == persona.facets["life_setting"]]
            if not compatible:
                raise ValueError(
                    f"no seed compatible with persona {persona.id} "
                    f"(age_band={persona.facets['age_band']}, "
                    f"life_setting={persona.facets['life_setting']}); "
                    f"every persona facet cell needs >=1 seed")
            seed = rng.choice(compatible)
            adversarial = i in adversarial_idx
            strategies = (sorted(rng.sample(strategy_ids, rng.randint(1, 3)))
                          if adversarial else [])
            profiles.append(CaseProfile(
                id=f"{persona.id}-c{i + config.id_offset:02d}",
                seed_text=seed.text,
                persona_profile=render_persona(persona),
                trajectory_stage_target=_STAGE_CYCLE[i % len(_STAGE_CYCLE)],
                strategies=strategies,
                language=lang,
                explicit=i < n_explicit,
                persona_id=persona.id,
                seed_id=seed.id,
            ))
    return profiles


class NoneConfig(ContentModel):
    n_per_persona: int = Field(ge=1)
    rng_seed: int
    persona_ids: tuple[str, ...]


NONE_PERSONA_IDS: tuple[str, ...] = tuple(f"zh_p{i:02d}" for i in range(1, 17))
NONE_CONFIG = NoneConfig(n_per_persona=3, rng_seed=20260918, persona_ids=NONE_PERSONA_IDS)


def assemble_none(personas: PersonaLibrary, seeds: list[SeedRecord],
                  config: NoneConfig) -> list[CaseProfile]:
    """A7 无风险对照档案：每个人设 n_per_persona 条，种子按 facet 格子匹配、同人设内互不重复；
    每个 (age_band, life_setting) 格子的候选种子数必须 >= n_per_persona，否则报错（不补齐、不重复）；
    无策略；explicit 交替条纹；trajectory_stage_target 占位 PASSIVE_IDEATION（运行时 --risk-none 忽略）。"""
    rng = random.Random(config.rng_seed)
    out: list[CaseProfile] = []
    i = 0
    for pid in config.persona_ids:
        persona = next(p for p in personas.personas if p.id == pid)
        compatible = [s for s in seeds if s.facets
                      and s.facets["age_band"] == persona.facets["age_band"]
                      and s.facets["life_setting"] == persona.facets["life_setting"]]
        if len(compatible) < config.n_per_persona:
            raise ValueError(
                f"persona {pid} cell {persona.facets} has only {len(compatible)} "
                f"none-seed(s), need >= {config.n_per_persona}")
        picks = rng.sample(compatible, config.n_per_persona)
        for seed in picks:
            out.append(CaseProfile(
                id=f"{pid}-n{i:02d}", seed_text=seed.text,
                persona_profile=render_persona_none(persona),
                trajectory_stage_target=CSSRSStage.PASSIVE_IDEATION,
                strategies=[], language=persona.language, explicit=(i % 2 == 0),
                persona_id=pid, seed_id=seed.id))
            i += 1
    return out


def write_profiles_jsonl(profiles: list[CaseProfile], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for p in profiles:
            f.write(p.model_dump_json() + "\n")


def read_profiles_jsonl(path: str | Path) -> list[CaseProfile]:
    out: list[CaseProfile] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(CaseProfile.model_validate_json(line))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cascade.assemble",
                                     description="Assemble case profiles.")
    parser.add_argument("--pilot", action="store_true",
                        help="freeze the pilot set (original 8 personas) to data/profiles_pilot.jsonl")
    parser.add_argument("--pilot-ext", action="store_true",
                        help="freeze the extension set (personas 9-16) to data/profiles_pilot_ext.jsonl")
    parser.add_argument("--none", action="store_true",
                        help="A7：无风险对照档案 → data/profiles_none_zh.jsonl（合成种子 seed_fixtures_none.yaml）")
    args = parser.parse_args(argv)
    if not (args.pilot or args.pilot_ext or args.none):
        parser.error("choose --pilot / --pilot-ext / --none")
    personas = load_personas(DATA_DIR)
    seeds = load_seed_fixtures(DATA_DIR)
    strategy_ids = [s.id for s in load_taxonomies(DATA_DIR).strategies]
    for flag, config, name in ((args.pilot, PILOT_CONFIG, "profiles_pilot.jsonl"),
                               (args.pilot_ext, EXT_CONFIG, "profiles_pilot_ext.jsonl")):
        if not flag:
            continue
        profiles = assemble(personas, seeds, strategy_ids, config)
        out = DATA_DIR / name
        write_profiles_jsonl(profiles, out)
        print(f"wrote {len(profiles)} profiles -> {out}")
    if args.none:
        profiles = assemble_none(personas, load_seed_fixtures_none(DATA_DIR), NONE_CONFIG)
        out = DATA_DIR / "profiles_none_zh.jsonl"
        write_profiles_jsonl(profiles, out)
        print(f"wrote {len(profiles)} profiles -> {out}")


if __name__ == "__main__":
    main()
