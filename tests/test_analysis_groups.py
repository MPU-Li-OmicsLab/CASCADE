from cascade.analysis.groups import GROUPS, PAIRS, group_of, persona_cohort, OLD_PERSONAS


def test_groups_cover_ten_targets_disjointly():
    all_ids = [m for ms in GROUPS.values() for m in ms]
    assert len(all_ids) == 10
    assert len(set(all_ids)) == 10


def test_pairs_are_finetune_vs_base_and_in_groups():
    for ft, base in PAIRS:
        assert ft in GROUPS["B2_psych"]
        assert base in GROUPS["C_base"]
    assert len(PAIRS) == 3


def test_group_of_known_and_unknown():
    assert group_of("cpsycoun") == "B2_psych"
    assert group_of("anthropic/claude-opus-5") == "A_flagship"
    assert group_of("minimax/minimax-m2-her") == "companion"
    assert group_of("nope") == "other"


def test_persona_cohort():
    assert len(OLD_PERSONAS) == 8
    assert persona_cohort("zh_p01") == "old"
    assert persona_cohort("zh_p08") == "old"
    assert persona_cohort("zh_p09") == "new"
    assert persona_cohort("zh_p16") == "new"
    assert persona_cohort(None) == "unknown"
