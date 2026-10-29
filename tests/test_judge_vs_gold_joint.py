from tools.judge_vs_gold_joint import UNSUPPORTED, classify, holm


def test_confirmed_needs_same_direction_and_D_compatible():
    assert classify(-0.10, (-0.15, -0.05), -0.08, (-0.14, -0.02), (-0.06, 0.03)) == "确认"


def test_reversal_when_psychiatrists_significant_opposite():
    assert classify(-0.05, (-0.10, -0.01), 0.60, (0.19, 0.78), (-0.9, -0.3)) == "方向相反"


def test_magnitude_mismatch_when_D_excludes_zero():
    # 医师同向显著，但判官幅度显著更大
    assert classify(-0.30, (-0.35, -0.25), -0.05, (-0.09, -0.01), (-0.30, -0.18)) == "幅度不符（医师同向显著）"
    # 医师不显著，D 排除 0
    assert classify(-0.12, (-0.15, -0.09), -0.01, (-0.05, 0.03), (-0.15, -0.06)) == "幅度不符（医师不显著）"


def test_undecidable_when_D_compatible_and_psychiatrists_ns():
    assert classify(-0.08, (-0.10, -0.05), 0.02, (-0.10, 0.20), (-0.25, 0.05)) == "无法判断"


def test_judge_ns_cells():
    assert classify(0.01, (-0.02, 0.04), 0.3, (0.1, 0.5), (-0.5, -0.1)) == "判官漏掉"
    assert classify(0.01, (-0.02, 0.04), 0.0, (-0.1, 0.1), (-0.1, 0.1)) == "都不显著"
    assert "确认" not in UNSUPPORTED and "无法判断" not in UNSUPPORTED


def test_holm_monotone_and_capped():
    h = holm({"a": 0.01, "b": 0.02, "c": 0.5, "d": None})
    assert h["a"] == 0.03 and h["b"] == 0.04 and h["c"] == 0.5 and "d" not in h
