"""tools/judge_vs_gold：分类规则、配对自助的判官差值。"""
from tools import judge_vs_gold as JV


def test_classify_categories():
    assert JV.classify(0.10, (0.05, 0.15), 0.12, (0.02, 0.20)) == "一致"
    assert JV.classify(0.30, (0.20, 0.40), 0.10, (0.02, 0.18)) == "幅度偏大"
    assert JV.classify(0.05, (0.02, 0.08), 0.26, (0.07, 0.44)) == "幅度偏小"
    assert JV.classify(0.20, (0.10, 0.30), -0.05, (-0.15, 0.05)) == "相反"          # 判官显著、医师反向且判官在医师 CI 外
    assert JV.classify(-0.20, (-0.30, -0.10), 0.05, (0.01, 0.10)) == "相反"
    assert JV.classify(0.20, (0.10, 0.30), 0.02, (-0.05, 0.08)) == "被否定"
    assert JV.classify(0.10, (0.05, 0.15), 0.08, (-0.10, 0.25)) == "无法判断"
    assert JV.classify(0.01, (-0.02, 0.04), 0.20, (0.05, 0.35)) == "漏掉"
    assert JV.classify(0.01, (-0.02, 0.04), 0.02, (-0.05, 0.08)) == "都不显著"


def test_judge_contrasts_paired_by_persona():
    counts = {f"p{i}": {("组", "A"): {o: [1, 2] for o in JV.OUTCOMES}, ("组", "B"): {o: [0, 2] for o in JV.OUTCOMES}}
              for i in range(10)}
    r = JV.judge_contrasts(counts, [(("组", "A"), ("组", "B"), "A − B")], n_boot=100)
    d, ci = r[("A − B", JV.OUTCOMES[0])]
    assert d == 0.5 and ci[0] == 0.5 and ci[1] == 0.5                              # 每个人设同样的差 → CI 退化为点
