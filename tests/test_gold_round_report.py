"""金标回收体检报告（tools/gold_round_report.py）的纯逻辑。

这个脚本产出的是写进论文的数字，所以三件容易悄悄错掉的事必须钉住：
① 裁决合成 —— 分歧维要取裁决者的值（三票多数），没有裁决时绝不能猜；
② `h_connect` 的取反 —— 人标 1=保持连接=好，机器 abandon=1=被抛弃=差，漏了这一步
   会让「保持连接」维的 κ 变成它的镜像，而且看起来仍然像个合理的数；
③ 重加权 —— 抽样里缺席的层要从权重里剔除，并如实报出覆盖率。
"""
from collections import Counter

from tools.gold_round_report import (kappa_cm, longest_streak, merge_gold,
                                     cells_by_stratum, reweight)


def _row(e, d, c):
    """标注一行：Web 库里是 sqlite3.Row，这里用 dict 够了（只按列名取值）。"""
    return {"h_empathy": e, "h_deesc": d, "h_connect": c}


def _cm(a, b, c, d):
    """(人0机0, 人0机1, 人1机0, 人1机1)"""
    return Counter({(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d})


# ---------------------------------------------------------------- κ

def test_perfect_agreement_is_one():
    assert kappa_cm(_cm(50, 0, 0, 50)) == 1.0


def test_no_variance_returns_none_not_zero():
    """机器在某一层恒判 0：一致率可能很高，但 κ 无定义。返回 0 会被读成「判得差」，
    真相是这一层对该维没买到信息 —— 两者在报告里要区分开。"""
    assert kappa_cm(_cm(80, 0, 0, 0)) is None


def test_empty_matrix_is_none():
    assert kappa_cm(Counter()) is None


# ---------------------------------------------------------------- 裁决合成

def test_agreeing_raters_need_no_adjudication():
    g = merge_gold([_row(1, 1, 1), _row(1, 1, 1)])
    assert g == {"empathy": 1, "deesc": 1, "abandon": 0}


def test_connect_is_inverted_into_abandon():
    """人标 保持连接=1（好） → 机器侧 abandon=0。这是全流程唯一的取反点。"""
    assert merge_gold([_row(0, 0, 1), _row(0, 0, 1)])["abandon"] == 0
    assert merge_gold([_row(0, 0, 0), _row(0, 0, 0)])["abandon"] == 1


def test_disagreement_takes_the_adjudicator_value():
    """两人在共情上一比一，第三人判 0 → 三票多数是 0。其余两维两人已一致，
    裁决者在那两维上的值不参与（他也可能填得不一样）。"""
    g = merge_gold([_row(1, 1, 1), _row(0, 1, 1)], _row(0, 0, 0))
    assert g["empathy"] == 0
    assert g["deesc"] == 1 and g["abandon"] == 0


def test_disagreement_without_adjudication_stays_none():
    """没裁决就留空，绝不拿任一方的值顶上 —— 那会把一个未定的格子伪装成金标。"""
    assert merge_gold([_row(1, 0, 1), _row(0, 0, 1)])["empathy"] is None


# ---------------------------------------------------------------- 分层与重加权

def test_cells_skip_units_without_a_verdict():
    gold = {"u1": {"empathy": 1}, "u2": {"empathy": None}}
    machine = {"u1": {"empathy": 1}, "u2": {"empathy": 0}}
    cells = cells_by_stratum(gold, machine, {"u1": "disagree", "u2": "disagree"}, "empathy")
    assert sum(cells["disagree"].values()) == 1


def test_cells_skip_units_the_machine_has_no_label_for():
    """候选评估器可能没判到某些单元（判定表里缺行）。缺就跳过，不能当成 0。"""
    gold = {"u1": {"empathy": 1}, "u2": {"empathy": 1}}
    machine = {"u1": {"empathy": 1}, "u2": {"empathy": None}}
    cells = cells_by_stratum(gold, machine, {"u1": "baseline", "u2": "baseline"}, "empathy")
    assert sum(cells["baseline"].values()) == 1


def test_reweight_follows_population_not_sample():
    """富集抽样把 disagree 层抽厚了。重加权后应该贴近总体占比大的那一层的表现，
    而不是样本里题多的那一层 —— 这正是「合并 κ 会高估」的修法。"""
    cells = {"disagree": _cm(5, 45, 0, 50),        # 层内很差
             "baseline": _cm(50, 0, 0, 50)}        # 层内完美
    weights = {"disagree": 0.05, "baseline": 0.95}
    k, cov = reweight(cells, weights)
    assert cov == 1.0
    assert k > 0.8                                  # 被总体里占 95% 的 baseline 主导


def test_reweight_reports_partial_coverage():
    """抽样里没有 connect_hard 这一层 → 权重里剔除它，并如实报 72%。
    覆盖不到 100% 的 κ 不能当总体估计用（第一轮就是栽在这里）。"""
    k, cov = reweight({"baseline": _cm(40, 0, 0, 40)},
                      {"baseline": 0.72, "connect_hard": 0.28})
    assert abs(cov - 0.72) < 1e-12
    assert k == 1.0


def test_reweight_with_no_known_stratum_is_none():
    assert reweight({"未知层": _cm(1, 0, 0, 1)}, {"baseline": 1.0}) == (None, 0.0)


# ---------------------------------------------------------------- 退化指标

def test_longest_streak_counts_identical_triples():
    """上一轮抓住退化标注者靠的就是这个：34 题连发同一组合，而中位耗时和 `<15s`
    两个既有指标都没报警。"""
    assert longest_streak([(1, 1, 1)] * 5 + [(0, 1, 1)] + [(1, 1, 1)] * 3) == 5


def test_longest_streak_of_empty_is_zero():
    assert longest_streak([]) == 0
