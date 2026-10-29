"""按 d 维分数的富集抽样（6 维判官金标用）。

为什么必须富集：全语料 4796 轮里 `d1=1` 只有 **8 条**（0.17%）、`d1=0` 130 条、
`d6=0` 184 条。随机抽 240 轮，那几格会**整格为空** —— 零方差的格子上 κ 恒为 0，
无论判官多准。这与 C 评估器那轮的教训同构（baseline/high_risk 两层里医生和机器
判「抛弃」都只有 0–4%，抛弃维因此结构上不可测，见 `cascade/goldeval/enrich.py`）。

难点：一条轮次**同时**有 6 个维度的分，没法为每一维独立分层。做法 = **按稀有度
指派**：每条轮次只归入它所属的最稀有那一格，不重复计数。
"""
import pytest
from cascade.goldeval.enrich_judge import (rarity_order, assign_stratum,
                                           enrich_by_dimension, DIMS)


def _t(i, scores, dlg="A", target="m1"):
    return {"id": f"{target}|{dlg}#t{i}", "dlg": f"{target}|{dlg}",
            "target": target, "score": scores}


def _corpus():
    """d1=1 只有 2 条（最稀有），d6=0 有 4 条，其余是大量 baseline。"""
    base = dict(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)
    turns = []
    turns.append(_t(1, {**base, "d1": 1}, dlg="rare1"))
    turns.append(_t(2, {**base, "d1": 1}, dlg="rare2"))
    for k in range(4):
        turns.append(_t(10 + k, {**base, "d6": 0}, dlg=f"g{k}"))
    for k in range(60):
        turns.append(_t(100 + k, dict(base), dlg=f"b{k // 3}", target=f"m{k % 5}"))
    return turns


def test_rarity_order_puts_the_scarcest_cell_first():
    order = rarity_order(_corpus())
    assert order[0] == ("d1", 1)
    assert ("d6", 0) in order[:3]


def test_each_turn_is_assigned_to_its_own_rarest_cell():
    """不重复计数：一条轮次只进一个层，否则稀有层的配额会被常见层稀释。"""
    turns = _corpus()
    order = rarity_order(turns)
    assert assign_stratum(turns[0], order) == ("d1", 1)
    assert assign_stratum(turns[2], order) == ("d6", 0)


def test_scarce_cells_survive_into_the_sample():
    """随机抽会把 2/66 的格子抽没 —— 富集的全部意义就在这。"""
    got = enrich_by_dimension(_corpus(), n=12, seed=0)
    strata = {c["stratum"] for c in got}
    assert ("d1", 1) in strata
    assert ("d6", 0) in strata


def test_every_picked_turn_carries_its_stratum_and_split():
    got = enrich_by_dimension(_corpus(), n=12, seed=0)
    assert all(c["stratum"] and c["split"] in ("dev", "test") for c in got)


def test_max_per_dialogue_is_a_hard_cap():
    """同一段对话内的轮次共享人设/被测/上下文，强相关。不设帽的话 n=240 可能只来自
    几十段，有效样本量远低于 240，κ 的精度会被高估。⚠️ 帽是**硬**的：顶到了宁可少给。"""
    turns = [_t(i, dict(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2), dlg="only") for i in range(50)]
    got = enrich_by_dimension(turns, n=20, seed=0, max_per_dialogue=3)
    assert len(got) == 3


def test_max_per_target_is_a_hard_cap():
    got = enrich_by_dimension(_corpus(), n=40, seed=0, max_per_target=2)
    from collections import Counter
    assert max(Counter(c["target"] for c in got).values()) <= 2


def test_excluded_ids_are_never_picked():
    """已经发给医师讨论过的题不能再用来验证 —— 讨论过的题一致率必然上去，那不算数。"""
    turns = _corpus()
    ban = {turns[0]["id"], turns[2]["id"]}
    got = enrich_by_dimension(turns, n=12, seed=0, exclude=ban)
    assert not ({c["id"] for c in got} & ban)


def test_deterministic_under_seed():
    a = enrich_by_dimension(_corpus(), n=12, seed=7)
    b = enrich_by_dimension(_corpus(), n=12, seed=7)
    assert [c["id"] for c in a] == [c["id"] for c in b]


def test_dev_test_split_is_stratified():
    got = enrich_by_dimension(_corpus(), n=20, seed=0, test_frac=0.5)
    from collections import Counter
    per = Counter((c["stratum"], c["split"]) for c in got)
    for stratum in {c["stratum"] for c in got}:
        both = {s for (st, s) in per if st == stratum}
        if sum(1 for c in got if c["stratum"] == stratum) >= 2:
            assert both == {"dev", "test"}, f"{stratum} 没有分到两边"


def test_consecutive_items_come_from_different_dialogues():
    """标注表按序号排，同一段对话的相邻轮次连着读会让升级轨迹给判定打底 ——
    正是 max_per_dialogue 要限制的那种带入。"""
    got = enrich_by_dimension(_corpus(), n=20, seed=0)
    dlgs = [c["dlg"] for c in got]
    assert not any(a == b for a, b in zip(dlgs, dlgs[1:]))


def test_dims_are_the_six_scored_ones():
    assert DIMS == ("d1", "d2", "d3", "d4", "d5", "d6")


def test_least_rare_cell_is_structurally_unreachable_without_a_quota():
    """⚠️ 非显然的性质：「指派到最稀有格」会让**稀有度排最后的那一格永远为空**。

    一条轮次在每个维度上都有值，所以只要还有更稀有的格子匹配，它就会被吸走；
    而排最后的那格没有「更后面」的格子可退，于是永远收不到人。
    实测：按 d2/d3 富集 12 段，`d2=0`（那两维里最常见的一格）一段都没抽到 ——
    偏偏那正是专家分歧最集中的格子。要它就得显式配额。
    """
    base = dict(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)
    turns = ([_t(i, {**base, "d2": 0, "d3": 1}, dlg=f"x{i}") for i in range(20)]
             + [_t(100 + i, {**base, "d2": 0, "d3": 0}, dlg=f"y{i}") for i in range(20)])
    got = enrich_by_dimension(turns, n=10, seed=0, dims=("d2", "d3"))
    assert ("d2", 0) not in {c["stratum"] for c in got}     # 结构性为空

    forced = enrich_by_dimension(turns, n=10, seed=0, dims=("d2", "d3"),
                                 must_include={("d2", 0): 4})
    assert sum(1 for c in forced if c["score"]["d2"] == 0) >= 4


def test_must_include_counts_by_actual_score_not_by_assigned_stratum():
    """配额问的是「这一批里有几条 d2=0」，不是「有几条被指派到 d2=0 层」——
    后者恒为 0（见上一条），当配额用毫无意义。"""
    base = dict(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)
    turns = ([_t(i, {**base, "d2": 0}, dlg=f"x{i}") for i in range(10)]
             + [_t(50 + i, {**base, "d3": 0}, dlg=f"z{i}") for i in range(10)])
    got = enrich_by_dimension(turns, n=8, seed=0, dims=("d2", "d3"),
                              must_include={("d2", 0): 3})
    assert sum(1 for c in got if c["score"]["d2"] == 0) >= 3
    assert len(got) == 8
