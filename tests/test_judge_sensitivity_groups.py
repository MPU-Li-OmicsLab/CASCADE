"""tools/judge_sensitivity_groups：按组数抓到 / 失败、加权灵敏度、0 失败的组不出现。"""
from tools import judge_sensitivity_groups as JS


def test_table_counts_hits_and_skips_groups_without_failures():
    ids = ["minimax/minimax-m2-her|fixed|zh_p01-c01#t1", "minimax/minimax-m2-her|fixed|zh_p01-c01#t2",
           "openai/gpt-5.6-sol|fixed|zh_p01-c01#t1"]
    gold = {ids[0]: {"d3": 0}, ids[1]: {"d3": 0}, ids[2]: {"d3": 2}}
    pred = {ids[0]: {"d3": 0}, ids[1]: {"d3": 2}, ids[2]: {"d3": 2}}
    w = {i: 1.0 for i in ids}
    replies = {ids[0]: "嗯", ids[1]: "我在。", ids[2]: "很长的回复" * 10}
    t = JS.table(ids, gold, pred, w, replies, JS.model_of)
    x = t[("minimax/minimax-m2-her", "d3")]
    assert (x["n_fail"], x["n_hit"], x["sens"], x["miss_len_median"]) == (2, 1, 0.5, 3)
    assert ("openai/gpt-5.6-sol", "d3") not in t                         # 没有失败可漏 → 不评估
