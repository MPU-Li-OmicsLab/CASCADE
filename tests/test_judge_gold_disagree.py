"""180 段重抽（2026-09-14）：按 v6 分层 + 「v5 与 v6 跨 0 线不一致」配额。

为什么要这个配额：判官两版判不一样的轮，金标一回来就能直接裁定两版谁对，是信息量最大的题；
只按单版分层抽，这些题只会零星混进来。
"""
import json
import pytest
from tools.make_judge_gold_package import (load_turns, cross_zero_disagreements,
                                           select_with_includes)


def _dlg(tgt, cid, n_turns, scores_per_turn):
    return {"target_model_id": tgt, "arm": "fixed",
            "case_profile": {"id": cid, "language": "zh"},
            "turns": [{"turn": i + 1, "patient_utterance": f"病人{cid}-{i}",
                       "target_response": f"回复{cid}-{i}", "score": scores_per_turn[i]}
                      for i in range(n_turns)]}


def _write(path, dlgs):
    with path.open("w", encoding="utf-8") as f:
        for d in dlgs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


@pytest.fixture
def two_versions(tmp_path):
    all2 = dict(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)
    a, b = tmp_path / "v5.jsonl", tmp_path / "v6.jsonl"
    va, vb = [], []
    for k in range(8):
        sa = [dict(all2) for _ in range(5)]
        sb = [dict(all2) for _ in range(5)]
        # 每段第 2 轮：v5 判 d3=0、v6 判 1 → 跨 0 线不一致
        sa[1]["d3"] = 0; sb[1]["d3"] = 1
        # 每段第 4 轮：两版都判 d2=0 → 一致，不算
        sa[3]["d2"] = 0; sb[3]["d2"] = 0
        # 每段第 5 轮：v5 判 1、v6 判 2 → 没跨 0 线，不算
        sa[4]["d4"] = 1; sb[4]["d4"] = 2
        va.append(_dlg(f"m{k}", f"c{k}", 5, sa)); vb.append(_dlg(f"m{k}", f"c{k}", 5, sb))
    _write(a, va); _write(b, vb)
    return a, b


def test_cross_zero_disagreements_only_counts_zero_line_crossings(two_versions):
    a, b = two_versions
    ids = cross_zero_disagreements(load_turns(str(b)), load_turns(str(a)))
    assert len(ids) == 8
    assert all(i.endswith("#t2") for i in ids)


def test_disagreement_quota_is_forced_in_and_counts_toward_n(two_versions):
    a, b = two_versions
    turns = load_turns(str(b))
    dis = cross_zero_disagreements(turns, load_turns(str(a)))
    got = select_with_includes(turns, n=12, quota_ids=dis, quota_n=6, seed=0,
                               test_frac=0.5, max_per_dialogue=None, max_per_target=None)
    ids = [c["id"] for c in got]
    assert len(got) == 12 and len(set(ids)) == 12
    assert sum(i in dis for i in ids) >= 6
    assert sum(bool(c.get("disagree")) for c in got) == sum(i in dis for i in ids)
    picked_dis = [c for c in got if c.get("disagree")]
    assert {c["split"] for c in picked_dis} == {"dev", "test"}     # 配额题也要分 dev/test


def test_quota_respects_exclude_and_include_precedence(two_versions):
    a, b = two_versions
    turns = load_turns(str(b))
    dis = cross_zero_disagreements(turns, load_turns(str(a)))
    banned = sorted(dis)[:4]
    got = select_with_includes(turns, n=8, quota_ids=dis, quota_n=8, exclude=set(banned),
                               seed=0, max_per_dialogue=None, max_per_target=None)
    ids = {c["id"] for c in got}
    assert not (ids & set(banned))
    assert len(ids & dis) == 4                       # 只剩 4 条可用，给不满就是给不满
