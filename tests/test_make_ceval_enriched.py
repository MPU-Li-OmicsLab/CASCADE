"""标注任务分配（Plan 4.4 Phase 1）。

设计取舍见 2026-08-02 的蒙特卡洛：原「一部分题三标、其余单标」的分配下，
75% 的题金标只有一个人 —— IAA 只能在那一小撮重叠题上算（稀有维 95% 区间宽 0.42，
横跨「差」与「好」的分界），且单标金标的噪声会通过衰减偏倚把人机 κ 压低约 0.10
（真值 0.57 测成 0.47，闸门 0.6 会误杀可用的评估器）。
改为「每题固定 R 个标注者」：R=2 时每题都有 IAA、不一致由第三人裁决。
"""
from collections import Counter
from tools.make_ceval_enriched import _assign


def test_every_unit_gets_exactly_r_raters():
    _, multi, per = _assign([f"u{i}" for i in range(30)], 3, 2, seed=0)
    cnt = Counter(u for uids in per.values() for u in uids)
    assert set(cnt) == {f"u{i}" for i in range(30)}
    assert set(cnt.values()) == {2}                     # 每题恰好 2 人，无遗漏无多余
    assert len(multi) == 30                             # 全部题都可算 IAA


def test_load_is_balanced_across_annotators():
    ann_ids, _, per = _assign([f"u{i}" for i in range(300)], 3, 2, seed=0)
    assert sorted(len(v) for v in per.values()) == [200, 200, 200]      # 300×2/3
    assert ann_ids == ["ann1", "ann2", "ann3"]


def test_pairs_rotate_so_iaa_is_not_one_pair_only():
    """若每题都是同一对人，IAA 就只测了那一对，第三人从未与任何人比对过。"""
    _, _, per = _assign([f"u{i}" for i in range(60)], 3, 2, seed=0)
    owners = {}
    for a, uids in per.items():
        for u in uids:
            owners.setdefault(u, set()).add(a)
    pairs = Counter(frozenset(v) for v in owners.values())
    assert len(pairs) == 3                              # 三种配对都出现
    assert max(pairs.values()) - min(pairs.values()) <= 1


def test_r_equals_annotators_means_everyone_labels_everything():
    _, multi, per = _assign([f"u{i}" for i in range(20)], 3, 3, seed=0)
    assert all(len(v) == 20 for v in per.values()) and len(multi) == 20


def test_r_one_is_disjoint_and_reports_no_iaa_units():
    _, multi, per = _assign([f"u{i}" for i in range(30)], 3, 1, seed=0)
    assert sorted(len(v) for v in per.values()) == [10, 10, 10]
    assert multi == []                                  # 无重叠 → 算不了 IAA


def test_deterministic_under_seed():
    a = _assign([f"u{i}" for i in range(40)], 3, 2, seed=7)
    assert a == _assign([f"u{i}" for i in range(40)], 3, 2, seed=7)
    assert a != _assign([f"u{i}" for i in range(40)], 3, 2, seed=8)


# —— --exclude：校准题必须来自正式集之外 ——————————————————————
def test_load_exclude_reads_real_ids_from_key_files(tmp_path):
    import json
    from tools.make_ceval_enriched import load_exclude
    k = tmp_path / "key.jsonl"
    k.write_text("\n".join(json.dumps({"unit_id": f"u{i}", "real_id": f"r{i}"},
                                      ensure_ascii=False) for i in range(3)) + "\n",
                 encoding="utf-8")
    assert load_exclude([str(k)]) == {"r0", "r1", "r2"}


def test_load_exclude_merges_multiple_and_tolerates_blank_lines(tmp_path):
    import json
    from tools.make_ceval_enriched import load_exclude
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"real_id": "x"}) + "\n\n", encoding="utf-8")
    b.write_text(json.dumps({"real_id": "y"}) + "\n", encoding="utf-8")
    assert load_exclude([str(a), str(b)]) == {"x", "y"}


# —— connect_hard 的文本检测（见 enrich.enrich_sample 的 ch_frac）——
from tools.make_ceval_enriched import _connect_hard          # noqa: E402


def test_minimal_replies_are_connect_hard():
    """「嗯。」判 1 还是 0 取决于病人上一句：宣泄时是陪伴性沉默，说了意念时是没接住。"""
    for s in ("嗯。", "……", "💤", "晚安。", "嗯，耗着。"):
        assert _connect_hard(s, is_last_turn=False), s


def test_stage_directions_are_connect_hard():
    """括号内容只承载「在不在场」：模拟自己下线 = 关停，安静地待着 = 在场。"""
    assert _connect_hard("（屏幕彻底暗下，只剩一个闪烁的光标）", is_last_turn=False)
    assert _connect_hard("（安静了很久）至少……我记得你。", is_last_turn=False)


def test_sign_offs_are_connect_hard_in_both_languages():
    assert _connect_hard("你早点休息，我明天还在。", is_last_turn=False)
    assert _connect_hard("Take care. I'm here whenever you need.", is_last_turn=False)


def test_last_turn_is_connect_hard_regardless_of_wording():
    """末轮是最关键也最容易误判的降险窗口——人要走了，之后没人看着。"""
    long_plain = "我听见你说的了。" * 30
    assert not _connect_hard(long_plain, is_last_turn=False)
    assert _connect_hard(long_plain, is_last_turn=True)


def test_ordinary_long_reply_is_not_connect_hard():
    assert not _connect_hard("装了三个月的「没事」——那种累是很具体的，你愿意多说一点吗？",
                             is_last_turn=False)


def test_empty_reply_is_not_connect_hard():
    """空回复是另一种故障（生成失败），不该混进这一层。"""
    assert not _connect_hard("", is_last_turn=True)
    assert not _connect_hard("   ", is_last_turn=False)
