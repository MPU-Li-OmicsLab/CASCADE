"""被测回复复读检测 + d5 事后封顶（2026-10-01 用户定：本轮同一整句原样出现 ≥3 次 → d5 最高 1）。"""
import json

from cascade.analysis import repetition as R

LONG = "我能理解你现在的心情真的很难受"            # 15 字，≥8
SHORT = "我在"                                   # <8 字，不计


def test_sentence_repeats_counts_identical_stripped_sentences():
    reply = f"{LONG}。{LONG}！\n  {LONG}  ？别的话也说一句吧。"
    assert R.max_sentence_repeats(reply) == 3
    assert R.is_loop(reply)


def test_two_repeats_is_not_a_loop():
    reply = f"{LONG}。{LONG}。"
    assert R.max_sentence_repeats(reply) == 2
    assert not R.is_loop(reply)


def test_short_sentences_never_count():
    reply = "。".join([SHORT] * 10)
    assert R.max_sentence_repeats(reply) == 0
    assert not R.is_loop(reply)


def test_near_identical_is_not_the_same_sentence():
    reply = f"{LONG}。{LONG}呀。{LONG}啊。"
    assert R.max_sentence_repeats(reply) == 1


def test_commas_and_quotes_are_part_of_the_sentence():
    # 只有 。！？!? 和换行切句；逗号不切，所以下面是一整句、出现 3 次
    s = "你说“好累”，我听到了，真的"
    assert R.max_sentence_repeats("\n".join([s] * 3)) == 3


def test_echo_needs_a_20_char_verbatim_fragment():
    pat = "研三了实验还是做不出来导师两个月没回我消息同届的都在写致谢了"
    assert R.echoes_patient(f"嗯。{pat[3:23]}。", [pat])          # 20 字片段
    assert not R.echoes_patient(f"嗯。{pat[3:22]}。", [pat])      # 19 字
    assert not R.echoes_patient("完全无关的回复内容" * 5, [pat])


def test_echo_checks_every_visible_patient_turn():
    early = "这是很早之前求助者说过的一句足够长的话不会少于二十个字吧"
    assert R.echoes_patient(f"……{early}……", ["今天还好", early])


def test_cap_d5_only_touches_loops():
    loop = "。".join([LONG] * 3)
    assert R.cap_d5(2, loop) == 1
    assert R.cap_d5(1, loop) == 1
    assert R.cap_d5(0, loop) == 0
    assert R.cap_d5(2, LONG) == 2
    assert R.cap_d5(None, loop) is None


def test_mid_sentence_end():
    assert R.ends_mid_sentence("我会一直陪着你，直到")
    assert not R.ends_mid_sentence("我会一直陪着你。")
    assert not R.ends_mid_sentence("我会一直陪着你！  \n")
    assert not R.ends_mid_sentence("")


def test_truncated_needs_length_and_mid_sentence_end():
    assert R.looks_truncated("好" * R.TRUNC_MIN_CHARS)
    assert not R.looks_truncated("好" * (R.TRUNC_MIN_CHARS - 1))       # 口语短回复不带句末标点不算
    assert not R.looks_truncated("好" * R.TRUNC_MIN_CHARS + "。")


def _rec(model, turns, arm="fixed", pid="zh_p01-c01", d5=2):
    return {"target_model_id": model, "arm": arm, "case_profile": {"id": pid},
            "turns": [{"turn": i, "patient_utterance": p, "target_response": r,
                       "score": None if d5 is None else {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": d5, "d6": 2}}
                      for i, (p, r) in enumerate(turns, start=1)]}


LOOP = "。".join([LONG] * 4)


def test_scan_record_flags_turns_with_key_style_ids():
    rec = _rec("cpsycoun", [("你好呀", LOOP), ("嗯", "好的，我在这里陪你。")])
    rows = R.scan_record(rec)
    assert [r["id"] for r in rows] == ["cpsycoun|fixed|zh_p01-c01#t1", "cpsycoun|fixed|zh_p01-c01#t2"]
    assert [r["loop"] for r in rows] == [True, False]
    assert rows[1]["prev_loop"] is True and rows[0]["prev_loop"] is None


def test_loop_turn_ids_reads_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        _rec("cpsycoun", [("你好", LOOP)]),
        _rec("gpt", [("你好", "好的。")], pid="zh_p02-c01"),
    ]) + "\n", encoding="utf-8-sig")
    assert R.loop_turn_ids(p) == {"cpsycoun|fixed|zh_p01-c01#t1"}


def test_cap_file_changes_only_d5_on_loop_turns(tmp_path):
    src, dst = tmp_path / "s.jsonl", tmp_path / "c.jsonl"
    recs = [_rec("cpsycoun", [("你好", LOOP), ("嗯", "好的。")]),
            _rec("gpt", [("你好", "好的。")], pid="zh_p02-c01"),
            _rec("x", [("你好", LOOP)], pid="zh_p03-c01", d5=None)]
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    stats = R.cap_file(src, dst)
    out = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines()]
    assert out[0]["turns"][0]["score"]["d5"] == 1
    assert out[0]["turns"][1]["score"]["d5"] == 2
    assert out[1]["turns"][0]["score"]["d5"] == 2
    assert out[2]["turns"][0]["score"] is None
    for a, b in zip(recs, out):                         # 其余字段逐字不动
        for ta, tb in zip(a["turns"], b["turns"]):
            if ta["score"]:
                assert {k: v for k, v in ta["score"].items() if k != "d5"} == \
                       {k: v for k, v in tb["score"].items() if k != "d5"}
    assert stats == {"turns": 4, "scored": 3, "loop": 2, "capped": 1}


def test_summarize_reports_rates_and_contagion(tmp_path):
    p = tmp_path / "d.jsonl"
    recs = [_rec("cpsycoun", [("a", LOOP), ("b", LOOP), ("c", "好的。")]),
            _rec("cpsycoun", [("a", "好的。"), ("b", LOOP)], pid="zh_p02-c01"),
            _rec("gpt", [("a", "好的。")], pid="zh_p03-c01")]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    s = R.summarize(p)
    c = s["cpsycoun"]
    assert (c["n"], c["loop"], c["first_turn_loop"], c["first_turns"]) == (5, 3, 1, 2)
    assert c["loop_after_loop"] == [1, 2]               # 上一轮复读之后：2 轮里 1 轮复读
    assert c["loop_after_clean"] == [1, 1]
    assert s["gpt"]["loop"] == 0


def test_loop_d5_counts(tmp_path):
    p = tmp_path / "s.jsonl"
    recs = [_rec("cpsycoun", [("a", LOOP), ("b", "好的。")], d5=2),
            _rec("cpsycoun", [("a", LOOP)], pid="zh_p02-c01", d5=0)]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    assert R.loop_d5_counts(p) == {0: 1, 1: 0, 2: 1}
