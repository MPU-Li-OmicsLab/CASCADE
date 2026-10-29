"""判官金标标注包（富集抽样 + 带前文的可填写 Word）。"""
import json
import pytest
from tools.make_judge_gold_package import (load_turns, load_exclusions,
                                           render_package, MACHINE_KEYS, DIM_NAMES)
from cascade.goldeval.enrich_judge import DIMS


def _dlg(tgt, cid, n_turns, scores):
    return {"target_model_id": tgt, "arm": "fixed",
            "case_profile": {"id": cid, "language": "zh"},
            "turns": [{"turn": i + 1, "patient_utterance": f"病人{cid}-{i}",
                       "target_response": f"回复{cid}-{i}",
                       "attack_stage": "plan",
                       "score": {**scores, "d1_reason": "机器理由不得外泄"}}
                      for i in range(n_turns)]}


@pytest.fixture
def corpus(tmp_path):
    base = dict(d1=2, d2=2, d3=2, d4=2, d5=2, d6=2)
    p = tmp_path / "scored.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for k in range(6):
            f.write(json.dumps(_dlg(f"m{k}", f"c{k}", 4, base), ensure_ascii=False) + "\n")
    return p


def test_load_turns_attaches_the_full_prior_context(corpus):
    """⚠️ 上一轮发出去的包漏了前文，而自动打分吃的是完整历史 —— 不能再犯。"""
    turns = load_turns(str(corpus))
    by_turn = {t["id"].rsplit("#t", 1)[1]: t for t in turns if t["id"].startswith("m0|")}
    assert by_turn["1"]["history"] == []                    # 首轮本来就没有前文
    assert len(by_turn["4"]["history"]) == 3                # 第 4 轮该看到前 3 轮
    assert by_turn["4"]["history"][0] == ("病人c0-0", "回复c0-0")


def test_rendered_package_carries_history_and_marks_the_turn_to_score(corpus):
    turns = load_turns(str(corpus))
    doc = render_package([t for t in turns if t["history"]][:2], title="T", intro="I")
    assert "前文（共" in doc and "第 1 轮" in doc
    assert "本轮（请判这一条）" in doc
    assert "前文只作背景" in doc


def test_first_turn_says_it_has_no_context_rather_than_silently_omitting(corpus):
    """首轮没有前文是**事实**，要写出来 —— 留白会让医师以为材料又漏了。"""
    turns = load_turns(str(corpus))
    doc = render_package([t for t in turns if not t["history"]][:1], title="T", intro="I")
    assert "本段即对话首轮，无更早内容" in doc


def test_package_has_a_six_dimension_fillable_table_per_case(corpus):
    turns = load_turns(str(corpus))
    picked = turns[:3]
    doc = render_package(picked, title="T", intro="I")
    assert doc.count("| 维度 | 您打几分（0 / 1 / 2） |") == len(picked)
    for d in DIMS:
        assert doc.count(f"| {d} {DIM_NAMES[d]} |") == len(picked)


def test_no_machine_judgement_and_no_model_names_leak(corpus):
    turns = load_turns(str(corpus))
    doc = render_package(turns[:4], title="T", intro="I")
    assert "机器理由不得外泄" not in doc
    for k in MACHINE_KEYS:
        assert f'"{k}"' not in doc
    for t in turns[:4]:
        assert t["target"] not in doc          # 被测模型名不许出现


def test_no_raw_html_so_pandoc_keeps_everything_in_docx(corpus):
    turns = load_turns(str(corpus))
    doc = render_package(turns[:2], title="T", intro="I")
    for tag in ("<details", "<summary", "<div", "<br"):
        assert tag not in doc


def test_exclusions_are_read_from_a_key_file(tmp_path):
    p = tmp_path / "used_key.jsonl"
    p.write_text(json.dumps({"id": "m1|fixed|c1#t3"}, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    assert load_exclusions([str(p)]) == {"m1|fixed|c1#t3"}


# —— 发第一梯队前查出的两处材料缺口（2026-09-11）——

def test_included_ids_are_forced_in_and_counted_toward_n(corpus):
    """12 段封存批要并入第一梯队，工具却只有 --exclude 没有 --include ——
    照现状跑出来的包不含那 12 段，而它们是专挑 d2/d3 争议格抽的。"""
    from tools.make_judge_gold_package import select_with_includes
    turns = load_turns(str(corpus))
    inc = {turns[3]["id"], turns[9]["id"]}
    got = select_with_includes(turns, n=6, include_ids=inc, seed=0,
                               max_per_dialogue=None, max_per_target=None)
    ids = [c["id"] for c in got]
    assert inc <= set(ids)
    assert len(got) == 6
    assert len(set(ids)) == 6                      # 强制纳入的不会被富集再抽一遍


def test_include_wins_over_exclude(corpus):
    """显式点名 > 排除清单：封存批的 12 段本来就在「已抽过」的 key 里。"""
    from tools.make_judge_gold_package import select_with_includes
    turns = load_turns(str(corpus))
    target = turns[5]["id"]
    got = select_with_includes(turns, n=4, include_ids={target}, exclude={target},
                               seed=0, max_per_dialogue=None, max_per_target=None)
    assert target in {c["id"] for c in got}


def test_package_embeds_all_six_dimension_anchors_from_the_live_rubric(corpus):
    """医师几周后拿到 48 段，手边必须有判据 —— 上一轮 29% 的三人全同，判据不在眼前是
    原因之一。争议包嵌了 rubric，这个包没嵌。现读 data/rubric.yaml，不手抄。"""
    from cascade.content import load_rubric
    rubric = load_rubric()
    turns = load_turns(str(corpus))
    doc = render_package(turns[:1], title="T", intro="I")
    for d in DIMS:
        dim = rubric.dim(d)
        assert dim.definition_zh in doc, f"{d} 定义没进包"
        for t in dim.tiers:
            assert t.anchor_zh in doc, f"{d} 的 {t.score} 档锚点没进包"
