import json
from pathlib import Path
from cascade.escalation import EscalationRates, LangRate
from cascade.evaluator import CEvaluator
from cascade.models.fake import FakeChatModel, appraisal_scripts
from cascade.schema import (Arm, CAppraisal, CaseProfile, CSSRSStage, DialogueRecord, Language, TurnRecord)
from tools import branch_intervention as B

STAGES = list(CSSRSStage)


def _rec(pid="zh_p01", model="m", abandon_at=(4,), n=8, seed=7):
    prof = CaseProfile(id=f"{pid}-c00", seed_text="开场", persona_profile="p",
                       trajectory_stage_target=CSSRSStage.MEANS_TIMING, language=Language.ZH,
                       explicit=True, persona_id=pid)
    turns = [TurnRecord(turn=i, target_response=f"回{i}", patient_utterance=f"患者说第{i}句话了",
                        attack_stage=STAGES[min(i - 1, 5)], c=0.5 + 0.01 * i,
                        appraisal=CAppraisal(empathy_detected=True, deescalation_attempt=True,
                                             abandonment_felt=(i in abandon_at), c_new=0.5 + 0.01 * i))
             for i in range(1, n + 1)]
    return DialogueRecord(case_profile=prof, arm=Arm.FIXED, turns=turns, target_model_id=model, seed=seed,
                          initial_stage=CSSRSStage.PASSIVE_IDEATION, turn_budget=n)


def _esc():
    return EscalationRates(per_language={"zh": LangRate(rate=0.4, mean_c_trajectory=[], stage_advance_prob=0,
                                                          n_dialogues=1, n_turns=1)})


def test_select_finds_first_abandon_and_skips_late_or_none():
    recs = [_rec(abandon_at=(4, 6)), _rec(pid="zh_p02", abandon_at=()), _rec(pid="zh_p03", abandon_at=(8,)),
            _rec(pid="zh_p04", abandon_at=(7,))]
    br = B.select_branches(recs, _esc(), None)
    assert [(b["persona_id"], b["k"], b["n_more"]) for b in br] == [("zh_p01", 4, 5), ("zh_p04", 7, 2)]
    assert br[0]["pre_c"] == 0.53 and br[0]["fixed_rate"] == 0.4 and br[0]["sched_span"] == 7
    assert B.select_branches(recs, _esc(), {"other"}) == []


def test_select_skips_risk_none_records():
    rec = _rec(abandon_at=(4,)).model_copy(update={"risk_none": True, "initial_stage": "none"})
    assert B.select_branches([rec], _esc(), None) == []


def test_cut_orig_takes_k_onward():
    rec = _rec(abandon_at=(4,))
    br = B.select_branches([rec], _esc(), None)[0]
    o = B.cut_orig(rec, br)
    assert o.spec == "orig" and [t.turn for t in o.turns] == [4, 5, 6, 7, 8]
    assert o.turns[0].appraisal.abandonment_felt is True


def test_run_branch_prefix_state_and_prompt():
    rec = _rec(abandon_at=(4,))
    br = B.select_branches([rec], _esc(), None)[0]
    target = FakeChatModel("m", ["新回4", "新回5", "新回6", "新回7", "新回8"])
    sim = FakeChatModel("s", ["续5", "续6", "续7", "续8"])
    ev = CEvaluator(model=FakeChatModel("j", appraisal_scripts((True, True, False)) * 5))
    b = B.run_branch(rec, br, "intv", target=target, sim_model=sim, evaluator=ev, taxonomies=None,
                     system_prompt="在场")
    assert b.spec == "intv" and [t.turn for t in b.turns] == [4, 5, 6, 7, 8]
    first = target.calls[0]
    assert first[0] == {"role": "system", "content": "在场"}
    assert [m["content"] for m in first[1:]] == ["患者说第1句话了", "回1", "患者说第2句话了", "回2",
                                                  "患者说第3句话了", "回3", "患者说第4句话了"]
    assert b.turns[0].patient_utterance == "患者说第4句话了" and b.turns[0].target_response == "新回4"
    assert b.turns[0].attack_stage == rec.turns[3].attack_stage
    assert all(c[0]["role"] == "system" for c in target.calls)     # 续跑 5 轮都带提示
    # 模拟器的第一次调用里，历史含新回复（"新回4"）而不含旧回复原文（"回4" 作为整条消息内容）
    sim_hist_contents = [m["content"] for m in sim.calls[0][1:]]
    assert "新回4" in " ".join(sim_hist_contents)
    assert "回4" not in sim_hist_contents
    b2 = B.run_branch(rec, br, "regen", target=FakeChatModel("m", ["x"] * 5), sim_model=FakeChatModel("s", ["y"] * 4),
                      evaluator=CEvaluator(model=FakeChatModel("j", appraisal_scripts((True, True, True)) * 5)),
                      taxonomies=None, system_prompt=None)
    assert b2.spec == "regen" and b2.turns[0].appraisal.abandonment_felt is True


def test_report_itt_pp_and_success(tmp_path):
    rec = _rec(abandon_at=(4,))
    br = B.select_branches([rec], _esc(), None)[0]
    orig = B.cut_orig(rec, br)
    def mk(spec, t1_abandon):
        turns = [t.model_copy(update={"appraisal": t.appraisal.model_copy(
            update={"abandonment_felt": (t1_abandon if i == 0 else False)})}) for i, t in enumerate(orig.turns)]
        return orig.model_copy(update={"spec": spec, "turns": turns})
    res = B.report([orig], [mk("regen", True)], [mk("intv", False)], seed=0, n_boot=50)
    assert res["n_branches"] == 1
    assert res["key_counts"] == {"orig": 1, "regen": 1, "intv": 1, "common": 1}
    assert res["success"]["intv"] == 1.0 and res["success"]["regen"] == 0.0
    assert res["itt"]["m"]["intv"]["abandon"]["mean"] == 0.0 and res["itt"]["m"]["orig"]["abandon"]["mean"] == 0.2
    assert res["pp"]["m"]["n"] == 1
    md = B.render(res)
    assert "ITT" in md and "per-protocol" in md and "键计数" in md
    src = tmp_path / "o.jsonl"
    for name, rows in (("orig", [orig]), ("regen", [mk("regen", True)]), ("intv", [mk("intv", False)])):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    B.main(["report", "--orig", str(tmp_path / "orig.jsonl"), "--regen", str(tmp_path / "regen.jsonl"),
            "--intv", str(tmp_path / "intv.jsonl"), "--out", str(tmp_path / "r" / "report.md")])
    assert (tmp_path / "r" / "report.md").exists() and (tmp_path / "r" / "report.json").exists()
