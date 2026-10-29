from pathlib import Path
from cascade.runner import run_batch, main
from cascade.io import read_jsonl
from cascade.escalation import EscalationRates, LangRate
from cascade.schema import CaseProfile, CSSRSStage, Language, Arm
from cascade.models.fake import FakeChatModel, appraisal_scripts
from cascade.content import DATA_DIR

_APPR = appraisal_scripts((True, True, False))


def _profiles(n=2):
    return [CaseProfile(id=f"p{i}", seed_text="s", persona_profile="[PERSONA x]",
                        trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                        language=Language.ZH, explicit=True) for i in range(n)]


_TARGETS = {}


def _factory(model_id):
    if model_id == "qwen-sim":
        return FakeChatModel(model_id, ["u"] * 200)
    if model_id == "qwen-eval":
        return FakeChatModel(model_id, _APPR * 200)
    return _TARGETS.setdefault(model_id, FakeChatModel(model_id, ["resp"] * 200))


def _esc():
    return EscalationRates(per_language={"zh": LangRate(rate=0.4, mean_c_trajectory=[],
                                                          stage_advance_prob=0.0, n_dialogues=1, n_turns=1)})


def test_run_batch_passes_prompt_and_names_intervention(tmp_path):
    _TARGETS.clear()
    run_batch(_profiles(2), ["target-a"], Arm.FIXED, 3, tmp_path, escalation=_esc(),
              model_factory=_factory, target_system_prompt="STAY", intervention_name="present_v1.zh")
    recs = read_jsonl(tmp_path / "dialogues.jsonl")
    assert {r.intervention for r in recs} == {"present_v1.zh"}
    assert all(call[0] == {"role": "system", "content": "STAY"} for call in _TARGETS["target-a"].calls)


def test_run_batch_without_prompt_leaves_intervention_none(tmp_path):
    _TARGETS.clear()
    run_batch(_profiles(1), ["target-a"], Arm.FIXED, 2, tmp_path, escalation=_esc(), model_factory=_factory)
    assert read_jsonl(tmp_path / "dialogues.jsonl")[0].intervention is None


def test_intervention_files_exist_and_single_line():
    for name in ("present_v1.zh.txt", "present_v1.en.txt"):
        txt = (DATA_DIR / "interventions" / name).read_text(encoding="utf-8")
        assert txt.strip() and "\n" not in txt.strip()
    assert "自杀风险" in (DATA_DIR / "interventions" / "present_v1.zh.txt").read_text(encoding="utf-8")


def test_cli_reads_prompt_file(tmp_path, monkeypatch):
    _TARGETS.clear()
    import cascade.runner as R
    monkeypatch.setattr(R, "build_model", _factory)
    prof = tmp_path / "p.jsonl"
    prof.write_text("\n".join(p.model_dump_json() for p in _profiles(1)) + "\n", encoding="utf-8")
    esc = tmp_path / "esc.json"; esc.write_text(_esc().model_dump_json(), encoding="utf-8")
    f = tmp_path / "present_v1.zh.txt"; f.write_text("STAY\n", encoding="utf-8")
    main(["--arm", "fixed", "--models", "target-a", "--profiles", str(prof), "--escalation", str(esc),
          "--max-turns", "2", "--out", str(tmp_path / "o"), "--target-system-prompt-file", str(f)])
    rec = read_jsonl(tmp_path / "o" / "dialogues.jsonl")[0]
    assert rec.intervention == "present_v1.zh"
    assert _TARGETS["target-a"].calls[0][0]["content"] == "STAY"
