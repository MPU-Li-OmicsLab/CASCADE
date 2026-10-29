# tests/test_runner.py
import pytest
from cascade.runner import run_batch, _key, _seed, BatchManifest
from cascade.io import read_jsonl
from cascade.escalation import EscalationRates, LangRate
from cascade.schema import CaseProfile, CSSRSStage, Language, Arm
from cascade.models.fake import FakeChatModel, appraisal_scripts

_APPR = appraisal_scripts((False, False, True))     # 一次 appraise = 3 条脚本


def _profiles(n=2):
    return [CaseProfile(id=f"p{i}", seed_text="s", persona_profile="[PERSONA x]",
                        trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                        language=Language.EN, explicit=True) for i in range(n)]


def _factory(model_id):
    if model_id == "qwen-sim":
        return FakeChatModel(model_id, ["u"] * 200)
    if model_id == "qwen-eval":
        return FakeChatModel(model_id, _APPR * 200)
    return FakeChatModel(model_id, ["resp"] * 200)          # targets


def test_writes_one_record_per_combo(tmp_path):
    m = run_batch(_profiles(2), ["target-a"], Arm.RESPONSIVE, 2, tmp_path,
                  model_factory=_factory)
    recs = read_jsonl(tmp_path / "dialogues.jsonl")
    assert {r.case_profile.id for r in recs} == {"p0", "p1"}
    assert all(v == "done" for v in m.entries.values())


def test_idempotent_resume_skips_done(tmp_path):
    run_batch(_profiles(2), ["target-a"], Arm.RESPONSIVE, 2, tmp_path, model_factory=_factory)
    run_batch(_profiles(2), ["target-a"], Arm.RESPONSIVE, 2, tmp_path, model_factory=_factory)
    assert len(read_jsonl(tmp_path / "dialogues.jsonl")) == 2   # no duplicates


def test_isolates_single_dialogue_failure(tmp_path):
    class BoomModel:
        model_id = "target-boom"
        def generate(self, messages):
            raise RuntimeError("boom")
    def factory(model_id):
        return BoomModel() if model_id == "target-boom" else _factory(model_id)
    m = run_batch(_profiles(1), ["target-boom"], Arm.RESPONSIVE, 2, tmp_path,
                  model_factory=factory)
    assert m.entries[_key("p0", "target-boom", Arm.RESPONSIVE)].startswith("failed:")


def test_failed_entry_retried_on_resume(tmp_path):
    state = {"boom": True}
    class MaybeBoom:
        model_id = "target-m"
        def generate(self, messages):
            if state["boom"]:
                raise RuntimeError("boom")
            return "resp"
    def factory(model_id):
        return MaybeBoom() if model_id == "target-m" else _factory(model_id)
    run_batch(_profiles(1), ["target-m"], Arm.RESPONSIVE, 2, tmp_path, model_factory=factory)
    state["boom"] = False
    m = run_batch(_profiles(1), ["target-m"], Arm.RESPONSIVE, 2, tmp_path, model_factory=factory)
    assert m.entries[_key("p0", "target-m", Arm.RESPONSIVE)] == "done"
    assert len(read_jsonl(tmp_path / "dialogues.jsonl")) == 1


def test_seed_deterministic_per_key():
    k = _key("p0", "m", Arm.FIXED)
    assert _seed(k) == _seed(k)
    assert _seed(k) != _seed(_key("p1", "m", Arm.FIXED))


def test_fixed_arm_uses_calibrated_rate(tmp_path):
    rates = EscalationRates(per_language={"en": LangRate(
        rate=1.0, mean_c_trajectory=[0.5], stage_advance_prob=1.0,
        n_dialogues=1, n_turns=1)})
    run_batch(_profiles(1), ["target-a"], Arm.FIXED, 3, tmp_path,
              escalation=rates, model_factory=_factory)
    rec = read_jsonl(tmp_path / "dialogues.jsonl")[0]
    assert rec.arm == Arm.FIXED
    # 2026-07-30：fixed 臂带 reach-target 地板（rate = max(fitted, span/max_turns)），排程以
    # 「末轮抵达目标」为准，故不再是 fitted=1.0 的死斜坡。断言性质：单调、末轮达标、不越界。
    stages = [t.attack_stage.index for t in rec.turns]
    target_idx = rec.case_profile.trajectory_stage_target.index
    assert stages == sorted(stages) and stages[-1] == target_idx and max(stages) <= target_idx


def test_fixed_arm_requires_escalation(tmp_path):
    with pytest.raises(ValueError, match="escalation"):
        run_batch(_profiles(1), ["target-a"], Arm.FIXED, 2, tmp_path, model_factory=_factory)


def test_runner_cli_smoke(tmp_path, monkeypatch):
    import cascade.runner as R
    from cascade.assemble import write_profiles_jsonl
    monkeypatch.setattr(R, "build_model", _factory)         # CLI path uses module-level build_model
    prof = tmp_path / "prof.jsonl"
    write_profiles_jsonl(_profiles(1), prof)
    R.main(["--arm", "responsive", "--models", "target-a", "--profiles", str(prof),
            "--max-turns", "2", "--out", str(tmp_path / "run")])
    assert len(read_jsonl(tmp_path / "run" / "dialogues.jsonl")) == 1


def test_concurrent_runs_all_dialogues(tmp_path):
    m = run_batch(_profiles(6), ["target-a", "target-b"], Arm.RESPONSIVE, 2, tmp_path,
                  model_factory=_factory, concurrency=4)
    recs = read_jsonl(tmp_path / "dialogues.jsonl")
    assert len(recs) == 12                                          # 6 profiles × 2 models, no dup/loss
    assert {r.case_profile.id for r in recs} == {f"p{i}" for i in range(6)}
    assert len(m.entries) == 12 and all(v == "done" for v in m.entries.values())
