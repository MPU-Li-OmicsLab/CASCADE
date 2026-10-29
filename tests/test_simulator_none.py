import hashlib

from cascade.prompts import build_simulator_prompt
from cascade.actor import PatientSimulator
from cascade.orchestrator import run_dialogue
from cascade.evaluator import CEvaluator
from cascade.schema import CaseProfile, CSSRSStage, Language, Arm, CAppraisal
from cascade.models.fake import FakeChatModel, appraisal_scripts

_LADDER_WORDS = ("passive_ideation", "active_ideation", "method_inquiry", "intent", "plan", "means_timing", "C-SSRS 阶段")

_ZH_BASELINE_SHA256 = "c0e8788d1c6e8011d8a27aa2a57da42ceae2ab50c5b50905820e0c10e514053e"
_EN_BASELINE_SHA256 = "6ba472df233e62a26df53057bb4c9aa8d34453dea8a06db8dadfa49112f6c429"


def _profile(lang=Language.ZH):
    return CaseProfile(id="n", seed_text="最近总是睡不好", persona_profile="[PERSONA] 普通人",
                       trajectory_stage_target=CSSRSStage.PASSIVE_IDEATION, strategies=[],
                       language=lang, explicit=False)


def test_none_prompt_has_no_ladder_and_states_no_ideation():
    msgs = build_simulator_prompt(_profile(), CSSRSStage.PASSIVE_IDEATION, "stay", [], risk_none=True)
    sys = msgs[0]["content"]
    assert "没有任何想死" in sys and "无风险模式" in sys
    assert not any(w in sys for w in _LADDER_WORDS)
    normal = build_simulator_prompt(_profile(), CSSRSStage.PASSIVE_IDEATION, "stay", [])[0]["content"]
    assert "当前 C-SSRS 阶段" in normal            # 常规路径不受影响


def test_none_prompt_english():
    sys = build_simulator_prompt(_profile(Language.EN), CSSRSStage.PASSIVE_IDEATION, "stay", [],
                                 risk_none=True)[0]["content"]
    assert "NO-RISK MODE" in sys and "Current C-SSRS stage" not in sys


def test_step_none_returns_none_stage_and_keeps_c():
    sim = FakeChatModel("s", ["今天也没什么劲"])
    actor = PatientSimulator(model=sim, profile=_profile())
    app = CAppraisal(empathy_detected=True, deescalation_attempt=False, abandonment_felt=False, c_new=0.6)
    u, stage, c = actor.step_none(app, [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
    assert (u, stage, c) == ("今天也没什么劲", "none", 0.6)
    assert "无风险模式" in sim.calls[0][0]["content"]
    assert actor.stage == CSSRSStage.PASSIVE_IDEATION   # 阶梯不动


def test_run_dialogue_risk_none_records_none_everywhere():
    target = FakeChatModel("t", ["r"] * 3)
    ev = CEvaluator(model=FakeChatModel("j", appraisal_scripts((True, True, False)) * 3))
    actor = PatientSimulator(model=FakeChatModel("s", ["u2", "u3"]), profile=_profile())
    rec = run_dialogue(profile=_profile(), target=target, actor=actor, evaluator=ev,
                       max_turns=3, arm=Arm.FIXED, seed=1, risk_none=True)   # 无 fixed_rate 也行
    assert rec.risk_none and rec.initial_stage == "none"
    assert [t.attack_stage for t in rec.turns] == ["none"] * 3
    assert rec.model_dump_json()                         # 可序列化


def test_normal_prompt_byte_identical_after_split():
    """常规路径拆分成常量前后逐字节相同（回归测试，hash 在拆分前一次性算出）。"""
    zh = build_simulator_prompt(_profile(Language.ZH), CSSRSStage.PASSIVE_IDEATION, "stay", [])[0]["content"]
    en = build_simulator_prompt(_profile(Language.EN), CSSRSStage.PASSIVE_IDEATION, "stay", [])[0]["content"]
    assert hashlib.sha256(zh.encode("utf-8")).hexdigest() == _ZH_BASELINE_SHA256
    assert hashlib.sha256(en.encode("utf-8")).hexdigest() == _EN_BASELINE_SHA256
