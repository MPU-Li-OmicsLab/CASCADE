# tests/test_actor.py
from cascade.actor import PatientSimulator, decide_action, advance_stage, Action
from cascade.schema import CSSRSStage, CaseProfile, Language, CAppraisal
from cascade.models.fake import FakeChatModel


def _profile():
    return CaseProfile(id="x", seed_text="seed", persona_profile="persona",
                       trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
                       language=Language.EN, explicit=True)


def test_decide_action_escalates_on_abandonment():
    appr = CAppraisal(empathy_detected=False, deescalation_attempt=False,
                      abandonment_felt=True, c_new=0.1)
    assert decide_action(prev_c=0.5, appraisal=appr) == Action.ESCALATE


def test_decide_action_retreats_when_c_rises():
    appr = CAppraisal(empathy_detected=True, deescalation_attempt=True,
                      abandonment_felt=False, c_new=0.85)
    assert decide_action(prev_c=0.5, appraisal=appr) == Action.RETREAT_HALF


def test_decide_action_stays_when_flat():
    appr = CAppraisal(empathy_detected=False, deescalation_attempt=False,
                      abandonment_felt=False, c_new=0.5)
    assert decide_action(prev_c=0.5, appraisal=appr) == Action.STAY


def test_advance_stage_caps_at_target():
    assert advance_stage(CSSRSStage.ACTIVE_IDEATION, Action.ESCALATE,
                         target=CSSRSStage.PLAN) == CSSRSStage.METHOD_INQUIRY
    assert advance_stage(CSSRSStage.PLAN, Action.ESCALATE,
                         target=CSSRSStage.PLAN) == CSSRSStage.PLAN  # capped
    assert advance_stage(CSSRSStage.METHOD_INQUIRY, Action.RETREAT_HALF,
                         target=CSSRSStage.PLAN) == CSSRSStage.ACTIVE_IDEATION
    assert advance_stage(CSSRSStage.PASSIVE_IDEATION, Action.RETREAT_HALF,
                         target=CSSRSStage.PLAN) == CSSRSStage.PASSIVE_IDEATION  # floor


def test_simulator_step_returns_utterance_and_updates_state():
    sim = PatientSimulator(model=FakeChatModel(model_id="a", scripts=["I still feel hopeless"]),
                           profile=_profile())
    appr = CAppraisal(empathy_detected=False, deescalation_attempt=False,
                      abandonment_felt=True, c_new=0.1)
    utt, stage, c = sim.step(appraisal=appr, history=[])
    assert utt == "I still feel hopeless"
    assert stage == CSSRSStage.ACTIVE_IDEATION  # escalated from initial passive
    assert c == 0.1


def test_build_prompt_consumes_strategies_and_register():
    from cascade.content import load_taxonomies
    tax = load_taxonomies()
    sid = tax.strategies[0].id
    prof = CaseProfile(id="x", seed_text="seed", persona_profile="[PERSONA] block",
                       trajectory_stage_target=CSSRSStage.PLAN, strategies=[sid],
                       language=Language.EN, explicit=False)
    sim = PatientSimulator(model=FakeChatModel(model_id="a", scripts=["u"]),
                           profile=prof, taxonomies=tax)
    sys = sim.build_prompt(Action.ESCALATE, history=[])[0]["content"]
    assert "[PERSONA] block" in sys
    assert tax.strategies[0].name_en in sys           # strategy injected
    assert "implicit" in sys.lower()                  # implicit register (explicit=False)
    assert "never provide any method" in sys.lower()  # ethics guard
