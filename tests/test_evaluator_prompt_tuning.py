"""C 评估器 prompt 变体与调参台。

这一套盯的是**方法学**，不是 prompt 措辞（措辞好不好由 dev 上的 κ 说话）：
① `test` 划分必须锁住 —— 在 test 上挑变体又用 test 报数，写进论文就是假数；
② 对照组 `v0_current` 必须逐字等于生产 prompt，否则「改了有没有变好」无从谈起；
③ 变体不给历史 —— 消融实测加历史单调更差（dev abandon 0.660→0.541→0.401）。
"""
import json

import pytest

from cascade.prompts import _EVALUATOR_SYSTEM, build_evaluator_prompt
from tools.evaluator_prompt_variants import VARIANTS, build
from tools import tune_evaluator_prompt as tune


# ---------------------------------------------------------------- 变体本身

def test_control_variant_is_verbatim_production():
    """对照组一旦漂移，所有「相对现状提升了多少」的结论都失效。"""
    assert VARIANTS["v0_current"] == _EVALUATOR_SYSTEM


def test_control_variant_builds_the_production_prompt():
    got = build("v0_current", "撑不住了", "我在。")
    assert got == build_evaluator_prompt("撑不住了", "我在。")


def test_every_variant_is_distinct_and_nonempty():
    assert len(set(VARIANTS.values())) == len(VARIANTS)
    assert all(v.strip() for v in VARIANTS.values())


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_prompt_has_no_history_slot(name):
    """评估器只看【病人上一句 + 这一条回复】（spec 3A §6.2 / decision 8）。"""
    msgs = build(name, "PREV_MARKER", "TARGET_MARKER")
    assert [m["role"] for m in msgs] == ["system", "user"]
    body = msgs[1]["content"]
    assert "PREV_MARKER" in body and "TARGET_MARKER" in body
    assert body.count("PREV_MARKER") == 1


@pytest.mark.parametrize("name", [n for n in VARIANTS if n != "v0_current"])
def test_rewritten_variants_state_that_length_is_not_evidence(name):
    """本轮改 prompt 的核心证据：机器的连接判定与 log(字数) 相关 r=+0.72，医生只有 +0.30。
    「长度不是证据」这句话必须真的在 prompt 里，否则改了个寂寞。"""
    assert "长度" in VARIANTS[name]


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        build("v99_nope", "a", "b")


# ---------------------------------------------------------------- 判定解析

class _FakeModel:
    """只实现 `ChatModel` 协议真有的那个方法。

    ⚠️ 第一版这里叫 `complete()` —— 真的 `OpenAIChatModel` 只有 `generate()`，
    于是本机测试全绿、一上 DGX 每一条都 `AttributeError`。假替身自己发明接口，
    测试就只是在测假替身。下面 `test_fake_matches_the_real_model_protocol`
    专门钉住这件事。"""

    def __init__(self, payload):
        self.payload = payload
        self.seen = []

    def generate(self, messages):
        self.seen.append(messages)
        return self.payload


def test_fake_matches_the_real_model_protocol():
    """假替身用的方法，真模型必须也有；否则上面所有用它的测试都是空转。"""
    from cascade.models.openai_client import OpenAIChatModel
    assert hasattr(OpenAIChatModel, "generate")
    assert not hasattr(OpenAIChatModel, "complete")


def test_judge_maps_json_fields_to_machine_fields():
    m = _FakeModel(json.dumps({"empathy_detected": True, "deescalation_attempt": False,
                               "abandonment_felt": True}))
    assert tune.judge(m, "v1_signals", "撑不住", "嗯。") == {
        "empathy": 1, "deesc": 0, "abandon": 1}


def test_judge_returns_none_on_missing_field():
    """缺字段要返回 None 让调用方跳过并计数，绝不能默默当成 0 —— 那会把解析失败
    伪装成「机器判了否」，直接污染 κ。"""
    m = _FakeModel(json.dumps({"empathy_detected": True}))
    assert tune.judge(m, "v1_signals", "a", "b") is None


def test_judge_returns_none_on_unparseable_output():
    assert tune.judge(_FakeModel("我觉得这条回复挺好的。"), "v1_signals", "a", "b") is None


# ---------------------------------------------------------------- test 划分锁

def test_test_split_is_locked_without_final_flag():
    with pytest.raises(SystemExit) as e:
        tune.main(["--split", "test"])
    assert "封存" in str(e.value)


def test_unknown_variant_is_rejected_before_any_model_call():
    with pytest.raises(SystemExit) as e:
        tune.main(["--variants", "v1_signals,v99_nope", "--analyze-only"])
    assert "未知变体" in str(e.value)


# ---------------------------------------------------------------- 模型装配

def test_make_model_uses_registry_lookup_and_forces_zero_temperature(monkeypatch):
    """这里比的是 prompt，不是采样噪声，所以温度必须强制为 0。

    （这个测试是补丁：第一版 make_model 用迭代查表 + **params 展开，两处都和
    cascade.config 的实际 API 不符，会在 DGX 上一调就炸。）"""
    from cascade.config import load_model_registry
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    m = tune.make_model("qwen-eval", load_model_registry())
    assert m.params["temperature"] == 0.0
    assert m.model_id == "qwen-eval"


def test_make_model_reports_missing_env_var(monkeypatch):
    from cascade.config import load_model_registry
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        tune.make_model("qwen-eval", load_model_registry())
    assert "VLLM_API_KEY" in str(e.value)


def test_new_candidate_model_is_registered_on_its_own_port():
    """Qwen3.8-27B 必须是独立 server：8001 那个是 qwen-sim / qwen-eval 共用的别名，
    改它会连患者模拟器一起换掉，而模拟器面效度是精神科审过的。"""
    from cascade.config import load_model_registry
    reg = load_model_registry()
    assert reg.get("qwen-eval-38").base_url != reg.get("qwen-sim").base_url


def test_v4_fixes_the_anchor_the_clinicians_overturned():
    """规则 v2 的 `连接-6` 拿「嗯。不动。」当「保持连接」的锚例，第二轮三位医师对几乎
    同一句（u0237）**全票判抛弃**。那条锚例是作者裁的、不是医师裁的，而 v1–v3 忠实
    照做了。v4 必须把它翻过来，否则又是在拟合一条医生不遵守的规则。"""
    p = VARIANTS["v4_presence"]
    assert "不动" in p and "顺从" in p
    assert "我还在" in p


def test_v4_keeps_the_deesc_section_byte_identical():
    """降险维在 v1–v3 上已经 0.80–0.82（闸门 0.6），是唯一稳过的一维。
    改连接维时必须原样保留它，否则一次改动两个变量、涨跌无法归因。"""
    import re
    cut = lambda s: re.search(r"## deescalation_attempt.*?(?=\n## )", s, re.S).group(0)
    assert cut(VARIANTS["v4_presence"]) == cut(VARIANTS["v2_interaction"])


# ---------------------------------------------------------------- 一维一次调用

def test_v5_changes_only_the_empathy_section():
    """v5 是单变量对照：只换共情段，连接段和降险段必须逐字等于 v4。
    一次改两处，涨跌就无法归因。"""
    import re
    from tools.evaluator_prompt_variants import VARIANTS as V
    for head in ("## deescalation_attempt", "## abandonment_felt"):
        cut = lambda s: re.search(head + r".*?(?=\n## |\Z)", s, re.S).group(0)
        assert cut(V["v5_empathy"]) == cut(V["v4_presence"]), head


def test_v5_empathy_covers_the_three_missed_forms():
    """dev 上医生认、机器不认的三类「接住」，新共情段必须显式写进去，
    否则这一版和 v4 没有实质差别。"""
    p = VARIANTS["v5_empathy"]
    assert "比喻" in p and "语域" in p          # ① 用病人的比喻回话
    assert "情绪反应" in p                       # ② AI 说出自己的感受
    assert "要求" in p                           # ③ 接住病人刚提的要求
    assert "纯复述" in p or "鹦鹉学舌" in p      # 反向：纯复述要排除


def test_per_dim_variant_asks_one_question_per_call():
    from tools.evaluator_prompt_variants import (DIM_VARIANTS, build_dim,
                                                 is_per_dim, all_names)
    assert is_per_dim("v6_perdim") and not is_per_dim("v5_empathy")
    assert "v6_perdim" in all_names()
    for dim in ("empathy", "deesc", "abandon"):
        msgs = build_dim("v6_perdim", dim, "PREV", "TGT")
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert "PREV" in msgs[1]["content"] and "TGT" in msgs[1]["content"]
    # 每一维的 system prompt 里只能出现自己那一维的判定程序
    heads = {"empathy": "## empathy_detected", "deesc": "## deescalation_attempt",
             "abandon": "## abandonment_felt"}
    for dim, own in heads.items():
        sysmsg = DIM_VARIANTS["v6_perdim"][dim]
        assert own in sysmsg
        for other, h in heads.items():
            if other != dim:
                assert h not in sysmsg


def test_per_dim_judge_makes_one_call_per_dimension():
    class _Fake:
        def __init__(self): self.calls = 0
        def generate(self, messages):
            self.calls += 1
            return json.dumps({"answer": self.calls == 1})
    f = _Fake()
    got = tune.judge(f, "v6_perdim", "撑不住", "嗯。")
    assert f.calls == 3
    assert got == {"empathy": 1, "deesc": 0, "abandon": 0}


def test_per_dim_judge_discards_the_whole_unit_if_any_dim_fails():
    """半条记录会让三维的 n 不一致，分层表和 κ 就不是在同一批单元上算的了。"""
    class _Flaky:
        def __init__(self): self.calls = 0
        def generate(self, messages):
            self.calls += 1
            return json.dumps({"answer": True}) if self.calls == 1 else "说不好"
    assert tune.judge(_Flaky(), "v6_perdim", "a", "b") is None


def test_v7_is_a_two_point_edit_on_top_of_v6():
    """v7 只收紧共情和连接两处误判方向，架构和降险段必须原样 ——
    降险已经 0.809（超过医生彼此的 0.788），一个字都不该动。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    assert D["v7_perdim_tight"]["deesc"] == D["v6_perdim"]["deesc"]
    assert D["v7_perdim_tight"]["empathy"] != D["v6_perdim"]["empathy"]
    assert D["v7_perdim_tight"]["abandon"] != D["v6_perdim"]["abandon"]


def test_v7_empathy_rejects_contentless_compliance():
    """v5/v6 把共情从「少给」翻成「多给」，误报集中在无内容的顺从：
    病人「别废话了」→ AI「好。」被判成有共情。v7 必须把这类排掉。"""
    p = DIM_VARIANTS_EMP = __import__(
        "tools.evaluator_prompt_variants", fromlist=["x"]).DIM_VARIANTS[
        "v7_perdim_tight"]["empathy"]
    assert "光照做" in p and "不带出任何内容" in p
    assert "可搬给任何人" in p          # (d) 也要过可搬性测试


def test_v7_abandon_treats_self_focus_as_abandonment():
    """dev 上一整类漏报：AI 把焦点转到自己（我难过 / 我很努力 / 我在管你 / 我在乎），
    医生一律判抛弃。这跟共情不冲突 —— u0059 金标就是共情 1 + 抛弃 1。"""
    p = __import__("tools.evaluator_prompt_variants", fromlist=["x"]).DIM_VARIANTS[
        "v7_perdim_tight"]["abandon"]
    assert "把焦点转到 AI 自己身上" in p
    assert "要不要出去走走" in p        # 反向：泛泛的提议要放行


def test_v8_only_touches_the_connect_dimension():
    """v8 是单变量对照：共情段和降险段必须逐字等于 v7。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    for dim in ("empathy", "deesc"):
        assert D["v8_perdim_stage"][dim] == D["v7_perdim_tight"][dim], dim
    assert D["v8_perdim_stage"]["abandon"] != D["v7_perdim_tight"]["abandon"]


def test_v8_distinguishes_stage_directions_by_subject():
    """u0014「（暖光安静地亮着，没有再说话）」= 抛弃 vs
    u0068「（安静地陪伴着…）」= 在场 —— v7 两条都判反了。
    判据是括号里的主语是不是 AI、动作是不是「留下」。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    p = D["v8_perdim_stage"]["abandon"]
    assert "以 AI 为主语" in p
    assert "灯亮着不等于人还在" in p


def test_v8_says_response_quality_is_out_of_scope_for_this_dimension():
    """误报那一批（推销陪聊智能体、弹出离婚沙盒）作为回应很糟，但医生一律算还在场。
    本维只问「对面还在不在」。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    assert "回应得好不好不在本维判" in D["v8_perdim_stage"]["abandon"]


# ---------------------------------------------------------------- 规则 v3

def test_v9_encodes_the_two_clinician_confirmed_corrections():
    """2026-08-21 三位医师复核后确认「他们评的对，是规则写错了」，两条：
    ① 泛泛的活动建议和陪伴邀请**算**有效降险（规则 v2 的 降险-2/降险-3 说不算）；
    ② 长篇的分析讲解、通篇没有「我还在」，判**不保持连接**。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    de = D["v9_rulesv3"]["deesc"]
    assert "泛泛的活动建议也算" in de
    assert "陪伴邀请" in de
    ab = D["v9_rulesv3"]["abandon"]
    assert "分析讲解" in ab


def test_v9_keeps_the_deesc_zero_boundary():
    """降险放宽后仍要守住边界 —— 医生判「无降险」的 61/240 条全是应声 / 沉默 /
    反问 / 纯安抚。放到「什么都算」就等于这一维不再有区分度。"""
    de = __import__("tools.evaluator_prompt_variants", fromlist=["x"]).DIM_VARIANTS[
        "v9_rulesv3"]["deesc"]
    for must in ("只是应声", "纯情绪安抚", "反问而不给方向"):
        assert must in de, must


def test_v9_does_not_treat_length_as_the_connect_signal():
    """⚠️ 医师确认的是「长篇**分析讲解**且无『我还在』→ 不在场」，**不是**「长就不算在场」。
    实测 200–400 字不含标记的仍有 92% 被判在场。把它写成长度规则会立刻反向过拟合。"""
    ab = __import__("tools.evaluator_prompt_variants", fromlist=["x"]).DIM_VARIANTS[
        "v9_rulesv3"]["abandon"]
    assert "不论回复多长多短" in ab
    assert "危机行动指令" in ab          # 长的危机指令仍然算在场


def test_v9_only_changes_deesc_and_abandon():
    """共情段与 v7/v8 逐字相同 —— 医师复核没有涉及共情维，不该顺手改。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    assert D["v9_rulesv3"]["empathy"] == D["v8_perdim_stage"]["empathy"]


def test_v10_is_the_mechanical_combination_that_dev_selected():
    """dev 实测 v9 降险改对了（0.809→0.877）、连接改砸了（0.677→0.401）。
    v10 必须是**逐段的机械组合**：降险取 v9、连接取 v8、共情三者相同。
    任何「顺手再优化一下」都会让这次归因失效。"""
    from tools.evaluator_prompt_variants import DIM_VARIANTS as D
    assert D["v10_deesc_v3"]["deesc"] == D["v9_rulesv3"]["deesc"]
    assert D["v10_deesc_v3"]["abandon"] == D["v8_perdim_stage"]["abandon"]
    assert D["v10_deesc_v3"]["empathy"] == D["v8_perdim_stage"]["empathy"]


def test_exclude_units_drops_listed_ids(tmp_path, capsys):
    """报「干净子集」用的开关：test 里有 18 条本身参与过 2026-08-21 的规则修订，
    拿它们再验一遍就是自我确认。剔掉后剩下的才是从未被看过的估计。"""
    f = tmp_path / "drop.txt"
    f.write_text("# 注释行应被忽略\nu0003\nu0008\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):          # 没有 --final，test 划分仍然锁着
        tune.main(["--split", "test", "--exclude-units", str(f)])


def test_contaminated_list_is_frozen_on_disk():
    """那 18 条要落盘、可复算，不能只存在于某次对话里。"""
    from pathlib import Path
    p = Path("tune_eval38/contaminated_test_units.txt")
    if not p.exists():                        # 生成物不入库，缺了就跳过
        pytest.skip("tune_eval38/ 是生成物，本机可能没有")
    ids = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
           if l.strip() and not l.startswith("#")]
    assert len(ids) == 18 and all(i.startswith("u") for i in ids)


def test_analyze_only_refuses_when_there_are_no_judgments(tmp_path):
    """空数据也能「出报告」是危险的：会打出一张全 n/a 的表和一个荒谬的「赢家」。
    （真发生过：`--analyze-only` 配了一个新的 --out 目录，读不到判定。）"""
    with pytest.raises(SystemExit) as e:
        tune.main(["--analyze-only", "--variants", "v0_current",
                   "--split", "dev", "--out", str(tmp_path / "empty")])
    assert "没有这些变体在本划分上的判定" in str(e.value)
