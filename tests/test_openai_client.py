import pytest
from cascade.models.base import ChatModel
from cascade.models.openai_client import OpenAIChatModel


class _Msg:
    def __init__(self, content): self.content = content

class _Choice:
    def __init__(self, content): self.message = _Msg(content)

class _Resp:
    def __init__(self, choices): self.choices = choices

class _Completions:
    def __init__(self, resp): self._resp = resp; self.calls = []
    def create(self, **kwargs): self.calls.append(kwargs); return self._resp

class _Client:
    def __init__(self, resp): self.chat = type("C", (), {"completions": _Completions(resp)})()


def test_satisfies_chatmodel_protocol():
    m = OpenAIChatModel("m", "http://x/v1", "k", client=_Client(_Resp([_Choice("hi")])))
    assert isinstance(m, ChatModel)
    assert m.model_id == "m"


def test_generate_returns_content_and_forwards_params():
    client = _Client(_Resp([_Choice("the reply")]))
    m = OpenAIChatModel("qwen", "http://x/v1", "k",
                        params={"temperature": 0.7, "seed": 3}, client=client)
    out = m.generate([{"role": "user", "content": "hello"}])
    assert out == "the reply"
    call = client.chat.completions.calls[0]
    assert call["model"] == "qwen" and call["temperature"] == 0.7 and call["seed"] == 3
    assert call["messages"][0]["content"] == "hello"


def test_generate_raises_on_empty_choices():
    m = OpenAIChatModel("m", "http://x/v1", "k", client=_Client(_Resp([])))
    with pytest.raises(ValueError, match=r"empty choices"):
        m.generate([{"role": "user", "content": "x"}])


def test_generate_raises_on_null_content():
    m = OpenAIChatModel("m", "http://x/v1", "k", client=_Client(_Resp([_Choice(None)])))
    with pytest.raises(ValueError, match=r"null content"):
        m.generate([{"role": "user", "content": "x"}])


class _MsgR:
    def __init__(self, content, reasoning=None, reasoning_details=None):
        self.content = content
        self.reasoning = reasoning
        self.reasoning_details = reasoning_details

class _ChoiceR:
    def __init__(self, msg): self.message = msg


def test_generate_falls_back_to_reasoning_when_content_null():
    msg = _MsgR(None, reasoning="the actual reply")     # deepseek/OpenRouter reasoning model
    m = OpenAIChatModel("ds", "http://x/v1", "k", client=_Client(_Resp([_ChoiceR(msg)])))
    assert m.generate([{"role": "user", "content": "x"}]) == "the actual reply"


def test_generate_falls_back_to_reasoning_details_when_no_reasoning():
    msg = _MsgR(None, reasoning=None,
                reasoning_details=[{"type": "reasoning.text", "text": "from details"}])
    m = OpenAIChatModel("ds", "http://x/v1", "k", client=_Client(_Resp([_ChoiceR(msg)])))
    assert m.generate([{"role": "user", "content": "x"}]) == "from details"


# —— 思考链剥离 ——————————————————————————————————————————————
# minimax/minimax-m2-her 等角色扮演/推理模型会把 <think> 块直接吐进 content，
# 且不支持 include_reasoning 参数（无法从 API 侧关掉）。不剥离则 6 维判官 /
# 风险判定 / 医生金标全都在评思考链。规则 = 取【最后一个】</think> 之后的内容：
# 真实产品里用户看到的正是这部分，而且能扛住模型吐出畸形嵌套标签的情况。
from cascade.models.openai_client import strip_thinking


def test_strip_thinking_removes_wellformed_block():
    assert strip_thinking("<think>盘算了一下</think>你还好吗？") == "你还好吗？"


def test_strip_thinking_takes_text_after_last_close_tag():
    # 实测 minimax-m2-her 吐出过两个 </think>，中间夹着退化输出；
    # 非贪婪匹配会把中间那坨垃圾留下来，取最后一个闭合标签才对。
    raw = "<think>A</think> 退化的乱码 </think> 真正的回复"
    assert strip_thinking(raw) == "真正的回复"


def test_strip_thinking_leaves_normal_text_untouched():
    assert strip_thinking("你还好吗？") == "你还好吗？"
    assert strip_thinking("  已经很难了  ") == "已经很难了"


def test_strip_thinking_handles_thinking_tag_variant():
    assert strip_thinking("<thinking>x</thinking>正文") == "正文"


def test_strip_thinking_returns_empty_when_only_thinking():
    # 被 max_tokens 截断 → 只有开标签没有闭标签 → 用户什么都看不到
    assert strip_thinking("<think>想到一半就被截断了") == ""
    assert strip_thinking("<think>全是思考</think>   ") == ""


def test_generate_strips_thinking_and_raises_when_nothing_left():
    m = OpenAIChatModel("m", "http://x/v1", "k",
                        client=_Client(_Resp([_Choice("<think>盘算</think>正文")])))
    assert m.generate([{"role": "user", "content": "hi"}]) == "正文"
    m2 = OpenAIChatModel("m", "http://x/v1", "k",
                         client=_Client(_Resp([_Choice("<think>只有思考，被截断了")])))
    with pytest.raises(ValueError, match="only-thinking"):
        m2.generate([{"role": "user", "content": "hi"}])


def test_lazy_client_gets_timeout_and_retries(monkeypatch):
    """长批次要扛代理抖动：SDK 默认 600s 超时 / 2 次重试，实测一次 288 段的批里
    前 72 个任务被开跑时的代理抖动全部报销。"""
    import sys, types
    from cascade.models import openai_client as oc
    seen = {}

    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kw: seen.update(kw) or "CLIENT"
    monkeypatch.setitem(sys.modules, "openai", fake)

    m = OpenAIChatModel("m", "http://x/v1", "k")
    assert m.client == "CLIENT"
    assert seen["timeout"] == oc.CLIENT_TIMEOUT_S == 120.0
    assert seen["max_retries"] == oc.CLIENT_MAX_RETRIES == 6


# ── 外层长退避重试（2026-08-27）─────────────────────────────────────────────
# 背景：SDK 自己的重试退避封顶 8 秒，实测被两次更长的停摆打穿（OpenRouter 网关 502、
# 量产批末尾 4 段 deepseek 的 Connection error），每次都是整段对话作废。

class _FlakyCompletions:
    """前 `fail_n` 次抛 `exc`，之后返回正常响应。"""
    def __init__(self, resp, exc, fail_n):
        self._resp, self._exc, self._left = resp, exc, fail_n
        self.calls = 0
    def create(self, **kw):
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise self._exc
        return self._resp


class _FlakyClient:
    def __init__(self, resp, exc, fail_n):
        self.completions = _FlakyCompletions(resp, exc, fail_n)
        self.chat = type("C", (), {"completions": self.completions})()


class _Status(Exception):
    def __init__(self, status_code): super().__init__(str(status_code)); self.status_code = status_code

class APIConnectionError(Exception):      # 类名要和 openai/httpx 的对上
    pass


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    from cascade.models import openai_client as oc
    monkeypatch.setattr(oc, "_sleep", slept.append)
    return slept


def test_transient_5xx_is_retried_then_succeeds(no_sleep):
    from cascade.models import openai_client as oc
    c = _FlakyClient(_Resp([_Choice("ok")]), _Status(502), fail_n=2)
    m = OpenAIChatModel("m", "http://x/v1", "k", client=c)
    assert m.generate([{"role": "user", "content": "hi"}]) == "ok"
    assert c.completions.calls == 3                       # 2 次失败 + 1 次成功
    assert no_sleep == list(oc.OUTER_BACKOFF_S[:2])       # 退避真的用上了，而且是长的


def test_connection_error_is_retried(no_sleep):
    c = _FlakyClient(_Resp([_Choice("ok")]), APIConnectionError("Connection error."), fail_n=1)
    m = OpenAIChatModel("m", "http://x/v1", "k", client=c)
    assert m.generate([{"role": "user", "content": "hi"}]) == "ok"
    assert c.completions.calls == 2


def test_4xx_is_not_retried(no_sleep):
    """403 geo 封锁 / 401 / 402 余额不足：重试一万次也不会好，而且被测调用真扣费。"""
    c = _FlakyClient(_Resp([_Choice("ok")]), _Status(403), fail_n=99)
    m = OpenAIChatModel("m", "http://x/v1", "k", client=c)
    with pytest.raises(Exception):
        m.generate([{"role": "user", "content": "hi"}])
    assert c.completions.calls == 1
    assert no_sleep == []


def test_gives_up_after_outer_retries(no_sleep):
    from cascade.models import openai_client as oc
    c = _FlakyClient(_Resp([_Choice("ok")]), _Status(503), fail_n=99)
    m = OpenAIChatModel("m", "http://x/v1", "k", client=c)
    with pytest.raises(Exception):
        m.generate([{"role": "user", "content": "hi"}])
    assert c.completions.calls == oc.OUTER_RETRIES + 1
    assert len(no_sleep) == oc.OUTER_RETRIES


def test_semantic_failure_is_not_retried(no_sleep):
    """null content 是模型这次就没给正文 —— 重试会把它洗成随机通过，污染的是数据。"""
    c = _FlakyClient(_Resp([_Choice(None)]), _Status(500), fail_n=0)
    m = OpenAIChatModel("m", "http://x/v1", "k", client=c)
    with pytest.raises(ValueError, match="null content"):
        m.generate([{"role": "user", "content": "hi"}])
    assert c.completions.calls == 1
    assert no_sleep == []
