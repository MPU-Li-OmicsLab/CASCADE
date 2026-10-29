"""ChatModel over any OpenAI-compatible endpoint: local vLLM servers, OpenAI,
OpenRouter, etc. The `openai` SDK is an optional dependency ([experiments] extra),
imported lazily so the core install and FakeChatModel tests never require it.
Inject `client` in tests to avoid both the import and any network call."""
from __future__ import annotations
import re
import time
from cascade.models.base import Message

# 长批次要扛住代理抖动：一次 288 段的批里，开跑头几分钟本地代理冷启动，
# 前 72 个任务全数 `Connection error` 报销（SDK 默认只重试 2 次）；另有 5 个任务
# 卡在 SDK 默认的 600 秒超时上白等 10 分钟。超时压到 120 秒 + 重试提到 6 次，
# 让瞬时抖动自己愈合，而不是把整段对弈丢进 manifest 的 failed 里。
CLIENT_TIMEOUT_S = 120.0
CLIENT_MAX_RETRIES = 6

# ⚡2026-08-27：SDK 内建重试**退避太短**，扛不住持续几十秒的故障 —— 它的退避上限是
# 8 秒，6 次加起来只覆盖约 30 秒。实测被打穿两次：OpenRouter 网关 502（4 个被测同时
# 报销）、以及量产批末尾 4 段 deepseek 的 Connection error。这两次都是**整段对话作废**，
# 而 fixed 臂 12 轮定长、每段更贵，丢一段的代价比 responsive 还高。
#
# 于是在 SDK 之外再套一层**长退避**重试：三次，间隔 15 / 45 / 90 秒，合计多覆盖 2.5 分钟。
# ⚠️ 只对**瞬时**故障重试（连接/超时/429/5xx）。4xx（鉴权、参数错、余额不足、geo 封锁）
#    立刻抛出 —— 那些重试一万次也不会好，而且被测调用是**真扣费**的。
OUTER_RETRIES = 3
OUTER_BACKOFF_S = (15.0, 45.0, 90.0)
_sleep = time.sleep          # 注入点：测试替换掉它，不真的睡

#: 按类名识别的瞬时故障。**刻意不 import openai** —— 它是 [experiments] 可选依赖，
#: 核心安装和 FakeChatModel 测试都不该因为这个模块而被迫装它。
_TRANSIENT_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "RemoteProtocolError",
})


def is_transient(exc: BaseException) -> bool:
    """这个异常值得等一会儿再试吗？

    有 `status_code` 就按 HTTP 语义判（429 限流 / 5xx 服务端）；没有就按类名认
    httpx / openai 的连接类异常。"""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in _TRANSIENT_NAMES

_CLOSE_THINK = re.compile(r"</think(?:ing)?>", re.I)
_OPEN_THINK = re.compile(r"<think(?:ing)?>", re.I)


def strip_thinking(text: str) -> str:
    """Drop a model's inline chain-of-thought, keeping only what a real user would see.

    Roleplay/reasoning targets (observed on `minimax/minimax-m2-her`) emit `<think>` blocks
    straight into `content` and expose no `include_reasoning` switch, so this can only be
    handled here. Left in, the thinking text is what the 6-dim judge, the risk labeler and
    the clinician gold would end up scoring.

    Rule: take everything after the LAST closing tag. A real sample carried two `</think>`
    with degenerate text between them — a non-greedy `<think>.*?</think>` strip keeps that
    garbage, the last-close rule does not. An opening tag with no close means the reply was
    cut off mid-thought and the user would see nothing: returns "".
    """
    matches = list(_CLOSE_THINK.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    if _OPEN_THINK.search(text):
        return ""
    return text.strip()


class OpenAIChatModel:
    def __init__(self, model_id: str, base_url: str, api_key: str,
                 params: dict | None = None, client=None):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.params = dict(params or {})
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import openai  # lazy: optional [experiments] dependency
            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key,
                                         timeout=CLIENT_TIMEOUT_S,
                                         max_retries=CLIENT_MAX_RETRIES)
        return self._client

    def _create(self, messages: list[Message]):
        """在 SDK 自己的秒级重试之外，再套一层分钟级长退避。

        只对瞬时故障退避重试；4xx 立刻抛出（重试不会好，而且被测调用真扣费）。
        本模块自己抛的 ValueError（null content / only-thinking）也不在这里重试 ——
        那是语义失败，重试会把「模型这次没给正文」洗成随机通过，污染数据。"""
        for attempt in range(OUTER_RETRIES + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model_id, messages=messages, **self.params)
            except Exception as exc:
                if attempt == OUTER_RETRIES or not is_transient(exc):
                    raise
                _sleep(OUTER_BACKOFF_S[attempt])

    def generate(self, messages: list[Message]) -> str:
        resp = self._create(messages)
        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError(f"empty choices from {self.model_id}: {resp!r}")
        msg = choices[0].message
        content = msg.content
        if not content:                       # reasoning models (deepseek/OpenRouter) sometimes
            content = getattr(msg, "reasoning", None)    # put the whole reply in `reasoning`, content=None
            if not content:
                details = getattr(msg, "reasoning_details", None) or []
                content = "\n".join(
                    ((d.get("text") if isinstance(d, dict) else getattr(d, "text", None)) or "")
                    for d in details).strip() or None
        if not content:
            raise ValueError(f"null content from {self.model_id}: {resp!r}")
        visible = strip_thinking(content)
        if not visible:                       # 整条回复都是思考链（通常是被 max_tokens 截断）→
            raise ValueError(                 # 用户什么都看不到，当失败处理，别把空串录进数据
                f"only-thinking content from {self.model_id} (raise max_tokens): {content[:200]!r}")
        return visible
