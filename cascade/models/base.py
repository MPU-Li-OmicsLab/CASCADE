# cascade/models/base.py
from __future__ import annotations
from typing import Protocol, runtime_checkable


Message = dict  # {"role": str, "content": str}


@runtime_checkable
class ChatModel(Protocol):
    model_id: str

    def generate(self, messages: list[Message]) -> str:
        """Return the assistant completion for `messages`."""
        ...
