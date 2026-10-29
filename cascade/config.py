# cascade/config.py
"""Model registry (data/models.yaml) + factory. Registry holds endpoints and
sampling params only — NEVER secrets; api_key_env names the environment variable."""
from __future__ import annotations
import os
from pathlib import Path
from pydantic import Field, model_validator
from cascade.content import DATA_DIR, ContentModel, _load_yaml
from cascade.models.base import ChatModel
from cascade.models.openai_client import OpenAIChatModel


class ModelSpec(ContentModel):
    model_id: str
    backend: str               # v1: only "openai"
    base_url: str
    api_key_env: str
    params: dict = Field(default_factory=dict)
    # ⚡2026-09-17：同一 API 模型以不同采样参数登记两次（被测 deepseek-v4-pro 温度 0.7 /
    # 判官 judge-deepseek 零温）时，registry 键与 API 模型名要分开。None = 与 model_id 相同。
    api_model: str | None = None


class ModelRegistry(ContentModel):
    version: int
    models: list[ModelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ModelRegistry":
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model_id in registry")
        return self

    def get(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.model_id == model_id:
                return m
        raise KeyError(f"unknown model_id: {model_id}")


def load_model_registry(path: Path = DATA_DIR / "models.yaml") -> ModelRegistry:
    return ModelRegistry(**_load_yaml(path))


def build_model(model_id: str, registry: ModelRegistry | None = None) -> ChatModel:
    registry = registry or load_model_registry()
    spec = registry.get(model_id)
    if spec.backend != "openai":
        raise ValueError(f"unsupported backend {spec.backend!r} (v1 supports 'openai')")
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        raise ValueError(
            f"env var {spec.api_key_env!r} for model {model_id!r} is unset/empty")
    return OpenAIChatModel(model_id=spec.api_model or spec.model_id, base_url=spec.base_url,
                           api_key=api_key, params=spec.params)
