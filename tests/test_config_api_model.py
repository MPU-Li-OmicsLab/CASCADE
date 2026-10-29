import os

from cascade.config import ModelRegistry, ModelSpec, build_model, load_model_registry


def test_api_model_alias_defaults_to_model_id(monkeypatch):
    reg = ModelRegistry(version=1, models=[
        ModelSpec(model_id="plain", backend="openai", base_url="http://x", api_key_env="K"),
        ModelSpec(model_id="alias", api_model="vendor/real", backend="openai", base_url="http://x",
                  api_key_env="K"),
    ])
    monkeypatch.setenv("K", "x")
    assert build_model("plain", reg).model_id == "plain"
    assert build_model("alias", reg).model_id == "vendor/real"


def test_registry_has_cross_family_judge():
    reg = load_model_registry()
    spec = reg.get("judge-deepseek")
    assert spec.api_model == "deepseek/deepseek-v4-pro"
    assert spec.params["temperature"] == 0.0
