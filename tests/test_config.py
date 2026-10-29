# tests/test_config.py
import pytest
from cascade.config import load_model_registry, build_model, ModelRegistry
from cascade.models.openai_client import OpenAIChatModel


def _registry():
    return ModelRegistry(version=1, models=[
        {"model_id": "m1", "backend": "openai", "base_url": "http://x/v1",
         "api_key_env": "CASCADE_TEST_KEY", "params": {"temperature": 0.7}}])


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError):
        _registry().get("nope")


def test_registry_rejects_duplicate_ids():
    with pytest.raises(Exception):
        ModelRegistry(version=1, models=[
            {"model_id": "d", "backend": "openai", "base_url": "u", "api_key_env": "E"},
            {"model_id": "d", "backend": "openai", "base_url": "u", "api_key_env": "E"}])


def test_build_model_reads_api_key(monkeypatch):
    monkeypatch.setenv("CASCADE_TEST_KEY", "secret")
    m = build_model("m1", _registry())
    assert isinstance(m, OpenAIChatModel)
    assert m.api_key == "secret" and m.params["temperature"] == 0.7


def test_build_model_missing_env_raises(monkeypatch):
    monkeypatch.delenv("CASCADE_TEST_KEY", raising=False)
    with pytest.raises(ValueError):
        build_model("m1", _registry())


def test_shipped_registry_loads():
    reg = load_model_registry()
    assert reg.version >= 1 and len(reg.models) >= 1
    assert all(m.backend == "openai" for m in reg.models)
