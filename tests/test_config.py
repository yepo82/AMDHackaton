import pytest

from config import AppConfig, load_config


def _clear_env(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)


def test_load_config_returns_app_config_with_valid_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a,model-b")

    config = load_config()

    assert config == AppConfig(
        api_key="secret-key",
        base_url="https://api.fireworks.ai/inference/v1",
        allowed_models=["model-a", "model-b"],
    )


def test_load_config_strips_whitespace_and_drops_empty_entries(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.com")
    monkeypatch.setenv("ALLOWED_MODELS", " model-a , , model-b ,")

    config = load_config()

    assert config.allowed_models == ["model-a", "model-b"]


def test_load_config_raises_when_api_key_missing(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.com")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a")

    with pytest.raises(ValueError):
        load_config()


def test_load_config_raises_when_base_url_missing(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-key")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a")

    with pytest.raises(ValueError):
        load_config()


def test_load_config_raises_when_allowed_models_missing(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.com")

    with pytest.raises(ValueError):
        load_config()


def test_load_config_raises_when_allowed_models_only_whitespace(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.com")
    monkeypatch.setenv("ALLOWED_MODELS", " , , ")

    with pytest.raises(ValueError):
        load_config()
