import pytest

from fireworks_client import FireworksClient


def _clear_env(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)


def test_construction_does_not_require_env_vars(monkeypatch):
    _clear_env(monkeypatch)

    FireworksClient()  # must not raise even with no Fireworks env vars set


def test_generate_raises_when_config_missing(monkeypatch):
    _clear_env(monkeypatch)
    client = FireworksClient()

    with pytest.raises(ValueError):
        client.generate("hello")


def test_generate_rejects_model_outside_allowed_list(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.com")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a,model-b")

    client = FireworksClient()

    with pytest.raises(ValueError):
        client.generate("hello", model="not-allowed")
