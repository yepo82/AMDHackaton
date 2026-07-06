from unittest.mock import MagicMock

import pytest

from config import AppConfig
from fireworks_client import SYSTEM_PROMPT, FireworksClient, choose_model


def _make_config(models):
    return AppConfig(api_key="secret-key", base_url="https://example.com/v1", allowed_models=models)


def _mock_openai(monkeypatch, response_content="mock answer"):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=response_content))]

    fake_create = MagicMock(return_value=fake_response)
    fake_client_instance = MagicMock()
    fake_client_instance.chat.completions.create = fake_create

    fake_openai_cls = MagicMock(return_value=fake_client_instance)
    monkeypatch.setattr("fireworks_client.OpenAI", fake_openai_cls)
    return fake_openai_cls, fake_create


def test_choose_model_prefers_keyword_match():
    models = ["big-70b-model", "llama-3-8b-instruct", "another-huge-model"]
    assert choose_model(models) == "llama-3-8b-instruct"


def test_choose_model_prefers_first_matching_model_in_order():
    models = ["qwen-2-72b", "generic-model"]
    assert choose_model(models) == "qwen-2-72b"


def test_choose_model_falls_back_to_first_when_no_keyword_matches():
    models = ["custom-model-alpha", "custom-model-beta"]
    assert choose_model(models) == "custom-model-alpha"


def test_client_is_built_with_config_api_key_and_base_url(monkeypatch):
    fake_openai_cls, _ = _mock_openai(monkeypatch)
    config = _make_config(["llama-3-8b-instruct"])

    FireworksClient(config)

    fake_openai_cls.assert_called_once_with(api_key="secret-key", base_url="https://example.com/v1")


def test_complete_sends_expected_request(monkeypatch):
    _, fake_create = _mock_openai(monkeypatch, response_content="42")
    config = _make_config(["big-model", "llama-3-8b-instruct"])
    client = FireworksClient(config)

    result = client.complete("What is 6*7?", max_tokens=100)

    assert result == "42"
    fake_create.assert_called_once_with(
        model="llama-3-8b-instruct",
        temperature=0,
        max_tokens=100,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is 6*7?"},
        ],
    )


def test_complete_uses_default_max_tokens(monkeypatch):
    _, fake_create = _mock_openai(monkeypatch)
    config = _make_config(["mini-model"])
    client = FireworksClient(config)

    client.complete("hello")

    assert fake_create.call_args.kwargs["max_tokens"] == 512


def test_complete_raises_clear_error_without_leaking_api_key(monkeypatch):
    fake_client_instance = MagicMock()
    fake_client_instance.chat.completions.create.side_effect = RuntimeError("upstream failure")
    monkeypatch.setattr("fireworks_client.OpenAI", MagicMock(return_value=fake_client_instance))

    config = _make_config(["mini-model"])
    client = FireworksClient(config)

    with pytest.raises(RuntimeError) as exc_info:
        client.complete("hello")

    assert "secret-key" not in str(exc_info.value)
