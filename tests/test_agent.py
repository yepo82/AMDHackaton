import pytest

import agent as agent_module
from agent import GeneralPurposeAgent


class FakeClient:
    def __init__(self, response="fake llm answer"):
        self.response = response
        self.calls = []

    def complete(self, prompt, *, max_tokens=512):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self.response


def test_math_prompt_solved_locally_without_loading_config(monkeypatch):
    def _boom():
        raise AssertionError("load_config should not be called for a locally solvable math task")

    monkeypatch.setattr(agent_module, "load_config", _boom)

    result = GeneralPurposeAgent().solve("What is 25% of 80?")

    assert result == "20"


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("What is 25% of 80?", "20"),
        ("What is 15 times 6?", "90"),
        ("What is 120 divided by 5?", "24"),
        ("Calculate 24 + 18", "42"),
        ("What is the average of 10 and 20?", "15"),
        ("What is the square root of 16?", "4"),
    ],
)
def test_simple_math_prompts_never_call_the_llm(monkeypatch, prompt, expected):
    def _boom(*args, **kwargs):
        raise AssertionError(f"LLM should never be called for a locally solvable prompt: {prompt!r}")

    monkeypatch.setattr(agent_module, "load_config", _boom)
    monkeypatch.setattr(agent_module, "FireworksClient", _boom)

    assert GeneralPurposeAgent().solve(prompt) == expected


def test_math_prompt_falls_back_to_llm_when_not_locally_solvable(monkeypatch):
    fake_client = FakeClient("42")
    monkeypatch.setattr(agent_module, "load_config", lambda: object())
    monkeypatch.setattr(agent_module, "FireworksClient", lambda config: fake_client)

    result = GeneralPurposeAgent().solve("Calculate the compound interest on $1000 at 5% for 3 years.")

    assert result == "42"
    assert fake_client.calls[0]["prompt"].startswith(
        "Solve the math problem. Return the final answer concisely."
    )


def test_factual_prompt_uses_llm_and_reuses_client_across_calls(monkeypatch):
    fake_client = FakeClient("Paris")
    calls = {"load_config": 0, "client_init": 0}

    def fake_load_config():
        calls["load_config"] += 1
        return object()

    def fake_client_factory(config):
        calls["client_init"] += 1
        return fake_client

    monkeypatch.setattr(agent_module, "load_config", fake_load_config)
    monkeypatch.setattr(agent_module, "FireworksClient", fake_client_factory)

    agent = GeneralPurposeAgent()
    first = agent.solve("What is the capital of France?")
    second = agent.solve("What is the tallest mountain?")

    assert first == "Paris"
    assert second == "Paris"
    assert calls["load_config"] == 1
    assert calls["client_init"] == 1
    assert len(fake_client.calls) == 2


def test_sentiment_prompt_dispatches_to_sentiment_handler(monkeypatch):
    fake_client = FakeClient("positive")
    monkeypatch.setattr(agent_module, "load_config", lambda: object())
    monkeypatch.setattr(agent_module, "FireworksClient", lambda config: fake_client)

    GeneralPurposeAgent().solve("Classify the sentiment of this review as positive or negative.")

    assert fake_client.calls[0]["prompt"].startswith(
        "Classify sentiment (positive, negative, or neutral) with one brief reason."
    )


def test_code_generation_prompt_dispatches_to_code_generation_handler(monkeypatch):
    fake_client = FakeClient("def reverse(s): return s[::-1]")
    monkeypatch.setattr(agent_module, "load_config", lambda: object())
    monkeypatch.setattr(agent_module, "FireworksClient", lambda config: fake_client)

    GeneralPurposeAgent().solve("Write a function that reverses a string.")

    assert fake_client.calls[0]["prompt"].startswith("Write correct, concise code for the spec.")


def test_llm_errors_are_not_swallowed_by_agent(monkeypatch):
    def fake_load_config():
        raise ValueError("FIREWORKS_API_KEY environment variable is not set")

    monkeypatch.setattr(agent_module, "load_config", fake_load_config)

    with pytest.raises(ValueError):
        GeneralPurposeAgent().solve("What is the capital of France?")
