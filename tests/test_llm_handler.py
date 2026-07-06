from handlers.code_handler import handle as code_handle
from handlers.code_handler import handle_code_debugging, handle_code_generation
from handlers.llm_handler import ask_llm, handle_factual
from handlers.logic_handler import handle_logic
from handlers.ner_handler import handle_ner
from handlers.sentiment_handler import handle_sentiment
from handlers.summary_handler import handle as summary_handle
from handlers.summary_handler import handle_summary


class FakeClient:
    def __init__(self, response="mock response"):
        self.response = response
        self.calls = []

    def complete(self, prompt, *, max_tokens=512):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self.response


def test_ask_llm_builds_compact_prompt_and_forwards_max_tokens():
    client = FakeClient("answer")

    result = ask_llm(client, "Do the thing.", "some task text", max_tokens=99)

    assert result == "answer"
    assert client.calls[0]["prompt"] == "Do the thing.\n\nTask:\nsome task text"
    assert client.calls[0]["max_tokens"] == 99


def test_handle_factual_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_factual("What is the capital of France?", client)

    call = client.calls[0]
    assert call["prompt"] == "Answer accurately and concisely.\n\nTask:\nWhat is the capital of France?"
    assert call["max_tokens"] == 256


def test_handle_summary_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_summary("Long article text...", client)

    call = client.calls[0]
    assert call["prompt"].startswith("Summarize according to the user instruction. Be concise.")
    assert call["max_tokens"] == 256


def test_handle_sentiment_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_sentiment("I love this product.", client)

    call = client.calls[0]
    assert call["prompt"].startswith(
        "Classify sentiment as positive, negative, or neutral. Give one brief justification."
    )
    assert call["max_tokens"] == 128


def test_handle_ner_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_ner("Barack Obama visited Paris in 2015.", client)

    call = client.calls[0]
    assert call["prompt"].startswith(
        "Extract named entities as valid JSON. Use labels PERSON, ORG, LOCATION, DATE. Return JSON only."
    )
    assert call["max_tokens"] == 256


def test_handle_code_debugging_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_code_debugging("def add(a, b): return a - b", client)

    call = client.calls[0]
    assert call["prompt"].startswith("Identify the bug briefly and provide corrected code.")
    assert call["max_tokens"] == 768


def test_handle_code_generation_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_code_generation("Write a function that reverses a string.", client)

    call = client.calls[0]
    assert call["prompt"].startswith(
        "Write correct concise code that satisfies the specification. "
        "Return only code unless explanation is requested."
    )
    assert call["max_tokens"] == 768


def test_handle_logic_uses_expected_instruction_and_max_tokens():
    client = FakeClient()

    handle_logic("Solve this logic puzzle...", client)

    call = client.calls[0]
    assert call["prompt"].startswith(
        "Solve the reasoning task carefully. Return the final answer with a brief explanation."
    )
    assert call["max_tokens"] == 512


def test_code_handle_dispatches_to_debugging_when_bug_keywords_present():
    client = FakeClient("fixed code")

    result = code_handle({"input": "There is a bug in this function, please debug it."}, client=client)

    assert result["status"] == "ok"
    assert client.calls[0]["prompt"].startswith("Identify the bug briefly and provide corrected code.")


def test_code_handle_dispatches_to_generation_by_default():
    client = FakeClient("new code")

    result = code_handle({"input": "Write a function that reverses a string."}, client=client)

    assert result["status"] == "ok"
    assert client.calls[0]["prompt"].startswith("Write correct concise code")


def test_handle_without_client_returns_error():
    result = summary_handle({"input": "text"})

    assert result["status"] == "error"
