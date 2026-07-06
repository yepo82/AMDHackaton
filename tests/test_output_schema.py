"""End-to-end checks against the real pipeline (main.run -> GeneralPurposeAgent):
every result must carry exactly task_id/answer, and failures must report the
exact fallback message -- all without any real Fireworks network calls."""
import json

import agent as agent_module
from main import UNABLE_TO_ANSWER, run


class _FakeClient:
    def complete(self, prompt, *, max_tokens=512):
        return "fake llm answer"


def test_results_always_have_task_id_and_answer_keys_without_fireworks_creds(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)

    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    tasks = [
        {"task_id": "t1", "prompt": "What is 15 times 6?"},  # solved locally
        {"task_id": "t2", "prompt": "What is the capital of France?"},  # needs LLM -> no creds -> fails
        {"task_id": "t3"},  # missing prompt -> fails
    ]
    input_path.write_text(json.dumps(tasks), encoding="utf-8")

    results = run(input_path=str(input_path), output_path=str(output_path))

    assert len(results) == len(tasks)
    for result in results:
        assert set(result.keys()) == {"task_id", "answer"}
        assert isinstance(result["answer"], str)

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == results


def test_failed_task_reports_exact_fallback_message(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)

    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(
        json.dumps([{"task_id": "t1", "prompt": "What is the capital of France?"}]),
        encoding="utf-8",
    )

    results = run(input_path=str(input_path), output_path=str(output_path))

    assert results == [{"task_id": "t1", "answer": UNABLE_TO_ANSWER}]


def test_results_have_correct_shape_on_successful_llm_run(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "load_config", lambda: object())
    monkeypatch.setattr(agent_module, "FireworksClient", lambda config: _FakeClient())

    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    tasks = [
        {"task_id": "t1", "prompt": "What is 15 times 6?"},
        {"task_id": "t2", "prompt": "What is the capital of France?"},
        {"task_id": "t3", "prompt": "Classify the sentiment of this review: great product!"},
    ]
    input_path.write_text(json.dumps(tasks), encoding="utf-8")

    results = run(input_path=str(input_path), output_path=str(output_path))

    assert results == [
        {"task_id": "t1", "answer": "90"},
        {"task_id": "t2", "answer": "fake llm answer"},
        {"task_id": "t3", "answer": "fake llm answer"},
    ]
    for result in results:
        assert set(result.keys()) == {"task_id", "answer"}
