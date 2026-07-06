import json

import pytest

from main import UNABLE_TO_ANSWER, MockAgent, load_tasks, main, process_task, run, save_results


def test_load_tasks_valid(tmp_path):
    path = tmp_path / "tasks.json"
    tasks = [{"task_id": "t1", "prompt": "hello"}]
    path.write_text(json.dumps(tasks), encoding="utf-8")

    assert load_tasks(str(path)) == tasks


def test_load_tasks_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_tasks("/nonexistent/path/tasks.json")


def test_load_tasks_malformed_json_raises(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_tasks(str(path))


def test_load_tasks_rejects_non_list(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"task_id": "t1"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_tasks(str(path))


def test_save_results_creates_output_dir(tmp_path):
    path = tmp_path / "output" / "results.json"
    results = [{"task_id": "t1", "answer": "hi"}]

    save_results(str(path), results)

    assert json.loads(path.read_text(encoding="utf-8")) == results


def test_process_task_success_uses_agent():
    result = process_task({"task_id": "t1", "prompt": "hello"}, MockAgent())

    assert result == {"task_id": "t1", "answer": "Mock answer for: hello"}


def test_process_task_missing_prompt_returns_fallback():
    result = process_task({"task_id": "t1"}, MockAgent())

    assert result == {"task_id": "t1", "answer": UNABLE_TO_ANSWER}


def test_process_task_agent_error_returns_fallback():
    class FailingAgent:
        def generate(self, prompt):
            raise RuntimeError("boom")

    result = process_task({"task_id": "t1", "prompt": "hello"}, FailingAgent())

    assert result == {"task_id": "t1", "answer": UNABLE_TO_ANSWER}


def test_run_end_to_end_writes_valid_output(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    tasks = [
        {"task_id": "t1", "prompt": "hello"},
        {"task_id": "t2"},  # missing prompt -> should not break the run
    ]
    input_path.write_text(json.dumps(tasks), encoding="utf-8")

    results = run(input_path=str(input_path), output_path=str(output_path))

    assert results == [
        {"task_id": "t1", "answer": "Mock answer for: hello"},
        {"task_id": "t2", "answer": UNABLE_TO_ANSWER},
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == results


def test_run_respects_env_var_overrides(tmp_path, monkeypatch):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(json.dumps([{"task_id": "t1", "prompt": "hi"}]), encoding="utf-8")

    monkeypatch.setenv("INPUT_PATH", str(input_path))
    monkeypatch.setenv("OUTPUT_PATH", str(output_path))

    results = run()

    assert results == [{"task_id": "t1", "answer": "Mock answer for: hi"}]
    assert json.loads(output_path.read_text(encoding="utf-8")) == results


def test_main_returns_zero_on_success(tmp_path, monkeypatch):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(json.dumps([{"task_id": "t1", "prompt": "hi"}]), encoding="utf-8")

    monkeypatch.setenv("INPUT_PATH", str(input_path))
    monkeypatch.setenv("OUTPUT_PATH", str(output_path))

    assert main() == 0
    assert output_path.exists()


def test_main_returns_nonzero_on_missing_input(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_PATH", str(tmp_path / "does_not_exist.json"))
    monkeypatch.setenv("OUTPUT_PATH", str(tmp_path / "results.json"))

    assert main() != 0


def test_main_returns_nonzero_on_malformed_input(tmp_path, monkeypatch):
    input_path = tmp_path / "tasks.json"
    input_path.write_text("not json", encoding="utf-8")

    monkeypatch.setenv("INPUT_PATH", str(input_path))
    monkeypatch.setenv("OUTPUT_PATH", str(tmp_path / "results.json"))

    assert main() != 0
