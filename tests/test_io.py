import json

from main import load_tasks, save_results


def test_load_tasks_from_list(tmp_path):
    path = tmp_path / "tasks.json"
    tasks = [{"id": "1", "type": "math", "input": "1+1"}]
    path.write_text(json.dumps(tasks), encoding="utf-8")

    assert load_tasks(str(path)) == tasks


def test_load_tasks_from_dict(tmp_path):
    path = tmp_path / "tasks.json"
    tasks = [{"id": "1", "type": "math", "input": "1+1"}]
    path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")

    assert load_tasks(str(path)) == tasks


def test_save_results_creates_file(tmp_path):
    path = tmp_path / "output" / "results.json"
    results = [{"id": "1", "status": "ok", "result": 2}]

    save_results(str(path), results)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == results
