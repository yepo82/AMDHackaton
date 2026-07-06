"""Entry point: reads tasks, processes them, writes results.

Input shape (list of tasks):
    [{"task_id": "t1", "prompt": "..."}, ...]

Output shape (list of results, always valid JSON, same order as input):
    [{"task_id": "t1", "answer": "..."}, ...]
"""
import json
import os
import sys
from typing import Optional

from config import Config

UNABLE_TO_ANSWER = "Unable to answer the task."


class MockAgent:
    """Placeholder standing in for the real Fireworks-backed agent.

    Exposes the same `generate(prompt) -> str` interface as
    `fireworks_client.FireworksClient`, so swapping it in later is a
    one-line change.
    """

    def generate(self, prompt: str) -> str:
        return f"Mock answer for: {prompt}"


def load_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of tasks in {path}, got {type(data).__name__}")
    return data


def save_results(path: str, results: list) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_task(task: dict, agent) -> dict:
    task_id = task.get("task_id")
    try:
        answer = agent.generate(task["prompt"])
    except Exception:
        answer = UNABLE_TO_ANSWER
    return {"task_id": task_id, "answer": answer}


def run(input_path: Optional[str] = None, output_path: Optional[str] = None, agent=None) -> list:
    config = Config()
    input_path = input_path or config.input_path
    output_path = output_path or config.output_path
    agent = agent or MockAgent()

    tasks = load_tasks(input_path)
    results = [process_task(task, agent) for task in tasks]

    save_results(output_path, results)
    return results


def main() -> int:
    try:
        run()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
