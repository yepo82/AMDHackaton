"""Entry point: reads tasks, processes them through the agent, writes results."""
import json
import os
from typing import Optional

from agent import Agent
from config import Config


def load_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("tasks", [])
    return data


def save_results(path: str, results: list) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run(input_path: Optional[str] = None, output_path: Optional[str] = None) -> list:
    config = Config()
    input_path = input_path or config.input_path
    output_path = output_path or config.output_path

    tasks = load_tasks(input_path)
    agent = Agent(config)
    results = [agent.process_task(task) for task in tasks]

    save_results(output_path, results)
    return results


if __name__ == "__main__":
    run()
