"""Generalist agent: routes a task to its handler and normalizes the response."""
from typing import Optional

from fireworks_client import FireworksClient
from router import route


class Agent:
    def __init__(self, client: Optional[FireworksClient] = None):
        self.client = client or FireworksClient()

    def process_task(self, task: dict) -> dict:
        task_id = task.get("id")
        task_type = task.get("type", "")

        try:
            handler = route(task_type)
            outcome = handler(task, client=self.client)
        except ValueError as exc:
            outcome = {"status": "error", "result": None, "error": str(exc)}

        return {"id": task_id, "type": task_type, **outcome}
