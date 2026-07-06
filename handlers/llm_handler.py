"""Generalist handler: delegates the task input to the Fireworks client."""


def handle(task: dict, client=None) -> dict:
    if client is None:
        return {"status": "error", "result": None, "error": "No Fireworks client configured"}

    prompt = task.get("input", "")
    try:
        result = client.complete(prompt)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}
