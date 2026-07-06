"""Logic and reasoning handler."""
from .llm_handler import ask_llm

INSTRUCTION = "Solve carefully. Give the final answer with a brief explanation."
MAX_TOKENS = 512


def handle_logic(prompt: str, client) -> str:
    return ask_llm(client, INSTRUCTION, prompt, max_tokens=MAX_TOKENS)


def handle(task: dict, client=None) -> dict:
    if client is None:
        return {"status": "error", "result": None, "error": "No Fireworks client configured"}

    try:
        result = handle_logic(task.get("input", ""), client)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}
