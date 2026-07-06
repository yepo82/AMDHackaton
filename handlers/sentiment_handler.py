"""Sentiment analysis handler."""
from .llm_handler import ask_llm

INSTRUCTION = "Classify sentiment as positive, negative, or neutral. Give one brief justification."
MAX_TOKENS = 128


def handle_sentiment(prompt: str, client) -> str:
    return ask_llm(client, INSTRUCTION, prompt, max_tokens=MAX_TOKENS)


def handle(task: dict, client=None) -> dict:
    if client is None:
        return {"status": "error", "result": None, "error": "No Fireworks client configured"}

    try:
        result = handle_sentiment(task.get("input", ""), client)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}
