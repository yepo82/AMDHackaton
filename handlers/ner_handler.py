"""Named entity recognition handler."""
from .llm_handler import ask_llm

INSTRUCTION = "Extract named entities as valid JSON. Use labels PERSON, ORG, LOCATION, DATE. Return JSON only."
MAX_TOKENS = 256


def handle_ner(prompt: str, client) -> str:
    return ask_llm(client, INSTRUCTION, prompt, max_tokens=MAX_TOKENS)


def handle(task: dict, client=None) -> dict:
    if client is None:
        return {"status": "error", "result": None, "error": "No Fireworks client configured"}

    try:
        result = handle_ner(task.get("input", ""), client)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}
