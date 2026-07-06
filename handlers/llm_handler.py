"""Generalist LLM handler: builds compact prompts and calls the Fireworks client."""

FACTUAL_INSTRUCTION = "Answer accurately and concisely."
FACTUAL_MAX_TOKENS = 256


def ask_llm(client, instruction: str, task: str, max_tokens: int = 512) -> str:
    prompt = f"{instruction}\n\nTask:\n{task}"
    return client.complete(prompt, max_tokens=max_tokens)


def handle_factual(prompt: str, client) -> str:
    return ask_llm(client, FACTUAL_INSTRUCTION, prompt, max_tokens=FACTUAL_MAX_TOKENS)


def handle(task: dict, client=None) -> dict:
    if client is None:
        return {"status": "error", "result": None, "error": "No Fireworks client configured"}

    prompt = task.get("input", "")
    try:
        result = handle_factual(prompt, client)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}
