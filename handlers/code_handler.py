"""Code handler: debugging and generation prompts."""
from .llm_handler import ask_llm

DEBUGGING_INSTRUCTION = "Identify the bug briefly and provide corrected code."
GENERATION_INSTRUCTION = (
    "Write correct concise code that satisfies the specification. "
    "Return only code unless explanation is requested."
)
MAX_TOKENS = 768

_DEBUGGING_KEYWORDS = ("debug", "bug", "fix this code", "error", "traceback", "exception", "broken code")


def handle_code_debugging(prompt: str, client) -> str:
    return ask_llm(client, DEBUGGING_INSTRUCTION, prompt, max_tokens=MAX_TOKENS)


def handle_code_generation(prompt: str, client) -> str:
    return ask_llm(client, GENERATION_INSTRUCTION, prompt, max_tokens=MAX_TOKENS)


def handle(task: dict, client=None) -> dict:
    if client is None:
        return {"status": "error", "result": None, "error": "No Fireworks client configured"}

    prompt = task.get("input", "")
    is_debugging = any(keyword in prompt.lower() for keyword in _DEBUGGING_KEYWORDS)
    try:
        result = handle_code_debugging(prompt, client) if is_debugging else handle_code_generation(prompt, client)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}
