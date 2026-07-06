"""Thin wrapper around the Fireworks AI OpenAI-compatible endpoint."""
from openai import OpenAI

from config import AppConfig

SYSTEM_PROMPT = "You are a precise AI agent. Answer directly."

_PREFERRED_KEYWORDS = ("small", "mini", "8b", "7b", "qwen", "llama")


def choose_model(models: list[str]) -> str:
    for model in models:
        lowered = model.lower()
        if any(keyword in lowered for keyword in _PREFERRED_KEYWORDS):
            return model
    return models[0]


class FireworksClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        model = choose_model(self.config.allowed_models)
        try:
            response = self._client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"Fireworks completion failed for model '{model}': {exc}") from exc
        return response.choices[0].message.content
