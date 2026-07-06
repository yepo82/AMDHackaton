"""Thin wrapper around the Fireworks AI OpenAI-compatible endpoint."""
from typing import Optional

import openai

from config import AppConfig, load_config


class FireworksClient:
    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config
        self._client: Optional[openai.OpenAI] = None

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            self._config = load_config()
        return self._config

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        config = self.config
        model = model or config.allowed_models[0]
        if model not in config.allowed_models:
            raise ValueError(f"Model '{model}' is not in ALLOWED_MODELS: {config.allowed_models}")

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
