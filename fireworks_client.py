"""Thin wrapper around the Fireworks AI OpenAI-compatible endpoint."""
from typing import Optional

import openai

from config import Config


class FireworksClient:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._client: Optional[openai.OpenAI] = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.config.fireworks_api_key,
                base_url=self.config.fireworks_base_url,
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.config.fireworks_model:
            raise ValueError("FIREWORKS_MODEL environment variable is not set")

        response = self.client.chat.completions.create(
            model=self.config.fireworks_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
