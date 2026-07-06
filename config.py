"""Fireworks AI configuration.

Values are read from environment variables only — never hardcoded and never
loaded from a .env file. `load_config()` validates and raises if something
is missing, so it must only be called when an actual LLM call is about to
happen. This keeps tests for math/router/io runnable without real
Fireworks credentials.
"""
import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]


def load_config() -> AppConfig:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models_raw = os.environ.get("ALLOWED_MODELS", "")

    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable is not set")
    if not base_url:
        raise ValueError("FIREWORKS_BASE_URL environment variable is not set")

    allowed_models = [model.strip() for model in allowed_models_raw.split(",") if model.strip()]
    if not allowed_models:
        raise ValueError("ALLOWED_MODELS environment variable must contain at least one model")

    return AppConfig(api_key=api_key, base_url=base_url, allowed_models=allowed_models)
