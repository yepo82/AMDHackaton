"""Environment-driven configuration. No secrets or model names are hardcoded here."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    fireworks_api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))
    fireworks_model: str = field(default_factory=lambda: os.environ.get("FIREWORKS_MODEL", ""))
    fireworks_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    input_path: str = field(default_factory=lambda: os.environ.get("INPUT_PATH", "/input/tasks.json"))
    output_path: str = field(default_factory=lambda: os.environ.get("OUTPUT_PATH", "/output/results.json"))
