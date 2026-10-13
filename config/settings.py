"""Runtime configuration for the stream-to-shorts pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Stable defaults for the local stream automation setup."""

    twitch_channel: str = "demo"
    output_dir: str = "output"
    virality_threshold: float = 5.0
    rate_limit_seconds: int = 300
    ffmpeg_path: str = "ffmpeg"
    ollama_base_url: str = "http://localhost:11434/v1"


def load_runtime_config() -> RuntimeConfig:
    """Build a config from environment variables with sensible defaults."""
    return RuntimeConfig(
        twitch_channel=os.getenv("TWITCH_CHANNEL", "demo"),
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        virality_threshold=float(os.getenv("VIRALITY_THRESHOLD", "5.0")),
        rate_limit_seconds=int(os.getenv("RATE_LIMIT_SECONDS", "300")),
        ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )
