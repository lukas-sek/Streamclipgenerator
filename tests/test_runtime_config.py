import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RuntimeConfig, load_runtime_config


def test_runtime_config_uses_defaults():
    config = RuntimeConfig()

    assert config.twitch_channel == "demo"
    assert config.rate_limit_seconds == 300
    assert config.virality_threshold == 5.0


def test_runtime_config_reads_environment(monkeypatch):
    monkeypatch.setenv("TWITCH_CHANNEL", "")
    monkeypatch.setenv("RATE_LIMIT_SECONDS", "600")
    monkeypatch.setenv("VIRALITY_THRESHOLD", "7.5")

    config = load_runtime_config()

    assert config.twitch_channel == ""
    assert config.rate_limit_seconds == 600
    assert config.virality_threshold == 7.5
