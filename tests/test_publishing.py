import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.publisher import PublishController
from models.performance import ClipPerformance


def test_engagement_rate_is_calculated_from_total_views():
    metrics = ClipPerformance(
        channel="example",
        output_path="output/final/example.mp4",
        views=100,
        likes=12,
        comments=3,
        watch_time_s=120.0,
    )

    assert metrics.engagement_rate == 15.0


def test_publish_controller_respects_rate_limit():
    controller = PublishController(rate_limit_seconds=300, now_factory=lambda: datetime(2024, 1, 1, 12, 0, 0))
    controller.last_published_at = datetime(2024, 1, 1, 11, 55, 0)

    assert controller.can_publish() is False
    assert controller.can_publish(now=datetime(2024, 1, 1, 12, 5, 0)) is True
