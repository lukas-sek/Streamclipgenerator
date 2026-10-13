import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.analytics import PerformanceAggregator
from models.performance import ClipPerformance


def test_aggregator_summarizes_channel_performance():
    aggregator = PerformanceAggregator()
    aggregator.add(
        ClipPerformance(
            channel="demo",
            output_path="output/final/demo_1.mp4",
            views=1000,
            likes=120,
            comments=30,
            watch_time_s=200.0,
        )
    )
    aggregator.add(
        ClipPerformance(
            channel="demo",
            output_path="output/final/demo_2.mp4",
            views=400,
            likes=24,
            comments=8,
            watch_time_s=150.0,
        )
    )

    summary = aggregator.summary_by_channel("demo")

    assert summary["channel"] == "demo"
    assert summary["clip_count"] == 2
    assert summary["total_views"] == 1400
    assert summary["avg_engagement_rate"] > 10.0


def test_aggregator_handles_empty_history():
    aggregator = PerformanceAggregator()

    assert aggregator.summary_by_channel("missing") == {
        "channel": "missing",
        "clip_count": 0,
        "total_views": 0,
        "avg_engagement_rate": 0.0,
    }
