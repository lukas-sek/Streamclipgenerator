import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.feedback import FeedbackLoop
from models.performance import ClipPerformance


def test_feedback_loop_adjusts_threshold_up_for_weak_performance():
    loop = FeedbackLoop()
    loop.record(
        ClipPerformance(
            channel="demo",
            output_path="output/final/demo.mp4",
            views=1000,
            likes=15,
            comments=5,
            watch_time_s=200.0,
        )
    )

    assert loop.recommend_threshold() > 4.0


def test_feedback_loop_keeps_threshold_low_for_strong_performance():
    loop = FeedbackLoop()
    loop.record(
        ClipPerformance(
            channel="demo",
            output_path="output/final/demo.mp4",
            views=1000,
            likes=120,
            comments=40,
            watch_time_s=200.0,
        )
    )

    assert loop.recommend_threshold() < 5.0
