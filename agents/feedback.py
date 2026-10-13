"""Collect published-clip feedback and tune trigger thresholds."""

from __future__ import annotations

from statistics import mean

from models.performance import ClipPerformance


class FeedbackLoop:
    """Small feedback loop that adjusts the next trigger threshold from actual performance."""

    def __init__(self, initial_threshold: float = 5.0):
        self.initial_threshold = float(initial_threshold)
        self.samples: list[ClipPerformance] = []

    def record(self, performance: ClipPerformance) -> float:
        """Append a completed clip and return the rolling average engagement rate."""
        self.samples.append(performance)
        return self.average_engagement_rate()

    def average_engagement_rate(self) -> float:
        """Return the mean engagement rate across recorded clips."""
        if not self.samples:
            return 0.0
        return float(mean(item.engagement_rate for item in self.samples))

    def recommend_threshold(self) -> float:
        """Return a new virality threshold based on recent clip quality."""
        avg_engagement = self.average_engagement_rate()

        if avg_engagement >= 15.0:
            return 3.5
        if avg_engagement >= 10.0:
            return 4.5
        if avg_engagement >= 6.0:
            return 5.0
        if avg_engagement >= 3.0:
            return 6.0
        return 7.0
