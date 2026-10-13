"""Aggregate clip performance into simple channel-level insights."""

from __future__ import annotations

from statistics import mean

from models.performance import ClipPerformance


class PerformanceAggregator:
    """Track published clips and summarize results by channel."""

    def __init__(self):
        self.records: list[ClipPerformance] = []

    def add(self, performance: ClipPerformance) -> None:
        """Store a clip result for later analytics."""
        self.records.append(performance)

    def summary_by_channel(self, channel: str) -> dict[str, float | int | str]:
        """Return aggregate metrics for one channel."""
        matching = [item for item in self.records if item.channel == channel]
        if not matching:
            return {
                "channel": channel,
                "clip_count": 0,
                "total_views": 0,
                "avg_engagement_rate": 0.0,
            }

        total_views = sum(item.views for item in matching)
        engagement_rates = [item.engagement_rate for item in matching]
        return {
            "channel": channel,
            "clip_count": len(matching),
            "total_views": total_views,
            "avg_engagement_rate": round(mean(engagement_rates), 2),
        }

    def top_channels(self, limit: int = 5) -> list[dict[str, float | int | str]]:
        """Return top channels ranked by total views."""
        grouped: dict[str, list[ClipPerformance]] = {}
        for record in self.records:
            grouped.setdefault(record.channel, []).append(record)

        channels = []
        for channel, items in grouped.items():
            total_views = sum(item.views for item in items)
            channels.append({
                "channel": channel,
                "clip_count": len(items),
                "total_views": total_views,
            })

        channels.sort(key=lambda item: item["total_views"], reverse=True)
        return channels[:limit]
