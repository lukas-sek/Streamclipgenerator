"""Metrics recorded after publishing a generated clip."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ClipPerformance(BaseModel):
    """Per-clip statistics used to tune publishing and trigger thresholds."""

    channel: str
    output_path: str
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    views: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    watch_time_s: float = Field(default=0.0, ge=0.0)

    @property
    def engagement_rate(self) -> float:
        if self.views <= 0:
            return 0.0
        return ((self.likes + self.comments) / self.views) * 100.0
