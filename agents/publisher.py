"""Publishing controls and feedback tracking for generated clips."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable


class PublishController:
    """Simple publisher with a minimum interval between submissions."""

    def __init__(self, rate_limit_seconds: int = 300, now_factory: Callable[[], datetime] | None = None):
        self.rate_limit_seconds = max(0, int(rate_limit_seconds))
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self.last_published_at: datetime | None = None

    def can_publish(self, now: datetime | None = None) -> bool:
        """Return True when the next publish is allowed."""
        current = now or self.now_factory()
        if self.last_published_at is None:
            return True
        return current - self.last_published_at > timedelta(seconds=self.rate_limit_seconds)

    def publish(self, now: datetime | None = None) -> bool:
        """Record a publish attempt and return whether it was accepted."""
        current = now or self.now_factory()
        if not self.can_publish(current):
            return False
        self.last_published_at = current
        return True
