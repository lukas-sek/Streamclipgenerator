"""Signals emitted by the stream monitoring stage."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatSpike(BaseModel):
    """A sudden increase in chat activity for one Twitch channel."""

    channel: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    msg_count: int = Field(description="Messages in the current five-second window")
    baseline: float = Field(description="Average messages per recent window")
    ratio: float = Field(description="Current message count divided by the baseline")
    trigger: Literal["velocity", "emote_flood"]

    @property
    def label(self) -> str:
        return (
            f"[{self.trigger}] #{self.channel} "
            f"ratio={self.ratio:.1f}x @ {self.timestamp.isoformat()}"
        )


class AudioAnalysis(BaseModel):
    """Audio measurements associated with a detected chat spike."""

    spike: ChatSpike
    segment_path: str = Field(description="Path to the raw downloaded stream segment")
    wav_path: str = Field(description="Path to the extracted 16 kHz mono WAV file")
    vad_segments: list[dict[str, float]] = Field(
        default_factory=list,
        description="Speech windows represented as start and end timestamps",
    )
    rms_peak: float = Field(default=0.0, description="Peak RMS value")
    rms_mean: float = Field(default=0.0, description="Mean RMS value")
    has_audio_event: bool = Field(
        default=False,
        description="Whether the audio passes the event threshold",
    )