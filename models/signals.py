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


class _LLMPackage(BaseModel):
    """Structured JSON returned by the local LLM packaging step."""

    title: str = Field(description="Five- to eight-word clip title")
    hook_text: str = Field(description="Five- to eight-word opening hook")
    virality_score: float = Field(ge=0.0, le=10.0, description="Predicted virality score")
    tags: list[str] = Field(description="Relevant content tags")

    @staticmethod
    def _ensure_list(value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(10.0, float(value)))

    @classmethod
    def model_validate(cls, value: object):
        if isinstance(value, dict):
            payload = dict(value)
            payload["virality_score"] = cls._clamp_score(float(payload.get("virality_score", 0.0)))
            payload["tags"] = cls._ensure_list(payload.get("tags", []))
            return super().model_validate(payload)
        return super().model_validate(value)


class ClipMetadata(BaseModel):
    """Phase-3 metadata used by the later video-editing pipeline."""

    audio_analysis: AudioAnalysis = Field(description="Output of the phase-2 detection graph")
    transcript_text: str = Field(default="", description="Full ASR transcript")
    word_timestamps: list[dict[str, object]] = Field(default_factory=list)
    language: str = Field(default="en", description="Detected language code")
    title: str = Field(default="", description="Five- to eight-word clip title")
    hook_text: str = Field(default="", description="Five- to eight-word opening hook")
    virality_score: float = Field(default=0.0, description="Predicted virality score")
    tags: list[str] = Field(default_factory=list)
    passed_threshold: bool = Field(default=False, description="Whether the clip passed the virality cutoff")