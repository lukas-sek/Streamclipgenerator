"""Audio detection graph for turning chat spikes into clip candidates."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import ffmpeg
import numpy as np
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from models.signals import AudioAnalysis, ChatSpike

logger = logging.getLogger(__name__)

SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "45"))
RMS_THRESHOLD = float(os.getenv("RMS_THRESHOLD", "0.05"))
RMS_WINDOW_MS = 500
OUTPUT_DIR = Path("output/segments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Phase2State(TypedDict):
    """State passed between the audio detection graph nodes."""

    spike: ChatSpike
    segment_path: str
    wav_path: str
    vad_segments: Annotated[list[dict[str, float]], lambda _old, new: new]
    rms_peak: float
    rms_mean: float
    has_audio_event: bool
    candidate: AudioAnalysis | None


async def download_segment(state: Phase2State) -> dict[str, str]:
    """Download a short Twitch segment and save it as an MPEG transport stream."""
    spike = state["spike"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    segment_path = OUTPUT_DIR / f"{spike.channel}_{timestamp}.ts"
    stream_url = f"https://twitch.tv/{spike.channel}"

    scripts_dir = Path(sys.executable).parent
    streamlink = scripts_dir / "streamlink.exe"
    streamlink_command = str(streamlink) if streamlink.exists() else "streamlink"
    stream_command = [streamlink_command, "--stdout", "--quiet", stream_url, "best"]
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        "pipe:0",
        "-t",
        str(SEGMENT_DURATION),
        "-c",
        "copy",
        str(segment_path),
    ]

    logger.info("Downloading %s seconds from %s", SEGMENT_DURATION, stream_url)

    def run_download() -> None:
        with subprocess.Popen(stream_command, stdout=subprocess.PIPE) as stream_process:
            with subprocess.Popen(
                ffmpeg_command,
                stdin=stream_process.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as ffmpeg_process:
                if stream_process.stdout:
                    stream_process.stdout.close()
                ffmpeg_process.wait(timeout=SEGMENT_DURATION + 30)
            stream_process.terminate()

    await asyncio.get_running_loop().run_in_executor(None, run_download)
    if not segment_path.exists():
        raise RuntimeError(f"Segment download failed: {segment_path}")

    return {"segment_path": str(segment_path)}


async def extract_audio(state: Phase2State) -> dict[str, str]:
    """Convert a segment to a 16 kHz mono PCM WAV file."""
    segment_path = state["segment_path"]
    wav_path = str(Path(segment_path).with_suffix(".wav"))

    def run_extraction() -> None:
        try:
            (
                ffmpeg.input(segment_path)
                .output(wav_path, ar=16000, ac=1, acodec="pcm_s16le", vn=None)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as error:
            stderr = error.stderr.decode(errors="replace") if error.stderr else ""
            raise RuntimeError(f"Audio extraction failed: {stderr[-400:]}") from error

        wav_file = Path(wav_path)
        if not wav_file.exists() or wav_file.stat().st_size < 44:
            raise RuntimeError(f"Audio extraction produced an empty file: {wav_path}")
        with wav_file.open("rb") as handle:
            if handle.read(4) != b"RIFF":
                raise RuntimeError(f"Audio extraction produced an invalid WAV: {wav_path}")

    await asyncio.get_running_loop().run_in_executor(None, run_extraction)
    return {"wav_path": wav_path}


async def run_vad(state: Phase2State) -> dict[str, list[dict[str, float]]]:
    """Detect speech windows in the extracted WAV using Silero VAD."""
    wav_path = state["wav_path"]

    def detect_speech() -> list[dict[str, float]]:
        import wave

        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        with wave.open(wav_path, "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            raw_audio = wav_file.readframes(wav_file.getnframes())

        if sample_width == 2:
            audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        model = load_silero_vad()
        timestamps = get_speech_timestamps(torch.from_numpy(audio), model, return_seconds=True)
        return [{"start": float(item["start"]), "end": float(item["end"])} for item in timestamps]

    vad_segments = await asyncio.get_running_loop().run_in_executor(None, detect_speech)
    return {"vad_segments": vad_segments}


async def analyze_rms(state: Phase2State) -> dict[str, float]:
    """Compute peak and mean RMS energy in 500 ms windows."""
    wav_path = state["wav_path"]

    def calculate_rms() -> tuple[float, float]:
        import wave

        with wave.open(wav_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            raw_audio = wav_file.readframes(wav_file.getnframes())

        dtype = np.int16 if sample_width == 2 else np.int32
        samples = np.frombuffer(raw_audio, dtype=dtype).astype(np.float32)
        if channels > 1:
            samples = samples[::channels]
        samples /= float(np.iinfo(dtype).max)

        window_size = int(sample_rate * RMS_WINDOW_MS / 1000)
        if window_size < 1 or len(samples) < window_size:
            return 0.0, 0.0

        window_count = len(samples) // window_size
        windows = samples[: window_count * window_size].reshape(window_count, window_size)
        rms_values = np.sqrt(np.mean(windows**2, axis=1))
        return float(rms_values.max()), float(rms_values.mean())

    rms_peak, rms_mean = await asyncio.get_running_loop().run_in_executor(None, calculate_rms)
    return {"rms_peak": rms_peak, "rms_mean": rms_mean}


def evaluate_signal(state: Phase2State) -> str:
    """Route strong audio events to a candidate and weak events to cleanup."""
    return "emit_candidate" if state["rms_peak"] >= RMS_THRESHOLD else "discard_segment"


async def emit_candidate(state: Phase2State) -> dict[str, object]:
    """Package the audio measurements as an accepted candidate."""
    candidate = AudioAnalysis(
        spike=state["spike"],
        segment_path=state["segment_path"],
        wav_path=state["wav_path"],
        vad_segments=state["vad_segments"],
        rms_peak=state["rms_peak"],
        rms_mean=state["rms_mean"],
        has_audio_event=True,
    )
    return {"has_audio_event": True, "candidate": candidate}


async def discard_segment(state: Phase2State) -> dict[str, object]:
    """Remove media files for a segment that does not pass the RMS threshold."""
    for key in ("segment_path", "wav_path"):
        path = Path(state.get(key, ""))
        if path.exists():
            path.unlink()
    return {"has_audio_event": False, "candidate": None}


def build_graph():
    """Build and compile the phase-2 audio detection graph."""
    graph = StateGraph(Phase2State)
    graph.add_node("download_segment", download_segment)
    graph.add_node("extract_audio", extract_audio)
    graph.add_node("run_vad", run_vad)
    graph.add_node("analyze_rms", analyze_rms)
    graph.add_node("emit_candidate", emit_candidate)
    graph.add_node("discard_segment", discard_segment)
    graph.add_edge(START, "download_segment")
    graph.add_edge("download_segment", "extract_audio")
    graph.add_edge("extract_audio", "run_vad")
    graph.add_edge("run_vad", "analyze_rms")
    graph.add_conditional_edges(
        "analyze_rms",
        evaluate_signal,
        {"emit_candidate": "emit_candidate", "discard_segment": "discard_segment"},
    )
    graph.add_edge("emit_candidate", END)
    graph.add_edge("discard_segment", END)
    return graph.compile()