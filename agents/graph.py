"""Audio detection graph for turning chat spikes into clip candidates."""

from __future__ import annotations

import asyncio
import gc
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

from agents.prompts import build_messages
from models.signals import AudioAnalysis, ChatSpike, ClipMetadata, _LLMPackage

logger = logging.getLogger(__name__)

SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "45"))
RMS_THRESHOLD = float(os.getenv("RMS_THRESHOLD", "0.05"))
RMS_WINDOW_MS = 500
VIRALITY_THRESHOLD = float(os.getenv("VIRALITY_THRESHOLD", "5.0"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "shorts-llm")
OUTPUT_DIR = Path("output/segments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Phase2State(TypedDict):
    """State passed through the audio and packaging stages."""

    spike: ChatSpike
    segment_path: str
    wav_path: str
    vad_segments: Annotated[list[dict[str, float]], lambda _old, new: new]
    rms_peak: float
    rms_mean: float
    has_audio_event: bool
    candidate: AudioAnalysis | None
    transcript_text: str
    word_timestamps: Annotated[list[dict[str, object]], lambda _old, new: new]
    language: str
    clip_metadata: ClipMetadata | None


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


async def transcribe_audio(state: Phase2State) -> dict[str, object]:
    """Transcribe the wav file with Faster-Whisper and return text plus word timestamps."""
    wav_path = state["wav_path"]

    def run_transcription() -> tuple[str, list[dict[str, object]], str]:
        from faster_whisper import WhisperModel

        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav_path, word_timestamps=True, beam_size=5)

        transcript_parts: list[str] = []
        word_timestamps: list[dict[str, object]] = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())
            for word in getattr(segment, "words", []) or []:
                word_timestamps.append(
                    {
                        "word": word.word,
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "probability": round(float(word.probability), 4),
                    }
                )

        transcript_text = " ".join(part for part in transcript_parts if part)
        language = getattr(info, "language", "en")
        del model
        gc.collect()
        return transcript_text, word_timestamps, language

    transcript_text, word_timestamps, language = await asyncio.get_running_loop().run_in_executor(
        None, run_transcription
    )
    return {
        "transcript_text": transcript_text,
        "word_timestamps": word_timestamps,
        "language": language,
    }


async def package_clip(state: Phase2State) -> dict[str, ClipMetadata]:
    """Package the transcript into structured clip metadata using the local LLM."""
    transcript = state["transcript_text"]

    def run_packaging() -> ClipMetadata:
        import json

        from openai import OpenAI

        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=build_messages(transcript),
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        raw_json = response.choices[0].message.content or "{}"
        try:
            llm_data = _LLMPackage.model_validate(json.loads(raw_json))
        except Exception:
            llm_data = _LLMPackage(
                title="Unknown Clip",
                hook_text="Something happened here",
                virality_score=0.0,
                tags=[],
            )

        audio_analysis = state["candidate"]
        return ClipMetadata(
            audio_analysis=audio_analysis,
            transcript_text=state["transcript_text"],
            word_timestamps=state["word_timestamps"],
            language=state["language"],
            title=llm_data.title,
            hook_text=llm_data.hook_text,
            virality_score=llm_data.virality_score,
            tags=llm_data.tags,
            passed_threshold=False,
        )

    clip_metadata = await asyncio.get_running_loop().run_in_executor(None, run_packaging)
    return {"clip_metadata": clip_metadata}


def evaluate_virality(state: Phase2State) -> str:
    """Accept or discard packaged clips based on virality threshold."""
    metadata = state.get("clip_metadata")
    if metadata is None:
        return "discard_transcript"
    return "finalize_clip" if metadata.virality_score >= VIRALITY_THRESHOLD else "discard_transcript"


async def finalize_clip(state: Phase2State) -> dict[str, ClipMetadata]:
    """Mark the clip as accepted and keep the metadata for later stages."""
    metadata = state["clip_metadata"]
    final_metadata = metadata.model_copy(update={"passed_threshold": True})
    return {"clip_metadata": final_metadata}


async def discard_transcript(state: Phase2State) -> dict[str, ClipMetadata | None]:
    """Drop transcript results that do not clear the virality threshold."""
    metadata = state.get("clip_metadata")
    if metadata is not None:
        return {"clip_metadata": metadata.model_copy(update={"passed_threshold": False})}
    return {"clip_metadata": None}


def build_graph():
    """Build and compile the phase-2 and phase-3 detection pipeline."""
    graph = StateGraph(Phase2State)
    graph.add_node("download_segment", download_segment)
    graph.add_node("extract_audio", extract_audio)
    graph.add_node("run_vad", run_vad)
    graph.add_node("analyze_rms", analyze_rms)
    graph.add_node("emit_candidate", emit_candidate)
    graph.add_node("discard_segment", discard_segment)
    graph.add_node("transcribe_audio", transcribe_audio)
    graph.add_node("package_clip", package_clip)
    graph.add_node("finalize_clip", finalize_clip)
    graph.add_node("discard_transcript", discard_transcript)
    graph.add_edge(START, "download_segment")
    graph.add_edge("download_segment", "extract_audio")
    graph.add_edge("extract_audio", "run_vad")
    graph.add_edge("run_vad", "analyze_rms")
    graph.add_conditional_edges(
        "analyze_rms",
        evaluate_signal,
        {"emit_candidate": "emit_candidate", "discard_segment": "discard_segment"},
    )
    graph.add_edge("emit_candidate", "transcribe_audio")
    graph.add_edge("transcribe_audio", "package_clip")
    graph.add_conditional_edges(
        "package_clip",
        evaluate_virality,
        {"finalize_clip": "finalize_clip", "discard_transcript": "discard_transcript"},
    )
    graph.add_edge("finalize_clip", END)
    graph.add_edge("discard_transcript", END)
    graph.add_edge("discard_segment", END)
    return graph.compile()