# Streamclipgenerator

A small Python project that turns Twitch chat spikes into short-form video clips.

## What it does

- monitors Twitch chat for a spike in velocity or emote activity
- downloads a short live segment when a trigger fires
- extracts and analyzes the audio signal
- transcribes and packages the best candidate clip
- reframes it as a vertical 9:16 edit with subtitles
- tracks publishing results and adapts the trigger threshold over time

## Project structure

- agents/: pipeline stages and runtime workers
- config/: runtime settings and local model configuration
- models/: data structures used across the pipeline
- tests/: focused regression tests for each milestone slice

## Technologies used

- Python 3.10+
- Twitch IRC for live chat monitoring
- FFmpeg for segment extraction, recomposition, and subtitle rendering
- Pydantic for model validation and structured data
- LangGraph for the state-based processing graph
- Whisper / faster-whisper for transcription
- Silero VAD for speech detection
- Ollama with local LLMs for clip packaging and virality scoring
- yt-dlp / streamlink for media acquisition
- pytest for automated validation

## Local setup

1. Create a virtual environment and install dependencies.
2. Ensure FFmpeg and Ollama are available locally.
3. Run the project-specific verification commands in the repo root.

## Verification

```bash
python -m pytest -q
```


