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

## Local setup

1. Create a virtual environment and install dependencies.
2. Ensure FFmpeg and Ollama are available locally.
3. Run the project-specific verification commands in the repo root.

## Verification

```bash
python -m pytest -q
```

The project follows a milestone-based structure so each chunk can be committed independently and reviewed as a clean feature slice.
