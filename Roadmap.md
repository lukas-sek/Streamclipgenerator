# Roadmap

## Milestone 1: Local environment

- Create a reproducible Python environment.
- Install CPU-friendly video, audio, and inference dependencies.
- Verify FFmpeg, Ollama, and required Python packages.

## Milestone 2: Stream ingestion and signal detection

- Connect to Twitch chat through IRC.
- Track message velocity and emote activity in rolling windows.
- Download short live segments when a spike is detected.
- Extract audio and score speech, silence, and energy peaks.

## Milestone 3: Transcription and clip packaging

- Transcribe candidate audio with word-level timestamps.
- Generate structured clip metadata with a local language model.
- Reject candidates below the virality threshold.

## Milestone 4: Vertical video editing

- Reframe source video to a 9:16 layout.
- Remove long dead-air sections.
- Burn dynamic subtitles into the final MP4.

## Milestone 5: Publishing and feedback

- Add publishing integrations and rate limiting.
- Store clip performance metrics.
- Use performance data to tune hooks, pacing, and trigger sensitivity.