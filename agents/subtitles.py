"""ASS subtitle generation for the clip editor stage."""

from __future__ import annotations

from pathlib import Path

_SCRIPT_INFO = """\
[Script Info]
Title: Stream-to-Shorts Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920
"""

_STYLES = """\
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,20,20,60,1
"""

_EVENTS_HEADER = """\
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds % 1) * 100))
    if centiseconds >= 100:
        centiseconds = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def compute_adjusted_words(word_timestamps: list[dict], cut_points: list[dict]) -> list[dict]:
    """Re-map word timestamps to the trimmed video timeline."""
    if not cut_points:
        return word_timestamps

    adjusted: list[dict] = []
    for word in word_timestamps:
        word_start = float(word["start"])
        word_end = float(word["end"])
        offset = 0.0
        in_cut = False
        cumulative_removed = 0.0
        previous_end = 0.0

        for cut_point in cut_points:
            cut_start = float(cut_point["start"])
            cut_end = float(cut_point["end"])
            cumulative_removed += cut_start - previous_end
            previous_end = cut_end

            if word_start >= cut_start and word_start < cut_end:
                in_cut = True
                offset = cumulative_removed
                break

        if not in_cut:
            continue

        new_start = word_start - offset
        new_end = min(word_end, previous_end) - offset
        if new_end <= new_start:
            continue

        adjusted.append(
            {
                "word": word["word"],
                "start": round(new_start, 3),
                "end": round(new_end, 3),
                "probability": word.get("probability", 1.0),
            }
        )

    return adjusted


def generate_ass(word_timestamps: list[dict], cut_points: list[dict], output_path: str) -> str:
    """Generate an ASS subtitle file for a trimmed clip."""
    adjusted = compute_adjusted_words(word_timestamps, cut_points)
    lines = [_SCRIPT_INFO, _STYLES, _EVENTS_HEADER]

    for word in adjusted:
        text = str(word["word"]).strip()
        if not text:
            continue
        start = _fmt_ass_time(float(word["start"]))
        end = _fmt_ass_time(float(word["end"]))
        subtitle_text = r"{\fad(150,50)}" + text.upper()
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{subtitle_text}")

    content = "\n".join(lines) + "\n"
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8-sig")
    return str(target_path.resolve())


def escape_ass_path(path: str) -> str:
    """Escape a Windows path so it can be used in an FFmpeg filter expression."""
    escaped = path.replace("\\", "/")
    if len(escaped) >= 2 and escaped[1] == ":":
        escaped = escaped[0] + "\\:" + escaped[2:]
    return escaped
