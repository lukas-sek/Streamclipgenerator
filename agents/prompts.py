"""Prompt templates for the packaging stage."""

SYSTEM_PROMPT = """\
You are a viral-clip analyst for gaming live streams.
Given a transcript from a Twitch clip, your job is to:
1. Write a punchy 5-8 word TITLE.
2. Write a punchy 5-8 word HOOK TEXT.
3. Score the clip's VIRALITY from 0.0 to 10.0.
4. Provide 3-6 lowercase tags.

Return only a single JSON object with keys:
{
  "title": "<5-8 words>",
  "hook_text": "<5-8 words>",
  "virality_score": <float 0.0-10.0>,
  "tags": ["tag1", "tag2", "tag3"]
}
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Transcript:\n"
            '"Oh my god, he is the last one alive. Four enemies left. He hits the corner. "'
            '"Headshot! Another one down. The crowd is insane. One more. He holds the angle. "'
            '"The whole team is dead. Chat is exploding. OMEGALUL KEKW PogChamp."'
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"title": "Insane 1v4 Clutch on 12 HP", '
            '"hook_text": "He wiped them on 12 HP", '
            '"virality_score": 9.2, '
            '"tags": ["clutch", "1v4", "fps", "gaming", "highlight"]}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Transcript:\n"
            '"Alright, so I am just farming here. Nothing crazy. Got a kill there. "'
            '"Chat is pretty quiet. Just a normal mid-game phase, managing resources."'
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"title": "Calm Mid-Game Resource Management", '
            '"hook_text": "Just a quiet farming session", '
            '"virality_score": 2.8, '
            '"tags": ["farming", "moba", "strategy", "gameplay"]}'
        ),
    },
]


def build_messages(transcript: str) -> list[dict[str, str]]:
    """Build a prompt that includes the system rules and few-shot examples."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": f"Transcript:\n\"{transcript.strip()}\""})
    return messages
