"""Monitor Twitch IRC chat for sudden activity spikes."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from collections import deque

from models.signals import ChatSpike

logger = logging.getLogger(__name__)

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667
WINDOW_SECS = 5
BASELINE_WINDOWS = int(os.getenv("BASELINE_WINDOWS", "30"))
SPIKE_MULTIPLIER = float(os.getenv("SPIKE_MULTIPLIER", "3.0"))
EMOTE_FLOOD_RATIO = 0.6
MIN_BASELINE_WINDOWS = 3

EMOTE_PATTERN = re.compile(
    r"\b(KEKW|PogChamp|LUL|OMEGALUL|Pog|PauseChamp|POGGERS|"
    r"monkaS|EZ|Clap|TriHard|HeyGuys)\b",
    re.IGNORECASE,
)


def _is_emote_flood(message: str) -> bool:
    """Return whether a message looks like emote spam or a caps flood."""
    letters = [character for character in message if character.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    return uppercase_ratio >= EMOTE_FLOOD_RATIO or bool(EMOTE_PATTERN.search(message))


class ScoutAgent:
    """Monitor one Twitch channel and emit detected chat spikes."""

    def __init__(self, channel: str, spike_queue: asyncio.Queue[ChatSpike]) -> None:
        self.channel = channel.lower().lstrip("#")
        self.spike_queue = spike_queue
        self._nick = f"justinfan{random.randint(10000, 99999)}"
        self._window_counts: deque[int] = deque(maxlen=BASELINE_WINDOWS)
        self._emote_counts: deque[int] = deque(maxlen=BASELINE_WINDOWS)
        self._current_msgs = 0
        self._current_emotes = 0
        self._window_start = time.monotonic()
        self._last_spike_at = 0.0
        self._spike_cooldown = 30.0

    def _tick_window(self) -> None:
        """Move the current window into the rolling baseline."""
        self._window_counts.append(self._current_msgs)
        self._emote_counts.append(self._current_emotes)
        self._current_msgs = 0
        self._current_emotes = 0
        self._window_start = time.monotonic()

    def _baseline(self) -> float:
        return sum(self._window_counts) / len(self._window_counts) if self._window_counts else 0.0

    def _emote_baseline(self) -> float:
        return sum(self._emote_counts) / len(self._emote_counts) if self._emote_counts else 0.0

    def _check_spike(self) -> ChatSpike | None:
        """Return a signal when the current window exceeds a threshold."""
        now = time.monotonic()
        if now - self._last_spike_at < self._spike_cooldown:
            return None
        if len(self._window_counts) < MIN_BASELINE_WINDOWS:
            return None

        baseline = self._baseline()
        if baseline < 1.0:
            return None

        ratio = self._current_msgs / baseline
        if ratio >= SPIKE_MULTIPLIER:
            self._last_spike_at = now
            return ChatSpike(
                channel=self.channel,
                msg_count=self._current_msgs,
                baseline=baseline,
                ratio=ratio,
                trigger="velocity",
            )

        emote_baseline = self._emote_baseline()
        if self._current_emotes >= 5 and (
            emote_baseline < 1.0 or self._current_emotes / emote_baseline >= 2.0
        ):
            self._last_spike_at = now
            return ChatSpike(
                channel=self.channel,
                msg_count=self._current_msgs,
                baseline=baseline,
                ratio=ratio,
                trigger="emote_flood",
            )

        return None

    def _handle_message(self, raw: str) -> None:
        """Count a raw Twitch IRC PRIVMSG line."""
        if raw.startswith("PING") or "PRIVMSG" not in raw:
            return

        try:
            text = raw.split("PRIVMSG", 1)[1].split(":", 1)[1].strip()
        except IndexError:
            return

        self._current_msgs += 1
        if _is_emote_flood(text):
            self._current_emotes += 1

    async def run(self) -> None:
        """Connect to Twitch IRC and process messages until cancelled."""
        logger.info("ScoutAgent connecting to #%s as %s", self.channel, self._nick)
        reconnect_delay = 5.0

        while True:
            try:
                reader, writer = await asyncio.open_connection(IRC_HOST, IRC_PORT)
                writer.write(f"NICK {self._nick}\r\n".encode())
                writer.write(f"USER {self._nick} 0 * :{self._nick}\r\n".encode())
                writer.write(f"JOIN #{self.channel}\r\n".encode())
                await writer.drain()

                while True:
                    elapsed = time.monotonic() - self._window_start
                    if elapsed >= WINDOW_SECS:
                        spike = self._check_spike()
                        self._tick_window()
                        if spike:
                            logger.info("Spike detected: %s", spike.label)
                            await self.spike_queue.put(spike)

                    try:
                        line_bytes = await asyncio.wait_for(
                            reader.readline(), timeout=max(0.1, WINDOW_SECS - elapsed)
                        )
                    except asyncio.TimeoutError:
                        continue

                    if not line_bytes:
                        logger.warning("ScoutAgent connection closed by server")
                        break

                    raw = line_bytes.decode("utf-8", errors="replace").rstrip()
                    if raw.startswith("PING"):
                        writer.write(f"{raw.replace('PING', 'PONG')}\r\n".encode())
                        await writer.drain()
                    else:
                        self._handle_message(raw)

            except (OSError, ConnectionResetError) as error:
                logger.warning("ScoutAgent connection error: %s", error)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)
            except asyncio.CancelledError:
                logger.info("ScoutAgent cancelled")
                return