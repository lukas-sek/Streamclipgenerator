"""Run the phase-2 stream-monitoring pipeline for one Twitch channel."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from agents.graph import Phase2State, build_graph
from agents.scout_agent import ScoutAgent
from models.signals import AudioAnalysis, ChatSpike

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

CANDIDATES_LOG = Path("output/candidates.jsonl")
CANDIDATES_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_candidate(candidate: AudioAnalysis) -> None:
    """Append one detected candidate to the JSONL output log."""
    with CANDIDATES_LOG.open("a", encoding="utf-8") as handle:
        handle.write(candidate.model_dump_json() + "\n")
    logger.info("Candidate written to %s", CANDIDATES_LOG)


async def process_spikes(
    spike_queue: asyncio.Queue[ChatSpike],
    graph,
    max_concurrent: int = 2,
) -> None:
    """Consume chat spikes and run the graph for each detected event."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _handle(spike: ChatSpike) -> None:
        async with semaphore:
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            initial_state: Phase2State = {
                "spike": spike,
                "segment_path": "",
                "wav_path": "",
                "vad_segments": [],
                "rms_peak": 0.0,
                "rms_mean": 0.0,
                "has_audio_event": False,
                "candidate": None,
            }
            logger.info("Invoking detection graph for spike: %s", spike.label)
            try:
                result = await graph.ainvoke(initial_state, config=config)
                candidate: AudioAnalysis | None = result.get("candidate")
                if candidate is not None:
                    _log_candidate(candidate)
                    logger.info(
                        "PASS  channel=%-20s rms_peak=%.4f  vad_segs=%d",
                        candidate.spike.channel,
                        candidate.rms_peak,
                        len(candidate.vad_segments),
                    )
                else:
                    logger.info("SKIP  channel=%s (below threshold)", spike.channel)
            except Exception as exc:  # pragma: no cover - logging path only
                logger.exception("Graph invocation failed for spike %s: %s", spike.label, exc)

    while True:
        spike = await spike_queue.get()
        asyncio.create_task(_handle(spike))
        spike_queue.task_done()


async def main(channel: str) -> None:
    """Start the scout and process messages from the queue."""
    logger.info("=== Phase 2 pipeline starting for channel: #%s ===", channel)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    spike_queue: asyncio.Queue[ChatSpike] = asyncio.Queue(maxsize=10)
    scout = ScoutAgent(channel=channel, spike_queue=spike_queue)

    scout_task = asyncio.create_task(scout.run(), name="scout")
    processor_task = asyncio.create_task(process_spikes(spike_queue, graph), name="processor")

    logger.info("Scout and processor tasks are running. Press Ctrl+C to stop.")
    try:
        await asyncio.gather(scout_task, processor_task)
    except asyncio.CancelledError:
        pass
    finally:
        scout_task.cancel()
        processor_task.cancel()
        logger.info("Pipeline shut down.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-to-Shorts phase 2 pipeline")
    parser.add_argument(
        "--channel",
        default=os.getenv("TWITCH_CHANNEL", ""),
        help="Twitch channel name to monitor, or set TWITCH_CHANNEL",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.channel:
        print("ERROR: provide --channel <name> or set TWITCH_CHANNEL env var", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(main(args.channel))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
