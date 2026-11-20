from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO
from uuid import UUID, uuid4, uuid5

from genui_quality_gate import telemetry
from genui_quality_gate.config import Settings
from genui_quality_gate.logging import LOGGER_NAME, configure_logging, task_context
from genui_quality_gate.queue import (
    QueueUnavailable,
    TaskPublisher,
    create_broker,
    declare_topology,
)
from genui_quality_gate.tasks import TaskEnvelope, TaskPayload

logger = logging.getLogger(LOGGER_NAME)


@dataclass
class Checkpoint:
    source_sha256: str
    batch_id: UUID
    offset: int = 0
    accepted: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as output:
            json.dump(
                {
                    "source_sha256": self.source_sha256,
                    "batch_id": str(self.batch_id),
                    "offset": self.offset,
                    "accepted": self.accepted,
                },
                output,
            )
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)


def load_checkpoint(source: Path, path: Path) -> Checkpoint:
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if not path.exists():
        return Checkpoint(digest, uuid4())
    state = json.loads(path.read_text())
    if state["source_sha256"] != digest:
        raise ValueError("Input file differs from the checkpoint source")
    return Checkpoint(digest, UUID(state["batch_id"]), state["offset"], state["accepted"])


async def publish_file(
    source: Path,
    checkpoint_path: Path,
    publisher: TaskPublisher,
    *,
    output: TextIO | None = None,
    retry_base: float = 0.02,
) -> dict[str, int]:
    checkpoint = await asyncio.to_thread(load_checkpoint, source, checkpoint_path)
    await asyncio.to_thread(checkpoint.save, checkpoint_path)
    rejected = 0
    with source.open("rb") as stream:
        stream.seek(checkpoint.offset)
        while True:
            offset = stream.tell()
            line = await asyncio.to_thread(stream.readline, 2_100_001)
            if not line:
                break
            if len(line) > 2_100_000:
                raise ValueError("A JSONL record exceeds the maximum task size")
            payload = TaskPayload.model_validate_json(line)
            task = TaskEnvelope(task_id=uuid5(checkpoint.batch_id, str(offset)), payload=payload)
            with (
                task_context(str(task.task_id)),
                telemetry.span(
                    "source.task",
                    attributes={"task.id": str(task.task_id), "batch.id": str(checkpoint.batch_id)},
                ) as span,
            ):
                retry = 0
                signals = telemetry.active.instruments
                while True:
                    try:
                        await publisher.publish(task)
                        break
                    except QueueUnavailable as error:
                        rejected += 1
                        if error.rejected:
                            signals.rejected.add(1)
                        delay = min(1.0, retry_base * 2 ** min(retry, 6))
                        retry += 1
                        span.add_event(
                            "broker.rejected" if error.rejected else "broker.unavailable",
                            {"retry": retry, "wait.seconds": delay},
                        )
                        with telemetry.span("source.backoff", attributes={"retry": retry}):
                            logger.warning(
                                "source_backoff",
                                extra={
                                    "attempt": retry,
                                    "wait_seconds": delay,
                                    "batch_id": str(checkpoint.batch_id),
                                    "broker_rejected": error.rejected,
                                },
                            )
                            started = perf_counter()
                            await asyncio.sleep(delay)
                            signals.waited.add(perf_counter() - started)
                signals.published.add(1)
                logger.info("source_published", extra={"batch_id": str(checkpoint.batch_id)})
            checkpoint.offset = stream.tell()
            checkpoint.accepted += 1
            await asyncio.to_thread(checkpoint.save, checkpoint_path)
            if output is not None:
                progress = {"accepted": checkpoint.accepted, "rejected": rejected}
                output.write(json.dumps(progress) + "\n")
                output.flush()
    return {"accepted": checkpoint.accepted, "rejected": rejected}


async def run(source: Path, checkpoint: Path) -> None:
    settings = Settings()
    async with telemetry.observe("source") as signals:
        configure_logging(signals.service)
        broker = create_broker(settings)
        try:
            await broker.connect()
            await declare_topology(broker, settings)
            started = perf_counter()
            result = await publish_file(source, checkpoint, TaskPublisher(broker, settings))
            logger.info(
                "batch_finished", extra=result | {"duration_seconds": perf_counter() - started}
            )
        except Exception:
            logger.exception("batch_failed")
            raise SystemExit(1) from None
        finally:
            await broker.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.checkpoint))


if __name__ == "__main__":
    main()
