from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import UUID

from genui_quality_gate.inference import Prediction
from genui_quality_gate.tasks import StoredTask

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_tasks (
    task_id UUID PRIMARY KEY,
    payload_hash TEXT NOT NULL,
    queue_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PROCESSING', 'SUCCEEDED', 'FAILED')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    result JSONB,
    error_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    finished_at TIMESTAMPTZ,
    inference_ms DOUBLE PRECISION,
    end_to_end_ms DOUBLE PRECISION
)
"""


@dataclass
class TaskOutcome:
    state: Literal["succeeded", "retry", "failed", "conflict"]
    attempts: int
    result: Prediction | None = None
    error_type: str | None = None


class TaskRepository(Protocol):
    async def execute(
        self,
        task_id: UUID,
        digest: str,
        work: Callable[[], Awaitable[Prediction]],
        created_at: datetime,
    ) -> TaskOutcome: ...

    async def get(self, task_id: UUID) -> StoredTask | None: ...


class PostgresTaskStore:
    def __init__(self, pool: Any, max_attempts: int = 3, queue_name: str = "genui.tasks") -> None:
        self.pool = pool
        self.max_attempts = max_attempts
        self.queue_name = queue_name

    async def initialize(self) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(SCHEMA)

    async def execute(
        self,
        task_id: UUID,
        digest: str,
        work: Callable[[], Awaitable[Prediction]],
        created_at: datetime,
    ) -> TaskOutcome:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO pipeline_tasks(task_id, payload_hash, status, created_at, queue_name) "
                "VALUES($1, $2, 'PROCESSING', $3, $4) ON CONFLICT(task_id) DO NOTHING",
                task_id,
                digest,
                created_at,
                self.queue_name,
            )
            record = await connection.fetchrow(
                "SELECT * FROM pipeline_tasks WHERE task_id=$1 FOR UPDATE", task_id
            )
            if record["payload_hash"] != digest:
                return TaskOutcome("conflict", record["attempts"], error_type="IdempotencyConflict")
            if record["status"] == "SUCCEEDED":
                return TaskOutcome(
                    "succeeded",
                    record["attempts"],
                    Prediction.model_validate_json(record["result"]),
                )
            if record["status"] == "FAILED":
                return TaskOutcome("failed", record["attempts"], error_type=record["error_type"])
            attempts = record["attempts"] + 1
            started = perf_counter()
            try:
                prediction = await work()
            except Exception as exc:
                state: Literal["failed", "retry"] = (
                    "failed" if attempts >= self.max_attempts else "retry"
                )
                await connection.execute(
                    "UPDATE pipeline_tasks SET attempts=$2, status=$3, error_type=$4, "
                    "finished_at=CASE WHEN $3='FAILED' THEN clock_timestamp() ELSE NULL END "
                    "WHERE task_id=$1",
                    task_id,
                    attempts,
                    "FAILED" if state == "failed" else "PROCESSING",
                    type(exc).__name__,
                )
                return TaskOutcome(state, attempts, error_type=type(exc).__name__)
            inference_ms = (perf_counter() - started) * 1000
            latency_ms = max(0, (datetime.now(UTC) - record["created_at"]).total_seconds() * 1000)
            await connection.execute(
                "UPDATE pipeline_tasks SET status='SUCCEEDED', attempts=$2, result=$3::jsonb, "
                "error_type=NULL, finished_at=clock_timestamp(), inference_ms=$4, "
                "end_to_end_ms=$5 WHERE task_id=$1",
                task_id,
                attempts,
                prediction.model_dump_json(),
                inference_ms,
                latency_ms,
            )
            return TaskOutcome("succeeded", attempts, prediction)

    async def get(self, task_id: UUID) -> StoredTask | None:
        async with self.pool.acquire() as connection:
            record = await connection.fetchrow(
                "SELECT * FROM pipeline_tasks WHERE task_id=$1", task_id
            )
        if record is None:
            return None
        values = dict(record)
        values.pop("payload_hash")
        values.pop("queue_name")
        if values["result"] is not None:
            values["result"] = json.loads(values["result"])
        return StoredTask.model_validate(values)
