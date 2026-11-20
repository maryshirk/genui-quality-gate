from unittest.mock import AsyncMock

import pytest
from faststream.rabbit import TestRabbitBroker

from genui_quality_gate.config import Settings
from genui_quality_gate.queue import create_broker, dead_letter_queue
from genui_quality_gate.task_store import TaskOutcome
from genui_quality_gate.tasks import TaskEnvelope, TaskPayload
from genui_quality_gate.worker import WorkerMetrics, register_handler
from tests.unit.test_inference import PAYLOAD


@pytest.mark.parametrize("state", ["succeeded", "retry", "failed", "conflict"])
async def test_faststream_handler_outcomes_and_dead_letters(state: str) -> None:
    settings = Settings(retry_delay_seconds=0.001)
    broker = create_broker(settings)
    store = AsyncMock()
    store.execute.return_value = TaskOutcome(state, 3, error_type="ValueError")
    predict = AsyncMock()
    counters = WorkerMetrics()
    register_handler(broker, settings, store, predict, counters)
    dead_letters = []

    @broker.subscriber(dead_letter_queue(settings))
    async def dead_letter(body: dict[str, object]) -> None:
        dead_letters.append(body)

    task = TaskEnvelope(payload=TaskPayload(**PAYLOAD))
    async with TestRabbitBroker(broker):
        await broker.publish(task.model_dump(mode="json"), queue=settings.queue_name)
    assert counters.received == 1
    assert counters.active == 0
    assert bool(dead_letters) == (state in {"failed", "conflict"})
    assert counters.retries == (state == "retry")
    store.execute.assert_awaited_once()
    assert store.execute.call_args.args[0] == task.task_id


async def test_faststream_validates_payload_inside_reliable_handler() -> None:
    settings = Settings(retry_delay_seconds=0.001)
    broker = create_broker(settings)
    store = AsyncMock()
    predict = AsyncMock()

    async def execute(task_id, digest, work, created_at):
        with pytest.raises(ValueError, match="Invalid task"):
            await work()
        return TaskOutcome("retry", 1)

    store.execute.side_effect = execute
    counters = WorkerMetrics()
    register_handler(broker, settings, store, predict, counters)
    async with TestRabbitBroker(broker):
        await broker.publish({"broken": True}, queue=settings.queue_name)
    predict.assert_not_called()
    assert counters.retries == 1


async def test_storage_failure_requeues_without_acknowledging_success() -> None:
    settings = Settings(retry_delay_seconds=0.001)
    broker = create_broker(settings)
    store = AsyncMock()
    store.execute.side_effect = ConnectionError("database offline")
    counters = WorkerMetrics()
    register_handler(broker, settings, store, AsyncMock(), counters)
    async with TestRabbitBroker(broker):
        await broker.publish(
            TaskEnvelope(payload=TaskPayload(**PAYLOAD)).model_dump(mode="json"),
            queue=settings.queue_name,
        )
    assert counters.active == 0
    assert counters.succeeded == 0
