import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from genui_quality_gate.config import Settings
from genui_quality_gate.queue import QueueUnavailable, TaskPublisher, task_queue
from genui_quality_gate.tasks import TaskEnvelope, TaskPayload
from tests.unit.test_inference import PAYLOAD


def test_bounded_queue_uses_reject_publish_and_persistence() -> None:
    queue = task_queue(Settings(queue_capacity=7))
    assert queue.durable
    assert queue.arguments["x-max-length"] == 7
    assert queue.arguments["x-overflow"] == "reject-publish"
    assert "x-max-length" not in task_queue(Settings(backpressure_enabled=False)).arguments


@pytest.mark.parametrize("prefetch", [0, 2, 100])
def test_prefetch_cannot_exceed_executor_concurrency(prefetch: int) -> None:
    with pytest.raises(ValidationError):
        Settings(prefetch_count=prefetch, worker_concurrency=1)
    assert Settings(prefetch_count=0, backpressure_enabled=False).prefetch_count == 0


async def test_publisher_bounds_outstanding_confirms_without_waiting_for_a_slot() -> None:
    broker = AsyncMock()
    held = asyncio.Event()
    broker.publish.side_effect = lambda *args, **kwargs: None

    async def publish(*args: object, **kwargs: object) -> None:
        await held.wait()

    broker.publish.side_effect = publish
    publisher = TaskPublisher(broker, Settings(api_publish_concurrency=1))
    task = TaskEnvelope(payload=TaskPayload(**PAYLOAD))
    first = asyncio.create_task(publisher.publish(task))
    await asyncio.sleep(0)
    with pytest.raises(QueueUnavailable, match="capacity"):
        await publisher.publish(task)
    held.set()
    await first
    assert publisher.inflight == 0


async def test_publisher_timeout_releases_slot() -> None:
    async def slow(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(2)

    broker = AsyncMock()
    broker.publish.side_effect = slow
    publisher = TaskPublisher(broker, Settings(publish_timeout_seconds=0.01))
    with pytest.raises(QueueUnavailable):
        await publisher.publish(TaskEnvelope(payload=TaskPayload(**PAYLOAD)))
    assert publisher.inflight == 0


async def test_publisher_enables_confirms_and_sends_stable_message_id() -> None:
    broker = AsyncMock()
    publisher = TaskPublisher(broker, Settings())
    task = TaskEnvelope(payload=TaskPayload(**PAYLOAD))
    await publisher.publish(task)
    kwargs = broker.publish.call_args.kwargs
    assert kwargs["message_id"] == str(task.task_id)
    assert kwargs["persist"] and kwargs["mandatory"]
