from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aio_pika.exceptions import AMQPException, DeliveryError
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from faststream.rabbit.schemas.queue import ClassicQueueArgs

from genui_quality_gate.config import Settings
from genui_quality_gate.tasks import TaskEnvelope


def task_queue(settings: Settings) -> RabbitQueue:
    arguments: ClassicQueueArgs = {}
    if settings.backpressure_enabled:
        arguments.update({"x-max-length": settings.queue_capacity, "x-overflow": "reject-publish"})
    return RabbitQueue(settings.queue_name, durable=True, arguments=arguments)


def dead_letter_queue(settings: Settings) -> RabbitQueue:
    return RabbitQueue(f"{settings.queue_name}.dlq", durable=True)


def create_broker(settings: Settings) -> RabbitBroker:
    return RabbitBroker(
        settings.rabbit_url.get_secret_value(),
        default_channel=Channel(
            prefetch_count=settings.prefetch_count, publisher_confirms=True, on_return_raises=True
        ),
        graceful_timeout=settings.worker_shutdown_seconds,
        logger=None,
    )


async def declare_topology(broker: RabbitBroker, settings: Settings) -> None:
    await broker.declare_queue(task_queue(settings))
    await broker.declare_queue(dead_letter_queue(settings))


class QueueUnavailable(Exception):
    pass


class TaskPublisher:
    def __init__(self, broker: RabbitBroker, settings: Settings) -> None:
        self.broker = broker
        self.settings = settings
        self.inflight = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self.inflight >= self.settings.api_publish_concurrency:
            raise QueueUnavailable("Publication capacity exceeded")
        self.inflight += 1
        try:
            yield
        finally:
            self.inflight -= 1

    async def publish(self, task: TaskEnvelope) -> None:
        try:
            async with self.slot(), asyncio.timeout(self.settings.publish_timeout_seconds):
                await self.broker.publish(
                    task.model_dump(mode="json"),
                    queue=self.settings.queue_name,
                    persist=True,
                    mandatory=True,
                    message_id=str(task.task_id),
                    timeout=self.settings.publish_timeout_seconds,
                )
        except (AMQPException, DeliveryError, TimeoutError, ConnectionError, OSError) as exc:
            raise QueueUnavailable(type(exc).__name__) from None
