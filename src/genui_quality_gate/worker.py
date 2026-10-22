from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any

import psutil
from fastapi import FastAPI
from faststream import AckPolicy
from faststream.rabbit import RabbitBroker
from faststream.rabbit.annotations import RabbitMessage

from genui_quality_gate.config import Settings
from genui_quality_gate.database import close_database_pool, create_database_pool
from genui_quality_gate.inference import LoadedModel, Prediction, load_model
from genui_quality_gate.logging import LOGGER_NAME, configure_logging
from genui_quality_gate.queue import create_broker, dead_letter_queue, declare_topology, task_queue
from genui_quality_gate.task_store import PostgresTaskStore, TaskRepository
from genui_quality_gate.tasks import TaskEnvelope, message_identity

logger = logging.getLogger(LOGGER_NAME)


@dataclass
class WorkerMetrics:
    active: int = 0
    received: int = 0
    succeeded: int = 0
    retries: int = 0
    failed: int = 0


def register_handler(
    broker: RabbitBroker,
    settings: Settings,
    store: TaskRepository,
    predict: Callable[[TaskEnvelope], Awaitable[Prediction]],
    metrics: WorkerMetrics | None = None,
) -> Callable[..., Awaitable[None]]:
    counters = metrics if metrics is not None else WorkerMetrics()

    @broker.subscriber(task_queue(settings), ack_policy=AckPolicy.MANUAL)
    async def handle(body: Any, message: RabbitMessage) -> None:
        counters.received += 1
        counters.active += 1
        task_id, digest = message_identity(body, message.message_id)
        created_at = datetime.now(UTC)
        try:
            envelope = TaskEnvelope.model_validate(body)
            created_at = envelope.created_at
        except Exception:
            envelope = None

        async def work() -> Prediction:
            if envelope is None:
                raise ValueError("Invalid task payload")
            return await predict(envelope)

        try:
            outcome = await store.execute(task_id, digest, work, created_at)
            if outcome.state == "retry":
                counters.retries += 1
                logger.warning(
                    "task_retry", extra={"task_id": str(task_id), "attempt": outcome.attempts}
                )
                await asyncio.sleep(settings.retry_delay_seconds)
                await message.nack(requeue=True)
            elif outcome.state in {"failed", "conflict"}:
                await broker.publish(
                    body,
                    queue=dead_letter_queue(settings).name,
                    persist=True,
                    mandatory=True,
                    message_id=str(task_id),
                    headers={"attempts": outcome.attempts, "error_type": outcome.error_type},
                    timeout=settings.publish_timeout_seconds,
                )
                await message.ack()
                counters.failed += 1
                logger.error(
                    "task_dead_lettered",
                    extra={"task_id": str(task_id), "attempt": outcome.attempts},
                )
            else:
                await message.ack()
                counters.succeeded += 1
                logger.info("task_succeeded", extra={"task_id": str(task_id)})
        except Exception as exc:
            logger.error(
                "task_delivery_failed",
                extra={"task_id": str(task_id), "error_type": type(exc).__name__},
            )
            await asyncio.sleep(settings.retry_delay_seconds)
            await message.nack(requeue=True)
        finally:
            counters.active -= 1

    return handle


def delayed_prediction(model: LoadedModel, task: TaskEnvelope, delay: float) -> Prediction:
    time.sleep(delay)
    return model.predict_sync(**task.payload.model_dump())


def create_worker_app(
    settings: Settings | None = None,
    *,
    model_loader: Callable[[Settings], LoadedModel] = load_model,
) -> FastAPI:
    configuration = settings if settings is not None else Settings()
    broker = create_broker(configuration)
    metrics = WorkerMetrics()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        pool = await create_database_pool(configuration)
        executor = ThreadPoolExecutor(max_workers=configuration.worker_concurrency)
        try:
            model = await asyncio.to_thread(model_loader, configuration)
            application.state.model = model
            store = PostgresTaskStore(pool, configuration.max_attempts, configuration.queue_name)
            await store.initialize()

            async def predict(task: TaskEnvelope) -> Prediction:
                return await asyncio.get_running_loop().run_in_executor(
                    executor,
                    partial(delayed_prediction, model, task, configuration.inference_delay_seconds),
                )

            register_handler(broker, configuration, store, predict, metrics)
            await broker.connect()
            await declare_topology(broker, configuration)
            await broker.start()
            application.state.ready = True
            logger.info(
                "worker_started",
                extra={
                    "model_name": model.metadata.name,
                    "model_version": model.metadata.version,
                    "model_alias": model.metadata.alias,
                    "model_run_id": model.metadata.run_id,
                    "prefetch_count": configuration.prefetch_count,
                    "worker_concurrency": configuration.worker_concurrency,
                },
            )
            yield
        finally:
            application.state.ready = False
            logger.info("worker_draining", extra={"active_tasks": metrics.active})
            await broker.stop()
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            await close_database_pool(pool, configuration.db_shutdown_timeout_seconds)
            logger.info("worker_stopped", extra={"active_tasks": metrics.active})

    application = FastAPI(title="GenUI worker", lifespan=lifespan)
    application.state.ready = False
    application.state.model = None
    application.state.broker = broker

    @application.get("/healthz")
    async def healthz() -> dict[str, Any]:
        model: LoadedModel | None = application.state.model
        return {
            "status": "ok" if application.state.ready else "starting",
            "model": model.metadata.model_dump() if model else None,
            "prefetch_count": configuration.prefetch_count,
            "concurrency": configuration.worker_concurrency,
            "active": metrics.active,
            "received": metrics.received,
            "succeeded": metrics.succeeded,
            "retries": metrics.retries,
            "failed": metrics.failed,
            "rss_bytes": psutil.Process().memory_info().rss,
        }

    return application
