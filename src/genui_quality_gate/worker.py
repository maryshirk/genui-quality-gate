from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any
from uuid import UUID

import psutil
from fastapi import FastAPI
from faststream import AckPolicy
from faststream.rabbit import RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from opentelemetry.trace import SpanKind, StatusCode, get_current_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import ValidationError

from genui_quality_gate import telemetry
from genui_quality_gate.config import Settings
from genui_quality_gate.database import close_database_pool, create_database_pool
from genui_quality_gate.inference import LoadedModel, Prediction, load_model
from genui_quality_gate.logging import LOGGER_NAME, configure_logging, task_context
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
        task_id, digest = message_identity(body, message.correlation_id or message.message_id)
        context = TraceContextTextMapPropagator().extract(message.headers)
        with (
            task_context(str(task_id)),
            telemetry.span(
                "process",
                kind=SpanKind.CONSUMER,
                context=context,
                attributes={
                    "task.id": str(task_id),
                    "messaging.destination.name": settings.queue_name,
                },
            ) as span,
        ):
            timestamp = message.headers.get("genui-published-at")
            if isinstance(timestamp, str):
                with suppress(ValueError):
                    span.set_attribute(
                        "messaging.queue.wait_seconds", max(0, time.time() - float(timestamp))
                    )
            redelivered = bool(message.raw_message.redelivered)
            span.set_attribute("messaging.redelivered", redelivered)
            if redelivered:
                telemetry.active.instruments.redelivered.add(1)
            await process(body, message, task_id, digest)

    async def process(body: Any, message: RabbitMessage, task_id: UUID, digest: str) -> None:
        signals = telemetry.active.instruments
        counters.received += 1
        counters.active += 1
        signals.inflight.set(counters.active)
        created_at = datetime.now(UTC)
        try:
            envelope = TaskEnvelope.model_validate(body)
            created_at = envelope.created_at
        except ValidationError:
            envelope = None

        async def work() -> Prediction:
            if envelope is None:
                raise ValueError("Invalid task payload")
            return await predict(envelope)

        try:
            outcome = await store.execute(task_id, digest, work, created_at)
            span = get_current_span()
            span.set_attribute("messaging.delivery.attempt", outcome.attempts)
            if outcome.state == "retry":
                counters.retries += 1
                signals.processed.add(1, {"outcome": "error"})
                logger.warning("task_retry", extra={"attempt": outcome.attempts})
                await asyncio.sleep(settings.retry_delay_seconds)
                await message.nack(requeue=True)
            elif outcome.state in {"failed", "conflict"}:
                span.set_status(StatusCode.ERROR)
                if outcome.state == "conflict":
                    telemetry.failure(span, ValueError("IdempotencyConflict"))
                with telemetry.span("publish DLQ", kind=SpanKind.PRODUCER):
                    headers: dict[str, Any] = {
                        "attempts": outcome.attempts,
                        "error_type": outcome.error_type,
                    }
                    TraceContextTextMapPropagator().inject(headers)
                    await broker.publish(
                        body,
                        queue=dead_letter_queue(settings).name,
                        persist=True,
                        mandatory=True,
                        message_id=str(task_id),
                        correlation_id=str(task_id),
                        headers=headers,
                        timeout=settings.publish_timeout_seconds,
                    )
                await message.ack()
                span.add_event("task.dead_lettered")
                counters.failed += 1
                signals.processed.add(1, {"outcome": "dlq"})
                logger.error(
                    "task_dead_lettered",
                    extra={"attempt": outcome.attempts},
                )
            else:
                await message.ack()
                counters.succeeded += 1
                if outcome.end_to_end_seconds is not None:
                    signals.processed.add(1, {"outcome": "ok"})
                    signals.task_time.record(outcome.end_to_end_seconds)
                logger.info("task_succeeded", extra={"attempt": outcome.attempts})
        except Exception as exc:
            telemetry.failure(get_current_span(), exc)
            signals.processed.add(1, {"outcome": "error"})
            logger.exception("task_delivery_failed")
            await asyncio.sleep(settings.retry_delay_seconds)
            await message.nack(requeue=True)
        finally:
            counters.active -= 1
            signals.inflight.set(counters.active)

    return handle


def delayed_prediction(model: LoadedModel, task: TaskEnvelope, delay: float) -> Prediction:
    started = time.perf_counter()
    with telemetry.span(
        "inference",
        attributes={"model.alias": model.metadata.alias, "model.version": model.metadata.version},
    ):
        try:
            time.sleep(delay)
            return model.predict_sync(**task.payload.model_dump())
        finally:
            telemetry.active.instruments.inference.record(time.perf_counter() - started)


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
        async with telemetry.observe("worker") as signals, resources(application, signals.service):
            yield

    @asynccontextmanager
    async def resources(application: FastAPI, service: str) -> AsyncIterator[None]:
        configure_logging(service)
        pool = await create_database_pool(configuration)
        executor = ThreadPoolExecutor(max_workers=configuration.worker_concurrency)
        try:
            with telemetry.span(
                "model.load", attributes={"model.alias": configuration.model_alias}
            ) as span:
                try:
                    model = await asyncio.to_thread(model_loader, configuration)
                except Exception:
                    logger.exception(
                        "model_load_failed", extra={"model_alias": configuration.model_alias}
                    )
                    raise
                span.set_attribute("model.version", model.metadata.version)
                logger.info(
                    "model_loaded",
                    extra={
                        "model_alias": model.metadata.alias,
                        "model_version": model.metadata.version,
                    },
                )
            application.state.model = model
            store = PostgresTaskStore(pool, configuration.max_attempts, configuration.queue_name)
            await store.initialize()

            async def predict(task: TaskEnvelope) -> Prediction:
                return await asyncio.get_running_loop().run_in_executor(
                    executor,
                    partial(
                        copy_context().run,
                        delayed_prediction,
                        model,
                        task,
                        configuration.inference_delay_seconds,
                    ),
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

    configure_logging("worker")
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
