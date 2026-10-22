from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg
import httpx
import pytest
from asgi_lifespan import LifespanManager
from faststream.rabbit.message import RabbitMessage
from pydantic import SecretStr
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from genui_quality_gate.config import Settings
from genui_quality_gate.inference import LoadedModel, Prediction
from genui_quality_gate.main import create_app
from genui_quality_gate.queue import (
    QueueUnavailable,
    TaskPublisher,
    create_broker,
    dead_letter_queue,
    declare_topology,
    task_queue,
)
from genui_quality_gate.source import publish_file
from genui_quality_gate.task_store import PostgresTaskStore
from genui_quality_gate.tasks import TaskEnvelope, TaskPayload
from genui_quality_gate.worker import WorkerMetrics, create_worker_app, register_handler
from tests.unit.test_inference import PAYLOAD, fake_estimator, model_metadata

pytestmark = pytest.mark.containers


@pytest.fixture(scope="session")
def services() -> Iterator[tuple[str, str]]:
    if os.getenv("RUN_CONTAINER_TESTS") != "1":
        pytest.skip("Set RUN_CONTAINER_TESTS=1 to run RabbitMQ Testcontainers tests")
    with (
        DockerContainer("rabbitmq:4.2-management")
        .with_exposed_ports(5672)
        .waiting_for(LogMessageWaitStrategy("Server startup complete")) as rabbit,
        DockerContainer("postgres:17-bookworm")
        .with_env("POSTGRES_PASSWORD", "pipeline-test")
        .with_env("POSTGRES_DB", "pipeline")
        .with_exposed_ports(5432)
        .waiting_for(
            LogMessageWaitStrategy("database system is ready to accept connections", times=2)
        ) as pg,
    ):
        rabbit_url = f"amqp://guest:guest@{rabbit.get_container_host_ip()}:"
        rabbit_url += f"{rabbit.get_exposed_port(5672)}/"
        dsn = f"postgresql://postgres:pipeline-test@{pg.get_container_host_ip()}:"
        dsn += f"{pg.get_exposed_port(5432)}/pipeline"
        yield rabbit_url, dsn


def settings_for(services: tuple[str, str], **kwargs: object) -> Settings:
    rabbit_url, dsn = services
    return Settings(
        rabbit_url=SecretStr(rabbit_url),
        database_url=SecretStr(dsn),
        queue_name=f"test.{uuid4().hex}",
        queue_capacity=2,
        retry_delay_seconds=0.02,
        **kwargs,
    )


async def wait_until(check) -> None:
    for _ in range(400):
        if await check():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("Expected broker state was not reached")


async def test_real_overflow_returns_503_without_pending_row(services) -> None:
    settings = settings_for(services, queue_enabled=True)
    broker = create_broker(settings)
    await broker.connect()
    pool = await asyncpg.create_pool(services[1])
    try:
        await declare_topology(broker, settings)
        store = PostgresTaskStore(pool)
        await store.initialize()
        publisher = TaskPublisher(broker, settings)
        for _ in range(2):
            await publisher.publish(TaskEnvelope(payload=TaskPayload(**PAYLOAD)))
        with pytest.raises(QueueUnavailable):
            await publisher.publish(TaskEnvelope(payload=TaskPayload(**PAYLOAD)))
        queue = await broker.declare_queue(task_queue(settings))
        await queue.declare()
        assert queue.declaration_result.message_count == 2
        app = create_app(settings)
        rejected_id = uuid4()
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            started = perf_counter()
            response = await client.post(
                "/process", json=PAYLOAD, headers={"Idempotency-Key": str(rejected_id)}
            )
            assert perf_counter() - started < 0.5
            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
        assert (
            await pool.fetchval("SELECT count(*) FROM pipeline_tasks WHERE task_id=$1", rejected_id)
            == 0
        )
    finally:
        await broker.stop()
        await pool.close()


async def test_batch_source_slows_to_worker_and_does_not_lose_tasks(
    services, tmp_path: Path
) -> None:
    settings = settings_for(services)
    consumer, producer = create_broker(settings), create_broker(settings)
    pool = await asyncpg.create_pool(services[1])
    store = PostgresTaskStore(pool, queue_name=settings.queue_name)
    await store.initialize()
    counters = WorkerMetrics()

    async def predict(task: TaskEnvelope) -> Prediction:
        await asyncio.sleep(0.06)
        return Prediction(winner="A", probability_a=0.8, model=model_metadata())

    register_handler(consumer, settings, store, predict, counters)
    source = tmp_path / "batch.jsonl"
    source.write_text((json.dumps(PAYLOAD) + "\n") * 12)
    try:
        await producer.connect()
        await declare_topology(producer, settings)
        await consumer.start()
        started = perf_counter()
        result = await publish_file(
            source, tmp_path / "checkpoint.json", TaskPublisher(producer, settings), retry_base=0.01
        )
        assert result["accepted"] == 12 and result["rejected"] > 0
        assert perf_counter() - started >= 0.4

        async def finished() -> bool:
            return counters.succeeded == 12

        await wait_until(finished)
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM pipeline_tasks WHERE queue_name=$1 AND status='SUCCEEDED'",
                settings.queue_name,
            )
            == 12
        )
    finally:
        await consumer.stop()
        await producer.stop()
        await pool.close()


async def test_redelivery_is_idempotent_and_poison_reaches_dlq(services) -> None:
    settings = settings_for(services)
    consumer, producer = create_broker(settings), create_broker(settings)
    pool = await asyncpg.create_pool(services[1])
    store = PostgresTaskStore(pool, max_attempts=3, queue_name=settings.queue_name)
    await store.initialize()
    counters = WorkerMetrics()
    predict = AsyncMock(
        return_value=Prediction(winner="A", probability_a=0.8, model=model_metadata())
    )
    register_handler(consumer, settings, store, predict, counters)
    good = TaskEnvelope(payload=TaskPayload(**PAYLOAD))
    bad_id = uuid4()
    try:
        await producer.connect()
        await declare_topology(producer, settings)
        await consumer.start()
        for _ in range(2):
            await producer.publish(
                good.model_dump(mode="json"), queue=settings.queue_name, persist=True
            )
        await producer.publish(
            {"task_id": str(bad_id), "payload": {"prompt": ""}},
            queue=settings.queue_name,
            persist=True,
            message_id=str(bad_id),
        )

        async def finished() -> bool:
            return counters.succeeded == 2 and counters.failed == 1

        await wait_until(finished)
        predict.assert_awaited_once()
        assert (await store.get(good.task_id)).attempts == 1
        failed = await store.get(bad_id)
        assert failed.status == "FAILED" and failed.attempts == 3
        queue = await producer.declare_queue(dead_letter_queue(settings))
        message = await queue.get(timeout=2)
        assert message.headers["attempts"] == 3
        assert json.loads(message.body)["task_id"] == str(bad_id)
        await message.ack()
    finally:
        await consumer.stop()
        await producer.stop()
        await pool.close()


async def test_worker_loads_once_and_health_responds_during_sync_inference(services) -> None:
    settings = settings_for(services, inference_delay_seconds=0)
    started, release = threading.Event(), threading.Event()
    estimator = fake_estimator()

    def slow(values):
        started.set()
        if not release.wait(5):
            raise TimeoutError()
        return [[0.2, 0.8]]

    estimator.predict_proba.side_effect = slow
    loader = Mock(return_value=LoadedModel(estimator, model_metadata("17")))
    app = create_worker_app(settings, model_loader=loader)
    producer = create_broker(settings)
    try:
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            await producer.connect()
            task = TaskEnvelope(payload=TaskPayload(**PAYLOAD))
            await TaskPublisher(producer, settings).publish(task)
            assert await asyncio.to_thread(started.wait, 2)
            response = await asyncio.wait_for(client.get("/healthz"), 0.2)
            assert response.json()["model"]["version"] == "17"
            assert response.json()["active"] == 1
            stop = asyncio.create_task(app.state.broker.stop())
            await asyncio.sleep(0.05)
            assert not stop.done()
            release.set()
            await asyncio.wait_for(stop, 3)
        loader.assert_called_once()
    finally:
        release.set()
        await producer.stop()


async def test_concurrent_duplicates_share_one_committed_result(services) -> None:
    from datetime import UTC, datetime

    settings = settings_for(services)
    pool = await asyncpg.create_pool(services[1])
    store = PostgresTaskStore(pool, queue_name=settings.queue_name)
    await store.initialize()
    calls = 0

    async def work() -> Prediction:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return Prediction(winner="A", probability_a=0.8, model=model_metadata())

    try:
        key = uuid4()
        outcomes = await asyncio.gather(
            *(store.execute(key, "same-input", work, datetime.now(UTC)) for _ in range(3))
        )
        assert calls == 1
        assert all(result.state == "succeeded" for result in outcomes)
        conflict = await store.execute(key, "different-input", work, datetime.now(UTC))
        assert conflict.state == "conflict"
        assert calls == 1
    finally:
        await pool.close()


async def test_ack_is_sent_after_result_is_visible_to_another_connection(
    services, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(services)
    pool = await asyncpg.create_pool(services[1])
    store = PostgresTaskStore(pool, queue_name=settings.queue_name)
    await store.initialize()
    consumer, producer = create_broker(settings), create_broker(settings)
    acknowledged = asyncio.Event()
    original_ack = RabbitMessage.ack
    task = TaskEnvelope(payload=TaskPayload(**PAYLOAD))

    async def checked_ack(message, **kwargs) -> None:
        row = await store.get(task.task_id)
        assert row is not None and row.status == "SUCCEEDED"
        await original_ack(message, **kwargs)
        acknowledged.set()

    monkeypatch.setattr(RabbitMessage, "ack", checked_ack)
    predict = AsyncMock(
        return_value=Prediction(winner="A", probability_a=0.8, model=model_metadata())
    )
    register_handler(consumer, settings, store, predict)
    try:
        await producer.connect()
        await declare_topology(producer, settings)
        await consumer.start()
        await TaskPublisher(producer, settings).publish(task)
        await asyncio.wait_for(acknowledged.wait(), 3)
    finally:
        await consumer.stop()
        await producer.stop()
        await pool.close()


async def test_retry_during_backpressure_does_not_drop_other_tasks(
    services, tmp_path: Path
) -> None:
    settings = settings_for(services)
    consumer, producer = create_broker(settings), create_broker(settings)
    pool = await asyncpg.create_pool(services[1])
    store = PostgresTaskStore(pool, queue_name=settings.queue_name)
    await store.initialize()
    counters = WorkerMetrics()

    async def predict(task: TaskEnvelope) -> Prediction:
        await asyncio.sleep(0.03)
        if task.payload.prompt == "poison":
            raise ValueError("Test failure")
        return Prediction(winner="A", probability_a=0.8, model=model_metadata())

    register_handler(consumer, settings, store, predict, counters)
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        json.dumps(PAYLOAD | {"prompt": "poison"}) + "\n" + (json.dumps(PAYLOAD) + "\n") * 12
    )
    try:
        await producer.connect()
        await declare_topology(producer, settings)
        await consumer.start()
        result = await publish_file(
            source, tmp_path / "checkpoint.json", TaskPublisher(producer, settings), retry_base=0.01
        )
        assert result["accepted"] == 13

        async def finished() -> bool:
            return counters.succeeded == 12 and counters.failed == 1

        await wait_until(finished)
        assert counters.retries == 2
    finally:
        await consumer.stop()
        await producer.stop()
        await pool.close()
