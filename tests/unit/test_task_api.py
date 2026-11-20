import asyncio
from time import perf_counter
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager

from genui_quality_gate.config import Settings
from genui_quality_gate.main import create_app
from genui_quality_gate.queue import QueueUnavailable
from tests.unit.test_app import FakePool
from tests.unit.test_inference import PAYLOAD


@pytest.mark.parametrize("route", ["/process", "/api/v1/tasks"])
async def test_task_is_published_before_202_and_idempotency_key_is_preserved(route: str) -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    publisher = AsyncMock()
    task_id = uuid4()
    async with LifespanManager(app):
        app.state.publisher = publisher
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                route, json=PAYLOAD, headers={"Idempotency-Key": str(task_id)}
            )
    assert response.status_code == 202
    assert response.json()["task_id"] == str(task_id)
    assert response.headers["location"] == f"/api/v1/tasks/{task_id}"
    task = publisher.publish.call_args.args[0]
    assert task.task_id == task_id
    assert task.payload.model_dump() == PAYLOAD
    publisher.publish.assert_awaited_once()


async def test_overload_is_fast_503_without_creating_database_record() -> None:
    pool = FakePool()
    app = create_app(
        Settings(inference_delay_seconds=2.1), pool_factory=AsyncMock(return_value=pool)
    )
    publisher = AsyncMock()
    publisher.publish.side_effect = QueueUnavailable("full")
    async with LifespanManager(app):
        app.state.publisher = publisher
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            start = perf_counter()
            response = await client.post("/process", json=PAYLOAD)
            assert perf_counter() - start < 0.2
            assert (await client.get("/healthz")).status_code == 200
    assert response.status_code == 503
    assert response.headers["retry-after"] == "3"
    assert pool.acquisitions == 0


async def test_unavailable_broker_returns_retry_after() -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/process", json=PAYLOAD)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


@pytest.mark.parametrize(
    "updates",
    [{"prompt": ""}, {"prompt": " "}, {"html_a": "x" * 1_000_001}, {"html_b": None}, {"x": 1}],
)
async def test_invalid_request_is_not_published(updates: dict[str, object]) -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    publisher = AsyncMock()
    async with LifespanManager(app):
        app.state.publisher = publisher
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/process", json=PAYLOAD | updates)).status_code == 422
    publisher.publish.assert_not_called()


async def test_result_lookup_handles_missing_storage_and_unknown_id() -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        app.state.task_store = None
        assert (await client.get(f"/api/v1/tasks/{uuid4()}")).status_code == 503
        app.state.task_store = AsyncMock()
        app.state.task_store.get.return_value = None
        assert (await client.get(f"/api/v1/tasks/{uuid4()}")).status_code == 404
        app.state.task_store.get.side_effect = ConnectionError()
        assert (await client.get(f"/api/v1/tasks/{uuid4()}")).status_code == 503


async def test_api_does_not_wait_for_worker_result() -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    published = asyncio.Event()
    async with LifespanManager(app):
        app.state.publisher = AsyncMock()
        app.state.publisher.publish.side_effect = lambda _: published.set()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await asyncio.wait_for(client.post("/process", json=PAYLOAD), 0.2)
    assert response.status_code == 202
    assert published.is_set()
