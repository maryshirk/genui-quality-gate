import asyncio
import os
from contextlib import AsyncExitStack

import asyncpg
import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from genui_quality_gate.config import Settings
from genui_quality_gate.main import create_app

pytestmark = pytest.mark.integration


def database_url() -> str:
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")
    return dsn


async def test_real_database_version_and_concurrent_health_checks() -> None:
    dsn = database_url()
    connection = await asyncpg.connect(dsn, timeout=5, command_timeout=5)
    try:
        expected_version = await connection.fetchval("SELECT current_setting('server_version')")
    finally:
        await connection.close(timeout=5)

    app = create_app(Settings(database_url=SecretStr(dsn)))
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            replies = await asyncio.gather(*(client.get("/api/v1/health") for _ in range(8)))
            for reply in replies:
                assert reply.status_code == 200
                database = reply.json()["components"]["postgresql"]
                assert database["version"] == expected_version
                assert database["status"] == "ok"
                assert database["latency_ms"] >= 0
                assert database["error"] is None
            assert (await client.get("/healthz")).json() == {"status": "ok"}

    assert app.state.pool is None


async def test_real_pool_recovers_after_acquisition_timeout() -> None:
    settings = Settings(database_url=SecretStr(database_url()))
    app = create_app(settings)
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pool = app.state.pool
            assert pool is not None
            async with AsyncExitStack() as held_connections:
                async with asyncio.timeout(10):
                    for _ in range(pool.get_max_size()):
                        await held_connections.enter_async_context(pool.acquire())
                app.state.settings = settings.model_copy(update={"health_timeout_seconds": 0.05})
                reply = await client.get("/api/v1/health")
                assert reply.status_code == 503
                assert reply.json()["components"]["postgresql"]["error"] == "timeout"
                assert (await client.get("/healthz")).status_code == 200

            app.state.settings = settings
            recovered = await client.get("/api/v1/health")
            assert recovered.status_code == 200
            assert recovered.json()["components"]["postgresql"]["status"] == "ok"

    assert app.state.pool is None
