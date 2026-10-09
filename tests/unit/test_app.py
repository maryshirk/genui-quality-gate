from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from time import perf_counter
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr, ValidationError

from genui_quality_gate.config import Settings
from genui_quality_gate.database import DatabaseConnection, DatabasePool, create_database_pool
from genui_quality_gate.main import create_app


class FakeConnection:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool

    async def fetchval(self, query: str) -> object:
        assert query == "SELECT current_setting('server_version')"
        await asyncio.sleep(self.pool.query_delay)
        if self.pool.error is not None:
            raise self.pool.error
        return self.pool.server_version


class FakePool:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.server_version: object = "17.6"
        self.acquire_delay = 0.0
        self.query_delay = 0.0
        self.close_delay = 0.0
        self.close_error: Exception | None = None
        self.acquisitions = 0
        self.releases = 0
        self.closed = False
        self.terminated = False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DatabaseConnection]:
        self.acquisitions += 1
        await asyncio.sleep(self.acquire_delay)
        try:
            yield FakeConnection(self)
        finally:
            self.releases += 1

    async def close(self) -> None:
        await asyncio.sleep(self.close_delay)
        if self.close_error is not None:
            raise self.close_error
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True


def test_settings_are_loaded_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENUI_DATABASE_URL", "postgresql://user:private@db/genui")
    monkeypatch.setenv("GENUI_HEALTH_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("GENUI_BUILD_SHA", "abc123")
    settings = Settings()
    assert settings.database_url.get_secret_value() == "postgresql://user:private@db/genui"
    assert "private" not in repr(settings)
    assert settings.health_timeout_seconds == 0.5
    assert settings.build_sha == "abc123"


@pytest.mark.parametrize("timeout", [0, -1, 31])
def test_settings_reject_unbounded_timeouts(timeout: float) -> None:
    with pytest.raises(ValidationError):
        Settings(health_timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_liveness_does_not_touch_database_and_version_matches_package() -> None:
    pool = FakePool()
    pool.error = ConnectionError("offline")
    settings = Settings(build_sha="a" * 40)
    factory = AsyncMock(return_value=pool)
    app = create_app(settings, pool_factory=factory)
    factory.assert_not_called()
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        healthz = await client.get("/healthz")
        release = await client.get("/api/v1/version")
        assert healthz.status_code == 200
        assert healthz.json() == {"status": "ok"}
        assert release.json() == {
            "service": "genui-quality-gate",
            "version": version("genui-quality-gate"),
            "revision": "a" * 40,
        }
        assert app.version == version("genui-quality-gate")
        assert pool.acquisitions == 0
    assert pool.closed
    assert app.state.pool is None


@pytest.mark.asyncio
async def test_version_reads_distribution_metadata_instead_of_api_version() -> None:
    with patch("genui_quality_gate.version.version", return_value="4.5.6") as metadata_version:
        app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    metadata_version.assert_called_once_with("genui-quality-gate")
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/api/v1/version")
        assert response.json()["version"] == "4.5.6"


@pytest.mark.asyncio
async def test_health_reports_server_version_and_end_to_end_latency() -> None:
    pool = FakePool()
    pool.acquire_delay = 0.01
    pool.query_delay = 0.01
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=pool))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        component = response.json()["components"]["postgresql"]
        assert component["status"] == "ok"
        assert component["version"] == "17.6"
        assert component["error"] is None
        assert component["latency_ms"] >= 20
        assert pool.releases == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (ConnectionRefusedError("secret"), "connection_error"),
        (RuntimeError("secret"), "database_error"),
    ],
)
async def test_database_failure_is_sanitized_and_recovers(
    failure: Exception, expected_error: str, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = FakePool()
    pool.error = failure
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=pool))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
        component = response.json()["components"]["postgresql"]
        assert component["status"] == "error"
        assert component["version"] is None
        assert component["error"] == expected_error
        assert component["latency_ms"] >= 0
        assert "secret" not in response.text
        assert (await client.get("/healthz")).status_code == 200
        pool.error = None
        assert (await client.get("/api/v1/health")).status_code == 200
    assert pool.releases == 2
    assert "secret" not in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_phase", ["acquire_delay", "query_delay"])
async def test_timeout_covers_acquisition_and_query(slow_phase: str) -> None:
    pool = FakePool()
    setattr(pool, slow_phase, 1)
    app = create_app(
        Settings(health_timeout_seconds=0.025), pool_factory=AsyncMock(return_value=pool)
    )
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        started = perf_counter()
        response = await client.get("/api/v1/health")
        assert perf_counter() - started < 0.5
        assert response.status_code == 503
        assert response.json()["components"]["postgresql"]["error"] == "timeout"
        assert (await client.get("/healthz")).status_code == 200
    assert pool.closed


@pytest.mark.asyncio
async def test_slow_database_does_not_block_liveness() -> None:
    pool = FakePool()
    pool.query_delay = 0.15
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=pool))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        check = asyncio.create_task(client.get("/api/v1/health"))
        await asyncio.sleep(0.01)
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert not check.done()
        assert (await check).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_version", [None, "", 123])
async def test_invalid_database_version_marks_dependency_unhealthy(bad_version: object) -> None:
    pool = FakePool()
    pool.server_version = bad_version
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=pool))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["components"]["postgresql"]["error"] == "database_error"


@pytest.mark.asyncio
async def test_pool_initialization_failure_keeps_app_alive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(side_effect=RuntimeError("secret DSN")))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/healthz")).status_code == 200
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["components"]["postgresql"]["error"] == "unavailable"
    assert "secret DSN" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_pool_initialization_has_a_deadline() -> None:
    async def hanging_factory(_settings: Settings) -> DatabasePool:
        await asyncio.sleep(1)
        return FakePool()

    app = create_app(Settings(health_timeout_seconds=0.01), pool_factory=hanging_factory)
    started = perf_counter()
    async with LifespanManager(app):
        assert app.state.pool is None
        assert perf_counter() - started < 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("close_error", [None, RuntimeError("private credentials")])
async def test_shutdown_is_bounded_and_terminates_unhealthy_pool(
    close_error: Exception | None, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = FakePool()
    if close_error is None:
        pool.close_delay = 1
    else:
        pool.close_error = close_error
    app = create_app(
        Settings(db_shutdown_timeout_seconds=0.01), pool_factory=AsyncMock(return_value=pool)
    )
    started = perf_counter()
    async with LifespanManager(app):
        pass
    assert perf_counter() - started < 0.5
    assert pool.terminated
    assert "private credentials" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_real_pool_factory_uses_lazy_bounded_pool() -> None:
    pool = FakePool()
    settings = Settings(database_url=SecretStr("postgresql://local/genui"))
    with patch(
        "genui_quality_gate.database.asyncpg.create_pool", new_callable=AsyncMock
    ) as factory:
        factory.return_value = pool
        assert await create_database_pool(settings) is pool
        factory.assert_awaited_once_with(
            dsn="postgresql://local/genui", min_size=0, max_size=5, timeout=2.0, command_timeout=2.0
        )


@pytest.mark.asyncio
async def test_request_logging_has_correlation_and_no_sensitive_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = create_app(pool_factory=AsyncMock(return_value=FakePool()))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get(
            "/healthz?token=private-token", headers={"x-request-id": "defense-demo-1"}
        )
        assert response.headers["x-request-id"] == "defense-demo-1"
        unknown = await client.get("/private-path-secret?token=hidden")
        assert unknown.status_code == 404
    output = capsys.readouterr().out
    assert "private-token" not in output
    assert "private-path-secret" not in output
    assert "hidden" not in output
    records = [json.loads(line) for line in output.splitlines()]
    access = [record for record in records if record["event"] == "request_completed"]
    assert len(access) == 2
    assert access[0]["request_id"] == "defense-demo-1"
    assert access[0]["method"] == "GET"
    assert access[0]["route"] == "/healthz"
    assert access[0]["status_code"] == 200
    assert access[0]["duration_ms"] >= 0
    assert access[1]["route"] == "<unmatched>"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", ["contains spaces", "x" * 65, "line\nbreak"])
async def test_invalid_request_id_is_replaced(invalid_id: str) -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/healthz", headers={"x-request-id": invalid_id})
        assert len(response.headers["x-request-id"]) == 32
        assert response.headers["x-request-id"] != invalid_id


@pytest.mark.asyncio
async def test_unexpected_exception_is_sanitized_and_logged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = create_app(Settings(), pool_factory=AsyncMock(return_value=FakePool()))

    @app.get("/explode")
    async def explode() -> None:
        raise RuntimeError("postgresql://user:private-password@host/db")

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/explode", headers={"x-request-id": "test-error"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error", "request_id": "test-error"}
        assert response.headers["x-request-id"] == "test-error"
    output = capsys.readouterr().out
    assert "private-password" not in output
    assert "request_failed" in output
    records = [json.loads(line) for line in output.splitlines()]
    failure = next(record for record in records if record["event"] == "request_failed")
    assert failure["error_type"] == "RuntimeError"
