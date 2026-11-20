import asyncio
import logging
from contextlib import AbstractAsyncContextManager
from time import perf_counter
from typing import Literal, Protocol, cast

import asyncpg
from pydantic import BaseModel, Field

from genui_quality_gate.config import Settings
from genui_quality_gate.logging import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
VERSION_QUERY = "SELECT current_setting('server_version')"
type HealthError = Literal["timeout", "connection_error", "database_error", "unavailable"]


class DatabaseConnection(Protocol):
    async def fetchval(self, query: str) -> object: ...


class DatabasePool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[DatabaseConnection]: ...

    async def close(self) -> None: ...

    def terminate(self) -> None: ...


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    version: str | None = None
    latency_ms: float = Field(ge=0)
    error: HealthError | None = None


async def create_database_pool(settings: Settings) -> DatabasePool:
    pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(),
        min_size=0,
        max_size=5,
        timeout=settings.health_timeout_seconds,
        command_timeout=settings.health_timeout_seconds,
    )
    return cast(DatabasePool, pool)


async def close_database_pool(pool: DatabasePool, timeout_seconds: float) -> None:
    try:
        async with asyncio.timeout(timeout_seconds):
            await pool.close()
    except Exception as exc:
        pool.terminate()
        logger.warning("database_shutdown_forced", extra={"error_type": type(exc).__name__})


async def check_database(pool: DatabasePool | None, timeout_seconds: float) -> ComponentHealth:
    started = perf_counter()
    error: HealthError | None = None
    server_version: str | None = None
    try:
        if pool is None:
            error = "unavailable"
        else:
            async with asyncio.timeout(timeout_seconds), pool.acquire() as connection:
                result = await connection.fetchval(VERSION_QUERY)
                if not isinstance(result, str) or not result:
                    raise ValueError("PostgreSQL did not return a server version")
                server_version = result
    except TimeoutError:
        error = "timeout"
    except OSError:
        error = "connection_error"
    except Exception:
        error = "database_error"

    if error is not None:
        logger.warning(
            "dependency_unhealthy", extra={"component": "postgresql", "error_type": error}
        )
    return ComponentHealth(
        status="ok" if error is None else "error",
        version=server_version if error is None else None,
        latency_ms=round((perf_counter() - started) * 1000, 3),
        error=error,
    )
