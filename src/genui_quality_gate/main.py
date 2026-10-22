import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from genui_quality_gate.api import router
from genui_quality_gate.config import Settings
from genui_quality_gate.database import DatabasePool, close_database_pool, create_database_pool
from genui_quality_gate.logging import LOGGER_NAME, configure_logging
from genui_quality_gate.middleware import RequestLoggingMiddleware
from genui_quality_gate.queue import TaskPublisher, create_broker, declare_topology
from genui_quality_gate.task_store import PostgresTaskStore
from genui_quality_gate.version import application_version

type PoolFactory = Callable[[Settings], Awaitable[DatabasePool]]
logger = logging.getLogger(LOGGER_NAME)


def create_app(
    settings: Settings | None = None,
    *,
    pool_factory: PoolFactory = create_database_pool,
    publisher_factory: Callable[..., TaskPublisher] = TaskPublisher,
) -> FastAPI:
    configuration = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        pool: DatabasePool | None = None
        try:
            async with asyncio.timeout(configuration.health_timeout_seconds):
                pool = await pool_factory(configuration)
        except Exception as exc:
            logger.error("database_pool_unavailable", extra={"error_type": type(exc).__name__})
        application.state.pool = pool
        broker = None
        try:
            if configuration.queue_enabled:
                try:
                    broker = create_broker(configuration)
                    async with asyncio.timeout(configuration.health_timeout_seconds):
                        await broker.connect()
                        await declare_topology(broker, configuration)
                    application.state.publisher = publisher_factory(broker, configuration)
                except Exception as exc:
                    logger.error("broker_unavailable", extra={"error_type": type(exc).__name__})
            if pool is not None:
                application.state.task_store = PostgresTaskStore(pool, configuration.max_attempts)
            logger.info("application_started")
            yield
        finally:
            application.state.publisher = None
            if broker is not None:
                await broker.stop()
            if pool is not None:
                await close_database_pool(pool, configuration.db_shutdown_timeout_seconds)
            application.state.pool = None
            logger.info("application_stopped")

    application = FastAPI(
        title="GenUI Quality Gate",
        version=application_version(),
        description="Asynchronous tasks for pairwise evaluation of generated interfaces.",
        lifespan=lifespan,
    )
    application.state.settings = configuration
    application.state.pool = None
    application.state.publisher = None
    application.state.task_store = None
    application.add_middleware(RequestLoggingMiddleware)
    application.include_router(router)
    return application


app = create_app()
