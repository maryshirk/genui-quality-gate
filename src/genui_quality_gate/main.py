import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from genui_quality_gate.api import router
from genui_quality_gate.config import Settings
from genui_quality_gate.database import DatabasePool, close_database_pool, create_database_pool
from genui_quality_gate.inference import LoadedModel, load_model
from genui_quality_gate.logging import LOGGER_NAME, configure_logging
from genui_quality_gate.middleware import RequestLoggingMiddleware
from genui_quality_gate.version import application_version

type PoolFactory = Callable[[Settings], Awaitable[DatabasePool]]
logger = logging.getLogger(LOGGER_NAME)


def create_app(
    settings: Settings | None = None,
    *,
    pool_factory: PoolFactory = create_database_pool,
    model_loader: Callable[[Settings], LoadedModel] = load_model,
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
        try:
            if configuration.model_enabled:
                try:
                    model = await asyncio.to_thread(model_loader, configuration)
                except Exception as exc:
                    logger.error("model_load_failed", extra={"error_type": type(exc).__name__})
                    raise RuntimeError("Unable to load the configured MLflow model") from None
                application.state.model = model
                logger.info(
                    "model_loaded",
                    extra={
                        "model_name": model.metadata.name,
                        "model_version": model.metadata.version,
                        "model_alias": model.metadata.alias,
                        "model_run_id": model.metadata.run_id,
                    },
                )
            logger.info("application_started")
            yield
        finally:
            application.state.model = None
            if pool is not None:
                await close_database_pool(pool, configuration.db_shutdown_timeout_seconds)
            application.state.pool = None
            logger.info("application_stopped")

    application = FastAPI(
        title="GenUI Quality Gate",
        version=application_version(),
        description="Service foundation for pairwise evaluation of generated interfaces.",
        lifespan=lifespan,
    )
    application.state.settings = configuration
    application.state.pool = None
    application.state.model = None
    application.add_middleware(RequestLoggingMiddleware)
    application.include_router(router)
    return application


app = create_app()
