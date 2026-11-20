import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from opentelemetry.trace import get_current_span

from genui_quality_gate.telemetry import stack_frames

LOGGER_NAME = "genui_quality_gate"
_TASK_ID: ContextVar[str | None] = ContextVar("task_id", default=None)
_EXTRA_FIELDS = (
    "request_id",
    "method",
    "route",
    "status_code",
    "duration_ms",
    "error_type",
    "component",
    "model_name",
    "model_version",
    "model_alias",
    "model_run_id",
    "task_id",
    "attempt",
    "active_tasks",
    "prefetch_count",
    "worker_concurrency",
    "accepted",
    "rejected",
    "duration_seconds",
    "wait_seconds",
    "batch_id",
    "broker_rejected",
)


@contextmanager
def task_context(task_id: str | None) -> Iterator[None]:
    token = _TASK_ID.set(task_id)
    try:
        yield
    finally:
        _TASK_ID.reset(token)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str = "api") -> None:
        super().__init__()
        self.service = service
        self.env = os.getenv("GENUI_ENVIRONMENT", "local")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage()
            if record.name.startswith(LOGGER_NAME)
            else "library_event",
            "logger": record.name,
            "service": self.service,
            "env": self.env,
        }
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        task_id = _TASK_ID.get()
        if task_id is not None:
            payload["task_id"] = task_id
        context = get_current_span().get_span_context()
        if context.is_valid:
            payload["trace_id"] = f"{context.trace_id:032x}"
            payload["span_id"] = f"{context.span_id:016x}"
        if record.exc_info is not None and record.exc_info[1] is not None:
            error = record.exc_info[1]
            payload["error_type"] = type(error).__name__
            payload["stacktrace"] = stack_frames(error)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(service: str = "api") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    for name in (
        "",
        LOGGER_NAME,
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "aio_pika",
        "aiormq",
        "mlflow",
        "opentelemetry",
        "faststream",
    ):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False
