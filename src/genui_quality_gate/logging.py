import json
import logging
import sys
from datetime import UTC, datetime

LOGGER_NAME = "genui_quality_gate"
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
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
