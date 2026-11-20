import asyncio
import json
import logging
import sys
from contextvars import copy_context
from unittest.mock import AsyncMock

import pytest
from faststream.rabbit import TestRabbitBroker
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from genui_quality_gate import telemetry
from genui_quality_gate.config import Settings
from genui_quality_gate.logging import LOGGER_NAME, JsonFormatter, task_context
from genui_quality_gate.queue import TaskPublisher, create_broker
from genui_quality_gate.task_store import TaskOutcome
from genui_quality_gate.tasks import TaskEnvelope, TaskPayload
from genui_quality_gate.worker import register_handler
from tests.unit.test_inference import PAYLOAD


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry.active, "tracer", provider.get_tracer("test"))
    return exporter


async def test_task_context_survives_rabbit_delivery_and_executor(
    spans: InMemorySpanExporter,
) -> None:
    settings = Settings()
    broker = create_broker(settings)
    store = AsyncMock()
    store.execute.return_value = TaskOutcome("succeeded", 1)
    task = TaskEnvelope(payload=TaskPayload(**PAYLOAD))

    def infer() -> None:
        with telemetry.span("test inference"):
            record = logging.LogRecord(LOGGER_NAME, logging.INFO, __file__, 0, "predict", (), None)
            payload = json.loads(JsonFormatter("worker").format(record))
            assert payload["task_id"] == str(task.task_id)
            assert len(payload["trace_id"]) == 32
            assert len(payload["span_id"]) == 16

    async def execute(task_id, digest, work, created_at):
        await work()
        return TaskOutcome("succeeded", 1)

    async def predict(_task):
        await asyncio.get_running_loop().run_in_executor(None, copy_context().run, infer)

    store.execute.side_effect = execute
    register_handler(broker, settings, store, predict)
    async with TestRabbitBroker(broker):
        with telemetry.span("request"):
            await TaskPublisher(broker, settings).publish(task)

    by_name = {span.name: span for span in spans.get_finished_spans()}
    for child, parent in (
        ("test inference", "process"),
        ("process", "publish"),
        ("publish", "request"),
    ):
        assert by_name[child].parent.span_id == by_name[parent].context.span_id
        assert by_name[child].context.trace_id == by_name[parent].context.trace_id
    assert by_name["process"].attributes["messaging.delivery.attempt"] == 1


def test_exception_message_is_absent_in_log_and_trace(spans: InMemorySpanExporter) -> None:
    secret = "postgresql://user:secret@hostname"
    with pytest.raises(ValueError), task_context("task"), telemetry.span("failure"):
        try:
            raise ValueError(secret)
        except ValueError:
            record = logging.LogRecord(
                LOGGER_NAME, logging.ERROR, __file__, 0, "failed", (), sys.exc_info()
            )
            line = JsonFormatter().format(record)
            assert secret not in line
            assert json.loads(line)["stacktrace"]
            raise
    span = spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == "exception"
    assert secret not in str(span.events[0].attributes)


def test_resource_defaults_can_be_overridden_by_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "custom-worker")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.version=demo,deployment.environment.name=test,service.instance.id=replica-2",
    )
    signals = telemetry.Telemetry("worker", enabled=True)
    try:
        assert signals.provider is not None
        attributes = signals.provider.resource.attributes
        assert attributes["service.name"] == "custom-worker"
        assert attributes["service.version"] == "demo"
        assert attributes["deployment.environment.name"] == "test"
        assert attributes["service.instance.id"] == "replica-2"
    finally:
        signals.shutdown()
