import asyncio
import os
import socket
import traceback
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Literal

import psutil
from opentelemetry import metrics, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import OTELResourceDetector, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, StatusCode
from opentelemetry.util.types import Attributes

from genui_quality_gate.version import application_version

SCOPE = "genui_quality_gate"
Service = Literal["api", "source", "worker"]
LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 5, 10, 30)
TASK_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 12, 20, 30, 60)


class Instruments:
    def __init__(self, meter: metrics.Meter) -> None:
        self.requests = meter.create_counter("genui_http_requests")
        self.http_time = meter.create_histogram(
            "genui_http_duration_seconds", explicit_bucket_boundaries_advisory=LATENCY_BUCKETS
        )
        self.published = meter.create_counter("genui_source_published")
        self.rejected = meter.create_counter("genui_source_rejected")
        self.waited = meter.create_counter("genui_source_wait_seconds")
        self.processed = meter.create_counter("genui_worker_tasks")
        self.redelivered = meter.create_counter("genui_worker_redelivered")
        self.inflight = meter.create_gauge("genui_worker_inflight")
        self.inference = meter.create_histogram(
            "genui_inference_duration_seconds", explicit_bucket_boundaries_advisory=LATENCY_BUCKETS
        )
        self.task_time = meter.create_histogram(
            "genui_task_duration_seconds", explicit_bucket_boundaries_advisory=TASK_BUCKETS
        )
        self.ready_limit = meter.create_gauge("genui_queue_ready_limit")


class Telemetry:
    def __init__(
        self,
        service: Service = "api",
        capacity: int | None = None,
        queue: str = "",
        *,
        enabled: bool = False,
    ) -> None:
        self.service = os.getenv("OTEL_SERVICE_NAME") or service
        self.provider: TracerProvider | None = None
        self.meter_provider: MeterProvider | None = None
        self.tracer: trace.Tracer = trace.NoOpTracer()
        meter: metrics.Meter = metrics.NoOpMeter(SCOPE)
        if enabled:
            resource = Resource.create(
                {
                    "service.name": self.service,
                    "service.version": application_version(),
                    "deployment.environment.name": "local",
                    "service.instance.id": socket.gethostname(),
                }
            ).merge(OTELResourceDetector().detect())
            self.service = str(resource.attributes["service.name"])
            if os.getenv("OTEL_TRACES_EXPORTER") != "none":
                self.provider = TracerProvider(resource=resource, shutdown_on_exit=False)
                self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                self.tracer = self.provider.get_tracer(SCOPE)
            if os.getenv("OTEL_METRICS_EXPORTER") != "none":
                labels = {"service.name": self.service}
                if service == "worker":
                    labels["service.instance.id"] = str(resource.attributes["service.instance.id"])
                self.meter_provider = MeterProvider(
                    resource=Resource(labels),
                    metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
                    shutdown_on_exit=False,
                )
                meter = self.meter_provider.get_meter(SCOPE)
                process = psutil.Process()
                meter.create_observable_gauge(
                    "genui_process_rss_bytes",
                    callbacks=[lambda _: [metrics.Observation(process.memory_info().rss)]],
                )
        self.instruments = Instruments(meter)
        if capacity is not None and service == "api":
            self.instruments.ready_limit.set(capacity, {"queue": queue})
        if service == "source":
            self.instruments.published.add(0)
            self.instruments.rejected.add(0)
            self.instruments.waited.add(0)
        if service == "worker":
            self.instruments.inflight.set(0)
            self.instruments.processed.add(0, {"outcome": "ok"})

    def shutdown(self) -> None:
        if self.meter_provider is not None:
            self.meter_provider.shutdown(timeout_millis=2000)
        if self.provider is not None:
            self.provider.shutdown()


active = Telemetry()


@asynccontextmanager
async def observe(
    service: Service, capacity: int | None = None, queue: str = ""
) -> AsyncIterator[Telemetry]:
    global active
    previous = active
    enabled = os.getenv("OTEL_SDK_DISABLED", "true").lower() != "true"
    active = Telemetry(service, capacity, queue, enabled=enabled)
    try:
        yield active
    finally:
        closing, active = active, previous
        await asyncio.to_thread(closing.shutdown)


def stack_frames(error: BaseException) -> list[str]:
    return [
        f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"
        for frame in traceback.extract_tb(error.__traceback__)
    ]


def failure(span: Span, error: BaseException) -> None:
    kind = type(error).__name__
    span.set_status(StatusCode.ERROR, kind)
    span.record_exception(
        Exception(kind),
        attributes={"exception.type": kind, "exception.stacktrace": "\n".join(stack_frames(error))},
    )


@contextmanager
def span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    context: Context | None = None,
    attributes: Attributes = None,
) -> Iterator[Span]:
    with active.tracer.start_as_current_span(
        name,
        context=context,
        kind=kind,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except Exception as error:
            failure(current, error)
            raise
