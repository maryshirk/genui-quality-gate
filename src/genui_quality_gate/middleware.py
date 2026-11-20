import logging
import re
from time import perf_counter
from uuid import uuid4

from opentelemetry.trace import Span, SpanKind, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from genui_quality_gate import telemetry
from genui_quality_gate.logging import LOGGER_NAME, task_context

logger = logging.getLogger(LOGGER_NAME)
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        path = scope.get("path")
        if path == "/healthz":
            await self.request(scope, receive, send, headers)
            return
        context = TraceContextTextMapPropagator().extract(dict(headers))
        with telemetry.span("HTTP request", kind=SpanKind.SERVER, context=context) as span:
            await self.request(scope, receive, send, headers, span)

    async def request(
        self, scope: Scope, receive: Receive, send: Send, headers: Headers, span: Span | None = None
    ) -> None:
        supplied_id = headers.get("x-request-id", "")
        request_id = supplied_id if _REQUEST_ID.fullmatch(supplied_id) else uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        status_code = 500
        response_started = False

        async def send_response(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_started = True
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_response)
        except Exception as exc:
            if span is not None:
                telemetry.failure(span, exc)
            logger.exception(
                "request_failed",
                extra={"request_id": request_id, "error_type": type(exc).__name__},
            )
            if response_started:
                raise
            response = JSONResponse(
                {"detail": "Internal server error", "request_id": request_id}, status_code=500
            )
            await response(scope, receive, send_response)
        finally:
            route = getattr(scope.get("route"), "path", "<unmatched>")
            duration = perf_counter() - started
            if span is not None:
                span.update_name(f"{scope['method']} {route}")
                span.set_attributes({"http.route": route, "http.response.status_code": status_code})
                if status_code >= 500:
                    span.set_status(StatusCode.ERROR)
                signals = telemetry.active.instruments
                labels = {"route": route, "status": str(status_code)}
                signals.requests.add(1, labels)
                signals.http_time.record(duration, labels)
            level = logging.DEBUG if route in {"/healthz", "/api/v1/health"} else logging.INFO
            with task_context(scope["state"].get("task_id")):
                logger.log(
                    level,
                    "request_completed",
                    extra={
                        "request_id": request_id,
                        "method": scope["method"],
                        "route": route,
                        "status_code": status_code,
                        "duration_ms": round(duration * 1000, 3),
                    },
                )
