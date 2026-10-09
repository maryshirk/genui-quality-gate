from unittest.mock import AsyncMock

import pytest
from starlette.types import Message, Receive, Scope, Send

from genui_quality_gate.logging import configure_logging
from genui_quality_gate.middleware import RequestLoggingMiddleware


@pytest.mark.asyncio
async def test_non_http_events_are_forwarded_unchanged() -> None:
    application = AsyncMock()
    receive, send = AsyncMock(), AsyncMock()
    scope: Scope = {"type": "websocket"}
    await RequestLoggingMiddleware(application)(scope, receive, send)
    application.assert_awaited_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_exception_after_headers_does_not_send_a_second_response() -> None:
    configure_logging()

    async def broken_stream(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("stream failed")

    sent: list[Message] = []

    async def collect(message: Message) -> None:
        sent.append(message)

    scope: Scope = {"type": "http", "method": "GET", "headers": []}
    with pytest.raises(RuntimeError, match="stream failed"):
        await RequestLoggingMiddleware(broken_stream)(scope, AsyncMock(), collect)
    assert len(sent) == 1
    assert sent[0]["status"] == 200
