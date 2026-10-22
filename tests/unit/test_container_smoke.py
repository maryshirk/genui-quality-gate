from http.client import RemoteDisconnected
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from scripts import container_smoke


@pytest.mark.parametrize(
    "error",
    [URLError("connection refused"), RemoteDisconnected("restart"), TimeoutError("startup")],
)
def test_wait_health_retries_transient_connection_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    payload = {"status": "ok"}
    request = Mock(side_effect=[error, (200, payload)])
    monkeypatch.setattr(container_smoke, "request", request)
    monkeypatch.setattr(container_smoke.time, "sleep", Mock())

    assert container_smoke.wait_health(200) == payload
    assert request.call_count == 2


def test_wait_health_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(container_smoke.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(container_smoke.time, "sleep", Mock())
    monkeypatch.setattr(
        container_smoke, "request", Mock(return_value=(503, {"status": "degraded"}))
    )

    with pytest.raises(AssertionError, match="Health did not reach HTTP 200"):
        container_smoke.wait_health(200, timeout=1)
