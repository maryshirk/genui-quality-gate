import json
import os
import subprocess
import time
import tomllib
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen


def compose(*args: str) -> str:
    return subprocess.check_output(["docker", "compose", *args], text=True).strip()


def request(path: str) -> tuple[int, dict[str, Any]]:
    base_url = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000")
    try:
        with urlopen(f"{base_url}{path}", timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def wait_health(expected_status: int, timeout: float = 45) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_result: object = None
    while time.monotonic() < deadline:
        try:
            status, payload = request("/api/v1/health")
            last_result = (status, payload)
            if status == expected_status:
                return payload
        except (OSError, HTTPException) as error:
            last_result = str(error)
        time.sleep(0.5)
    raise AssertionError(f"Health did not reach HTTP {expected_status}: {last_result}")


def main() -> None:
    with Path("pyproject.toml").open("rb") as file:
        expected_version = tomllib.load(file)["project"]["version"]
    assert request("/healthz")[0] == 200
    status, version = request("/api/v1/version")
    assert status == 200
    assert version["version"] == expected_version, version
    expected_sha = os.environ.get("BUILD_SHA", "development")
    assert version["revision"] == expected_sha, version
    assert compose("exec", "-T", "app", "id", "-u") == "10001"
    image_id = compose("images", "-q", "app")
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", image_id], text=True)
    )[0]
    labels = image["Config"]["Labels"]
    assert labels["org.opencontainers.image.revision"] == expected_sha, labels
    if os.environ.get("APP_VERSION"):
        assert labels["org.opencontainers.image.version"] == expected_version, labels

    healthy = wait_health(200)
    postgres = healthy["components"]["postgresql"]
    assert postgres["status"] == "ok", healthy
    assert postgres["version"], healthy
    assert postgres["latency_ms"] >= 0, healthy
    print(f"Container ready: version={expected_version}, revision={expected_sha}")

    try:
        compose("stop", "postgres")
        unhealthy = wait_health(503)
        assert unhealthy["components"]["postgresql"]["status"] == "error", unhealthy
        assert request("/healthz")[0] == 200, "Database outage must not fail process liveness"
        print("Database outage: readiness=503, liveness=200")
    finally:
        compose("start", "postgres")
    wait_health(200)
    print("Database recovered: readiness=200")


if __name__ == "__main__":
    main()
