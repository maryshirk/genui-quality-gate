from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import asyncpg
import httpx
from pydantic import SecretStr

from genui_quality_gate.config import Settings
from genui_quality_gate.queue import (
    QueueUnavailable,
    TaskPublisher,
    create_broker,
    declare_topology,
)
from genui_quality_gate.task_store import PostgresTaskStore
from genui_quality_gate.tasks import TaskEnvelope, TaskPayload
from scripts.pipeline_experiment import compose, container_client


async def result_ready(store: PostgresTaskStore, task_id: Any) -> Any:
    for _ in range(300):
        task = await store.get(task_id)
        if task and task.status in {"SUCCEEDED", "FAILED"}:
            return task
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Task did not finish: {task_id}")


def worker_url(worker: Any) -> str:
    worker.reload()
    port = worker.attrs["NetworkSettings"]["Ports"]["8001/tcp"][0]["HostPort"]
    return f"http://127.0.0.1:{port}/healthz"


async def wait_active(http: httpx.AsyncClient, worker: Any, active: bool) -> dict[str, Any]:
    for _ in range(150):
        try:
            response = await http.get(worker_url(worker))
            response.raise_for_status()
            body = response.json()
            if body["status"] == "ok" and bool(body["active"]) == active:
                return dict(body)
        except (httpx.HTTPError, KeyError, IndexError):
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError("Worker state did not become ready")


async def run() -> None:
    queue_name = f"verify.{uuid4().hex[:8]}"
    env = dict(os.environ) | {
        "GENUI_QUEUE_NAME": queue_name,
        "GENUI_QUEUE_CAPACITY": "3",
        "GENUI_INFERENCE_DELAY_SECONDS": "2",
        "GENUI_BACKPRESSURE_ENABLED": "true",
        "GENUI_PREFETCH_COUNT": "1",
        "GENUI_WORKER_CONCURRENCY": "1",
    }
    await asyncio.to_thread(
        compose,
        "up",
        "-d",
        "--no-deps",
        "--wait",
        "--wait-timeout",
        "180",
        "--scale",
        "worker=1",
        "app",
        "worker",
        env=env,
    )
    rabbit_url = os.getenv("GENUI_RABBIT_URL") or (
        f"amqp://genui:{os.getenv('RABBITMQ_PASSWORD', 'local-rabbit-change-me')}"
        f"@127.0.0.1:{os.getenv('RABBITMQ_PORT', '5672')}/"
    )
    configuration = Settings(
        queue_name=queue_name, queue_capacity=3, rabbit_url=SecretStr(rabbit_url)
    )
    broker = create_broker(configuration)
    pool = await asyncpg.create_pool(configuration.database_url.get_secret_value())
    store = PostgresTaskStore(pool, queue_name=queue_name)
    client = container_client()
    worker_id = await asyncio.to_thread(compose, "ps", "-q", "worker", env=env)
    worker = await asyncio.to_thread(client.containers.get, worker_id)
    example = await asyncio.to_thread(Path("artifacts/mlflow/example_request.json").read_text)
    payload = TaskPayload.model_validate_json(example)
    report: dict[str, Any] = {"queue": queue_name}
    try:
        await broker.connect()
        await declare_topology(broker, configuration)
        publisher = TaskPublisher(broker, configuration)
        async with httpx.AsyncClient(timeout=5) as http:
            await wait_active(http, worker, False)
            accepted = []
            await asyncio.to_thread(worker.pause)
            try:
                for _ in range(8):
                    task = TaskEnvelope(payload=payload)
                    try:
                        await publisher.publish(task)
                        accepted.append(task)
                    except QueueUnavailable:
                        break
                rejected_id = uuid4()
                start = perf_counter()
                response = await http.post(
                    "http://127.0.0.1:8000/process",
                    json=payload.model_dump(),
                    headers={"Idempotency-Key": str(rejected_id)},
                )
                report["overload_latency_ms"] = (perf_counter() - start) * 1000
                assert response.status_code == 503, response.text
                assert response.headers["retry-after"] == "2"
                assert await store.get(rejected_id) is None
                report["overload_status"] = response.status_code
                report["retry_after"] = response.headers["retry-after"]
                report["rejected_task_absent"] = True
            finally:
                await asyncio.to_thread(worker.unpause)
            await asyncio.gather(*(result_ready(store, task.task_id) for task in accepted))
            await publisher.publish(accepted[0])
            await asyncio.sleep(0.2)
            duplicate = await store.get(accepted[0].task_id)
            assert duplicate is not None
            assert duplicate.attempts == 1
            report["duplicate_result_rows"] = await pool.fetchval(
                "SELECT count(*) FROM pipeline_tasks WHERE task_id=$1", duplicate.task_id
            )
            poison_id = uuid4()
            await broker.publish(
                {"task_id": str(poison_id), "payload": {"prompt": ""}},
                queue=queue_name,
                persist=True,
                message_id=str(poison_id),
            )
            failed = await result_ready(store, poison_id)
            assert failed.status == "FAILED" and failed.attempts == 3
            report["poison_attempts"] = failed.attempts
            report["poison_task_id"] = str(poison_id)
            await wait_active(http, worker, False)

            graceful = TaskEnvelope(payload=payload)
            await publisher.publish(graceful)
            health = await wait_active(http, worker, True)
            started = perf_counter()
            await asyncio.to_thread(worker.kill, signal="SIGTERM")
            exit_result = await asyncio.to_thread(worker.wait, timeout=15)
            finished = await result_ready(store, graceful.task_id)
            assert finished.status == "SUCCEEDED"
            assert exit_result["StatusCode"] in {0, 143}
            report["graceful_exit_code"] = exit_result["StatusCode"]
            report["graceful_seconds"] = perf_counter() - started
            report["graceful_active_task_completed"] = True
            report["model"] = health["model"]
            await asyncio.to_thread(worker.start)
            await wait_active(http, worker, False)

            interrupted = TaskEnvelope(payload=payload)
            await publisher.publish(interrupted)
            await wait_active(http, worker, True)
            await asyncio.to_thread(worker.kill, signal="SIGKILL")
            killed = await asyncio.to_thread(worker.wait, timeout=10)
            assert killed["StatusCode"] == 137
            await asyncio.to_thread(worker.start)
            await wait_active(http, worker, False)
            recovered = await result_ready(store, interrupted.task_id)
            assert recovered.status == "SUCCEEDED"
            report["sigkill_redelivery_succeeded"] = True
            report["sigkill_result_rows"] = await pool.fetchval(
                "SELECT count(*) FROM pipeline_tasks WHERE task_id=$1", interrupted.task_id
            )
        directory = Path("docs/experiments")
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            (directory / "reliability.json").write_text, json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        await broker.stop()
        await pool.close()
        client.close()
        await asyncio.to_thread(
            compose,
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "--wait-timeout",
            "180",
            "--scale",
            "worker=1",
            "app",
            "worker",
        )


if __name__ == "__main__":
    asyncio.run(run())
