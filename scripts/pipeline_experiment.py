from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import asyncpg
import docker
import httpx
import matplotlib
import pandas as pd
import psutil

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from genui_quality_gate.config import Settings
from genui_quality_gate.ml.data import INPUT_COLUMNS, load_dataset


def compose(*args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["docker", "compose", *args], env=env, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def container_client() -> Any:
    endpoint = os.getenv("DOCKER_HOST")
    if not endpoint:
        context = (
            os.getenv("DOCKER_CONTEXT")
            or subprocess.check_output(["docker", "context", "show"], text=True).strip()
        )
        value = subprocess.check_output(["docker", "context", "inspect", context], text=True)
        endpoint = json.loads(value)[0]["Endpoints"]["docker"]["Host"]
    return docker.DockerClient(base_url=endpoint)


def prepare(path: Path, count: int) -> None:
    example = Path("artifacts/mlflow/example_request.json")
    if example.exists():
        payload = json.loads(example.read_text())
    else:
        frame, _ = load_dataset(Path("data/genui"))
        payload = frame[INPUT_COLUMNS].iloc[0].to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("w") as stream:
        for _ in range(count):
            stream.write(line)
    print(json.dumps({"input": str(path), "tasks": count, "bytes_per_task": len(line.encode())}))


def broker_memory(client: Any, container_id: str) -> float:
    statistics = client.api.stats(container_id, stream=False, one_shot=True)
    memory = statistics["memory_stats"]
    cache = memory.get("stats", {}).get("inactive_file", 0)
    return float(max(0, memory.get("usage", 0) - cache)) / 1024**2


async def measure(
    label: str,
    bounded: bool,
    workers: int,
    source: Path,
    count: int,
    delay: float,
    capacity: int,
    output: Path,
) -> dict[str, Any]:
    queue_name = f"experiment.{label}.{uuid4().hex[:8]}"
    env = dict(os.environ) | {
        "GENUI_QUEUE_NAME": queue_name,
        "GENUI_QUEUE_CAPACITY": str(capacity),
        "GENUI_BACKPRESSURE_ENABLED": str(bounded).lower(),
        "GENUI_PREFETCH_COUNT": "1" if bounded else "0",
        "GENUI_WORKER_CONCURRENCY": "1",
        "GENUI_INFERENCE_DELAY_SECONDS": str(delay),
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
        f"worker={workers}",
        "worker",
        env=env,
    )
    ids = (await asyncio.to_thread(compose, "ps", "-q", "worker", env=env)).splitlines()
    inspected = json.loads(
        await asyncio.to_thread(
            subprocess.check_output,
            ["docker", "inspect", *ids],
            text=True,
        )
    )
    urls = [
        f"http://127.0.0.1:{item['NetworkSettings']['Ports']['8001/tcp'][0]['HostPort']}/healthz"
        for item in inspected
    ]
    rabbit_id = await asyncio.to_thread(compose, "ps", "-q", "rabbitmq")
    settings = Settings()
    source_env = env | {
        "GENUI_RABBIT_URL": (
            f"amqp://genui:{os.getenv('RABBITMQ_PASSWORD', 'local-rabbit-change-me')}"
            f"@127.0.0.1:{os.getenv('RABBITMQ_PORT', '5672')}/"
        )
    }
    local = Path("artifacts/backpressure") / queue_name
    local.mkdir(parents=True, exist_ok=True)
    checkpoint = local / "checkpoint.json"
    pool = await asyncpg.create_pool(settings.database_url.get_secret_value())
    client = container_client()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    model: dict[str, str] = {}
    with (local / "source.log").open("w") as source_log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "genui_quality_gate.source",
            str(source),
            "--checkpoint",
            str(checkpoint),
            env=source_env,
            stdout=source_log,
            stderr=subprocess.STDOUT,
        )
        try:
            management = f"http://127.0.0.1:{os.getenv('RABBITMQ_MANAGEMENT_PORT', '15672')}"
            async with httpx.AsyncClient(
                auth=("genui", os.getenv("RABBITMQ_PASSWORD", "local-rabbit-change-me")),
                timeout=5,
            ) as http:
                while True:
                    elapsed = time.perf_counter() - started
                    if elapsed > max(120, count * delay * 4):
                        raise TimeoutError(f"Experiment {label} did not finish")
                    queue_response = await http.get(
                        f"{management}/api/queues/%2F/{quote(queue_name, safe='')}"
                    )
                    queue_response.raise_for_status()
                    queue = queue_response.json()
                    responses = await asyncio.gather(*(http.get(url) for url in urls))
                    health = [response.json() for response in responses]
                    if health:
                        model = health[0]["model"]
                    accepted = (
                        json.loads(checkpoint.read_text())["accepted"] if checkpoint.exists() else 0
                    )
                    completed = await pool.fetchval(
                        "SELECT count(*) FROM pipeline_tasks "
                        "WHERE queue_name=$1 AND status='SUCCEEDED'",
                        queue_name,
                    )
                    try:
                        source_rss = psutil.Process(process.pid).memory_info().rss / 1024**2
                    except psutil.NoSuchProcess:
                        source_rss = 0.0
                    rabbit_memory = await asyncio.to_thread(broker_memory, client, rabbit_id)
                    rows.append(
                        {
                            "seconds": elapsed,
                            "ready": queue.get("messages_ready", 0),
                            "unacked": queue.get("messages_unacknowledged", 0),
                            "accepted": accepted,
                            "completed": completed,
                            "source_rss_mb": source_rss,
                            "worker_rss_mb": sum(item["rss_bytes"] for item in health) / 1024**2,
                            "broker_memory_mb": rabbit_memory,
                        }
                    )
                    if process.returncode is not None:
                        if process.returncode != 0:
                            raise RuntimeError(f"Source failed; see {local / 'source.log'}")
                        if completed == count and not queue.get("messages", 0):
                            break
                    await asyncio.sleep(0.25)
            records = await pool.fetch(
                "SELECT extract(epoch from finished_at) AS finished_epoch, inference_ms, "
                "end_to_end_ms, attempts FROM pipeline_tasks "
                "WHERE queue_name=$1 ORDER BY finished_at",
                queue_name,
            )
        finally:
            if process.returncode is None:
                process.terminate()
                await asyncio.wait_for(process.wait(), 10)
            await pool.close()
            client.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{label}.csv", index=False)
    latencies = pd.DataFrame([dict(row) for row in records])
    latencies.to_csv(output / f"{label}_tasks.csv", index=False)
    if len(frame) > 1:
        frame["source_rate"] = frame.accepted.diff() / frame.seconds.diff()
        frame["worker_rate"] = frame.completed.diff() / frame.seconds.diff()
    if bounded and frame.ready.max() > capacity:
        raise AssertionError(f"Queue capacity exceeded in {label}")
    active = frame[(frame.accepted >= capacity) & (frame.accepted < count)]
    rates = {"steady_source_rate": None, "steady_worker_rate": None}
    if len(active) > 2:
        first, last = active.iloc[0], active.iloc[-1]
        duration = last.seconds - first.seconds
        rates = {
            "steady_source_rate": (last.accepted - first.accepted) / duration,
            "steady_worker_rate": (last.completed - first.completed) / duration,
        }
    result = {
        "label": label,
        "queue": queue_name,
        "bounded": bounded,
        "workers": workers,
        "prefetch": 1 if bounded else 0,
        "capacity": capacity if bounded else None,
        "tasks": count,
        "delay_seconds": delay,
        "duration_seconds": float(frame.seconds.iloc[-1]),
        "max_ready": int(frame.ready.max()),
        "max_unacked": int(frame.unacked.max()),
        "source_peak_mb": float(frame.source_rss_mb.max()),
        "worker_peak_mb": float(frame.worker_rss_mb.max()),
        "broker_peak_mb": float(frame.broker_memory_mb.max()),
        "p50_task_ms": float(latencies.end_to_end_ms.median()),
        "p95_task_ms": float(latencies.end_to_end_ms.quantile(0.95)),
        "mean_inference_ms": float(latencies.inference_ms.mean()),
        "completed": int(frame.completed.iloc[-1]),
        "model": model,
        **rates,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def plots(output: Path, results: list[dict[str, Any]]) -> None:
    for kind in ("queue", "memory", "rates", "latency"):
        fig, axes = plt.subplots(1, len(results), figsize=(13, 3.8), squeeze=False)
        for ax, result in zip(axes[0], results, strict=True):
            frame = pd.read_csv(output / f"{result['label']}.csv")
            x = frame.seconds
            if kind == "queue":
                ax.plot(x, frame.ready, label="Ready")
                ax.plot(x, frame.unacked, label="Unacknowledged")
                if result["bounded"]:
                    ax.axhline(result["capacity"], color="black", linestyle=":", label="Limit")
                ax.set_ylabel("Messages")
            elif kind == "memory":
                for name in ("source_rss_mb", "worker_rss_mb", "broker_memory_mb"):
                    ax.plot(x, frame[name], label=name.replace("_mb", ""))
                ax.set_ylabel("MiB")
            elif kind == "rates":
                delta = frame.seconds.diff()
                source_rate = (frame.accepted.diff() / delta).rolling(3, min_periods=1).mean()
                worker_rate = (frame.completed.diff() / delta).rolling(3, min_periods=1).mean()
                ax.plot(x, source_rate, label="Source")
                ax.plot(x, worker_rate, label="Workers")
                ax.set_ylabel("Tasks / second")
            else:
                tasks = pd.read_csv(output / f"{result['label']}_tasks.csv")
                completed_at = tasks.finished_epoch - tasks.finished_epoch.iloc[0]
                ax.plot(completed_at, tasks.end_to_end_ms / 1000)
                ax.set_ylabel("End-to-end seconds")
            ax.set_title(result["label"].replace("_", " "))
            ax.set_xlabel("Seconds")
            if kind != "latency":
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"{kind}.png", dpi=130)
        plt.close(fig)


def refresh_report(output: Path) -> None:
    path = output / "summary.json"
    report = json.loads(path.read_text())
    for result in report["results"]:
        log = Path("artifacts/backpressure") / result["queue"] / "source.log"
        source = json.loads(log.read_text().splitlines()[-1])
        result["source_duration_seconds"] = source["duration_seconds"]
        result["source_mean_rate"] = source["accepted"] / source["duration_seconds"]
        result["source_rejections"] = source["rejected"]
    path.write_text(json.dumps(report, indent=2) + "\n")
    plots(output, report["results"])


async def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for label, bounded, workers in (
            ("unbounded_1_worker", False, 1),
            ("bounded_1_worker", True, 1),
            ("bounded_2_workers", True, 2),
        ):
            results.append(
                await measure(
                    label,
                    bounded,
                    workers,
                    args.input,
                    args.count,
                    args.delay,
                    args.capacity,
                    args.output,
                )
            )
        (args.output / "summary.json").write_text(
            json.dumps(
                {"measured_at": datetime.now(UTC).isoformat(), "results": results},
                indent=2,
            )
            + "\n"
        )
        refresh_report(args.output)
    finally:
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
            "worker",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--input", type=Path, default=Path("data/pipeline/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments"))
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--capacity", type=int, default=20)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.input, args.count)
    elif args.action == "report":
        refresh_report(args.output)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
