import argparse
import json
import os
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from mlflow import MlflowClient

from genui_quality_gate.ml.train import run_experiments
from scripts.container_smoke import compose, wait_health


def synthetic_pairs() -> pd.DataFrame:
    rows = []
    for group in range(30):
        prompt = f"Build an accessible event registration page for conference {group}."
        complete = (
            "<html lang='en'><head><title>Event registration</title></head><body>"
            "<main><h1>Event registration</h1><p>Register for the conference.</p>"
            "<form><label for='email'>Email</label><input id='email' type='email'>"
            "<button>Register</button></form></main></body></html>"
        )
        incomplete = "<html><body><div>Event</div></body></html>"
        for target in (0, 1):
            rows.append(
                {
                    "pair_id": f"synthetic-{group}-{target}",
                    "prompt_id": f"synthetic-{group}",
                    "prompt": prompt,
                    "html_a": complete if target else incomplete,
                    "html_b": incomplete if target else complete,
                    "target": target,
                    "agreement": 1.0,
                }
            )
    return pd.DataFrame(rows)


def process(payload: dict[str, str]) -> tuple[int, dict[str, Any]]:
    base_url = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000")
    request = Request(
        f"{base_url}/process",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            accepted = json.load(response)
        for _ in range(200):
            try:
                with urlopen(f"{base_url}{accepted['status_url']}", timeout=5) as response:
                    result = json.load(response)
                if result["status"] == "SUCCEEDED":
                    return 200, result["result"]
                if result["status"] == "FAILED":
                    return 500, result
            except HTTPError as error:
                if error.code != 404:
                    raise
            time.sleep(0.1)
        raise TimeoutError("Queued inference did not finish")
    except HTTPError as error:
        return error.code, json.load(error)


def verify(tracking_uri: str, model_name: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    champion = client.get_model_version_by_alias(model_name, "champion")
    challenger = client.get_model_version_by_alias(model_name, "challenger")
    assert champion.version != challenger.version
    payload = synthetic_pairs().iloc[1][["prompt", "html_a", "html_b"]].to_dict()
    wait_health(200)
    status, first = process(payload)
    assert status == 200, first
    assert first["model"]["name"] == model_name, first
    assert first["model"]["version"] == champion.version, first
    assert first["model"]["run_id"] == champion.run_id, first
    assert first["winner"] in {"A", "B"}, first
    assert 0 <= first["probability_a"] <= 1, first
    assert process({"prompt": "", "html_a": "", "html_b": ""})[0] == 422
    try:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
        status, unchanged = process(payload)
        assert status == 200 and unchanged["model"] == first["model"], unchanged
        compose("restart", "worker")
        time.sleep(4)
        wait_health(200, timeout=120)
        status, switched = process(payload)
        assert status == 200, switched
        assert switched["model"]["version"] == challenger.version, switched
        assert switched["model"]["run_id"] == challenger.run_id, switched
        try:
            compose("stop", "mlflow")
            status, offline = process(payload)
            assert status == 200 and offline == switched, offline
        finally:
            compose("up", "-d", "--wait", "--wait-timeout", "120", "mlflow")
    finally:
        client.set_registered_model_alias(model_name, "champion", champion.version)
        compose("restart", "worker")
        time.sleep(4)
        wait_health(200, timeout=120)
    print(
        f"MLflow verified: model={model_name}, versions={champion.version}/{challenger.version}, "
        "alias pinned until restart, inference works while tracking server is offline"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["seed", "verify"])
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
    )
    parser.add_argument("--model-name", default="genui-pairwise-quality-smoke")
    args = parser.parse_args()
    if args.action == "seed":
        result = run_experiments(
            synthetic_pairs(),
            tracking_uri=args.tracking_uri,
            experiment_name="genui-synthetic-smoke",
            model_name=args.model_name,
            source="https://example.invalid/genui-ci-pairs-v1",
            revision="synthetic-v1",
        )
        print(json.dumps(result, indent=2))
    else:
        verify(args.tracking_uri, args.model_name)


if __name__ == "__main__":
    main()
