from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from genui_quality_gate.ml.data import content_digest
from genui_quality_gate.ml.train import evaluate, log_dataset, main, run_experiments


def training_frame() -> pd.DataFrame:
    rich = "<main><h1>Register</h1><form><input><button>Submit</button></form></main>"
    plain = "<div>Empty</div>"
    return pd.DataFrame(
        [
            {
                "pair_id": f"{index}-{target}",
                "prompt_id": str(index),
                "prompt": f"Register {index}",
                "html_a": rich if target else plain,
                "html_b": plain if target else rich,
                "target": target,
                "agreement": 1.0,
            }
            for index in range(20)
            for target in (0, 1)
        ]
    )


def test_metrics_use_probabilities_and_fixed_threshold() -> None:
    metrics = evaluate(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["balanced_accuracy"] == 1
    assert metrics["roc_auc"] == 1
    assert metrics["brier_score"] == pytest.approx(0.025)
    baseline = evaluate(pd.Series([0, 1]), np.array([0.5, 0.5]))
    assert baseline["balanced_accuracy"] == 0.5


def test_dataset_logging_links_content_digest_source_and_context() -> None:
    frame = training_frame()
    with (
        patch("genui_quality_gate.ml.train.from_pandas") as dataset,
        patch("genui_quality_gate.ml.train.mlflow.log_input") as log,
    ):
        log_dataset(frame, "https://example.invalid/dataset", "training")
    assert dataset.call_args.kwargs["digest"] == content_digest(frame)[:32]
    assert dataset.call_args.kwargs["source"] == "https://example.invalid/dataset"
    assert dataset.call_args.kwargs["targets"] == "target"
    log.assert_called_once_with(dataset.return_value, context="training")


def test_experiments_log_three_models_and_register_two_versions() -> None:
    frame = training_frame()
    functions = (
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "set_tags",
        "set_tag",
        "log_params",
        "log_metrics",
        "log_dict",
        "log_text",
        "log_figure",
        "register_model",
    )
    with ExitStack() as stack:
        mocks = {
            name: stack.enter_context(patch(f"genui_quality_gate.ml.train.mlflow.{name}"))
            for name in functions
        }
        dataset = stack.enter_context(patch("genui_quality_gate.ml.train.log_dataset"))
        client = stack.enter_context(patch("genui_quality_gate.ml.train.MlflowClient"))
        log_model = stack.enter_context(
            patch("genui_quality_gate.ml.train.mlflow.sklearn.log_model")
        )
        mocks["start_run"].return_value.__enter__.return_value.info.run_id = "run-id"
        log_model.return_value.model_uri = "models:/model-id"
        mocks["register_model"].side_effect = [
            SimpleNamespace(version="1"),
            SimpleNamespace(version="2"),
        ]
        result = run_experiments(
            frame,
            tracking_uri="http://localhost:5001",
            experiment_name="test-experiment",
            model_name="test-model",
            source="https://example.invalid/data",
            revision="test-revision",
        )
    assert set(result["run_ids"]) == {"baseline", "structure", "content"}
    assert result["champion_version"] in {"1", "2"}
    assert result["champion_version"] != result["challenger_version"]
    assert result["test_metrics"]["balanced_accuracy"] == 1
    assert result["beats_baseline"]
    assert log_model.call_count == 3
    assert log_model.call_args.kwargs["serialization_format"] == "skops"
    assert log_model.call_args.kwargs["skops_trusted_types"] == [
        "genui_quality_gate.ml.features.PairFeatures"
    ]
    assert mocks["register_model"].call_count == 2
    assert client.return_value.set_registered_model_alias.call_count == 2
    contexts = [call.args[2] for call in dataset.call_args_list]
    assert contexts.count("test") == 1
    assert contexts.count("training_original") == 3
    assert contexts.count("training_mirrored") == 3
    for call in dataset.call_args_list:
        if call.args[2] == "training_mirrored":
            assert call.args[0]["target"].mean() == 0.5
    figure_paths = {call.args[1] for call in mocks["log_figure"].call_args_list}
    assert {
        "eda/distributions.png",
        "eda/splits.png",
        "eda/feature_preferences.png",
        "eda/feature_correlations.png",
        "diagnostics/validation.png",
        "diagnostics/test.png",
    } <= figure_paths
    lineage = [
        call.args[0]
        for call in mocks["log_dict"].call_args_list
        if call.args[1] == "data/lineage.json"
    ]
    assert all(item["parent_digest"] == content_digest(frame) for item in lineage)


def test_cli_loads_real_dataset_and_exports_local_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    result = {"eda_run_id": "eda", "champion_version": "2"}
    frame = training_frame()
    with (
        patch("sys.argv", ["train", "--data-dir", str(tmp_path), "--output-dir", str(output)]),
        patch("genui_quality_gate.ml.train.load_dataset", return_value=(frame, {"counts": {}})),
        patch("genui_quality_gate.ml.train.run_experiments", return_value=result) as training,
        patch("genui_quality_gate.ml.train.mlflow.artifacts.download_artifacts") as download,
        patch("builtins.print") as printing,
    ):
        main()
    assert training.call_args.kwargs["model_name"] == "genui-pairwise-quality"
    assert json.loads((output / "summary.json").read_text()) == result
    download.assert_called_once_with(run_id="eda", artifact_path="eda", dst_path=str(output))
    printing.assert_called_once()
