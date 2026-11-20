from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from numpy.typing import NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from genui_quality_gate.ml.data import (
    DATASET_REVISION,
    DATASET_SOURCE,
    INPUT_COLUMNS,
    content_digest,
    load_dataset,
    mirror_training_pairs,
    split_dataset,
)
from genui_quality_gate.ml.eda import build_analysis, log_analysis
from genui_quality_gate.ml.features import build_pipeline

VARIANTS = ("baseline", "structure", "content")


def evaluate(target: pd.Series, probability: NDArray[np.float64]) -> dict[str, float]:
    prediction = probability >= 0.5
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "accuracy": float(accuracy_score(target, prediction)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def log_dataset(frame: pd.DataFrame, source: str, context: str) -> None:
    dataset = from_pandas(
        frame,
        source=source,
        name=f"genui-{context}",
        targets="target",
        digest=content_digest(frame)[:32],
    )
    mlflow.log_input(dataset, context=context)


def log_eda(frame: pd.DataFrame, splits: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> None:
    log_analysis(build_analysis(frame, splits, manifest))
    assignments = pd.concat(
        [
            part[["pair_id", "prompt_id", "target"]].assign(split=name)
            for name, part in splits.items()
        ]
    )
    mlflow.log_text(assignments.to_csv(index=False), "data/split_assignments.csv")


def log_diagnostics(
    frame: pd.DataFrame, probability: NDArray[np.float64], context: str, pipeline: Pipeline
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ConfusionMatrixDisplay.from_predictions(
        frame["target"], probability >= 0.5, display_labels=["B", "A"], ax=axes[0], colorbar=False
    )
    RocCurveDisplay.from_predictions(frame["target"], probability, ax=axes[1])
    fig.tight_layout()
    mlflow.log_figure(fig, f"diagnostics/{context}.png")
    plt.close(fig)
    predictions = frame[["pair_id", "prompt_id", "target", "agreement"]].copy()
    predictions["probability_a"] = probability
    predictions["predicted_a"] = probability >= 0.5
    mlflow.log_text(predictions.to_csv(index=False), f"diagnostics/{context}_predictions.csv")
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        coefficients = pd.DataFrame(
            {
                "feature": pipeline.named_steps["features"].get_feature_names_out(),
                "coefficient": classifier.coef_[0],
            }
        ).sort_values("coefficient")
        mlflow.log_text(coefficients.to_csv(index=False), "diagnostics/coefficients.csv")


def run_experiments(
    frame: pd.DataFrame,
    *,
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    source: str,
    revision: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    splits = split_dataset(frame)
    train = mirror_training_pairs(splits["train"])
    validation = splits["validation"]
    manifest = summary or {}
    digest = content_digest(frame)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=False, capture_output=True, text=True
        )
        dirty = str(bool(status.stdout.strip()) or status.returncode != 0).lower()
    except OSError:
        dirty = "unknown"
    provenance = {
        "ml_source_sha256": source_hash,
        "source_tree_dirty": dirty,
    }
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="eda") as eda_run:
        eda_run_id = eda_run.info.run_id
        mlflow.set_tags({"dataset_revision": revision, "dataset_digest": digest, "kind": "eda"})
        mlflow.set_tags(provenance)
        log_dataset(frame, source, "eda")
        log_dataset(splits["train"], source, "eda_training")
        log_eda(frame, splits, manifest)
    results: dict[str, dict[str, Any]] = {}
    pipelines = {}
    requirements = [
        f"{name}=={version(name)}"
        for name in ("scikit-learn", "pandas", "numpy", "beautifulsoup4", "skops")
    ]
    for variant in VARIANTS:
        pipeline = build_pipeline(variant)
        pipeline.fit(train[INPUT_COLUMNS], train["target"])
        probability = pipeline.predict_proba(validation[INPUT_COLUMNS])[:, 1]
        metrics = evaluate(validation["target"], probability)
        with mlflow.start_run(run_name=variant) as run:
            mlflow.set_tags(
                {
                    "dataset_revision": revision,
                    "dataset_digest": digest,
                    "selection_metric": "validation_balanced_accuracy",
                    "target": "1=A preferred, 0=B preferred",
                    "kind": "training",
                    **provenance,
                }
            )
            mlflow.log_params(
                {
                    "variant": variant,
                    "seed": 42,
                    "split": "60/20/20 by prompt",
                    "train_pairs": len(splits["train"]),
                    "augmented_train_pairs": len(train),
                    "validation_pairs": len(validation),
                    "C": 0.1 if variant != "baseline" else "not_applicable",
                    "fit_intercept": False,
                    "decision_threshold": 0.5,
                    "feature_count": len(pipeline.named_steps["features"].get_feature_names_out()),
                }
            )
            log_dataset(splits["train"], source, "training_original")
            log_dataset(train, source, "training_mirrored")
            log_dataset(validation, source, "validation")
            mlflow.log_dict(
                {
                    "source": source,
                    "revision": revision,
                    "parent_digest": digest,
                    "mlflow_digest": digest[:32],
                    "digest_algorithm": "first 32 hex characters of SHA256; full hashes retained",
                    "training_digest": content_digest(splits["train"]),
                    "augmented_training_digest": content_digest(train),
                    "transformations": manifest.get(
                        "transformations", ["prompt split", "A/B reversal"]
                    ),
                },
                "data/lineage.json",
            )
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            log_diagnostics(validation, probability, "validation", pipeline)
            example = train[INPUT_COLUMNS].iloc[:1]
            model = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=infer_signature(example, pipeline.predict_proba(example)),
                input_example=example,
                pip_requirements=requirements,
                code_paths=[str(Path(__file__).resolve().parents[1])],
                pyfunc_predict_fn="predict_proba",
                serialization_format="skops",
                skops_trusted_types=["genui_quality_gate.ml.features.PairFeatures"],
            )
            results[variant] = {
                "run_id": run.info.run_id,
                "model_uri": model.model_uri,
                "metrics": metrics,
            }
            pipelines[variant] = pipeline
    ranked = sorted(
        ("structure", "content"),
        key=lambda name: (
            -results[name]["metrics"]["balanced_accuracy"],
            results[name]["metrics"]["log_loss"],
        ),
    )
    champion, challenger = ranked
    client = MlflowClient()
    for variant in ("structure", "content"):
        registered = mlflow.register_model(results[variant]["model_uri"], model_name)
        results[variant]["version"] = registered.version
        client.set_model_version_tag(model_name, registered.version, "dataset_digest", digest)
        client.set_model_version_tag(model_name, registered.version, "variant", variant)
    for alias, variant in (("champion", champion), ("challenger", challenger)):
        client.set_registered_model_alias(model_name, alias, results[variant]["version"])
    test = splits["test"]
    probability = pipelines[champion].predict_proba(test[INPUT_COLUMNS])[:, 1]
    test_metrics = evaluate(test["target"], probability)
    with mlflow.start_run(run_id=results[champion]["run_id"]):
        log_dataset(test, source, "test")
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        log_diagnostics(test, probability, "test", pipelines[champion])
        mlflow.set_tag("selected_alias", "champion")
        mlflow.log_dict(results, "comparison.json")
    return {
        "model_name": model_name,
        "eda_run_id": eda_run_id,
        "champion": champion,
        "champion_version": results[champion]["version"],
        "challenger_version": results[challenger]["version"],
        "dataset_digest": digest,
        "run_ids": {name: result["run_id"] for name, result in results.items()},
        "validation_metrics": {name: result["metrics"] for name, result in results.items()},
        "test_metrics": test_metrics,
        "beats_baseline": results[champion]["metrics"]["balanced_accuracy"]
        > results["baseline"]["metrics"]["balanced_accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    )
    parser.add_argument("--experiment", default="genui-quality")
    parser.add_argument("--model-name", default="genui-pairwise-quality")
    parser.add_argument("--data-dir", type=Path, default=Path("data/genui"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mlflow"))
    arguments = parser.parse_args()
    frame, manifest = load_dataset(arguments.data_dir)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "example_request.json").write_text(
        json.dumps(frame[INPUT_COLUMNS].iloc[0].to_dict(), indent=2) + "\n"
    )
    result = run_experiments(
        frame,
        tracking_uri=arguments.tracking_uri,
        experiment_name=arguments.experiment,
        model_name=arguments.model_name,
        source=DATASET_SOURCE,
        revision=DATASET_REVISION,
        summary=manifest,
    )
    (arguments.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    mlflow.artifacts.download_artifacts(
        run_id=result["eda_run_id"], artifact_path="eda", dst_path=str(arguments.output_dir)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
