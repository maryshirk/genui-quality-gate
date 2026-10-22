import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from genui_quality_gate.config import Settings
from genui_quality_gate.inference import LoadedModel, ModelMetadata, load_model

PAYLOAD = {"prompt": "Create a shop", "html_a": "<h1>Shop A</h1>", "html_b": "<h1>Shop B</h1>"}


def model_metadata(version: str = "2") -> ModelMetadata:
    return ModelMetadata(
        name="genui-pairwise-quality", version=version, alias="champion", run_id="training-run"
    )


def fake_estimator(probabilities: object = None) -> Mock:
    estimator = Mock()
    estimator.classes_ = [0, 1]
    estimator.predict_proba.return_value = [[0.2, 0.8]] if probabilities is None else probabilities
    return estimator


def test_model_settings_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENUI_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("GENUI_MODEL_NAME", "new-model")
    monkeypatch.setenv("GENUI_MODEL_ALIAS", "challenger")
    settings = Settings()
    assert settings.tracking_uri == "http://mlflow:5000"
    assert settings.model_name == "new-model"
    assert settings.model_alias == "challenger"


def test_loader_resolves_alias_before_loading_pinned_version() -> None:
    settings = Settings()
    with (
        patch("mlflow.MlflowClient") as client_factory,
        patch("mlflow.set_tracking_uri") as tracking,
        patch("mlflow.set_registry_uri") as registry,
        patch("mlflow.sklearn.load_model", return_value=fake_estimator()) as download,
    ):
        client = client_factory.return_value
        client.get_model_version_by_alias.return_value = SimpleNamespace(version="7", run_id="run")
        model = load_model(settings)
    tracking.assert_called_once_with(settings.tracking_uri)
    registry.assert_called_once_with(settings.tracking_uri)
    client.get_model_version_by_alias.assert_called_once_with(settings.model_name, "champion")
    download.assert_called_once_with("models:/genui-pairwise-quality/7")
    assert model.metadata.version == "7"
    assert model.metadata.run_id == "run"


def test_loader_rejects_model_without_source_run() -> None:
    with (
        patch("mlflow.MlflowClient") as client_factory,
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_registry_uri"),
        patch("mlflow.sklearn.load_model") as download,
    ):
        client_factory.return_value.get_model_version_by_alias.return_value = SimpleNamespace(
            version="7", run_id=None
        )
        with pytest.raises(ValueError, match="training run"):
            load_model(Settings())
    download.assert_not_called()


@pytest.mark.parametrize("classes", [[0], [1], [0, 2], [0, 1, 2]])
def test_wrong_model_classes_are_rejected(classes: list[int]) -> None:
    estimator = fake_estimator()
    estimator.classes_ = classes
    with pytest.raises(ValueError, match="binary preference"):
        LoadedModel(estimator, model_metadata())


@pytest.mark.parametrize(
    ("classes", "probabilities", "probability_a", "winner"),
    [([0, 1], [[0.2, 0.8]], 0.8, "A"), ([1, 0], [[0.2, 0.8]], 0.2, "B")],
)
async def test_prediction_uses_positive_class_and_preserves_input_columns(
    classes: list[int], probabilities: list[list[float]], probability_a: float, winner: str
) -> None:
    estimator = fake_estimator(probabilities)
    estimator.classes_ = classes
    model = LoadedModel(estimator, model_metadata())
    response = await model.predict(**PAYLOAD)
    assert response.probability_a == probability_a
    assert response.winner == winner
    values = estimator.predict_proba.call_args.args[0]
    assert list(values.columns) == ["prompt", "html_a", "html_b"]
    assert values.to_dict(orient="records") == [PAYLOAD]


@pytest.mark.parametrize(
    "probabilities",
    [[], [[0.5]], [[0.5, 0.5], [0.2, 0.8]], [[-1, 2]], [[0.1, 0.1]], [[float("nan"), 0.5]]],
)
async def test_invalid_probabilities_are_rejected(probabilities: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="invalid"):
        await LoadedModel(fake_estimator(probabilities), model_metadata()).predict(**PAYLOAD)


async def test_prediction_does_not_block_event_loop() -> None:
    started, release = threading.Event(), threading.Event()
    estimator = fake_estimator()

    def slow_prediction(_values: object) -> list[list[float]]:
        started.set()
        if not release.wait(timeout=3):
            raise TimeoutError("Prediction was not released")
        return [[0.2, 0.8]]

    estimator.predict_proba.side_effect = slow_prediction
    model = LoadedModel(estimator, model_metadata())
    task = asyncio.create_task(model.predict(**PAYLOAD))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.wait_for(asyncio.sleep(0), 0.1)
        assert not task.done()
    finally:
        release.set()
    assert (await task).winner == "A"
