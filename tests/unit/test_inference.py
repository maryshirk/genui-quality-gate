import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from asgi_lifespan import LifespanManager

from genui_quality_gate.config import Settings
from genui_quality_gate.inference import LoadedModel, ModelMetadata, load_model
from genui_quality_gate.main import create_app

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


def fake_pool() -> SimpleNamespace:
    return SimpleNamespace(close=AsyncMock(), terminate=Mock())


def test_model_settings_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENUI_MODEL_ENABLED", "true")
    monkeypatch.setenv("GENUI_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("GENUI_MODEL_NAME", "new-model")
    monkeypatch.setenv("GENUI_MODEL_ALIAS", "challenger")
    settings = Settings()
    assert settings.model_enabled is True
    assert settings.tracking_uri == "http://mlflow:5000"
    assert settings.model_name == "new-model"
    assert settings.model_alias == "challenger"


def test_loader_resolves_alias_before_loading_pinned_version() -> None:
    settings = Settings(model_enabled=True)
    registered_version = SimpleNamespace(version="7", run_id="source-run")
    with (
        patch("mlflow.MlflowClient") as client_factory,
        patch("mlflow.set_tracking_uri") as tracking,
        patch("mlflow.set_registry_uri") as registry,
        patch("mlflow.sklearn.load_model", return_value=fake_estimator()) as download,
    ):
        client = client_factory.return_value
        client.get_model_version_by_alias.return_value = registered_version
        model = load_model(settings)
    tracking.assert_called_once_with(settings.tracking_uri)
    registry.assert_called_once_with(settings.tracking_uri)
    client_factory.assert_called_once_with(
        tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri
    )
    client.get_model_version_by_alias.assert_called_once_with(settings.model_name, "champion")
    download.assert_called_once_with("models:/genui-pairwise-quality/7")
    assert model.metadata.version == "7"
    assert model.metadata.run_id == "source-run"


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
            load_model(Settings(model_enabled=True))
    download.assert_not_called()


@pytest.mark.parametrize("classes", [[0], [1], [0, 2], [0, 1, 2]])
def test_wrong_model_classes_are_rejected(classes: list[int]) -> None:
    estimator = fake_estimator()
    estimator.classes_ = classes
    with pytest.raises(ValueError, match="binary preference"):
        LoadedModel(estimator, model_metadata())


@pytest.mark.asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probabilities",
    [[], [[0.5]], [[0.5, 0.5], [0.2, 0.8]], [[-1, 2]], [[0.1, 0.1]], [[float("nan"), 0.5]]],
)
async def test_invalid_probabilities_are_rejected(probabilities: list[list[float]]) -> None:
    model = LoadedModel(fake_estimator(probabilities), model_metadata())
    with pytest.raises(ValueError, match="invalid"):
        await model.predict(**PAYLOAD)


@pytest.mark.asyncio
async def test_disabled_model_is_not_loaded_and_process_returns_503() -> None:
    loader = Mock()
    app = create_app(
        Settings(model_enabled=False),
        pool_factory=AsyncMock(return_value=fake_pool()),
        model_loader=loader,
    )
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/process", json=PAYLOAD)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model is not loaded"
        assert "model" not in (await client.get("/api/v1/version")).json()
    loader.assert_not_called()


@pytest.mark.asyncio
async def test_process_preserves_original_html_and_prompt() -> None:
    payload = {
        "prompt": "  Create a shop\n",
        "html_a": "\n  <h1>Shop A</h1>\t\n",
        "html_b": "\t<h1>Shop B</h1>  ",
    }
    estimator = fake_estimator()
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(return_value=fake_pool()),
        model_loader=Mock(return_value=LoadedModel(estimator, model_metadata())),
    )
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/process", json=payload)
        assert response.status_code == 200
    values = estimator.predict_proba.call_args.args[0]
    assert values.to_dict(orient="records") == [payload]


@pytest.mark.asyncio
async def test_startup_loads_once_outside_event_loop_and_requests_use_same_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    estimator = fake_estimator()
    loaded = LoadedModel(estimator, model_metadata("3"))
    thread_ids = []

    def loader(_settings: Settings) -> LoadedModel:
        thread_ids.append(threading.get_ident())
        return loaded

    factory = Mock(side_effect=loader)
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(return_value=fake_pool()),
        model_loader=factory,
    )
    factory.assert_not_called()
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post("/process", json=PAYLOAD)
        factory.side_effect = lambda _: LoadedModel(fake_estimator(), model_metadata("4"))
        second = await client.post("/process", json=PAYLOAD)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert first.json()["model"] == model_metadata("3").model_dump()
        assert (await client.get("/api/v1/version")).json()["model"]["version"] == "3"
    factory.assert_called_once()
    assert len(thread_ids) == 1
    assert thread_ids[0] != threading.get_ident()
    assert estimator.predict_proba.call_count == 2
    assert app.state.model is None
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    loaded_log = next(record for record in records if record["event"] == "model_loaded")
    assert loaded_log["model_name"] == "genui-pairwise-quality"
    assert loaded_log["model_version"] == "3"
    assert loaded_log["model_alias"] == "champion"
    assert loaded_log["model_run_id"] == "training-run"


@pytest.mark.asyncio
async def test_failed_model_startup_is_sanitized_and_closes_pool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool = fake_pool()
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(return_value=pool),
        model_loader=Mock(side_effect=RuntimeError("password-in-tracking-uri")),
    )
    with pytest.raises(RuntimeError, match="Unable to load the configured MLflow model"):
        async with LifespanManager(app):
            pytest.fail("Startup must fail without a model")
    pool.close.assert_awaited_once()
    assert app.state.model is None
    assert app.state.pool is None
    assert "password-in-tracking-uri" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_restart_loads_new_version_from_alias() -> None:
    loader = Mock(return_value=LoadedModel(fake_estimator(), model_metadata("1")))
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(side_effect=lambda _: fake_pool()),
        model_loader=loader,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with LifespanManager(app):
            assert (await client.post("/process", json=PAYLOAD)).json()["model"]["version"] == "1"
            loader.return_value = LoadedModel(fake_estimator(), model_metadata("2"))
            assert (await client.post("/process", json=PAYLOAD)).json()["model"]["version"] == "1"
        async with LifespanManager(app):
            assert (await client.post("/process", json=PAYLOAD)).json()["model"]["version"] == "2"
    assert loader.call_count == 2


@pytest.mark.asyncio
async def test_slow_prediction_does_not_block_liveness() -> None:
    started = threading.Event()
    release = threading.Event()

    def predict(_values: object) -> list[list[float]]:
        started.set()
        if not release.wait(timeout=3):
            raise TimeoutError("Test prediction was not released")
        return [[0.2, 0.8]]

    estimator = fake_estimator()
    estimator.predict_proba.side_effect = predict
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(return_value=fake_pool()),
        model_loader=Mock(return_value=LoadedModel(estimator, model_metadata())),
    )
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        prediction = asyncio.create_task(client.post("/process", json=PAYLOAD))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            assert (await asyncio.wait_for(client.get("/healthz"), timeout=0.5)).status_code == 200
            assert not prediction.done()
        finally:
            release.set()
        assert (await prediction).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"prompt": ""},
        {"prompt": " "},
        {"prompt": "x" * 10_001},
        {"html_a": "x" * 1_000_001},
        {"html_b": None},
        {"extra": "field"},
    ],
)
async def test_invalid_input_is_rejected_without_prediction(updates: dict[str, object]) -> None:
    estimator = fake_estimator()
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(return_value=fake_pool()),
        model_loader=Mock(return_value=LoadedModel(estimator, model_metadata())),
    )
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/process", json=PAYLOAD | updates)
        assert response.status_code == 422
    estimator.predict_proba.assert_not_called()


@pytest.mark.asyncio
async def test_prediction_error_is_sanitized(capsys: pytest.CaptureFixture[str]) -> None:
    estimator = fake_estimator()
    estimator.predict_proba.side_effect = ValueError("private-html-input")
    app = create_app(
        Settings(model_enabled=True),
        pool_factory=AsyncMock(return_value=fake_pool()),
        model_loader=Mock(return_value=LoadedModel(estimator, model_metadata())),
    )
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/process", json=PAYLOAD)
        assert response.status_code == 500
        assert response.json()["detail"] == "Model inference failed"
        assert "private-html-input" not in response.text
        assert (await client.get("/healthz")).status_code == 200
    output = capsys.readouterr().out
    assert "private-html-input" not in output
    assert PAYLOAD["html_a"] not in output
    assert PAYLOAD["prompt"] not in output
