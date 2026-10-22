from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from genui_quality_gate.config import Settings

if TYPE_CHECKING:
    from pandas import DataFrame


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    alias: str
    run_id: str


class Prediction(BaseModel):
    winner: Literal["A", "B"]
    probability_a: float = Field(ge=0, le=1)
    model: ModelMetadata


class ProbabilityModel(Protocol):
    classes_: Iterable[int]

    def predict_proba(self, values: DataFrame) -> Sequence[Sequence[float]]: ...


@dataclass
class LoadedModel:
    estimator: ProbabilityModel
    metadata: ModelMetadata
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if sorted(self.estimator.classes_) != [0, 1]:
            raise ValueError("Model must predict binary preference labels 0 and 1")

    def predict_sync(self, prompt: str, html_a: str, html_b: str) -> Prediction:
        import pandas as pd

        values = pd.DataFrame([{"prompt": prompt, "html_a": html_a, "html_b": html_b}])
        probabilities = self.estimator.predict_proba(values)
        if len(probabilities) != 1 or len(probabilities[0]) != 2:
            raise ValueError("Model returned an invalid probability shape")
        row = [float(value) for value in probabilities[0]]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in row) or not math.isclose(
            sum(row), 1.0, abs_tol=1e-6
        ):
            raise ValueError("Model returned invalid probabilities")
        probability_a = row[list(self.estimator.classes_).index(1)]
        return Prediction(
            winner="A" if probability_a >= 0.5 else "B",
            probability_a=probability_a,
            model=self.metadata,
        )

    async def predict(self, prompt: str, html_a: str, html_b: str) -> Prediction:
        async with self._lock:
            return await asyncio.to_thread(self.predict_sync, prompt, html_a, html_b)


def load_model(settings: Settings) -> LoadedModel:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_registry_uri(settings.tracking_uri)
    client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
    version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    run_id = version.run_id
    if not run_id:
        raise ValueError("Registered model must reference a training run")
    metadata = ModelMetadata(
        name=settings.model_name,
        version=version.version,
        alias=settings.model_alias,
        run_id=run_id,
    )
    estimator = mlflow.sklearn.load_model(f"models:/{metadata.name}/{metadata.version}")
    return LoadedModel(estimator=cast(ProbabilityModel, estimator), metadata=metadata)
