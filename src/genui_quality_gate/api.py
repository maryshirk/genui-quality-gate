import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from genui_quality_gate.database import ComponentHealth, check_database
from genui_quality_gate.inference import LoadedModel, ModelMetadata, Prediction
from genui_quality_gate.logging import LOGGER_NAME
from genui_quality_gate.version import SERVICE_NAME

router = APIRouter()
logger = logging.getLogger(LOGGER_NAME)


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    service: str = SERVICE_NAME
    version: str
    revision: str
    model: ModelMetadata | None = None


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=10_000)
    html_a: str = Field(min_length=1, max_length=1_000_000)
    html_b: str = Field(min_length=1, max_length=1_000_000)

    @field_validator("prompt", "html_a", "html_b")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not contain only whitespace")
        return value


class HealthComponents(BaseModel):
    postgresql: ComponentHealth


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: HealthComponents


@router.get(
    "/healthz",
    tags=["operations"],
    description="Process liveness, not API versioning or dependency readiness.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/api/v1/version",
    tags=["operations"],
    description="Application package version and revision, with the loaded model identity.",
    response_model_exclude_none=True,
)
async def version(request: Request) -> VersionResponse:
    model: LoadedModel | None = request.app.state.model
    return VersionResponse(
        version=request.app.version,
        revision=request.app.state.settings.build_sha,
        model=model.metadata if model is not None else None,
    )


@router.get(
    "/api/v1/health",
    tags=["operations"],
    description="Query the actual PostgreSQL server and report its version and latency.",
    responses={503: {"model": HealthResponse, "description": "Dependency unavailable"}},
)
async def health(request: Request, response: Response) -> HealthResponse:
    component = await check_database(
        request.app.state.pool, request.app.state.settings.health_timeout_seconds
    )
    healthy = component.status == "ok"
    response.status_code = 200 if healthy else 503
    return HealthResponse(
        status="ok" if healthy else "degraded",
        components=HealthComponents(postgresql=component),
    )


@router.post("/process", tags=["inference"])
async def process(payload: ProcessRequest, request: Request) -> Prediction:
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        return await model.predict(payload.prompt, payload.html_a, payload.html_b)
    except Exception as exc:
        logger.error(
            "inference_failed",
            extra={"error_type": type(exc).__name__, "request_id": request.state.request_id},
        )
        raise HTTPException(status_code=500, detail="Model inference failed") from None
