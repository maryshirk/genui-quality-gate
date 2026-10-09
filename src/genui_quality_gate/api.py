from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from genui_quality_gate.database import ComponentHealth, check_database
from genui_quality_gate.version import SERVICE_NAME

router = APIRouter()


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    service: str = SERVICE_NAME
    version: str
    revision: str


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
    description="Application package version, not the v1 API contract or the model version.",
)
async def version(request: Request) -> VersionResponse:
    return VersionResponse(
        version=request.app.version, revision=request.app.state.settings.build_sha
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
