import logging
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel

from genui_quality_gate.database import ComponentHealth, check_database
from genui_quality_gate.logging import LOGGER_NAME
from genui_quality_gate.queue import QueueUnavailable, TaskPublisher
from genui_quality_gate.task_store import TaskRepository
from genui_quality_gate.tasks import AcceptedTask, StoredTask, TaskEnvelope, TaskPayload
from genui_quality_gate.version import SERVICE_NAME

router = APIRouter()
logger = logging.getLogger(LOGGER_NAME)


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
    description="Application package version and build revision.",
    response_model_exclude_none=True,
)
async def version(request: Request) -> VersionResponse:
    return VersionResponse(
        version=request.app.version,
        revision=request.app.state.settings.build_sha,
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


@router.post("/process", status_code=202, tags=["tasks"])
@router.post("/api/v1/tasks", status_code=202, tags=["tasks"])
async def process(
    payload: TaskPayload,
    request: Request,
    response: Response,
    idempotency_key: Annotated[UUID | None, Header()] = None,
) -> AcceptedTask:
    publisher: TaskPublisher | None = request.app.state.publisher
    retry_after = str(request.app.state.settings.retry_after_seconds)
    if publisher is None:
        raise HTTPException(
            status_code=503,
            detail="Task queue unavailable",
            headers={"Retry-After": retry_after},
        )
    task = TaskEnvelope(payload=payload)
    if idempotency_key is not None:
        task.task_id = idempotency_key
    try:
        await publisher.publish(task)
    except QueueUnavailable:
        logger.warning("task_load_shed", extra={"request_id": request.state.request_id})
        raise HTTPException(
            status_code=503,
            detail="Task queue full or temporarily unavailable",
            headers={"Retry-After": retry_after, "X-Task-ID": str(task.task_id)},
        ) from None
    location = f"/api/v1/tasks/{task.task_id}"
    response.headers["Location"] = location
    return AcceptedTask(task_id=task.task_id, status_url=location)


@router.get("/api/v1/tasks/{task_id}", tags=["tasks"])
async def task_result(task_id: UUID, request: Request) -> StoredTask:
    store: TaskRepository | None = request.app.state.task_store
    if store is None:
        raise HTTPException(status_code=503, detail="Result storage unavailable")
    try:
        result = await store.get(task_id)
    except Exception:
        raise HTTPException(status_code=503, detail="Result storage unavailable") from None
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Result not available yet or task ID unknown",
        )
    return result
