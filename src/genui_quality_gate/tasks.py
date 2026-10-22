from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from genui_quality_gate.inference import Prediction


class TaskPayload(BaseModel):
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


class TaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID = Field(default_factory=uuid4)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: TaskPayload


class AcceptedTask(BaseModel):
    task_id: UUID
    status: Literal["ACCEPTED"] = "ACCEPTED"
    status_url: str


class StoredTask(BaseModel):
    task_id: UUID
    status: Literal["PROCESSING", "SUCCEEDED", "FAILED"]
    attempts: int
    result: Prediction | None = None
    error_type: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    inference_ms: float | None = None
    end_to_end_ms: float | None = None


def message_identity(body: Any, message_id: str) -> tuple[UUID, str]:
    content = body.get("payload", body) if isinstance(body, dict) else body
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
    candidate = body.get("task_id", message_id) if isinstance(body, dict) else message_id
    try:
        task_id = UUID(str(candidate))
    except (ValueError, TypeError):
        task_id = uuid5(UUID("8b564731-7c3a-4adc-b662-cc30746381fe"), message_id or digest)
    return task_id, digest
