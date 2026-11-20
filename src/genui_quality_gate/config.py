from __future__ import annotations

import math

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENUI_", extra="ignore")

    database_url: SecretStr = SecretStr("postgresql://localhost/genui")
    health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    db_shutdown_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    build_sha: str = "unknown"
    tracking_uri: str = "http://127.0.0.1:5001"
    model_name: str = Field(default="genui-pairwise-quality", min_length=1)
    model_alias: str = Field(default="champion", min_length=1)
    queue_enabled: bool = False
    rabbit_url: SecretStr = SecretStr("amqp://guest:guest@127.0.0.1:5672/")
    queue_name: str = Field(default="genui.tasks", pattern=r"^[a-zA-Z0-9_.-]+$")
    queue_capacity: int = Field(default=20, ge=1, le=100_000)
    backpressure_enabled: bool = True
    worker_concurrency: int = Field(default=1, ge=1, le=32)
    prefetch_count: int = Field(default=1, ge=0, le=1024)
    inference_delay_seconds: float = Field(default=0.25, ge=0, le=30)
    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=0.25, gt=0, le=30)
    publish_timeout_seconds: float = Field(default=0.25, gt=0, le=5)
    api_publish_concurrency: int = Field(default=8, ge=1, le=128)
    worker_shutdown_seconds: float = Field(default=60, ge=5, le=300)

    @model_validator(mode="after")
    def validate_worker_limits(self) -> Settings:
        if self.backpressure_enabled and not 1 <= self.prefetch_count <= self.worker_concurrency:
            raise ValueError("prefetch_count must be between 1 and worker_concurrency")
        if self.worker_shutdown_seconds <= self.inference_delay_seconds + 5:
            raise ValueError("worker_shutdown_seconds must allow inference and result persistence")
        return self

    @property
    def retry_after_seconds(self) -> int:
        return max(1, math.ceil(self.inference_delay_seconds / self.worker_concurrency))
