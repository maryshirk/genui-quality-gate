from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENUI_", extra="ignore")

    database_url: SecretStr = SecretStr("postgresql://localhost/genui")
    health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    db_shutdown_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    build_sha: str = "unknown"
