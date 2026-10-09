from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENUI_", extra="ignore")

    database_url: SecretStr = SecretStr("postgresql://localhost/genui")
    health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    db_shutdown_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    build_sha: str = "unknown"
    model_enabled: bool = False
    tracking_uri: str = "http://127.0.0.1:5001"
    model_name: str = Field(default="genui-pairwise-quality", min_length=1)
    model_alias: str = Field(default="champion", min_length=1)
