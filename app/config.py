"""Application configuration settings."""

from __future__ import annotations

from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration container for environment-driven settings."""

    stats_api_base_url: str = Field(
        "https://api.balldontlie.io/v1",
        description="Base URL for the free NBA stats provider.",
    )
    stats_api_key: str | None = Field(
        default=None,
        description="Optional API key for providers that require authentication.",
    )
    request_timeout_seconds: int = Field(
        10, description="HTTP timeout when calling external stats APIs."
    )
    default_lookback_games: int = Field(
        10, description="Default number of historical games to build features from."
    )
    provider_max_retries: int = Field(
        1, description="Number of times to retry a failed stats API request."
    )

    class Config:
        env_prefix = "MLPICKS_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
