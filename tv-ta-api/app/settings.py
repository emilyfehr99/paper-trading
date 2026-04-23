from __future__ import annotations

import os

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "tv-ta-api"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "30"))
    redis_url: str | None = os.getenv("REDIS_URL") or None  # e.g. redis://localhost:6379/0
    ws_push_interval_seconds: int = int(os.getenv("WS_PUSH_INTERVAL_SECONDS", "5"))


settings = Settings()

