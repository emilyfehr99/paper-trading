from __future__ import annotations

import os

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "tv-ta-api"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "30"))
    # Completed sessions (history end_ts in the past) — safe to cache longer for replay.
    historical_cache_ttl_seconds: int = int(
        os.getenv("HISTORICAL_CACHE_TTL_SECONDS", "3600")
    )
    # Live tip windows (end near now / count-only): keep short so post-close cache
    # cannot serve a pre-close tip for a full CACHE_TTL_SECONDS after :05.
    live_tip_cache_ttl_seconds: int = int(os.getenv("LIVE_TIP_CACHE_TTL_SECONDS", "5"))
    redis_url: str | None = os.getenv("REDIS_URL") or None  # e.g. redis://localhost:6379/0
    ws_push_interval_seconds: int = int(os.getenv("WS_PUSH_INTERVAL_SECONDS", "5"))


settings = Settings()

