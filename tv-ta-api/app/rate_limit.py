"""Adaptive pacing so bulk TV fetches stay under TradingView throttle thresholds.

Goal: never *trip* hard rate limits — slow down before TV does, speed up when healthy.
This is not a ToS bypass; it is controlled concurrency with backoff.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class AdaptiveTvLimiter:
    """Token-gated spacing between TV WebSocket segment fetches."""

    min_interval: float = field(default_factory=lambda: _env_float("TVTA_BULK_MIN_INTERVAL", 0.35))
    max_interval: float = field(default_factory=lambda: _env_float("TVTA_BULK_MAX_INTERVAL", 12.0))
    interval: float = field(default_factory=lambda: _env_float("TVTA_BULK_START_INTERVAL", 0.75))
    min_concurrency: int = field(default_factory=lambda: _env_int("TVTA_BULK_MIN_CONCURRENCY", 1))
    max_concurrency: int = field(default_factory=lambda: _env_int("TVTA_BULK_MAX_CONCURRENCY", 2))
    concurrency: int = field(default_factory=lambda: _env_int("TVTA_BULK_START_CONCURRENCY", 1))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _next_at: float = field(default=0.0, repr=False)
    _success_streak: int = field(default=0, repr=False)
    _throttle_events: int = field(default=0, repr=False)

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)

    def success(self) -> None:
        self._success_streak += 1
        # Gently speed up after a healthy streak.
        if self._success_streak >= 4:
            self.interval = max(self.min_interval, self.interval * 0.88)
            if self._success_streak >= 8 and self.concurrency < self.max_concurrency:
                self.concurrency += 1
                self._success_streak = 0
                logger.info(
                    "TV bulk limiter speed-up interval=%.2fs concurrency=%d",
                    self.interval,
                    self.concurrency,
                )

    def throttle(self, reason: str = "") -> None:
        self._throttle_events += 1
        self._success_streak = 0
        self.interval = min(self.max_interval, max(self.interval * 2.0, self.min_interval * 2))
        self.concurrency = max(self.min_concurrency, self.concurrency - 1)
        logger.warning(
            "TV bulk limiter throttle reason=%s interval=%.2fs concurrency=%d events=%d",
            reason or "unknown",
            self.interval,
            self.concurrency,
            self._throttle_events,
        )

    def snapshot(self) -> dict[str, float | int]:
        return {
            "interval": round(self.interval, 3),
            "concurrency": self.concurrency,
            "throttle_events": self._throttle_events,
        }
