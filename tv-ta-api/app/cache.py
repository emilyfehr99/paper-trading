from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


@dataclass(frozen=True)
class CacheEntry:
    expires_at: float
    value: Any


class InMemoryTTLCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheEntry(expires_at=time.time() + self.ttl_seconds, value=value)


class Cache:
    def __init__(self, ttl_seconds: int, redis_url: str | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self._memory = InMemoryTTLCache(ttl_seconds=ttl_seconds)
        self._redis = None
        if redis_url and redis is not None:
            try:
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get_json(self, key: str) -> Optional[dict[str, Any]]:
        if self._redis is not None:
            raw = self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        val = self._memory.get(key)
        return val

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        if self._redis is not None:
            self._redis.setex(key, self.ttl_seconds, json.dumps(value, separators=(",", ":")))
            return
        self._memory.set(key, value)

