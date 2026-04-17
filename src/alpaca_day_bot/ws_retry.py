"""Helpers for Alpaca websocket reconnect behavior."""

from __future__ import annotations


def is_connection_or_rate_limit(exc: BaseException) -> bool:
    msg = (str(exc) + " " + repr(exc)).lower()
    return (
        "connection limit" in msg
        or "rate limit" in msg
        or "too many connection" in msg
        or "max connection" in msg
        or ("limit" in msg and "exceed" in msg)
    )
