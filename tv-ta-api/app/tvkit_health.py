"""Probe TradingView / tvkit auth — used by /health/tradingview."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def probe_tradingview_auth(
    *,
    symbol: str = "MNQ1",
    timeout_sec: float = 25.0,
) -> dict[str, Any]:
    """
    Fetch a tiny OHLCV slice via tvkit. Success implies live TradingView auth.

    Uses persisted session file (TVKIT_SESSION_FILE) — refresh via HTTP, not browser.
    """
    from .tvkit_session import refresh_session

    refresh_result = await refresh_session(allow_browser_bootstrap=False)
    if not refresh_result.get("ok"):
        return {
            "ok": False,
            "reason": refresh_result.get("reason", "session_refresh_failed"),
            "symbol": symbol,
        }

    # Session HTTP refresh succeeded → auth is valid. Bar fetch is a bonus check
    # (MNQ1! often returns empty on weekends / thin globex windows).
    from .data_provider import normalize_resolution
    from .tradingview_provider import (
        _fetch_count_once,
        _fetch_range_once,
        normalize_symbol_for_tradingview,
    )

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=3)
    resolution = normalize_resolution("15")
    tv_symbol = normalize_symbol_for_tradingview(symbol)
    auth_mode = _auth_mode_label()

    for fetch in (
        lambda: _fetch_range_once(symbol, resolution, start, end, tv_symbol=tv_symbol),
        lambda: _fetch_count_once(symbol, resolution, 50),
    ):
        try:
            df = await asyncio.wait_for(fetch(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            continue
        except Exception as exc:
            logger.debug("TradingView probe fetch skipped: %s", exc)
            continue
        if df is not None and not df.empty:
            last = df.index[-1]
            return {
                "ok": True,
                "symbol": tv_symbol,
                "bars": int(len(df)),
                "last_bar": last.isoformat() if hasattr(last, "isoformat") else str(last),
                "auth_mode": auth_mode,
            }

    return {
        "ok": True,
        "symbol": tv_symbol,
        "bars": 0,
        "auth_mode": auth_mode,
        "note": "session_refresh_ok; bar probe empty (weekend/globex)",
    }


def _auth_mode_label() -> str:
    import os

    from .tvkit_session import load_session, session_file_path

    session = load_session()
    if session.get("cookies"):
        return f"session_file:{session_file_path().name}"
    if session.get("auth_token"):
        return "session_token"
    if os.getenv("TVKIT_BROWSER", "").strip():
        return f"browser:{os.getenv('TVKIT_BROWSER', '').strip()}"
    if os.getenv("TVKIT_AUTH_TOKEN", "").strip():
        return "static_token"
    return "anonymous"
