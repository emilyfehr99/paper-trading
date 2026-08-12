"""Persist TradingView session cookies + auth_token — refresh without per-request browser reads."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SESSION = (
    Path(__file__).resolve().parents[2] / "state" / "tradingview_session.json"
)
_MIN_TOKEN_LENGTH = 10
# TradingView blocks bare httpx requests (403); browser UA is required for cookie refresh.
_TV_HOMEPAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def session_file_path() -> Path:
    raw = os.getenv("TVKIT_SESSION_FILE", "").strip()
    return Path(raw) if raw else _DEFAULT_SESSION


def load_session() -> dict[str, Any]:
    path = session_file_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read session file %s: %s", path, exc)
        return {}


def save_session(data: dict[str, Any]) -> None:
    path = session_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated_at"] = datetime.now(tz=UTC).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _valid_cached_token(session: dict[str, Any] | None = None) -> str | None:
    """Return persisted auth_token when it looks usable (avoids cookie→homepage refresh)."""
    data = session if session is not None else load_session()
    token = str(data.get("auth_token") or os.getenv("TVKIT_AUTH_TOKEN", "").strip())
    if len(token) >= _MIN_TOKEN_LENGTH:
        return token
    return None


def get_live_auth_kwargs() -> dict[str, Any]:
    """
    Auth kwargs for tvkit OHLCV — prefer persisted session over live browser reads.

    Order: session file auth_token → session file cookies → TVKIT_AUTH_TOKEN env
    → TVKIT_BROWSER (optional bootstrap fallback).

    Auth token is preferred over cookies because cookie refresh hits the TradingView
    homepage (bot-blocked without a browser User-Agent) while the token works on WS.
    """
    session = load_session()
    token = _valid_cached_token(session)
    if token:
        return {"auth_token": token}

    cookies = session.get("cookies")
    if isinstance(cookies, dict) and cookies.get("sessionid"):
        return {"cookies": {str(k): str(v) for k, v in cookies.items()}}

    token = os.getenv("TVKIT_AUTH_TOKEN", "").strip()
    if token:
        return {"auth_token": str(token)}

    browser = os.getenv("TVKIT_BROWSER", "").strip() or None
    profile = os.getenv("TVKIT_BROWSER_PROFILE", "").strip() or None
    if browser:
        kwargs: dict[str, Any] = {"browser": browser}
        if profile:
            kwargs["browser_profile"] = profile
        return kwargs

    return {}


async def _fetch_profile_from_cookies(cookies: dict[str, str]) -> dict[str, Any]:
    """Fetch TradingView user profile — uses browser UA (bare httpx gets 403)."""
    import httpx
    from tvkit.auth.profile_parser import ProfileParser

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        cookies=cookies,
        headers=_TV_HOMEPAGE_HEADERS,
        follow_redirects=True,
    ) as client:
        response = await client.get("https://www.tradingview.com/")
    if response.status_code >= 500:
        response.raise_for_status()
    return ProfileParser.parse(response.text)


async def refresh_session(*, allow_browser_bootstrap: bool = False) -> dict[str, Any]:
    """
    Refresh auth_token from persisted sessionid via HTTP (no browser).

    If cookies are stale and allow_browser_bootstrap, try one browser extract.
    """
    from tvkit.auth.cookie_provider import CookieProvider

    path = session_file_path()
    session = load_session()
    cookies: dict[str, str] = {}
    if isinstance(session.get("cookies"), dict):
        cookies = {str(k): str(v) for k, v in session["cookies"].items()}

    if cookies.get("sessionid"):
        try:
            profile = await _fetch_profile_from_cookies(cookies)
            auth_token = str(profile["auth_token"])
            if len(auth_token) < _MIN_TOKEN_LENGTH:
                raise ValueError("auth_token missing or too short after profile parse")
            session["cookies"] = cookies
            session["auth_token"] = auth_token
            save_session(session)
            logger.info("TradingView session refreshed from disk cookies (%s)", path)
            return {"ok": True, "source": "session_file", "path": str(path)}
        except Exception as exc:
            logger.warning("Session cookie refresh failed: %s", exc)

    cached = _valid_cached_token(session)
    if cached:
        logger.info("Using cached auth_token (cookie refresh unavailable)")
        return {"ok": True, "source": "cached_auth_token", "path": str(path)}

    browser = os.getenv("TVKIT_BROWSER", "chrome" if allow_browser_bootstrap else "").strip()
    if allow_browser_bootstrap and browser:
        try:
            cp = CookieProvider()
            cookies = cp.extract(browser, os.getenv("TVKIT_BROWSER_PROFILE", "").strip() or None)
            profile = await _fetch_profile_from_cookies(cookies)
            session = {
                "cookies": cookies,
                "auth_token": str(profile["auth_token"]),
            }
            save_session(session)
            logger.info("TradingView session bootstrapped from browser=%s", browser)
            return {"ok": True, "source": "browser_bootstrap", "path": str(path)}
        except Exception as exc:
            logger.warning("Browser bootstrap failed: %s", exc)
            return {"ok": False, "reason": str(exc)[:200]}

    if not cookies.get("sessionid"):
        return {
            "ok": False,
            "reason": "no_session_file — run: python3 scripts/tvkit_refresh_session.py --bootstrap",
        }
    return {"ok": False, "reason": "session_expired — re-run --bootstrap after logging into TradingView"}
