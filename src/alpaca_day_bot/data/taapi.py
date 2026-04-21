from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("alpaca_day_bot.taapi")

_TAAPI_BASE = "https://api.taapi.io"
_TAAPI_BULK = f"{_TAAPI_BASE}/bulk"


@dataclass(frozen=True)
class TaapiIndicators:
    rsi_1m: float | None
    rsi_15m: float | None
    macd_1m: float | None
    macd_signal_1m: float | None


def _get_json(url: str) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "alpaca-paper-day-bot/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        # Never log the secret token (it is part of the URL in direct mode).
        redacted = url
        if "secret=" in redacted:
            redacted = redacted.split("secret=", 1)[0] + "secret=<redacted>"
        log.warning("taapi_get_failed err=%s url=%s", e, redacted)
        return None


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": "alpaca-paper-day-bot/0.1",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.warning("taapi_post_failed err=%s endpoint=%s", e, url)
        return None


def fetch_taapi_indicators_for_stock(*, secret: str, symbol: str) -> TaapiIndicators:
    """
    TAAPI stocks docs:
    - RSI: https://api.taapi.io/rsi?symbol=AAPL&interval=1h&type=stocks&secret=APIKEY
    - MACD: https://api.taapi.io/macd?... same params
    """
    sym = (symbol or "").strip().upper()
    sec = (secret or "").strip()
    if not sym or not sec:
        return TaapiIndicators(None, None, None, None)

    # Prefer bulk to reduce request count (and avoid 429s).
    bulk_1m = _post_json(
        _TAAPI_BULK,
        {
            "secret": sec,
            "construct": {
                "type": "stocks",
                "symbol": sym,
                "interval": "1m",
                "indicators": [
                    {"indicator": "rsi", "period": 14, "id": "rsi_1m"},
                    {"indicator": "macd", "id": "macd_1m"},
                ],
            },
        },
    )
    bulk_15m = _post_json(
        _TAAPI_BULK,
        {
            "secret": sec,
            "construct": {
                "type": "stocks",
                "symbol": sym,
                "interval": "15m",
                "indicators": [
                    {"indicator": "rsi", "period": 14, "id": "rsi_15m"},
                ],
            },
        },
    )

    def _bulk_get(d: dict[str, Any] | None, id_: str) -> dict[str, Any] | None:
        if not d or not isinstance(d, dict):
            return None
        data = d.get("data")
        if not isinstance(data, list):
            return None
        for row in data:
            if not isinstance(row, dict):
                continue
            if str(row.get("id")) == id_:
                res = row.get("result")
                return res if isinstance(res, dict) else None
        return None

    rsi1 = _bulk_get(bulk_1m, "rsi_1m")
    macd1 = _bulk_get(bulk_1m, "macd_1m")
    rsi15 = _bulk_get(bulk_15m, "rsi_15m")

    def f(d: dict | None, k: str) -> float | None:
        if not d:
            return None
        try:
            return float(d.get(k))
        except Exception:
            return None

    return TaapiIndicators(
        rsi_1m=f(rsi1, "value"),
        rsi_15m=f(rsi15, "value"),
        macd_1m=f(macd1, "valueMACD"),
        macd_signal_1m=f(macd1, "valueMACDSignal"),
    )

