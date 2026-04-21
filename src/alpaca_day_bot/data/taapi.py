from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("alpaca_day_bot.taapi")

_TAAPI_BASE = "https://api.taapi.io"


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
        log.warning("taapi_get_failed err=%s url=%s", e, url)
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

    def url(endpoint: str, interval: str) -> str:
        q = urllib.parse.urlencode({"secret": sec, "symbol": sym, "interval": interval, "type": "stocks"})
        return f"{_TAAPI_BASE}/{endpoint}?{q}"

    rsi1 = _get_json(url("rsi", "1m"))
    rsi15 = _get_json(url("rsi", "15m"))
    macd1 = _get_json(url("macd", "1m"))

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

