from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("alpaca_day_bot.tvta")


@dataclass(frozen=True)
class TvtaIndicators:
    rsi_1m: float | None
    rsi_15m: float | None
    macd_1m: float | None
    macd_signal_1m: float | None
    raw_1m: dict[str, float]
    raw_15m: dict[str, float]


def _norm_base(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("missing base_url")
    return b


def _tv_symbol(sym: str, prefix: str) -> str:
    s = (sym or "").strip().upper()
    p = (prefix or "").strip().upper()
    if not s:
        return s
    if ":" in s:
        return s
    if not p:
        return s
    return f"{p}:{s}"


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 25.0,
    retries: int = 2,
    backoff_s: float = 0.75,
) -> dict[str, Any] | None:
    try:
        data = json.dumps(payload).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(int(retries) + 1):
            try:
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
                with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                    return json.loads(resp.read().decode())
            except Exception as e:
                last_err = e
                if attempt < int(retries):
                    time.sleep(float(backoff_s) * (2**attempt))
                    continue
                break
    except Exception as e:
        log.warning("tvta_post_failed err=%s url=%s", e, url)
        return None
    if last_err is not None:
        log.warning("tvta_post_failed err=%s url=%s", last_err, url)
    return None


def fetch_tvta_indicators_for_stock(
    *,
    base_url: str,
    symbol: str,
    symbol_prefix: str = "NYSE",
) -> TvtaIndicators:
    """
    Calls your tv-ta-api batch endpoint twice (1m + 15m) to fetch:
      - rsi:14
      - macd (12/26/9)

    Maps results into the same key-shape we used for TAAPI so downstream code/ML can consume it.
    """
    b = _norm_base(base_url)
    tvsym = _tv_symbol(symbol, symbol_prefix)
    if not tvsym:
        return TvtaIndicators(None, None, None, None, raw_1m={}, raw_15m={})

    url = f"{b}/api/ta/batch"

    def _parse(resp: dict[str, Any] | None) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if not resp or not isinstance(resp, dict):
            return out
        results = resp.get("results")
        if not isinstance(results, list):
            return out
        for row in results:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or "").strip()
            inds = row.get("indicators")
            if not sym or not isinstance(inds, dict):
                continue
            clean: dict[str, float] = {}
            for k, v in inds.items():
                try:
                    clean[str(k)] = float(v)
                except Exception:
                    continue
            out[sym] = clean
        return out

    def _fetch(resolution: str, indicators: list[str]) -> dict[str, float]:
        resp = _post_json(
            url,
            {"resolution": resolution, "items": [{"symbol": tvsym, "indicators": indicators}]},
        )
        return _parse(resp).get(tvsym, {})

    # Intraday data providers sometimes fail for specific symbols/resolutions
    # (especially on cold starts / near session boundaries). Prefer 1m/15m
    # but fall back to 5m/30m so the bot still has usable momentum proxies.

    # 1m: RSI + MACD (fallback 5m)
    m1 = _fetch("1", ["rsi:14", "macd"])
    if not m1:
        m1 = _fetch("5", ["rsi:14", "macd"])

    # 15m: RSI only (fallback 30m)
    m15 = _fetch("15", ["rsi:14"])
    if not m15:
        m15 = _fetch("30", ["rsi:14"])

    rsi_1m = m1.get("rsi_14")
    rsi_15m = m15.get("rsi_14")
    macd_1m = m1.get("macd")
    macd_signal_1m = m1.get("macd_signal")

    return TvtaIndicators(
        rsi_1m=rsi_1m,
        rsi_15m=rsi_15m,
        macd_1m=macd_1m,
        macd_signal_1m=macd_signal_1m,
        raw_1m=m1,
        raw_15m=m15,
    )

