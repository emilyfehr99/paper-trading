from __future__ import annotations

import json
import logging
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
        log.warning("tvta_post_failed err=%s url=%s", e, url)
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

    # 1m: RSI + MACD
    resp_1m = _post_json(
        url,
        {
            "resolution": "1",
            "items": [{"symbol": tvsym, "indicators": ["rsi:14", "macd"]}],
        },
    )
    m1 = _parse(resp_1m).get(tvsym, {})

    # 15m: RSI only (keep it cheap)
    resp_15m = _post_json(
        url,
        {
            "resolution": "15",
            "items": [{"symbol": tvsym, "indicators": ["rsi:14"]}],
        },
    )
    m15 = _parse(resp_15m).get(tvsym, {})

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

