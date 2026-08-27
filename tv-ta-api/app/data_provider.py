from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yfinance as yf

from .symbols import normalize_symbol_for_yfinance

logger = logging.getLogger(__name__)

# Bound concurrent TV/yfinance pulls. Unbounded 4-bot tip scans used to pile up
# inside uvicorn's threadpool and starve /health → false recycle stampedes.
_OHLCV_SLOTS = max(1, int(os.getenv("TVTA_OHLCV_CONCURRENCY", "3")))
_ohlcv_sem = threading.Semaphore(_OHLCV_SLOTS)

Resolution = Literal[
    # TradingView-style
    "1S",
    "5S",
    "15S",
    "30S",
    "1",
    "5",
    "15",
    "30",
    "60",
    "120",
    "240",
    "1D",
    "1W",
    # Back-compat (also accepted)
    "1s",
    "5s",
    "15s",
    "30s",
    "1m",
    "5m",
    "15m",
    "30m",
    "1H",
    "2H",
    "4H",
]


def normalize_resolution(resolution: str) -> Resolution:
    r = resolution.strip()
    mapping: dict[str, str] = {
        "1s": "1S",
        "5s": "5S",
        "15s": "15S",
        "30s": "30S",
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "1H": "60",
        "2h": "120",
        "2H": "120",
        "4h": "240",
        "4H": "240",
        "D": "1D",
        "W": "1W",
    }
    r = mapping.get(r, r)
    allowed = {
        "1S",
        "5S",
        "15S",
        "30S",
        "1",
        "5",
        "15",
        "30",
        "60",
        "120",
        "240",
        "1D",
        "1W",
    }
    if r not in allowed:
        raise ValueError(f"Unsupported resolution: {resolution}")
    return r  # type: ignore[return-value]


@dataclass(frozen=True)
class Bars:
    df: pd.DataFrame  # index is tz-aware timestamps


def ohlcv_provider_mode() -> str:
    """auto | yfinance | tradingview"""
    return os.getenv("TVTA_OHLCV_PROVIDER", "auto").strip().lower()


def _use_tradingview(symbol: str, start_ts: int | None, end_ts: int | None) -> bool:
    mode = ohlcv_provider_mode()
    if mode == "yfinance":
        return False
    if mode == "tradingview":
        return True
    # auto
    if start_ts is not None and end_ts is not None:
        return True
    from .tradingview_provider import is_futures_like_symbol

    return is_futures_like_symbol(symbol)


def _yf_interval(resolution: Resolution) -> str:
    resolution = normalize_resolution(resolution)
    return {
        "1S": "1m",
        "5S": "1m",
        "15S": "1m",
        "30S": "1m",
        "1": "1m",
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "60": "60m",
        "120": "60m",  # will be resampled
        "240": "60m",  # will be resampled
        "1D": "1d",
        "1W": "1wk",
    }[resolution]


def _resample_rule(resolution: Resolution) -> str | None:
    return {
        "120": "2H",
        "240": "4H",
    }.get(resolution)


def _yf_period(resolution: Resolution, count: int) -> str:
    resolution = normalize_resolution(resolution)
    if resolution in {"1S", "5S", "15S", "30S", "1"}:
        return "7d"
    if resolution in {"5", "15", "30"}:
        return "60d"
    if resolution in {"60", "120", "240"}:
        return "60d"
    if resolution == "1D":
        return "2y"
    return "5y"


def _fetch_yfinance(
    symbol: str,
    resolution: Resolution,
    count: int,
    extra_bars: int,
) -> pd.DataFrame:
    yf_symbol = normalize_symbol_for_yfinance(symbol)
    resolution = normalize_resolution(resolution)
    interval = _yf_interval(resolution)
    period = _yf_period(resolution, count=count)

    df = yf.download(
        yf_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns.to_list()]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index, utc=True)
    if "close" in df.columns:
        df = df.dropna(subset=["close"])

    rule = _resample_rule(resolution)
    if rule:
        o = df["open"].resample(rule).first()
        h = df["high"].resample(rule).max()
        l = df["low"].resample(rule).min()
        c = df["close"].resample(rule).last()
        v = df["volume"].resample(rule).sum() if "volume" in df.columns else None
        out = pd.concat([o, h, l, c], axis=1).dropna()
        out.columns = ["open", "high", "low", "close"]
        if v is not None:
            out["volume"] = v
        df = out

    return df.tail(count + extra_bars)


def fetch_ohlcv(
    symbol: str,
    resolution: Resolution,
    count: int,
    extra_bars: int = 200,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Bars:
    """
    Fetch OHLCV bars for indicator computation.

    Provider selection (``TVTA_OHLCV_PROVIDER``):
      - ``auto`` (default): TradingView for futures + explicit date ranges; yfinance otherwise
      - ``tradingview``: always use tvkit / TradingView WebSocket
      - ``yfinance``: legacy 60-day intraday cap

    Deep history (Feb/Mar etc.) uses named quarterly contracts automatically via tvkit;
    no login required. Continuous symbols (MNQ1!) are a secondary fallback.
    """
    with _ohlcv_sem:
        return _fetch_ohlcv_unlocked(
            symbol,
            resolution,
            count,
            extra_bars,
            start_ts=start_ts,
            end_ts=end_ts,
        )


def _fetch_ohlcv_unlocked(
    symbol: str,
    resolution: Resolution,
    count: int,
    extra_bars: int = 200,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Bars:
    resolution = normalize_resolution(resolution)

    if _use_tradingview(symbol, start_ts, end_ts):
        from .tradingview_provider import fetch_ohlcv_tradingview

        df = fetch_ohlcv_tradingview(
            symbol,
            resolution,
            count,
            extra_bars,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if not df.empty:
            logger.info(
                "TradingView OHLCV %s %s bars=%d (%s → %s)",
                symbol,
                resolution,
                len(df),
                df.index.min(),
                df.index.max(),
            )
            return Bars(df=df)
        if ohlcv_provider_mode() == "tradingview":
            return Bars(df=pd.DataFrame())
        logger.warning("TradingView empty for %s — falling back to yfinance", symbol)

    df = _fetch_yfinance(symbol, resolution, count, extra_bars)

    if start_ts is not None:
        df = df[df.index >= pd.to_datetime(start_ts, unit="s", utc=True)]
    if end_ts is not None:
        df = df[df.index <= pd.to_datetime(end_ts, unit="s", utc=True)]

    return Bars(df=df)
