from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yfinance as yf

from .symbols import normalize_symbol_for_yfinance


Resolution = Literal[
    # TradingView-style
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


def _yf_interval(resolution: Resolution) -> str:
    resolution = normalize_resolution(resolution)
    return {
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
    # yfinance intraday has limits; keep it simple for a demo provider.
    resolution = normalize_resolution(resolution)
    if resolution in {"1", "5", "15", "30", "60", "120", "240"}:
        return "60d"
    if resolution == "1D":
        return "2y"
    return "5y"


def fetch_ohlcv(symbol: str, resolution: Resolution, count: int, extra_bars: int = 200) -> Bars:
    """
    Fetch OHLCV bars for indicator computation.
    Returns at least `count` rows (usually more due to indicator warmup needs).
    """
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
        return Bars(df=pd.DataFrame())

    # yfinance may return a MultiIndex columns frame (e.g. ("Close","NVDA")).
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        # Prefer the price field names (Open/High/Low/Close/Volume) as the column names.
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

    df = df.tail(count + extra_bars)
    return Bars(df=df)

