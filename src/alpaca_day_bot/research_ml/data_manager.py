from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class OHLCV:
    df: pd.DataFrame  # index=DatetimeIndex (UTC), columns=open/high/low/close/volume


def fetch_ohlcv_yfinance(
    *,
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    interval: str = "1m",
) -> OHLCV:
    """
    Fetch OHLCV via yfinance.

    Notes:
    - yfinance intraday has lookback limits; for longer history use 5m/15m/1h/1d.
    - Returns UTC-indexed dataframe with standard column names.
    """
    import yfinance as yf

    t = yf.Ticker(symbol)
    df = t.history(start=start, end=end, interval=interval, auto_adjust=False, actions=False)
    if df is None or df.empty:
        return OHLCV(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))

    df = df.copy()
    # normalize column names
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    # keep only OHLCV
    keep = []
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            keep.append(c)
    df = df[keep]

    # ensure UTC index
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df = df.sort_index()
    return OHLCV(df)

