from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data_provider import Resolution, fetch_ohlcv, normalize_resolution


TopMetric = Literal["daytrade_score", "dollar_volume", "volatility"]


DEFAULT_DAYTRADE_SYMBOLS: list[str] = [
    # US large caps / high-volume names often daytraded (demo defaults)
    "NASDAQ:NVDA",
    "NASDAQ:TSLA",
    "NASDAQ:AMD",
    "NASDAQ:AAPL",
    "NASDAQ:MSFT",
    "NASDAQ:AMZN",
    "NASDAQ:META",
    "NASDAQ:GOOGL",
    "NASDAQ:QQQ",
    "NYSE:SPY",
    # Common liquid, often lower-priced daytrading candidates (price varies over time)
    "NYSE:F",
    "NYSE:T",
    "NYSE:PFE",
    "NYSE:NOK",
    "NYSE:SOFI",
    "NYSE:SNAP",
    "NYSE:NIO",
    "NYSE:LCID",
    "NYSE:RIVN",
    "NYSE:PLUG",
    "NYSE:CCL",
    "NYSE:BB",
    "NYSE:U",
    "NYSE:AFRM",
    "NYSE:PATH",
    "NYSE:HOOD",
    # Leveraged/volatility ETFs (often active daytrading vehicles; price varies)
    "NYSE:UVXY",
    "NYSE:VXX",
    "NYSE:SQQQ",
    "NYSE:TZA",
    "NYSE:LABU",
    "NYSE:SOXS",
]


def _to_ticker(symbol: str) -> str:
    # Keep original symbol (exchange prefix) in API response, but yfinance normalization
    # is handled by the data provider.
    return symbol.strip()


def _realized_volatility(close: pd.Series) -> float:
    r = np.log(close).diff().dropna()
    if r.empty:
        return float("nan")
    return float(r.std() * np.sqrt(252.0))


def _avg_dollar_volume(close: pd.Series, volume: pd.Series, window: int) -> float:
    if close.empty or volume.empty:
        return float("nan")
    dv = (close * volume).dropna()
    if dv.empty:
        return float("nan")
    return float(dv.tail(window).mean())


def score_daytrading(df: pd.DataFrame) -> dict[str, float]:
    """
    Returns a few features for ranking:
      - dollar_volume_5: mean(close*volume) over last 5 bars
      - vol_ann: realized volatility (annualized) over available window
    """
    if df is None or df.empty:
        return {"dollar_volume_5": float("nan"), "vol_ann": float("nan"), "last_close": float("nan")}
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    return {
        "dollar_volume_5": _avg_dollar_volume(close, volume, window=5),
        "vol_ann": _realized_volatility(close.tail(30)),
        "last_close": float(close.dropna().iloc[-1]) if not close.dropna().empty else float("nan"),
    }


def rank_top_daytrading(
    symbols: list[str],
    resolution: Resolution,
    limit: int,
    metric: TopMetric,
    max_price: float | None = None,
) -> dict:
    resolution = normalize_resolution(resolution)

    rows: list[dict] = []
    latest_ts = 0
    for s in symbols:
        sym = _to_ticker(s)
        bars = fetch_ohlcv(symbol=sym, resolution=resolution, count=120, extra_bars=0)
        df = bars.df
        if not df.empty:
            latest_ts = max(latest_ts, int(pd.Timestamp(df.index[-1]).timestamp()))
        feats = score_daytrading(df)
        rows.append({"symbol": sym, **feats})

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return {"timestamp": 0, "resolution": resolution, "metric": metric, "results": []}

    if max_price is not None:
        out_df = out_df[out_df["last_close"].astype(float) <= float(max_price)]
        if out_df.empty:
            return {
                "timestamp": latest_ts,
                "resolution": resolution,
                "metric": metric,
                "results": [],
            }

    # Ranking: higher is better. Use percentile ranks then combine.
    out_df["rank_dv"] = out_df["dollar_volume_5"].rank(pct=True, ascending=True)
    out_df["rank_vol"] = out_df["vol_ann"].rank(pct=True, ascending=True)

    if metric == "dollar_volume":
        out_df["score"] = out_df["dollar_volume_5"]
    elif metric == "volatility":
        out_df["score"] = out_df["vol_ann"]
    else:
        # Daytrading-friendly: prefer BOTH high liquidity and high movement.
        # Multiplicative blend strongly penalizes low liquidity or low vol.
        out_df["score"] = (out_df["rank_dv"] * out_df["rank_vol"]).fillna(0.0)

    out_df = out_df.sort_values("score", ascending=False).head(limit)

    results = []
    for _, r in out_df.iterrows():
        results.append(
            {
                "symbol": r["symbol"],
                "score": float(r["score"]) if pd.notna(r["score"]) else None,
                "dollar_volume_5": float(r["dollar_volume_5"]) if pd.notna(r["dollar_volume_5"]) else None,
                "vol_ann": float(r["vol_ann"]) if pd.notna(r["vol_ann"]) else None,
                "last_close": float(r["last_close"]) if pd.notna(r["last_close"]) else None,
            }
        )

    return {
        "timestamp": latest_ts,
        "resolution": resolution,
        "metric": metric,
        "results": results,
    }

