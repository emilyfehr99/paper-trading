"""
STOCH V RSI overlay — matches TradingView study ``STOCH V RSI``.

Pine reference::
    rsi1 = rsi(src, lengthRSI)
    k = sma(stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK)
    d = sma(k, smoothD)

White line = ``k``, orange line = ``d``.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def stoch_v_rsi_series(
    close: pd.Series,
    *,
    length_rsi: int = 14,
    length_stoch: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(overlay_rsi, k, d)`` per bar."""
    rsi = ta.rsi(close, length=length_rsi)
    if rsi is None:
        rsi = pd.Series(index=close.index, dtype=float)
    lowest = rsi.rolling(length_stoch, min_periods=length_stoch).min()
    highest = rsi.rolling(length_stoch, min_periods=length_stoch).max()
    denom = highest - lowest
    raw = rsi - lowest
    stoch_raw = raw.divide(denom).mul(100.0)
    stoch_raw = stoch_raw.where(denom != 0, 0.0)
    k = stoch_raw.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return rsi, k, d


def stoch_v_rsi_latest(
    df: pd.DataFrame,
    *,
    length_rsi: int = 14,
    length_stoch: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, float]:
    """Latest bar + prior bars for cross detection."""
    nan = float("nan")
    out = {
        "stochrsi_k": nan,
        "stochrsi_d": nan,
        "stochrsi_k_prev": nan,
        "stochrsi_d_prev": nan,
        "stochrsi_k_prev2": nan,
        "stochrsi_d_prev2": nan,
        "rsi_14": nan,
    }
    if df.empty or "close" not in df.columns:
        return out

    rsi, k, d = stoch_v_rsi_series(
        df["close"],
        length_rsi=length_rsi,
        length_stoch=length_stoch,
        smooth_k=smooth_k,
        smooth_d=smooth_d,
    )

    def _last(s: pd.Series, n: int = 0) -> float:
        s = s.dropna()
        if len(s) <= n:
            return nan
        return float(s.iloc[-1 - n])

    out["rsi_14"] = _last(rsi)
    out["stochrsi_k"] = _last(k)
    out["stochrsi_d"] = _last(d)
    out["stochrsi_k_prev"] = _last(k, 1)
    out["stochrsi_d_prev"] = _last(d, 1)
    out["stochrsi_k_prev2"] = _last(k, 2)
    out["stochrsi_d_prev2"] = _last(d, 2)
    return out
