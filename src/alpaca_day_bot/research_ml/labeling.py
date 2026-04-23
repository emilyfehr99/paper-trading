from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabeledFrame:
    X: pd.DataFrame
    y_tb: pd.Series  # triple-barrier label: 1 (tp), -1 (sl), 0 (timeout)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI(14), MACD(12,26,9), Bollinger Bands(20,2).
    Expects columns open/high/low/close/volume, index DatetimeIndex.
    """
    import pandas_ta as ta

    out = df.copy()
    out["rsi_14"] = ta.rsi(out["close"], length=14)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        # columns often: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        for c in macd.columns:
            if c.lower().startswith("macd"):
                out["macd"] = macd[c]
                break
        for c in macd.columns:
            if "macds" in c.lower() or "signal" in c.lower():
                out["macd_signal"] = macd[c]
                break
        for c in macd.columns:
            if "macdh" in c.lower() or "hist" in c.lower():
                out["macd_hist"] = macd[c]
                break

    bb = ta.bbands(out["close"], length=20, std=2.0)
    if bb is not None:
        # columns often: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        for c in bb.columns:
            cl = c.lower()
            if cl.startswith("bbl"):
                out["bb_low"] = bb[c]
            elif cl.startswith("bbm"):
                out["bb_mid"] = bb[c]
            elif cl.startswith("bbu"):
                out["bb_high"] = bb[c]
            elif cl.startswith("bbp"):
                out["bb_pct"] = bb[c]

    # simple returns / volatility context
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["vol_20"] = out["ret_1"].rolling(20).std()
    return out


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    max_bars: int = 5,
) -> pd.Series:
    """
    Label each bar t by scanning the next `max_bars` bars:
    - 1 if high hits entry*(1+tp_pct) first
    - -1 if low hits entry*(1-sl_pct) first
    - 0 if neither hit within max_bars (timeout)
    """
    if df is None or df.empty:
        return pd.Series([], dtype=int)
    if not all(c in df.columns for c in ("high", "low", "close")):
        raise ValueError("df must contain high, low, close")

    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    n = len(df)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            y[i] = 0
            continue
        tp = entry * (1.0 + float(tp_pct))
        sl = entry * (1.0 - float(sl_pct))

        # lookahead window (exclusive of i)
        j_end = min(n, i + 1 + int(max_bars))
        out = 0
        for j in range(i + 1, j_end):
            hi = high[j]
            lo = low[j]
            if np.isfinite(hi) and hi >= tp:
                out = 1
                break
            if np.isfinite(lo) and lo <= sl:
                out = -1
                break
        y[i] = out

    return pd.Series(y, index=df.index, name="y_tb", dtype=int)


def build_dataset(
    df: pd.DataFrame,
    *,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    max_bars: int = 5,
) -> LabeledFrame:
    feat = add_technical_features(df)
    y_tb = triple_barrier_labels(feat, tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars)

    # Drop rows with missing feature core
    X = feat[
        [
            "close",
            "volume",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_low",
            "bb_mid",
            "bb_high",
            "bb_pct",
            "ret_1",
            "ret_5",
            "vol_20",
        ]
    ].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return LabeledFrame(X=X, y_tb=y_tb)

