"""Williams Alligator + MACD histogram (TradingView-style bar colors)."""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def williams_alligator_latest(df: pd.DataFrame) -> dict[str, float]:
    """
    Bill Williams Alligator via pandas_ta (SMMA jaws/teeth/lips with TV displacements).
    Keys match alpaca_day_bot local feature names.
    """
    nan = float("nan")
    out = {
        "alligator_jaw": nan,
        "alligator_teeth": nan,
        "alligator_lips": nan,
        "alligator_stack_bullish": 0.0,
    }
    if df.empty or "close" not in df.columns:
        return out

    ag = ta.alligator(df["close"])
    if ag is None or ag.empty:
        return out

    jaw_c = next((c for c in ag.columns if str(c).startswith("AGj")), None)
    teeth_c = next((c for c in ag.columns if str(c).startswith("AGt")), None)
    lips_c = next((c for c in ag.columns if str(c).startswith("AGl")), None)
    if not (jaw_c and teeth_c and lips_c):
        return out

    def _last(col: str) -> float:
        s = ag[col].dropna()
        return float(s.iloc[-1]) if not s.empty else nan

    jaw, teeth, lips = _last(jaw_c), _last(teeth_c), _last(lips_c)
    out["alligator_jaw"] = jaw
    out["alligator_teeth"] = teeth
    out["alligator_lips"] = lips
    if jaw == jaw and teeth == teeth and lips == lips:
        out["alligator_stack_bullish"] = 1.0 if (lips > teeth > jaw) else 0.0
    return out


def macd_histogram_latest(
    df: pd.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """
    MACD line, signal, histogram (green/red bars on TradingView), and flip/fade flags.
    """
    nan = float("nan")
    out = {
        "macd": nan,
        "macd_signal": nan,
        "macd_hist": nan,
        "macd_hist_prev": nan,
        "macd_line_prev": nan,
        "macd_signal_prev": nan,
        "macd_hist_rising": 0.0,
        "macd_hist_fading": 0.0,
        "macd_cross_green": 0.0,
        "macd_cross_red": 0.0,
        "macd_line_cross_green": 0.0,
        "macd_line_cross_red": 0.0,
        "macd_bar_green": 0.0,
    }
    if df.empty or "close" not in df.columns:
        return out

    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        return out

    macd_col = next((c for c in macd_df.columns if str(c).startswith("MACD_")), None)
    sig_col = next((c for c in macd_df.columns if str(c).startswith("MACDs_")), None)
    hist_col = next((c for c in macd_df.columns if str(c).startswith("MACDh_")), None)

    def _last_series(col: str | None) -> float:
        if not col:
            return nan
        s = macd_df[col].dropna()
        return float(s.iloc[-1]) if not s.empty else nan

    line_s = macd_df[macd_col].dropna() if macd_col else pd.Series(dtype=float)
    sig_s = macd_df[sig_col].dropna() if sig_col else pd.Series(dtype=float)
    if line_s.empty or sig_s.empty:
        return out

    line_curr = float(line_s.iloc[-1])
    line_prev = float(line_s.iloc[-2]) if len(line_s) >= 2 else line_curr
    sig_curr = float(sig_s.iloc[-1])
    sig_prev = float(sig_s.iloc[-2]) if len(sig_s) >= 2 else sig_curr
    out["macd"] = line_curr
    out["macd_signal"] = sig_curr
    out["macd_line_prev"] = line_prev
    out["macd_signal_prev"] = sig_prev
    out["macd_line_cross_green"] = 1.0 if (line_prev <= sig_prev and line_curr > sig_curr) else 0.0
    out["macd_line_cross_red"] = 1.0 if (line_prev >= sig_prev and line_curr < sig_curr) else 0.0

    if not hist_col:
        return out

    hist = macd_df[hist_col].dropna()
    if len(hist) < 1:
        return out

    curr = float(hist.iloc[-1])
    prev = float(hist.iloc[-2]) if len(hist) >= 2 else curr
    out["macd_hist"] = curr
    out["macd_hist_prev"] = prev
    rising = curr > prev
    out["macd_hist_rising"] = 1.0 if rising else 0.0
    out["macd_bar_green"] = 1.0 if curr > 0 else 0.0
    out["macd_cross_green"] = 1.0 if (prev <= 0 < curr) else 0.0
    out["macd_cross_red"] = 1.0 if (prev >= 0 > curr) else 0.0
    # Lighter green: still positive but histogram shrinking
    out["macd_hist_fading"] = 1.0 if (curr > 0 and not rising) else 0.0
    return out
