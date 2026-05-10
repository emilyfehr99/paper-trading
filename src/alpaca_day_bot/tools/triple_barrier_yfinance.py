"""
Triple-barrier labeling from OHLCV (yfinance or any frame with high/low/close).

Mirrors ``main._label_signals_triple_barrier`` bar-walk semantics.
"""
from __future__ import annotations

import pandas as pd


def triple_barrier_outcome_from_bars(
    dfx: pd.DataFrame,
    *,
    act: str,
    entry_f: float,
    tp_f: float,
    sl_f: float,
) -> tuple[str, float]:
    """
    Walk bars in chronological order; first touch of TP or SL wins (BUY vs SHORT rules).

    Returns ``(outcome, px_at_eval)`` where ``px_at_eval`` is the last bar's close (mark at end of path).
    """
    if dfx is None or getattr(dfx, "empty", True):
        return "timeout", float("nan")
    if not all(c in dfx.columns for c in ("high", "low", "close")):
        return "timeout", float("nan")

    dfx = dfx.copy()
    dfx.index = pd.to_datetime(dfx.index, utc=True, errors="coerce")
    dfx = dfx.sort_index()
    outcome = "timeout"
    # Same as live bot: mark-to-market at last bar in window, even if TP/SL hit earlier.
    px_at_eval = float(dfx["close"].iloc[-1])
    act_u = (act or "").strip().upper()
    try:
        for _idx, row in dfx.iterrows():
            hi = float(row["high"])
            lo = float(row["low"])
            if act_u == "SHORT":
                if lo <= tp_f:
                    outcome = "tp"
                    break
                if hi >= sl_f:
                    outcome = "sl"
                    break
            else:
                if hi >= tp_f:
                    outcome = "tp"
                    break
                if lo <= sl_f:
                    outcome = "sl"
                    break
    except Exception:
        outcome = "timeout"
        px_at_eval = float(dfx["close"].iloc[-1])
    return outcome, px_at_eval


def realized_return_from_last_close(*, act: str, entry_f: float, px_at_eval: float) -> float:
    act_u = (act or "").strip().upper()
    if act_u == "SHORT":
        return (entry_f - px_at_eval) / entry_f
    return (px_at_eval - entry_f) / entry_f
