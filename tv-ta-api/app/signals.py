from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

from .indicators import compute_latest, parse_indicator_list


@dataclass(frozen=True)
class TradePlan:
    bias: str  # long|short|neutral
    reason: str
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    risk_per_share: float | None
    r_multiple: float | None
    option_bias: str  # call|put|none


def _last_non_nan(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def build_trade_plan(
    df: pd.DataFrame,
    *,
    rsi_len: int = 14,
    donchian_len: int = 20,
    bb_len: int = 20,
    bb_std: float = 2.0,
    willr_len: int = 14,
    atr_len: int = 14,
    atr_mult: float = 1.5,
    rr: float = 2.0,
) -> tuple[dict[str, float], TradePlan]:
    """
    Rule-based signal (educational). Uses:
      - VWAP positioning (above/below)
      - Donchian breakout (above upper / below lower)
      - RSI / Williams %R as momentum confirmation
      - ATR for stop-loss sizing
    """
    if df is None or df.empty:
        return {}, TradePlan("neutral", "no_data", None, None, None, None, None, "none")

    # Compute a stable set of keys
    indicators_csv = f"rsi:{rsi_len},bbands:{bb_len}-{bb_std},donchian:{donchian_len},willr:{willr_len},vwap,atr:{atr_len}"
    latest = compute_latest(df, parse_indicator_list(indicators_csv))

    close = float(df["close"].dropna().iloc[-1])

    # Pull values with fallbacks
    vwap = latest.get("vwap")
    dcu = latest.get("donchian_upper")
    dcl = latest.get("donchian_lower")
    rsi = latest.get(f"rsi_{rsi_len}")
    willr = latest.get(f"willr_{willr_len}")

    # ATR via pandas_ta directly (compute_latest generic will name it, but we keep it explicit)
    atr_s = ta.atr(df["high"], df["low"], df["close"], length=atr_len)
    atr = _last_non_nan(atr_s) if atr_s is not None else None

    above_vwap = (vwap is not None) and (close > float(vwap))
    below_vwap = (vwap is not None) and (close < float(vwap))

    # Use cross vs previous bar to avoid flagging as "breakout" on every bar sitting above the band.
    prev_close = float(df["close"].dropna().iloc[-2]) if len(df["close"].dropna()) >= 2 else close
    breakout_up = (dcu is not None) and (prev_close < float(dcu)) and (close >= float(dcu))
    breakout_dn = (dcl is not None) and (prev_close > float(dcl)) and (close <= float(dcl))

    rsi_ok_long = (rsi is not None) and (float(rsi) >= 55)
    rsi_ok_short = (rsi is not None) and (float(rsi) <= 45)

    # Williams %R: closer to 0 is stronger; below -80 is oversold.
    willr_ok_long = (willr is not None) and (float(willr) >= -50)
    willr_ok_short = (willr is not None) and (float(willr) <= -50)

    bias = "neutral"
    reason_parts: list[str] = []

    if breakout_up and above_vwap and rsi_ok_long and willr_ok_long:
        bias = "long"
        reason_parts = ["donchian_breakout_up", "above_vwap", "rsi_confirm", "willr_confirm"]
    elif breakout_dn and below_vwap and rsi_ok_short and willr_ok_short:
        bias = "short"
        reason_parts = ["donchian_breakout_down", "below_vwap", "rsi_confirm", "willr_confirm"]
    else:
        # softer setups (no breakout) — keep neutral
        if above_vwap:
            reason_parts.append("above_vwap")
        if below_vwap:
            reason_parts.append("below_vwap")
        if breakout_up:
            reason_parts.append("donchian_breakout_up")
        if breakout_dn:
            reason_parts.append("donchian_breakout_down")

    entry = close
    if atr is None or atr <= 0:
        return latest, TradePlan(
            bias=bias,
            reason=",".join(reason_parts) if reason_parts else "no_setup",
            entry=entry,
            stop_loss=None,
            take_profit=None,
            risk_per_share=None,
            r_multiple=None,
            option_bias="call" if bias == "long" else ("put" if bias == "short" else "none"),
        )

    risk = float(atr) * float(atr_mult)
    if bias == "long":
        stop = entry - risk
        tp = entry + (risk * float(rr))
    elif bias == "short":
        stop = entry + risk
        tp = entry - (risk * float(rr))
    else:
        stop = None
        tp = None

    return latest, TradePlan(
        bias=bias,
        reason=",".join(reason_parts) if reason_parts else "no_setup",
        entry=entry,
        stop_loss=stop,
        take_profit=tp,
        risk_per_share=risk if bias != "neutral" else None,
        r_multiple=float(rr) if bias != "neutral" else None,
        option_bias="call" if bias == "long" else ("put" if bias == "short" else "none"),
    )

