"""Full-bar enrichment for backtest (MACD histogram + Alligator + helpers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from .bill_williams import macd_histogram_latest, williams_alligator_latest
from .stoch_v_rsi import stoch_v_rsi_series
from .tv_aroon import tv_aroon

# Columns returned per bar for bot backtest / TVTA confirm (flat dict per timestamp).
ENRICH_OUTPUT_KEYS = (
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_line_prev",
    "macd_signal_prev",
    "macd_hist",
    "macd_hist_prev",
    "macd_hist_rising",
    "macd_hist_falling_bars",
    "macd_bar_green",
    "macd_cross_green",
    "macd_cross_red",
    "macd_line_cross_green",
    "macd_line_cross_red",
    "macd_hist_fading",
    "alligator_jaw",
    "alligator_teeth",
    "alligator_lips",
    "alligator_stack_bullish",
    "alligator_awake",
    "alligator_sleeping",
    "atr",
    "rvol",
    "adx",
    "stochrsi_k",
    "stochrsi_d",
    "stochrsi_k_prev",
    "stochrsi_d_prev",
    "stochrsi_k_prev2",
    "stochrsi_d_prev2",
    "aroon_up",
    "aroon_down",
    "aroon_osc",
)


def bars_to_dataframe(bars: list[dict]) -> pd.DataFrame:
    """Convert API OHLCV payloads to a sorted UTC DataFrame."""
    if not bars:
        return pd.DataFrame()
    rows = []
    for b in bars:
        ts = pd.to_datetime(int(b["t"]), unit="s", utc=True)
        rows.append(
            {
                "timestamp": ts,
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "volume": float(b.get("v") or b.get("volume") or 0),
            }
        )
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="last")]


def enrich_ohlcv_frame(
    df: pd.DataFrame,
    *,
    alligator_mode: str = "williams",
    awake_mult: float = 0.1,
    sleep_mult: float = 0.3,
) -> pd.DataFrame:
    """
    Compute per-bar indicators on supplied OHLCV (no yfinance).

    alligator_mode:
      - williams: pandas_ta Alligator (TVTA batch / TradingView-style)
      - sma_shift: shifted SMA 13/8/5 (legacy bot strategy lines)
    """
    if df is None or len(df) < 3:
        return pd.DataFrame()

    out = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            return pd.DataFrame()

    close = out["close"]
    rsi_overlay, stoch_k, stoch_d = stoch_v_rsi_series(close)
    out["rsi_14"] = rsi_overlay
    out["stochrsi_k"] = stoch_k
    out["stochrsi_d"] = stoch_d
    out["stochrsi_k_prev"] = stoch_k.shift(1)
    out["stochrsi_d_prev"] = stoch_d.shift(1)
    out["stochrsi_k_prev2"] = stoch_k.shift(2)
    out["stochrsi_d_prev2"] = stoch_d.shift(2)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        out["macd"] = macd["MACD_12_26_9"]
        out["macd_signal"] = macd["MACDs_12_26_9"]
        out["macd_hist"] = macd["MACDh_12_26_9"]
    else:
        out["macd"] = np.nan
        out["macd_signal"] = np.nan
        out["macd_hist"] = np.nan

    out["atr"] = ta.atr(out["high"], out["low"], close, length=14)
    out["vol_sma_20"] = ta.sma(out["volume"], length=20)
    out["rvol"] = out["volume"] / out["vol_sma_20"].replace(0, np.nan)
    adx_df = ta.adx(out["high"], out["low"], close, length=14)
    out["adx"] = adx_df["ADX_14"] if adx_df is not None else np.nan

    if alligator_mode == "sma_shift":
        out["alligator_jaw"] = ta.sma(close, length=13).shift(8)
        out["alligator_teeth"] = ta.sma(close, length=8).shift(5)
        out["alligator_lips"] = ta.sma(close, length=5).shift(3)
        lips, teeth, jaw = out["alligator_lips"], out["alligator_teeth"], out["alligator_jaw"]
        out["alligator_stack_bullish"] = (
            (lips > teeth) & (teeth > jaw)
        ).astype(float)
    else:
        ag = ta.alligator(close)
        if ag is not None and not ag.empty:
            jaw_c = next((c for c in ag.columns if str(c).startswith("AGj")), None)
            teeth_c = next((c for c in ag.columns if str(c).startswith("AGt")), None)
            lips_c = next((c for c in ag.columns if str(c).startswith("AGl")), None)
            if jaw_c and teeth_c and lips_c:
                out["alligator_jaw"] = ag[jaw_c]
                out["alligator_teeth"] = ag[teeth_c]
                out["alligator_lips"] = ag[lips_c]
                out["alligator_stack_bullish"] = (
                    (out["alligator_lips"] > out["alligator_teeth"])
                    & (out["alligator_teeth"] > out["alligator_jaw"])
                ).astype(float)

    out["macd_line_prev"] = out["macd"].shift(1)
    out["macd_signal_prev"] = out["macd_signal"].shift(1)
    out["macd_hist_prev"] = out["macd_hist"].shift(1)
    out["macd_hist_rising"] = out["macd_hist"] > out["macd_hist_prev"]
    falling = (out["macd_hist"] < out["macd_hist_prev"]).astype(int)
    count = 0
    falling_bars: list[int] = []
    for v in falling:
        count = count + 1 if v else 0
        falling_bars.append(count)
    out["macd_hist_falling_bars"] = falling_bars

    hist = out["macd_hist"]
    prev = out["macd_hist_prev"]
    out["macd_bar_green"] = (hist > 0).astype(float)
    out["macd_cross_green"] = ((prev <= 0) & (hist > 0)).astype(float)
    out["macd_cross_red"] = ((prev >= 0) & (hist < 0)).astype(float)
    mlp = out["macd_line_prev"]
    msp = out["macd_signal_prev"]
    out["macd_line_cross_green"] = ((mlp <= msp) & (out["macd"] > out["macd_signal"])).astype(
        float
    )
    out["macd_line_cross_red"] = ((mlp >= msp) & (out["macd"] < out["macd_signal"])).astype(
        float
    )
    out["macd_hist_fading"] = ((hist > 0) & (~out["macd_hist_rising"])).astype(float)

    spread = out[["alligator_jaw", "alligator_teeth", "alligator_lips"]].max(axis=1) - out[
        ["alligator_jaw", "alligator_teeth", "alligator_lips"]
    ].min(axis=1)
    out["alligator_awake"] = spread > (awake_mult * out["atr"].replace(0, np.nan))
    out["alligator_sleeping"] = spread < (sleep_mult * out["atr"].replace(0, np.nan))

    aroon = tv_aroon(out["high"], out["low"], length=14)
    out["aroon_up"] = aroon["aroon_up"]
    out["aroon_down"] = aroon["aroon_down"]
    out["aroon_osc"] = aroon["aroon_osc"]

    return out


def frame_to_points(df: pd.DataFrame, *, include_ohlcv: bool = True) -> list[dict]:
    """Export OHLCV + indicator columns per bar for JSON responses."""
    if df.empty:
        return []
    pts: list[dict] = []
    for ts, row in df.iterrows():
        ind: dict[str, float] = {}
        for k in ENRICH_OUTPUT_KEYS:
            if k not in row.index:
                continue
            v = row[k]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv:
                ind[k] = fv
        pt: dict = {"t": int(pd.Timestamp(ts).timestamp()), "indicators": ind}
        if include_ohlcv:
            for src, dst in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v")):
                if src not in row.index:
                    continue
                try:
                    fv = float(row[src])
                except (TypeError, ValueError):
                    continue
                if fv == fv:
                    pt[dst] = fv
        pts.append(pt)
    return pts


def indicators_at_timestamp(df: pd.DataFrame, as_of: int) -> tuple[int, dict[str, float]]:
    """Latest indicator values at or before ``as_of`` unix seconds."""
    if df.empty:
        return 0, {}
    ts = pd.to_datetime(as_of, unit="s", utc=True)
    sub = df[df.index <= ts]
    if sub.empty:
        return 0, {}
    last = sub.iloc[-1]
    ts_out = int(pd.Timestamp(sub.index[-1]).timestamp())
    ind = {}
    for k in ENRICH_OUTPUT_KEYS:
        if k not in last.index:
            continue
        try:
            fv = float(last[k])
        except (TypeError, ValueError):
            continue
        if fv == fv:
            ind[k] = fv
    return ts_out, ind
