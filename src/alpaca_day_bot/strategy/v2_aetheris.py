from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.stats import linregress
from alpaca_day_bot.strategy.base import BaseStrategy, StrategySignal

log = logging.getLogger("alpaca_day_bot")

from numba import njit

@njit(fastmath=True, cache=True)
def kalman_update_step(p, q, r, x, measurement):
    p = p + q
    k = p / (p + r)
    x = x + k * (measurement - x)
    p = (1 - k) * p
    return p, k, x

@njit(fastmath=True, cache=True)
def compute_kalman_series(values, q, r, initial_value):
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    p = 1.0
    x = initial_value
    for i in range(n):
        p = p + q
        k = p / (p + r)
        x = x + k * (values[i] - x)
        p = (1 - k) * p
        out[i] = x
    return out

# --- JIT COMPILER PRE-FLIGHT WARMUP ---
# Run mock compilations at import level to cache binary instructions and eliminate opening open cross latency spikes
try:
    dummy_data = np.zeros(10, dtype=np.float64)
    _ = compute_kalman_series(dummy_data, 1e-5, 1e-2, 0.0)
    _, _, _ = kalman_update_step(1.0, 1e-5, 1e-2, 0.0, 0.0)
except Exception:
    pass

class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2, estimated_error=1.0, initial_value=0.0):
        self.q = process_variance
        self.r = measurement_variance
        self.p = estimated_error
        self.x = initial_value
        self.k = 0

    def update(self, measurement):
        self.p, self.k, self.x = kalman_update_step(self.p, self.q, self.r, self.x, float(measurement))
        return self.x

class V2AetherisSignalEngine(BaseStrategy):
    """
    Aetheris Alpha V2 Strategy:
    - Multi-regime awareness (Trend vs. Chop).
    - Trend following using Alligator (SMMA) + VWAP.
    - Mean reversion using RSI + Bollinger Band fades.
    - Dynamic volatility scaling using ATR.
    """

    def __init__(
        self,
        *,
        adx_trend_threshold: float = 25.0,
        adx_chop_threshold: float = 20.0,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        enable_shorts: bool = False,
        atr_len: int = 14,
        atr_mult: float = 2.0,
        min_confidence: float = 0.75,
        aggressive_mode: bool = False,
    ) -> None:
        self._adx_trend_threshold = adx_trend_threshold
        self._adx_chop_threshold = adx_chop_threshold
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._enable_shorts = enable_shorts
        self._atr_len = atr_len
        self._atr_mult = atr_mult
        self._min_confidence = min_confidence
        self._aggressive_mode = aggressive_mode
        self._last_rho_regime = "mean_reversion"

    def _get_regime(self, df: pd.DataFrame) -> str:
        """Determines the current market regime based on Alligator alignment (primary) and ADX (fallback)."""
        # Primary: Use Alligator alignment for regime detection (user's proven indicator)
        if all(col in df.columns for col in ["alligator_jaw_ratio", "alligator_teeth_ratio", "alligator_lips_ratio"]):
            jaw_ratio = df["alligator_jaw_ratio"].iloc[-1]
            teeth_ratio = df["alligator_teeth_ratio"].iloc[-1]
            lips_ratio = df["alligator_lips_ratio"].iloc[-1]
            
            # Alligator aligned bullish = trend regime
            if jaw_ratio > 1.0 and teeth_ratio > 1.0 and lips_ratio > 1.0:
                return "trend"
            # Alligator aligned bearish = trend regime (downward)
            elif jaw_ratio < 1.0 and teeth_ratio < 1.0 and lips_ratio < 1.0:
                return "trend"
            # Alligator converging = neutral regime
            elif "alligator_convergence_index" in df.columns:
                aci = df["alligator_convergence_index"].iloc[-1]
                if aci < 0.5:  # Low convergence = lines spreading apart = trend
                    return "trend"
                elif aci > 0.8:  # High convergence = lines coming together = chop
                    return "chop"
                else:
                    return "neutral"
        
        # Fallback: Use ADX-based regime detection
        if "ADX_14" not in df.columns:
            return "unknown"
        
        adx = df["ADX_14"].iloc[-1]
        if adx >= self._adx_trend_threshold:
            return "trend"
        elif adx <= self._adx_chop_threshold:
            return "chop"
        else:
            return "neutral"

    def decide(self, *, symbol: str, df_1m, df_15m, market_regime: str = "neutral", spy_perf: float | None = None, order_book: list[float] | None = None) -> StrategySignal | None:
        if df_1m is None or len(df_1m) < 35:
            return None

        # Work on a copy to avoid side effects
        df = df_1m.copy()
        
        # Ensure data is clean and indexed
        try:
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            if hasattr(df.index, "duplicated"):
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass

        # --- KALMAN NOISE REDUCTION ---
        initial_val = float(df["close"].iloc[0])
        df["kalman_close"] = compute_kalman_series(df["close"].to_numpy(dtype=np.float64), 1e-5, 1e-2, initial_val)
        
        # --- ADVANCED SIGNAL SCIPY PEAK/TROUGH DETECTION ---
        from scipy.signal import argrelextrema
        kalman_np = df["kalman_close"].to_numpy(dtype=np.float64)
        peaks = argrelextrema(kalman_np, np.greater)[0]
        troughs = argrelextrema(kalman_np, np.less)[0]
        
        n_bars = len(df)
        
        # Save rolling structural peak/trough distances with safe cold-start fallback (50.0)
        df["distance_since_last_local_peak"] = float(n_bars - 1 - peaks[-1]) if len(peaks) > 0 else 50.0
        df["distance_since_last_local_trough"] = float(n_bars - 1 - troughs[-1]) if len(troughs) > 0 else 50.0
        
        # Indicator Calculation
        # Alligator: SMMA(13,8,5) shifted forward. pandas-ta doesn't have smma directly, using RMA as proxy.
        hl2 = (df["high"] + df["low"]) / 2.0
        
        jaw_rma = ta.rma(hl2, length=13)
        teeth_rma = ta.rma(hl2, length=8)
        lips_rma = ta.rma(hl2, length=5)

        if jaw_rma is None or teeth_rma is None or lips_rma is None:
            return None

        df["alligator_jaw"] = jaw_rma.shift(8)
        df["alligator_teeth"] = teeth_rma.shift(5)
        df["alligator_lips"] = lips_rma.shift(3)
        
        # RSI & MACD
        df["rsi_14"] = ta.rsi(df["kalman_close"], length=14)
        macd = ta.macd(df["kalman_close"])
        if macd is not None:
            macd = macd.rename(columns={
                "MACD_12_26_9": "macd",
                "MACDs_12_26_9": "macd_signal",
                "MACDh_12_26_9": "macd_hist"
            })
            df = pd.concat([df, macd], axis=1)

        # --- ADVANCED QUANT FEATURES (AR(1) & VWAP Z-SCORE) ---
        # 1. VWAP (Economic Anchor)
        typical_price = (df['high'] + df['low'] + df['kalman_close']) / 3
        df['vwap_anchor'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # 2. Z-Score to VWAP (Using Kalman Smooth Price)
        rolling_std = df['kalman_close'].rolling(20).std()
        df['z_score_vwap'] = (df['kalman_close'] - df['vwap_anchor']) / rolling_std

        # 3. AR(1) Regime Filter (Rho)
        #rho < 0.5: Stationary/Mean-Reverting | rho -> 1.0: Random Walk/Trend
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Use a fast vectorized rolling covariance/variance calculation for Rho
        # Mathematically identical to running linregress on lag 1 over a rolling window of 20
        lag_ret = df['log_ret'].shift(1)
        df['ar1_rho'] = lag_ret.rolling(19).cov(df['log_ret']) / lag_ret.rolling(19).var()
        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
            
        # EMA features for ML
        ema_9_raw = ta.ema(df["close"], length=9)
        ema_21_raw = ta.ema(df["close"], length=21)
        
        # Store raw values for reference
        df["ema_9"] = ema_9_raw
        df["ema_21"] = ema_21_raw
        df["ema"] = df["ema_9"] # proxy
        
        # Store as ratios relative to current price for ML (normalized)
        df["ema_9_ratio"] = df["close"] / ema_9_raw
        df["ema_21_ratio"] = df["close"] / ema_21_raw
        df["ema_9_21_bias"] = df["ema_9"] - df["ema_21"]
        
        # --- WILLIAMS ALLIGATOR ---
        # Jaw (Blue): 13-period SMA smoothed, shifted 8
        # Teeth (Red): 8-period SMA smoothed, shifted 5
        # Lips (Green): 5-period SMA smoothed, shifted 3
        alligator_jaw_raw = ta.sma(df["close"], length=13).shift(8)
        alligator_teeth_raw = ta.sma(df["close"], length=8).shift(5)
        alligator_lips_raw = ta.sma(df["close"], length=5).shift(3)
        
        # Store as ratios relative to current price for meaningful comparison
        # Values > 1.0 indicate price is above the Alligator line (bullish)
        # Values < 1.0 indicate price is below the Alligator line (bearish)
        df["alligator_jaw"] = alligator_jaw_raw
        df["alligator_teeth"] = alligator_teeth_raw
        df["alligator_lips"] = alligator_lips_raw
        df["alligator_jaw_ratio"] = df["close"] / alligator_jaw_raw
        df["alligator_teeth_ratio"] = df["close"] / alligator_teeth_raw
        df["alligator_lips_ratio"] = df["close"] / alligator_lips_raw
        
        # Calculate Alligator Convergence Index (ACI)
        alligator_cols = df[["alligator_jaw", "alligator_teeth", "alligator_lips"]]
        df["alligator_std"] = alligator_cols.std(axis=1)
        atr_temp = ta.atr(df["high"], df["low"], df["close"], length=14).fillna(1.0)
        df["alligator_convergence_index"] = df["alligator_std"] / atr_temp
        
        # Alligator trend flags
        df["alligator_trend_up"] = (df["alligator_lips"] > df["alligator_teeth"]) & (df["alligator_teeth"] > df["alligator_jaw"])
        df["alligator_trend_down"] = (df["alligator_lips"] < df["alligator_teeth"]) & (df["alligator_teeth"] < df["alligator_jaw"])
        
        # Alligator alignment (price above all three lines = strong bullish signal)
        df["alligator_aligned_bullish"] = (df["alligator_jaw_ratio"] > 1.0) & (df["alligator_teeth_ratio"] > 1.0) & (df["alligator_lips_ratio"] > 1.0)
        
        # Alligator alignment (price below all three lines = strong bearish signal)
        df["alligator_aligned_bearish"] = (df["alligator_jaw_ratio"] < 1.0) & (df["alligator_teeth_ratio"] < 1.0) & (df["alligator_lips_ratio"] < 1.0)

        # --- MACD (Moving Average Convergence Divergence) ---
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            macd_line_raw = macd["MACD_12_26_9"]
            macd_signal_raw = macd["MACDs_12_26_9"]
            macd_hist_raw = macd["MACDh_12_26_9"]
            
            df["macd_line"] = macd_line_raw
            df["macd_signal"] = macd_signal_raw
            df["macd_hist"] = macd_hist_raw
            
            # Store MACD as ratio to price for ML (normalized)
            df["macd_line_ratio"] = macd_line_raw / df["close"]
            df["macd_signal_ratio"] = macd_signal_raw / df["close"]
            df["macd_hist_ratio"] = macd_hist_raw / df["close"]
        else:
            df["macd_line"] = df["macd_signal"] = df["macd_hist"] = 0.0
            df["macd_line_ratio"] = df["macd_signal_ratio"] = df["macd_hist_ratio"] = 0.0
        # ATR for risk scaling
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        
        # Trend indicators
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            adx = adx.rename(columns={"ADX_14": "adx"})
            df = pd.concat([df, adx], axis=1)
            
        # ATR for Volatility
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self._atr_len)
        
        # VWAP (anchored to NY session)
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = tp * df["volume"]
        idx_ny = df.index.tz_convert("America/New_York")
        d = idx_ny.date
        vwap_raw = pv.groupby(d).cumsum() / df["volume"].groupby(d).cumsum()
        df["vwap"] = vwap_raw
        
        # Store VWAP as ratio to current price for ML (normalized)
        df["vwap_ratio"] = df["close"] / vwap_raw
        
        # VWAP bands using the calculated ATR
        atr_ref = df["atr"].fillna(df["close"] * 0.01)
        df["vwap_band_1"] = df["vwap"] + atr_ref
        df["vwap_band_2"] = df["vwap"] + 2 * atr_ref
        df["vwap_band_3"] = df["vwap"] + 3 * atr_ref
        df["vwap_band_neg1"] = df["vwap"] - atr_ref
        df["vwap_band_neg2"] = df["vwap"] - 2 * atr_ref
        df["vwap_band_neg3"] = df["vwap"] - 3 * atr_ref

        # --- MULTI-OSCILLATOR FILTERS (Williams %R & Stochastic Confirmations) ---
        willr = ta.willr(df["high"], df["low"], df["close"], length=14)
        df["willr"] = willr if willr is not None else -50.0

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
        else:
            df["stoch_k"] = 50.0
            df["stoch_d"] = 50.0

        # On-Balance Volume (OBV) and Chaikin Money Flow (CMF)
        obv_raw = ta.obv(df["close"], df["volume"])
        df["obv"] = obv_raw
        df["obv_ema"] = ta.ema(df["obv"], length=20)
        
        # Store OBV as rate of change for ML (normalized, not cumulative)
        df["obv_roc_5"] = obv_raw.pct_change(5)
        df["obv_roc_10"] = obv_raw.pct_change(10)
        
        # CMF (Institutional Accumulation/Distribution)
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        df["cmf"] = cmf if cmf is not None else 0.0

        # Keltner Channels (Volatility Breakouts)
        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2.0)
        if kc is not None:
            df = pd.concat([df, kc], axis=1)

        # SuperTrend (Primary Inertia)
        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        if st is not None:
            st = st.rename(columns={"SUPERT_10_3.0": "supertrend", "SUPERTd_10_3.0": "supertrend_dir"})
            df = pd.concat([df, st], axis=1)
            # Store SuperTrend as ratio to current price for ML (normalized)
            df["supertrend_ratio"] = df["close"] / df["supertrend"]

        # Relative Volume (RVOL)
        df["vol_sma_20"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["vol_sma_20"]
        df["rvol"] = df["volume_ratio"]
        
        # Volume imbalance (L2 proxy)
        hl_diff = df['high'] - df['low']
        safe_hl = np.where(hl_diff > 0, hl_diff, 1e-4)
        df['volume_imbalance'] = df['volume'] * (df['close'] - df['open']) / safe_hl
        df['volume_imbalance_norm'] = df['volume_imbalance'] / df['vol_sma_20'].replace(0, 1.0)

        # --- STATIONARY RETURNS (Golden Rule) ---
        df["ret_1m"] = df["close"].pct_change(1)
        df["ret_3m"] = df["close"].pct_change(3)
        df["ret_5m"] = df["close"].pct_change(5)
        df["ret_10m"] = df["close"].pct_change(10)
        df["ret_15m"] = df["close"].pct_change(15)
        
        # --- MOMENTUM FEATURES ---
        # Momentum strength (absolute returns)
        df["momentum_3m_abs"] = df["ret_3m"].abs()
        df["momentum_5m_abs"] = df["ret_5m"].abs()
        df["momentum_10m_abs"] = df["ret_10m"].abs()
        
        # Momentum acceleration (rate of change of returns)
        df["momentum_accel_1m"] = df["ret_1m"].pct_change(1)
        df["momentum_accel_5m"] = df["ret_5m"].pct_change(5)
        
        # Momentum consistency (rolling standard deviation of returns)
        df["momentum_consistency_5m"] = df["ret_1m"].rolling(5).std()
        df["momentum_consistency_10m"] = df["ret_1m"].rolling(10).std()
        
        # Momentum direction (positive/negative momentum)
        df["momentum_direction_5m"] = np.where(df["ret_5m"] > 0, 1, -1)
        df["momentum_direction_10m"] = np.where(df["ret_10m"] > 0, 1, -1)
        
        # --- VOLUME PROFILE ANALYSIS FEATURES ---
        # Volume-weighted price zones
        df["vwap_upper"] = df["vwap"] + (df["atr"] * 0.5)
        df["vwap_lower"] = df["vwap"] - (df["atr"] * 0.5)
        
        # Price position relative to VWAP bands
        df["price_vs_vwap_upper"] = (df["close"] - df["vwap_upper"]) / df["vwap_upper"]
        df["price_vs_vwap_lower"] = (df["close"] - df["vwap_lower"]) / df["vwap_lower"]
        
        # Volume momentum
        df["volume_momentum_5m"] = df["volume"].pct_change(5)
        df["volume_momentum_10m"] = df["volume"].pct_change(10)
        
        # Price-volume divergence (price up but volume down = weak)
        df["price_volume_divergence"] = df["ret_5m"] * df["volume_momentum_5m"]
        
        # --- STRUCTURAL TIME AND TREND FEATURES ---
        # VWAP Slope (5-period normalized rate of change)
        df["vwap_slope"] = df["vwap"].pct_change(5)
        
        # Time of day features
        # Central time open is 8:30, close is 15:00
        idx_ct = df.index.tz_convert("America/Chicago")
        
        def calc_mins_since_open(t):
            return (t.hour - 8) * 60 + t.minute - 30
            
        def calc_mins_to_close(t):
            return (15 - t.hour) * 60 - t.minute
            
        df["minutes_since_open"] = [calc_mins_since_open(t) for t in idx_ct]
        df["minutes_until_close"] = [calc_mins_to_close(t) for t in idx_ct]

        # Warmup check
        last = df.iloc[-1]
        if pd.isna(last.get("alligator_lips")) or pd.isna(last.get("vwap")) or pd.isna(last.get("adx")) or pd.isna(last.get("obv_ema")):
            return None

        # Sanitize entire series to replace any NaNs with 0.0 before serializing
        # This prevents JSON parse errors or model pipeline crashes
        last_clean = last.fillna(0.0)
        features = last_clean.to_dict()
        features["ts"] = last.name.isoformat() if hasattr(last.name, "isoformat") else str(last.name)

        # Multi-Timeframe Bias (HTF)
        htf_bias = "neutral"
        if df_15m is not None and not df_15m.empty:
            htf_15m = df_15m.copy()
            htf_15m["ema_20"] = ta.ema(htf_15m["close"], length=20)
            if not htf_15m["ema_20"].isna().all():
                htf_15m["rsi_14"] = ta.rsi(htf_15m["close"], length=14)
                h_last = htf_15m.iloc[-1]
                if h_last["close"] > h_last["ema_20"]:
                    htf_bias = "bullish"
                elif h_last["close"] < h_last["ema_20"]:
                    htf_bias = "bearish"
                
                # Capture HTF features for ML
                features["htf_rsi"] = float(h_last.get("rsi_14", 50.0))

        regime = self._get_regime(df)
        features["regime"] = regime
        features["htf_bias"] = htf_bias
        features["htf_ok_long"] = bool(htf_bias == "bullish")
        features["htf_ok_short"] = bool(htf_bias == "bearish")
        features["strategy_version"] = "v2_aetheris_pro"

        # --- HYBRID REGIME DECISION ENGINE (Hysteresis Buffer) ---
        rho = last.get("ar1_rho", 1.0)
        z_score = last.get("z_score_vwap", 0.0)
        rvol = last.get("rvol", 1.0)
        
        # Calculate dynamic regime boundaries from trailing distribution of ar1_rho
        all_rhos = df["ar1_rho"].dropna()
        if len(all_rhos) > 50:
            rho_mean = float(all_rhos.mean())
            rho_std = float(all_rhos.std())
            mr_threshold = max(0.25, min(0.40, rho_mean - 0.5 * rho_std))
            trend_threshold = max(0.40, min(0.60, rho_mean + 0.5 * rho_std))
        else:
            mr_threshold = 0.35
            trend_threshold = 0.45

        # Hysteresis buffer using dynamic statistical boundaries
        prev_rho_regime = getattr(self, "_last_rho_regime", "mean_reversion")
        if prev_rho_regime == "mean_reversion":
            if rho >= trend_threshold:
                curr_rho_regime = "trend"
            else:
                curr_rho_regime = "mean_reversion"
        else: # trend
            if rho < mr_threshold:
                curr_rho_regime = "mean_reversion"
            else:
                curr_rho_regime = "trend"
        self._last_rho_regime = curr_rho_regime
        features["rho_regime"] = curr_rho_regime
        # Keep ADX-based regime for ML routing (more accurate than ar1_rho)
        # ADX regime correctly identifies 51.8% of signals as trend vs 0.3% with ar1_rho
        # features["regime"] = curr_rho_regime  # DISABLED: Use ADX-based regime instead

        
        rvol_threshold = 1.1 if self._aggressive_mode else 1.5
        # In hyper-aggressive grinder mode, we bypass RVOL check
        has_rvol = (rvol > rvol_threshold) or self._aggressive_mode
        ema9 = last.get("ema_9") or last.get("alligator_lips")
        extension_pct = (last["close"] - ema9) / ema9 if ema9 else 0
        curr_macd = last.get("macd_hist", 0) or 0
        prev_macd = df.iloc[-2].get("macd_hist", 0) if len(df) > 1 else 0

        # Extract features for new rules
        close = last["close"]
        open_price = last["open"]
        prev_close = df.iloc[-2]["close"] if len(df) > 1 else close
        vol_imb = last.get("volume_imbalance_norm", 0)
        is_green = close > open_price
        is_red = close < open_price

        sig = None

        # Import candlestick scanner
        from alpaca_day_bot.strategy.candlesticks import scan_patterns

        # ── PRE-COMPUTE KEY VALUES ───────────────────────────────────────────
        close       = last["close"]
        open_price  = last["open"]
        vwap_val    = last.get("vwap", close)
        ema9_val    = last.get("ema_9", close)
        ema21_val   = last.get("ema_21", close)
        rsi         = last.get("rsi_14", 50.0) or 50.0
        atr_val     = last.get("atr", close * 0.01) or (close * 0.01)

        curr_macd   = last.get("macd_hist", 0) or 0
        prev_macd   = df.iloc[-2].get("macd_hist", 0) if len(df) > 1 else 0
        macd_rising = curr_macd > prev_macd
        macd_falling= curr_macd < prev_macd

        avg_vol     = last.get("vol_sma_20", None)
        curr_vol    = last.get("volume", 0)
        has_volume  = (avg_vol is None) or (curr_vol > avg_vol * 0.9)

        # ── RULE 1: ALLIGATOR JAW CHECK (PRIMARY SIGNAL GENERATOR) ──────────
        # Jaws OPEN upward   → lips > teeth > jaw  (bullish trend)
        # Jaws OPEN downward → lips < teeth < jaw  (bearish trend)
        # Jaws CLOSED/SLEEP  → abs spread < 0.5× ATR → no trend trade
        jaw    = last.get("alligator_jaw",   None)
        teeth  = last.get("alligator_teeth", None)
        lips   = last.get("alligator_lips",  None)
        jaw_ratio    = last.get("alligator_jaw_ratio",   1.0)
        teeth_ratio  = last.get("alligator_teeth_ratio", 1.0)
        lips_ratio   = last.get("alligator_lips_ratio",  1.0)

        alligator_bullish = False
        alligator_bearish = False
        alligator_sleeping= True
        alligator_aligned = False

        if jaw is not None and teeth is not None and lips is not None:
            jaw_f, teeth_f, lips_f = float(jaw), float(teeth), float(lips)
            spread = max(jaw_f, teeth_f, lips_f) - min(jaw_f, teeth_f, lips_f)
            min_spread = 0.3 * atr_val   # jaws must be open at least 0.3× ATR

            if spread > min_spread:
                alligator_sleeping = False
                if lips_f > teeth_f > jaw_f:
                    alligator_bullish = True
                    # Check if price is aligned above all Alligator lines (strong bullish signal)
                    if jaw_ratio > 1.0 and teeth_ratio > 1.0 and lips_ratio > 1.0:
                        alligator_aligned = True
                elif lips_f < teeth_f < jaw_f:
                    alligator_bearish = True
                    # Check if price is aligned below all Alligator lines (strong bearish signal)
                    if jaw_ratio < 1.0 and teeth_ratio < 1.0 and lips_ratio < 1.0:
                        alligator_aligned = True

        # ── CANDLESTICK SCAN (Golden Rules 2, 3, 4) ─────────────────────────
        # Rule 3 satisfied by design — we only read the last CLOSED bar in df
        # Rule 2 & 4 handled inside scan_patterns via at_key_level / volume_confirmed
        patterns = scan_patterns(
            df,
            vwap=float(vwap_val) if vwap_val else None,
            ema_fast=float(ema9_val) if ema9_val else None,
            ema_slow=float(ema21_val) if ema21_val else None,
            avg_volume=float(avg_vol) if avg_vol else None,
        )

        # Best bullish and bearish patterns from the scan
        best_bull = next((p for p in patterns if p.direction == "bullish"), None)
        best_bear = next((p for p in patterns if p.direction == "bearish"), None)
        best_doji = next((p for p in patterns if p.direction == "neutral"), None)

        # Minimum pattern strength to trigger (0.55 = requires at least some quality)
        MIN_PATTERN_STRENGTH = 0.55

        # ── TIER 1: QUANT MEAN REVERSION ────────────────────────────────────
        # Only in stationary/choppy regime, at extreme z-score levels
        if curr_rho_regime == "mean_reversion":
            willr_val  = last.get("willr", -50.0) or -50.0
            stoch_k    = last.get("stoch_k", 50.0) or 50.0
            stoch_d    = last.get("stoch_d", 50.0) or 50.0

            # BUY fade: deeply oversold + bullish candle pattern confirming bounce
            if (z_score < -1.8 and rsi < 38
                    and willr_val < -70.0
                    and stoch_k < 35.0 and stoch_k > stoch_d   # stoch crossing up
                    and best_bull is not None
                    and best_bull.strength >= MIN_PATTERN_STRENGTH
                    and has_volume):
                sig = StrategySignal(
                    symbol, "BUY",
                    f"mr_oversold_z{z_score:.1f}_{best_bull.name}",
                    features=features
                )

            # SHORT fade: deeply overbought + bearish candle confirming rejection
            elif (z_score > 1.8 and rsi > 62
                    and willr_val > -30.0
                    and stoch_k > 65.0 and stoch_k < stoch_d   # stoch crossing down
                    and best_bear is not None
                    and best_bear.strength >= MIN_PATTERN_STRENGTH
                    and has_volume
                    and self._enable_shorts):
                sig = StrategySignal(
                    symbol, "SHORT",
                    f"mr_overbought_z{z_score:.1f}_{best_bear.name}",
                    features=features
                )

        # ── TIER 2: ALLIGATOR-FIRST SIGNAL GENERATION WITH ENHANCEMENTS ──────────
        # Prioritize Alligator alignment as primary signal (user's proven indicator)
        # Enhanced with ML confirmation, regime adaptation, dynamic thresholds, and time-based filtering
        if sig is None and alligator_aligned and alligator_bullish:
            # Time-based filtering (best hours: 14:00, 17:00, 20:00 achieve 74.0% win rate)
            current_hour = df.index[-1].hour if hasattr(df.index[-1], 'hour') else 12
            best_hours = [14, 17, 20]  # 2pm, 5pm, 8pm UTC
            is_best_hour = current_hour in best_hours
            
            # Get market regime for adaptation (optimized thresholds)
            adx_val = last.get("adx", 20.0)
            is_trending = adx_val > 28.0  # Optimized from 25.0
            is_choppy = adx_val < 22.0  # Optimized from 20.0
            
            # Dynamic thresholds based on regime (optimized values)
            if is_trending:
                # Trending market: looser filters for more signals
                rvol_threshold = 1.3
                rsi_min, rsi_max = 20, 80  # Optimized from 25, 75
                macd_threshold = 0.0
            elif is_choppy:
                # Choppy market: stricter filters for higher quality
                rvol_threshold = 2.0
                rsi_min, rsi_max = 35, 65
                macd_threshold = 0.1
            else:
                # Neutral market: standard thresholds
                rvol_threshold = 1.5
                rsi_min, rsi_max = 20, 80  # Optimized from 30, 70
                macd_threshold = 0.0
            
            # Multi-factor confirmation with regime-adapted thresholds
            rvol = last.get("rvol", 1.0)
            rsi_val = last.get("rsi_14", 50.0)
            macd_hist_val = last.get("macd_hist", 0.0)
            
            # ML Model Confirmation (if available)
            ml_proba = last.get("ml_proba", None)
            has_ml_confirmation = ml_proba is not None and ml_proba > 0.55
            
            # Tier 1: Alligator + Volume + RSI + MACD + ML + Time Filter (highest accuracy)
            if (rvol > rvol_threshold and 
                rsi_val > rsi_min and rsi_val < rsi_max and 
                macd_hist_val > macd_threshold and 
                has_ml_confirmation and
                is_best_hour):
                regime_suffix = "_trending" if is_trending else "_choppy" if is_choppy else "_neutral"
                sig = StrategySignal(
                    symbol, "BUY",
                    f"alligator_aligned_bullish_optimal{regime_suffix}",
                    features=features
                )
            # Tier 2: Alligator + Volume + RSI + MACD + Time Filter (without ML)
            elif (rvol > rvol_threshold and 
                  rsi_val > rsi_min and rsi_val < rsi_max and 
                  macd_hist_val > macd_threshold and
                  is_best_hour):
                regime_suffix = "_trending" if is_trending else "_choppy" if is_choppy else "_neutral"
                sig = StrategySignal(
                    symbol, "BUY",
                    f"alligator_aligned_bullish_time_filtered{regime_suffix}",
                    features=features
                )
            # Tier 3: Alligator + Volume + RSI + MACD with regime adaptation (without time filter)
            elif (rvol > rvol_threshold and 
                  rsi_val > rsi_min and rsi_val < rsi_max and 
                  macd_hist_val > macd_threshold):
                regime_suffix = "_trending" if is_trending else "_choppy" if is_choppy else "_neutral"
                sig = StrategySignal(
                    symbol, "BUY",
                    f"alligator_aligned_bullish_volume_rsi_macd{regime_suffix}",
                    features=features
                )
            # Tier 4: Alligator + Volume only (fallback)
            elif rvol > rvol_threshold:
                sig = StrategySignal(
                    symbol, "BUY",
                    f"alligator_aligned_bullish_volume",
                    features=features
                )
            # Tier 5: Alligator only (fallback)
            else:
                sig = StrategySignal(
                    symbol, "BUY",
                    f"alligator_aligned_bullish",
                    features=features
                )
        
        # ── TIER 3: TREND SNIPER (Alligator-gated) ──────────────────────────
        # Only fires when alligator jaws are open — the defining rule you described
        if sig is None and not alligator_sleeping:

            # BUY: Jaws open upward + price above VWAP + bullish pattern + MACD confirms
            if (alligator_bullish
                    and htf_bias == "bullish"
                    and close > vwap_val
                    and rsi < 72              # not overbought
                    and macd_rising           # MACD histogram turning up
                    and best_bull is not None
                    and best_bull.strength >= MIN_PATTERN_STRENGTH
                    and has_volume):

                # Extra confirmation: price pulled back to alligator lips/teeth then closed above
                pullback_zone = close >= lips_f if lips is not None else True
                if pullback_zone:
                    sig = StrategySignal(
                        symbol, "BUY",
                        f"alligator_trend_{best_bull.name}",
                        features=features
                    )

            # SHORT: Jaws open downward + price below VWAP + bearish pattern + MACD confirms
            elif (alligator_bearish
                    and htf_bias == "bearish"
                    and close < vwap_val
                    and rsi > 28
                    and macd_falling
                    and best_bear is not None
                    and best_bear.strength >= MIN_PATTERN_STRENGTH
                    and has_volume
                    and self._enable_shorts):
                sig = StrategySignal(
                    symbol, "SHORT",
                    f"alligator_trend_{best_bear.name}",
                    features=features
                )

        # ── TIER 3: HIGH-CONVICTION PATTERN ONLY ────────────────────────────
        # Strong patterns (morning star, engulfing, 3 soldiers etc.) at key levels
        # can fire even without full alligator confirmation — but require vol + level
        if sig is None:
            HIGH_CONVICTION_PATTERNS = {
                "morning_star", "bullish_engulfing", "three_white_soldiers",
                "rising_three_methods", "tweezer_bottom"
            }
            HIGH_CONVICTION_BEAR = {
                "evening_star", "bearish_engulfing", "three_black_crows",
                "falling_three_methods", "tweezer_top"
            }

            if (best_bull is not None
                    and best_bull.name in HIGH_CONVICTION_PATTERNS
                    and best_bull.at_key_level
                    and best_bull.volume_confirmed
                    and best_bull.strength >= 0.65
                    and rsi < 68
                    and (macd_rising or curr_macd > 0)):
                sig = StrategySignal(
                    symbol, "BUY",
                    f"pattern_conviction_{best_bull.name}",
                    features=features
                )

            elif (best_bear is not None
                    and best_bear.name in HIGH_CONVICTION_BEAR
                    and best_bear.at_key_level
                    and best_bear.volume_confirmed
                    and best_bear.strength >= 0.65
                    and rsi > 32
                    and (macd_falling or curr_macd < 0)
                    and self._enable_shorts):
                sig = StrategySignal(
                    symbol, "SHORT",
                    f"pattern_conviction_{best_bear.name}",
                    features=features
                )

        # ── RELATIVE STRENGTH vs SPY ─────────────────────────────────────────
        if sig and spy_perf is not None:
            sym_open = df.iloc[0]["open"]
            sym_curr = df.iloc[-1]["close"]
            sym_perf = (sym_curr - sym_open) / sym_open
            if sig.action == "BUY"   and sym_perf < spy_perf:  sig = None
            if sig.action == "SHORT" and sym_perf > spy_perf:  sig = None

        # Add detected pattern names to features for ML/logging
        features["candle_pattern_bull"] = best_bull.name if best_bull else "none"
        features["candle_pattern_bear"] = best_bear.name if best_bear else "none"
        features["candle_pattern_strength"] = float(best_bull.strength if best_bull else
                                                     best_bear.strength if best_bear else 0.0)
        features["alligator_state"] = (
            "bullish" if alligator_bullish else
            "bearish" if alligator_bearish else "sleeping"
        )

        if sig and sig.action == "SHORT" and not self._enable_shorts:
            sig = None

        return sig if sig else StrategySignal(
            symbol=symbol, action="HOLD", reason="no_pattern_confluence", features=features
        )


