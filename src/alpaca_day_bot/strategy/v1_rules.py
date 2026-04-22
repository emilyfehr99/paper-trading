from __future__ import annotations

from typing import Any

from alpaca_day_bot.strategy.base import BaseStrategy, StrategySignal


class V1RulesSignalEngine(BaseStrategy):
    """
    Conservative long-only intraday rules (hybrid momentum / mean-reversion filter):
    - RSI(14) pullback condition
    - MACD(12,26,9) bullish cross confirmation
    - Above VWAP filter
    - Volume confirmation vs SMA(20)
    - Multi-timeframe bias: 15m RSI > 50 trend alignment
    - Volatility regime filter: ATR(14) not > 2x its recent average
    """

    def __init__(
        self,
        *,
        rsi_pullback_max: float = 35.0,
        ema_trend_len: int = 20,
        volume_sma_len: int = 20,
        volume_confirm_mult: float = 1.2,
        htf_rsi_len: int = 14,
        htf_rsi_min: float = 50.0,
        htf_rsi_max_short: float = 55.0,
        rsi_rebound_min_short: float = 58.0,
        enable_shorts: bool = False,
        aggressive_mode: bool = False,
        atr_len: int = 14,
        atr_regime_lookback: int = 50,
        atr_regime_max_mult: float = 2.0,
    ) -> None:
        self._rsi_pullback_max = rsi_pullback_max
        self._ema_trend_len = ema_trend_len
        self._volume_sma_len = volume_sma_len
        self._volume_confirm_mult = volume_confirm_mult
        self._htf_rsi_len = htf_rsi_len
        self._htf_rsi_min = htf_rsi_min
        self._htf_rsi_max_short = htf_rsi_max_short
        self._rsi_rebound_min_short = rsi_rebound_min_short
        self._enable_shorts = bool(enable_shorts)
        self._aggressive_mode = bool(aggressive_mode)
        self._atr_len = atr_len
        self._atr_regime_lookback = atr_regime_lookback
        self._atr_regime_max_mult = atr_regime_max_mult

    def evaluate_setup(self, *, symbol: str, df_1m, df_15m) -> dict[str, Any]:
        """
        Diagnostics for live runs: how close this symbol is to a V1 BUY (0–5 score).
        Always returns a dict (even when bars are missing).
        """
        import pandas as pd
        import pandas_ta as ta

        base: dict[str, Any] = {
            "symbol": symbol,
            "bars_1m": 0,
            "bars_15m": 0,
            "blocked": None,
            "rsi_14": None,
            "htf_rsi": None,
            "checks": {
                "rsi_pullback": False,
                "above_ema": False,
                "macd_cross": False,
                "above_vwap": False,
                "volume_confirm": False,
            },
            "buy_score": 0,
            "would_action": "HOLD",
        }

        if df_1m is None or getattr(df_1m, "empty", True):
            base["blocked"] = "no_1m_bars"
            return base

        df = df_1m.copy()
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass
        if hasattr(df.index, "duplicated"):
            df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        base["bars_1m"] = int(len(df))

        for c in ("open", "high", "low", "close", "volume"):
            if c not in df.columns:
                base["blocked"] = "bad_ohlcv"
                return base

        df["ema_20"] = ta.ema(df["close"], length=int(self._ema_trend_len))
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        df["vwap_calc"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["volume_sma"] = df["volume"].rolling(int(self._volume_sma_len)).mean()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=int(self._atr_len))
        df["atr_avg"] = df["atr"].rolling(int(self._atr_regime_lookback)).mean()

        last = df.iloc[-1]
        if pd.isna(last.get("rsi")) or pd.isna(last.get("ema_20")) or pd.isna(last.get("vwap_calc")):
            base["blocked"] = "warmup_indicators"
            return base

        base["rsi_14"] = float(last["rsi"])

        if df_15m is None or getattr(df_15m, "empty", True):
            base["blocked"] = "no_15m_bars"
            return base
        htf = df_15m.copy()
        try:
            htf.index = pd.to_datetime(htf.index, utc=True)
        except Exception:
            pass
        if hasattr(htf.index, "duplicated"):
            htf = htf[~htf.index.duplicated(keep="last")]
        htf = htf.sort_index()
        base["bars_15m"] = int(len(htf))

        if len(htf) < (self._htf_rsi_len + 2):
            base["blocked"] = "htf_not_ready"
            return base

        htf["rsi"] = ta.rsi(htf["close"], length=int(self._htf_rsi_len))
        htf_last = htf.iloc[-1]
        if pd.isna(htf_last.get("rsi")):
            base["blocked"] = "htf_rsi_nan"
            return base

        htf_rsi_v = float(htf_last["rsi"])
        base["htf_rsi"] = htf_rsi_v
        if htf_rsi_v < float(self._htf_rsi_min):
            base["blocked"] = "htf_bias_rsi"
            return base

        atr = float(last.get("atr")) if not pd.isna(last.get("atr")) else None
        atr_avg = float(last.get("atr_avg")) if not pd.isna(last.get("atr_avg")) else None
        if atr is None or atr_avg is None:
            base["blocked"] = "atr_not_ready"
            return base
        if atr_avg > 1e-12 and atr > atr_avg * float(self._atr_regime_max_mult):
            base["blocked"] = "atr_regime_spike"
            return base

        macd_col = "MACD_12_26_9"
        macds_col = "MACDs_12_26_9"
        if macd_col not in df.columns or macds_col not in df.columns or len(df) < 2:
            base["blocked"] = "macd_not_ready"
            return base
        macd_now = df[macd_col].iloc[-1]
        macds_now = df[macds_col].iloc[-1]
        macd_prev = df[macd_col].iloc[-2]
        macds_prev = df[macds_col].iloc[-2]
        if pd.isna(macd_now) or pd.isna(macds_now) or pd.isna(macd_prev) or pd.isna(macds_prev):
            base["blocked"] = "macd_not_ready"
            return base
        macd_bull_cross = (macd_now > macds_now) and (macd_prev <= macds_prev)

        vol_sma = last.get("volume_sma")
        if pd.isna(vol_sma) or float(vol_sma) <= 0:
            base["blocked"] = "volume_not_ready"
            return base

        rsi_pullback = float(last["rsi"]) <= float(self._rsi_pullback_max)
        above_ema = float(last["close"]) > float(last["ema_20"])
        above_vwap = float(last["close"]) > float(last["vwap_calc"])
        volume_confirm = float(last["volume"]) > float(vol_sma) * float(self._volume_confirm_mult)

        chk = base["checks"]
        chk["rsi_pullback"] = bool(rsi_pullback)
        chk["above_ema"] = bool(above_ema)
        chk["macd_cross"] = bool(macd_bull_cross)
        chk["above_vwap"] = bool(above_vwap)
        chk["volume_confirm"] = bool(volume_confirm)

        score = sum(1 for v in chk.values() if v)
        base["buy_score"] = int(score)

        if not rsi_pullback:
            base["blocked"] = "rsi_no_pullback"
        elif not above_ema:
            base["blocked"] = "below_ema"
        elif not macd_bull_cross:
            base["blocked"] = "macd_no_cross"
        elif not above_vwap:
            base["blocked"] = "below_vwap"
        elif not volume_confirm:
            base["blocked"] = "no_volume_confirm"
        else:
            base["blocked"] = None
            base["would_action"] = "BUY"

        return base

    def decide(self, *, symbol: str, df_1m, df_15m) -> StrategySignal | None:
        """
        df_1m: 1-minute OHLCV DataFrame indexed by UTC timestamp.
        df_15m: 15-minute OHLCV DataFrame indexed by UTC timestamp.
        """
        if df_1m is None or getattr(df_1m, "empty", True):
            return None

        import pandas as pd
        import pandas_ta as ta

        df = df_1m.copy()
        # pandas-ta VWAP requires an ordered DatetimeIndex; enforce monotonic UTC index.
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass
        if hasattr(df.index, "duplicated"):
            df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        # Ensure expected cols exist
        for c in ("open", "high", "low", "close", "volume"):
            if c not in df.columns:
                return None

        # Indicators (1m)
        df["ema_20"] = ta.ema(df["close"], length=int(self._ema_trend_len))
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        df["vwap_calc"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["volume_sma"] = df["volume"].rolling(int(self._volume_sma_len)).mean()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=int(self._atr_len))
        df["atr_avg"] = df["atr"].rolling(int(self._atr_regime_lookback)).mean()

        last = df.iloc[-1]
        # Guard against NaNs from warmup
        if pd.isna(last.get("rsi")) or pd.isna(last.get("ema_20")) or pd.isna(last.get("vwap_calc")):
            return None

        # Higher timeframe bias: 15m RSI must be > threshold (longs) or < max threshold (shorts)
        if df_15m is None or getattr(df_15m, "empty", True) or len(df_15m) < (self._htf_rsi_len + 2):
            return StrategySignal(symbol, "HOLD", "htf_not_ready")
        htf = df_15m.copy()
        htf["rsi"] = ta.rsi(htf["close"], length=int(self._htf_rsi_len))
        htf_last = htf.iloc[-1]
        if pd.isna(htf_last.get("rsi")):
            return StrategySignal(symbol, "HOLD", "htf_bias_rsi")
        htf_rsi_v = float(htf_last["rsi"])
        htf_ok_long = htf_rsi_v >= float(self._htf_rsi_min)
        htf_ok_short = htf_rsi_v <= float(self._htf_rsi_max_short)

        # Volatility regime guard: skip if ATR > k * average ATR
        atr = float(last.get("atr")) if not pd.isna(last.get("atr")) else None
        atr_avg = float(last.get("atr_avg")) if not pd.isna(last.get("atr_avg")) else None
        if atr is None or atr_avg is None:
            return StrategySignal(symbol, "HOLD", "atr_not_ready")
        if atr_avg > 1e-12 and atr > atr_avg * float(self._atr_regime_max_mult):
            return StrategySignal(symbol, "HOLD", "atr_regime_spike")

        # Core entry conditions
        rsi_pullback = float(last["rsi"]) <= float(self._rsi_pullback_max)
        above_ema = float(last["close"]) > float(last["ema_20"])
        below_ema = float(last["close"]) < float(last["ema_20"])

        # MACD cross: MACD line crosses above signal line on last bar
        macd_col = "MACD_12_26_9"
        macds_col = "MACDs_12_26_9"
        if macd_col not in df.columns or macds_col not in df.columns or len(df) < 2:
            return StrategySignal(symbol, "HOLD", "macd_not_ready")
        macd_now = df[macd_col].iloc[-1]
        macds_now = df[macds_col].iloc[-1]
        macd_prev = df[macd_col].iloc[-2]
        macds_prev = df[macds_col].iloc[-2]
        if pd.isna(macd_now) or pd.isna(macds_now) or pd.isna(macd_prev) or pd.isna(macds_prev):
            return StrategySignal(symbol, "HOLD", "macd_not_ready")
        macd_bull_cross = (macd_now > macds_now) and (macd_prev <= macds_prev)
        macd_bear_cross = (macd_now < macds_now) and (macd_prev >= macds_prev)
        macd_bull = (macd_now > macds_now)
        macd_bear = (macd_now < macds_now)

        above_vwap = float(last["close"]) > float(last["vwap_calc"])
        below_vwap = float(last["close"]) < float(last["vwap_calc"])
        vol_sma = last.get("volume_sma")
        if pd.isna(vol_sma) or float(vol_sma) <= 0:
            return StrategySignal(symbol, "HOLD", "volume_not_ready")
        volume_ratio = float(last["volume"]) / float(vol_sma)
        volume_confirm = volume_ratio > float(self._volume_confirm_mult)

        features = {
            "close": float(last["close"]),
            "rsi_14": float(last["rsi"]),
            "ema": float(last["ema_20"]),
            "macd": (None if pd.isna(macd_now) else float(macd_now)),
            "macd_signal": (None if pd.isna(macds_now) else float(macds_now)),
            "vwap": float(last["vwap_calc"]),
            "volume": float(last["volume"]),
            "volume_sma": (None if pd.isna(vol_sma) else float(vol_sma)),
            "volume_ratio": volume_ratio,
            "htf_rsi": float(htf_last["rsi"]) if not pd.isna(htf_last.get("rsi")) else None,
            "atr": atr,
            "atr_avg": atr_avg,
            "htf_ok_long": bool(htf_ok_long),
            "htf_ok_short": bool(htf_ok_short),
        }

        # Long entry
        macd_ok_long = macd_bull_cross if not self._aggressive_mode else (macd_bull_cross or macd_bull)
        vwap_ok_long = above_vwap if not self._aggressive_mode else True
        vol_ok_long = volume_confirm if not self._aggressive_mode else (volume_ratio >= max(0.80, float(self._volume_confirm_mult) * 0.80))

        # Setup A: pullback entry (original)
        if htf_ok_long and rsi_pullback and above_ema and macd_ok_long and vwap_ok_long and vol_ok_long:
            return StrategySignal(symbol, "BUY", "long_rsi_macd_vwap_volume", features=features)

        # Setup B (aggressive): momentum continuation
        if self._aggressive_mode:
            rsi_momo_long = float(last["rsi"]) >= 55.0
            if htf_ok_long and rsi_momo_long and above_ema and macd_bull and vol_ok_long:
                return StrategySignal(symbol, "BUY", "long_momo", features=features)

            # Setup C (more aggressive): "3-of-4" confirmation.
            # Goal: increase trade frequency under small notional caps. This is intentionally looser
            # than cross-only MACD and will increase false positives.
            checks = 0
            checks += 1 if above_ema else 0
            checks += 1 if above_vwap else 0
            checks += 1 if volume_ratio >= max(0.70, float(self._volume_confirm_mult) * 0.70) else 0
            checks += 1 if macd_bull else 0
            rsi_ok = 45.0 <= float(last["rsi"]) <= 78.0
            if htf_ok_long and rsi_ok and checks >= 3:
                return StrategySignal(symbol, "BUY", "long_aggr_3of4", features=features)

            # Setup D (max frequency): "2-of-4" confirmation.
            # This is intentionally very loose to drive trade count; expect more noise.
            if htf_ok_long and rsi_ok and checks >= 2:
                return StrategySignal(symbol, "BUY", "long_aggr_2of4", features=features)

        # Short entry (optional)
        if self._enable_shorts:
            rsi_rebound = float(last["rsi"]) >= float(self._rsi_rebound_min_short)
            macd_ok_short = macd_bear_cross if not self._aggressive_mode else (macd_bear_cross or macd_bear)
            vwap_ok_short = below_vwap if not self._aggressive_mode else True
            vol_ok_short = volume_confirm if not self._aggressive_mode else (volume_ratio >= max(0.80, float(self._volume_confirm_mult) * 0.80))

            if htf_ok_short and rsi_rebound and below_ema and macd_ok_short and vwap_ok_short and vol_ok_short:
                return StrategySignal(symbol, "SHORT", "short_rsi_macd_vwap_volume", features=features)

            # Setup B (aggressive): downside momentum continuation
            if self._aggressive_mode:
                rsi_momo_short = float(last["rsi"]) <= 45.0
                if htf_ok_short and rsi_momo_short and below_ema and macd_bear and vol_ok_short:
                    return StrategySignal(symbol, "SHORT", "short_momo", features=features)

                # Setup C (more aggressive): "3-of-4" confirmation for shorts.
                checks = 0
                checks += 1 if below_ema else 0
                checks += 1 if below_vwap else 0
                checks += 1 if volume_ratio >= max(0.70, float(self._volume_confirm_mult) * 0.70) else 0
                checks += 1 if macd_bear else 0
                rsi_ok = 22.0 <= float(last["rsi"]) <= 58.0
                if htf_ok_short and rsi_ok and checks >= 3:
                    return StrategySignal(symbol, "SHORT", "short_aggr_3of4", features=features)

                # Setup D (max frequency): "2-of-4" confirmation for shorts.
                if htf_ok_short and rsi_ok and checks >= 2:
                    return StrategySignal(symbol, "SHORT", "short_aggr_2of4", features=features)

        # HOLD reason (keep it informative)
        if not htf_ok_long and not (self._enable_shorts and htf_ok_short):
            return StrategySignal(symbol, "HOLD", "htf_bias_rsi", features=features)
        if not rsi_pullback:
            return StrategySignal(symbol, "HOLD", "rsi_no_pullback", features=features)
        if not above_ema and not (self._enable_shorts and below_ema):
            return StrategySignal(symbol, "HOLD", "ema_filter", features=features)
        if not (macd_bull_cross or (self._enable_shorts and macd_bear_cross)):
            return StrategySignal(symbol, "HOLD", "macd_no_cross", features=features)
        if not (above_vwap or (self._enable_shorts and below_vwap)):
            return StrategySignal(symbol, "HOLD", "vwap_filter", features=features)
        if not volume_confirm:
            return StrategySignal(symbol, "HOLD", "no_volume_confirm", features=features)
        return StrategySignal(symbol, "HOLD", "no_setup", features=features)

