import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

log = logging.getLogger("MacroEngine")

class MacroEngine:
    """
    Provides real-time institutional macro context (SPY, QQQ, VIX, 10-Year Yield) 
    using yfinance for high-velocity updates.
    Supports Pillar 2 (Macro Conditioning) and Pillar 3 (HTF Trend Alignment).
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_fetch: datetime | None = None
        self._refresh_interval = timedelta(minutes=1) # High frequency for day trading

    def get_macro_context(self) -> Dict[str, float]:
        """
        Fetches the latest macro indicators:
        - SPY: S&P 500 Index ETF (5m returns, 1h & 4h Trend)
        - QQQ: Nasdaq 100 Index ETF (5m returns, 1h & 4h Trend)
        - ^VIX: CBOE Volatility Index (Level & 5m Change)
        - ^TNX: 10-Year Treasury Yield
        """
        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch) < self._refresh_interval:
            return self._cache

        try:
            # Fetch 5 days of 5-minute data to calculate EMAs and returns
            tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
            data = yf.download(tickers, period="5d", interval="5m", progress=False)
            
            if not data.empty and len(data) > 30:
                # Resolve multi-index column names from yfinance
                close_df = data["Close"]
                
                vix_series = close_df["^VIX"].ffill()
                tnx_series = close_df["^TNX"].ffill()
                spy_series = close_df["SPY"].ffill()
                qqq_series = close_df["QQQ"].ffill()
                
                # 1. Broad-Market Returns (Pillar 2)
                spy_ret_5m = float(spy_series.pct_change(1).iloc[-1])
                qqq_ret_5m = float(qqq_series.pct_change(1).iloc[-1])
                vix_roc_5m = float(vix_series.pct_change(1).iloc[-1])
                
                vix = float(vix_series.iloc[-1])
                tnx = float(tnx_series.iloc[-1])
                spy_close = float(spy_series.iloc[-1])
                qqq_close = float(qqq_series.iloc[-1])
                
                # 2. Multi-Timeframe Trend Alignment (Pillar 3)
                # 1 Hour = 12 5-minute bars. EMA(20) on 1-hour bars = EMA(240) on 5-minute bars
                # 4 Hours = 48 5-minute bars. EMA(20) on 4-hour bars = EMA(960) on 5-minute bars
                spy_ema_1h = spy_series.ewm(span=240, adjust=False).mean().iloc[-1]
                spy_ema_4h = spy_series.ewm(span=960, adjust=False).mean().iloc[-1]
                qqq_ema_1h = qqq_series.ewm(span=240, adjust=False).mean().iloc[-1]
                qqq_ema_4h = qqq_series.ewm(span=960, adjust=False).mean().iloc[-1]
                
                spy_trend_1h = 1.0 if spy_close > spy_ema_1h else 0.0
                spy_trend_4h = 1.0 if spy_close > spy_ema_4h else 0.0
                qqq_trend_1h = 1.0 if qqq_close > qqq_ema_1h else 0.0
                qqq_trend_4h = 1.0 if qqq_close > qqq_ema_4h else 0.0
                
                self._cache = {
                    "vix": vix,
                    "tnx": tnx,
                    "vix_zscore": (vix - 20.0) / 5.0,
                    "spy_ret_5m": spy_ret_5m,
                    "qqq_ret_5m": qqq_ret_5m,
                    "vix_roc_5m": vix_roc_5m,
                    "spy_trend_1h": spy_trend_1h,
                    "spy_trend_4h": spy_trend_4h,
                    "qqq_trend_1h": qqq_trend_1h,
                    "qqq_trend_4h": qqq_trend_4h,
                    "updated_at": now.isoformat()
                }
                self._last_fetch = now
                log.info(f"Macro Sync: VIX={vix:.2f}, SPY_5m={spy_ret_5m:.2%}, SPY_1H_Trend={spy_trend_1h}")
                return self._cache
            else:
                log.warning("Empty macro data from yfinance.")
                return self._cache or self._get_fallback_context()

        except Exception as e:
            log.error(f"Macro Sync Failed: {e}", exc_info=True)
            return self._cache or self._get_fallback_context()

    def _get_fallback_context(self) -> Dict[str, float]:
        return {
            "vix": 20.0,
            "tnx": 4.0,
            "vix_zscore": 0.0,
            "spy_ret_5m": 0.0,
            "qqq_ret_5m": 0.0,
            "vix_roc_5m": 0.0,
            "spy_trend_1h": 1.0,
            "spy_trend_4h": 1.0,
            "qqq_trend_1h": 1.0,
            "qqq_trend_4h": 1.0
        }

