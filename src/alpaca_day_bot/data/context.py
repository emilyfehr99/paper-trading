from __future__ import annotations
import logging
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

log = logging.getLogger("alpaca_day_bot.context")

class MacroContextCollector:
    """
    Collects institutional macro context (VIX, 10Y Yield) and sentiment.
    This feeds the 'Big Big Model' to help it understand WHY a trade failed.
    """
    def __init__(self):
        self._cache = {}
        self._last_fetch = None
        self._refresh_interval = timedelta(minutes=15)

    def get_context(self) -> Dict[str, Any]:
        """
        Fetch latest macro metrics. Uses a 15-minute cache to avoid API throttling.
        """
        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch) < self._refresh_interval:
            return self._cache

        try:
            # VIX - Volatility Index (Fear Gauge)
            # TNX - 10-Year Treasury Note Yield
            data = yf.download(["^VIX", "^TNX"], period="1d", interval="1m", progress=False)
            
            if not data.empty:
                vix = float(data["Close"]["^VIX"].iloc[-1])
                tnx = float(data["Close"]["^TNX"].iloc[-1])
                
                self._cache = {
                    "vix": vix,
                    "tnx_10y": tnx,
                    "timestamp_utc": now.isoformat(),
                    # Placeholder for sentiment - will be integrated with News API or PRAW
                    "sentiment_score": 0.0 
                }
                self._last_fetch = now
                log.info(f"Macro Context Updated: VIX={vix:.2f}, 10Y={tnx:.2f}")
            else:
                log.warning("Macro data fetch returned empty results.")
        except Exception as e:
            log.error(f"Failed to fetch macro context: {e}")
            # Return last valid cache if available, else empty
            
        return self._cache
