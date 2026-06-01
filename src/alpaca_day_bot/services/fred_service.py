import os
import logging
from datetime import datetime, timedelta, timezone
from fredapi import Fred

log = logging.getLogger("alpaca_day_bot.fred")

class FredLiquidityService:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.fred = None
        self._cached_value = 0.15
        self._last_update_time = None
        
        if self.api_key:
            try:
                self.fred = Fred(api_key=self.api_key)
                log.info("FRED Liquidity Service successfully initialized with API Key.")
            except Exception as e:
                log.error(f"Failed to initialize FRED API Client: {e}")

    def get_fed_liquidity_momentum(self) -> float:
        """
        Fetches the latest 10-Year vs 2-Year Treasury Spread (T10Y2Y) or SOFR rate.
        Returns a liquidity proxy float. Treasury spreads change once daily, so they are
        cached for 6 hours to prevent rate limits and thread-blocking streaming network latency.
        """
        now = datetime.now(timezone.utc)
        
        # Return cached spread if inside 6-hour window
        if self._last_update_time is not None and (now - self._last_update_time) < timedelta(hours=6):
            return self._cached_value
            
        if not self.fred:
            log.debug("FRED API Key missing. Returning default fallback liquidity momentum (0.15).")
            return self._cached_value
            
        try:
            series_id = "T10Y2Y"
            data = self.fred.get_series(series_id)
            if data is not None and len(data) > 0:
                latest_value = float(data.dropna().iloc[-1])
                self._cached_value = latest_value
                self._last_update_time = now
                log.info(f"FRED Yield Spread (T10Y2Y) successfully updated: {latest_value:.4f}")
        except Exception as e:
            log.warning(f"Error fetching data from FRED: {e}. Defaulting to cached value: {self._cached_value:.4f}")
            
        return self._cached_value
