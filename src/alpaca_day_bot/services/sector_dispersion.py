import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

log = logging.getLogger("alpaca_day_bot.services.sector_dispersion")

class SectorDispersionService:
    """
    Tracks intraday rolling returns of key S&P 500 sectors to calculate Sector Dispersion.
    Directly informs the ML models of stock-specific vs macro correlated regimes.
    """
    def __init__(self, api_key_id: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key_id, secret_key)
        self.sector_etfs = ["XLK", "XLF", "XLY", "XLE", "XLV"]
        self._last_update_time = None
        self._cached_dispersion = 0.015 # Historical baseline dispersion

    def get_sector_dispersion(self) -> float:
        """
        Returns the latest calculated sector dispersion factor.
        """
        now = datetime.now(timezone.utc)
        
        # Limit API calls to once every 3 minutes
        if self._last_update_time is not None and (now - self._last_update_time) < timedelta(minutes=3):
            return self._cached_dispersion

        try:
            end = now
            start = end - timedelta(minutes=25) # Grab enough lookback for a robust 5m window
            
            req = StockBarsRequest(
                symbol_or_symbols=self.sector_etfs,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.IEX
            )
            
            bars = self.client.get_stock_bars(req)
            data_dict = bars.data
            
            records = []
            for etf, etf_bars in data_dict.items():
                for b in etf_bars:
                    records.append({
                        "timestamp": b.timestamp,
                        "etf": etf,
                        "close": float(b.close)
                    })
                    
            if records:
                df_raw = pd.DataFrame(records)
                # Pivot to index by timestamp and align ETFs across columns
                df_pivot = df_raw.pivot(index="timestamp", columns="etf", values="close")
                df_pivot = df_pivot.sort_index()
                
                # Guarantee all 5 sector ETFs exist as columns
                df_pivot = df_pivot.reindex(columns=self.sector_etfs)
                
                # Production Fix: forward-fill minor delays and backward-fill gaps
                df_pivot = df_pivot.ffill().bfill()
                
                # Calculate 5-minute rolling percent changes
                df_pct = df_pivot.pct_change(5)
                df_pct = df_pct.ffill().fillna(0.0)
                
                if len(df_pct) >= 6:
                    # Relative weights based on active relative market cap footprints in S&P 500:
                    # XLK (Tech) = 0.45, XLF (Financials) = 0.18, XLY (Cons. Disc) = 0.13, XLE (Energy) = 0.07, XLV (Healthcare) = 0.17
                    weights = np.array([0.45, 0.18, 0.13, 0.07, 0.17])
                    latest_returns = df_pct.iloc[-1].to_numpy()
                    
                    # Compute Weighted Sector Variance and standard deviation
                    weighted_mean = np.sum(latest_returns * weights)
                    weighted_var = np.sum(weights * (latest_returns - weighted_mean)**2)
                    latest_disp = float(np.sqrt(weighted_var))
                    
                    self._cached_dispersion = latest_disp
                    self._last_update_time = now
                    log.debug(f"Updated Weighted Sector Dispersion: {self._cached_dispersion:.6f} across aligned ETFs.")
                else:
                    log.warning("Insufficient time ticks in sector DataFrame. Using cached dispersion.")
            else:
                log.warning("No sector bars fetched from Alpaca. Using cached dispersion.")
                
        except Exception as e:
            log.error(f"Sector Dispersion query failed: {e}. Defaulting to cached value.")
            
        return self._cached_dispersion
