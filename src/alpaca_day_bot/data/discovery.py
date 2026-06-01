import requests
import logging
import pandas as pd
from typing import List, Set
from datetime import datetime

log = logging.getLogger("alpaca_day_bot.discovery")

class DiscoveryEngine:
    """
    Scrapes and aggregates stock ideas from Finviz, StockTwits, and AltIndex.
    Expands the Sniper's universe from dozens to thousands of high-conviction symbols.
    """
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.headers = {"User-Agent": self.user_agent}

    def get_finviz_movers(self) -> Set[str]:
        """
        Scrapes Finviz for Top Gainers and New Highs.
        """
        symbols = set()
        try:
            # Finviz Top Gainers
            url = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                # Basic scraping - in production we'd use a robust parser
                # For now, we'll use a regex or string find for ticker patterns
                import re
                tickers = re.findall(r't=([A-Z]{1,5})', res.text)
                symbols.update(tickers)
            
            # Finviz New Highs
            url = "https://finviz.com/screener.ashx?v=111&s=ta_newhigh"
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                tickers = re.findall(r't=([A-Z]{1,5})', res.text)
                symbols.update(tickers)
                
            log.info(f"Finviz Discovery: Found {len(symbols)} momentum symbols.")
        except Exception as e:
            log.error(f"Finviz Discovery Failed: {e}")
            
        return symbols

    def get_stocktwits_trending(self) -> Set[str]:
        """
        Pulls trending tickers from StockTwits public API.
        """
        symbols = set()
        try:
            url = "https://api.stocktwits.com/api/2/trending/symbols.json"
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for item in data.get("symbols", []):
                    symbols.add(item.get("symbol"))
            log.info(f"StockTwits Discovery: Found {len(symbols)} trending symbols.")
        except Exception as e:
            log.error(f"StockTwits Discovery Failed: {e}")
            
        return symbols

    def build_expanded_universe(self) -> List[str]:
        """
        Aggregates all discovery sources into a single high-conviction universe.
        """
        universe = set()
        universe.update(self.get_finviz_movers())
        universe.update(self.get_stocktwits_trending())
        
        # Add basic liquid S&P 500 if the lists are too small
        if len(universe) < 50:
            # Fallback to a core liquid set
            universe.update(["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "AMZN", "META", "GOOGL"])
            
        return sorted(list(universe))
