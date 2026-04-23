from __future__ import annotations


def normalize_symbol_for_yfinance(symbol: str) -> str:
    """
    Accepts TradingView-like symbols (e.g. NASDAQ:NVDA) and normalizes for yfinance.
    Extend this mapping for other venues, futures, FX, crypto, etc.
    """
    s = symbol.strip()
    if ":" in s:
        _, ticker = s.split(":", 1)
        return ticker.strip()
    return s

