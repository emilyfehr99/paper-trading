from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from alpaca_day_bot.storage.ledger import Ledger

log = logging.getLogger("alpaca_day_bot.options_sim")


def _pnl_virtual_long_option(
    *,
    side: str,
    notional_usd: float,
    leverage: float,
    underlying_entry: float,
    underlying_exit: float,
) -> float:
    """
    Very simple mock model:
    - Treat the option like a leveraged long/short on the underlying.
    - Loss is capped at -notional (like long premium).
    - Profit is unbounded (linear) for simplicity.
    """
    if underlying_entry <= 0:
        return 0.0
    ret = (underlying_exit - underlying_entry) / underlying_entry
    s = (side or "").strip().lower()
    if s == "put":
        ret = -ret
    pnl = float(notional_usd) * float(leverage) * float(ret)
    return max(-float(notional_usd), pnl)


def close_open_virtual_options(
    *,
    ledger: Ledger,
    apca_api_key_id: str,
    apca_api_secret_key: str,
    ts_close: datetime | None = None,
) -> dict[str, int]:
    """
    Close any open virtual options using the latest available minute close from Alpaca IEX bars.
    This runs in the market-close workflow so EOD reports include realized virtual-option P&L.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    now = ts_close or datetime.now(tz=timezone.utc)
    open_rows = ledger.list_open_virtual_option_trades()
    if not open_rows:
        return {"closed": 0}

    syms = sorted({r["symbol"] for r in open_rows if r.get("symbol")})
    client = StockHistoricalDataClient(apca_api_key_id, apca_api_secret_key)

    # Ask for a small window near "now" and take last close we can find per symbol.
    end = now
    start = end - timedelta(hours=6)

    closes: dict[str, float] = {}
    try:
        req = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            limit=10000,
        )
        bars = client.get_stock_bars(req)
        df = getattr(bars, "df", None)
        if df is not None and not getattr(df, "empty", True):
            try:
                import pandas as pd

                if isinstance(df.index, pd.MultiIndex):
                    for sym in syms:
                        try:
                            sdf = df.xs(sym, level=0)
                        except Exception:
                            continue
                        if "close" in sdf.columns and not sdf["close"].dropna().empty:
                            closes[str(sym)] = float(sdf["close"].dropna().iloc[-1])
                else:
                    sym = str(syms[0]) if syms else ""
                    if sym and "close" in df.columns and not df["close"].dropna().empty:
                        closes[sym] = float(df["close"].dropna().iloc[-1])
            except Exception:
                closes = {}
    except Exception as e:
        log.warning("virtual_options_close_price_fetch_failed err=%s", e)

    closed = 0
    for r in open_rows:
        sym = str(r["symbol"])
        px = closes.get(sym)
        if px is None or px <= 0:
            continue
        pnl = _pnl_virtual_long_option(
            side=str(r.get("side") or "call"),
            notional_usd=float(r.get("notional_usd") or 0.0),
            leverage=float(r.get("leverage") or 1.0),
            underlying_entry=float(r.get("underlying_entry") or 0.0),
            underlying_exit=float(px),
        )
        ledger.close_virtual_option_trade(
            trade_id=int(r["id"]),
            ts_close=now,
            underlying_exit=float(px),
            pnl_usd=float(pnl),
        )
        closed += 1

    return {"closed": int(closed)}

