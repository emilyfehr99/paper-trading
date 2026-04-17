from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from alpaca_day_bot.config import Settings
from alpaca_day_bot.data.stream import BarBuffer, BarEvent, _to_float, _ts_utc

log = logging.getLogger("alpaca_day_bot.rest_bars")


class RestBarPoller:
    """
    Fill BarBuffer using REST historical bars instead of the market-data websocket.

    Alpaca often allows only one concurrent market-data websocket per account; this bot
    also uses the trading updates websocket, so streaming bars frequently hits
    "connection limit exceeded". REST polling avoids the second MD websocket entirely.
    """

    def __init__(self, settings: Settings, buffer: BarBuffer) -> None:
        self._settings = settings
        self._buffer = buffer
        self._interval_s = float(settings.rest_bar_poll_interval_s)
        self._wide_first_fetch = True

    def _row_to_event(self, symbol: str, ts, row) -> BarEvent:
        return BarEvent(
            symbol=symbol,
            ts=_ts_utc(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts),
            open=_to_float(row.get("open")),
            high=_to_float(row.get("high")),
            low=_to_float(row.get("low")),
            close=_to_float(row.get("close")),
            volume=_to_float(row.get("volume")),
            vwap=(None if row.get("vwap") is None else _to_float(row.get("vwap"))),
        )

    def _fetch_events(self) -> list[BarEvent]:
        import pandas as pd
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        lag_m = float(self._settings.rest_bar_end_lag_minutes)
        end = datetime.now(tz=timezone.utc) - timedelta(minutes=lag_m)
        # First fetch: enough 1m history for indicators; later: small window to save payload.
        mins = 180 if self._wide_first_fetch else 20
        start = end - timedelta(minutes=mins)

        client = StockHistoricalDataClient(
            self._settings.apca_api_key_id,
            self._settings.apca_api_secret_key,
        )
        req = StockBarsRequest(
            symbol_or_symbols=list(self._settings.symbols),
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is None or getattr(df, "empty", True):
            return []

        out: list[BarEvent] = []
        if isinstance(df.index, pd.MultiIndex):
            for sym in self._settings.symbols:
                try:
                    sdf = df.xs(sym, level=0)
                except Exception:
                    continue
                for ts, row in sdf.iterrows():
                    out.append(self._row_to_event(str(sym), ts, row))
        else:
            sym = str(self._settings.symbols[0])
            for ts, row in df.iterrows():
                out.append(self._row_to_event(sym, ts, row))

        if out:
            self._wide_first_fetch = False
        return out

    def warm_buffer(self, *, rounds: int = 2, pause_s: float = 1.0) -> int:
        """Fetch REST bars synchronously (for CI / scheduled ticks; no background thread)."""
        total = 0
        for _ in range(max(1, rounds)):
            try:
                events = self._fetch_events()
            except Exception as e:
                log.warning("warm_buffer fetch failed: %s", e, exc_info=True)
                events = []
            if events:

                async def _push() -> None:
                    for e in events:
                        await self._buffer.append(e)

                asyncio.run(_push())
                total += len(events)
            time.sleep(pause_s)
        return total

    def run_forever(self) -> None:
        log.info(
            "rest bar poller started",
            extra={
                "extra_json": {
                    "interval_s": self._interval_s,
                    "symbols": list(self._settings.symbols),
                }
            },
        )
        while True:
            t0 = time.monotonic()
            try:
                events = self._fetch_events()
                if events:

                    async def _push() -> None:
                        for e in events:
                            await self._buffer.append(e)

                    asyncio.run(_push())
            except Exception as e:
                log.warning("rest bar fetch failed: %s", e, exc_info=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(1.0, self._interval_s - elapsed)
            time.sleep(sleep_s)
