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
        from alpaca.data.timeframe import TimeFrame
        from concurrent.futures import ThreadPoolExecutor, as_completed

        asset_class = (getattr(self._settings, "asset_class", "equity") or "equity").strip().lower()
        is_crypto = asset_class == "crypto"

        lag_m = float(self._settings.crypto_rest_bar_end_lag_minutes if is_crypto else self._settings.rest_bar_end_lag_minutes)
        end = datetime.now(tz=timezone.utc) - timedelta(minutes=lag_m)
        # First fetch: include multiple market sessions so 15m RSI(14) can be ready
        # even early in the day (a same-day minute window may be too short).
        # Later fetches keep a small window to reduce payload.
        start = (end - timedelta(days=7)) if self._wide_first_fetch else (end - timedelta(minutes=20))

        if is_crypto:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            client = CryptoHistoricalDataClient(
                self._settings.apca_api_key_id,
                self._settings.apca_api_secret_key,
            )
        else:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.enums import DataFeed
            client = StockHistoricalDataClient(
                self._settings.apca_api_key_id,
                self._settings.apca_api_secret_key,
            )

        def chunks(xs: list[str], n: int):
            for i in range(0, len(xs), n):
                yield xs[i : i + n]

        symbols = []
        for s in self._settings.symbols:
            if is_crypto:
                if "/" in s:
                    symbols.append(s)
            else:
                if "/" not in s:
                    symbols.append(s)
        # Alpaca endpoints can reject very large symbol lists; batch conservatively.
        batch_n = 200
        batches = list(chunks(symbols, batch_n))
        out: list[BarEvent] = []

        def _fetch_batch(batch):
            try:
                if is_crypto:
                    req = CryptoBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Minute,
                        start=start,
                        end=end,
                        limit=10000,
                    )
                    bars = client.get_crypto_bars(req)
                else:
                    req = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Minute,
                        start=start,
                        end=end,
                        feed=DataFeed.IEX,
                        extended_hours=True,
                        limit=10000,
                    )
                    bars = client.get_stock_bars(req)

                df = bars.df
                if df is None or getattr(df, "empty", True):
                    return []

                batch_events = []
                if isinstance(df.index, pd.MultiIndex):
                    for sym in batch:
                        try:
                            sdf = df.xs(sym, level=0)
                        except Exception:
                            continue
                        for ts, row in sdf.iterrows():
                            batch_events.append(self._row_to_event(str(sym), ts, row))
                else:
                    sym = str(batch[0])
                    for ts, row in df.iterrows():
                        batch_events.append(self._row_to_event(sym, ts, row))
                return batch_events
            except Exception as e:
                log.warning("batch fetch failed for symbols %s: %s", batch[:5], e)
                return []

        # Run all batches concurrently (I/O bound Alpaca calls)
        # Using 8 workers so we fetch 1500 symbols in parallel instantly.
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="rest_poller") as executor:
            futures = {executor.submit(_fetch_batch, b): b for b in batches}
            for fut in as_completed(futures):
                batch_res = fut.result()
                if batch_res:
                    out.extend(batch_res)

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
                for e in events:
                    self._buffer.append(e)
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
                    for e in events:
                        self._buffer.append(e)
            except Exception as e:
                log.warning("rest bar fetch failed: %s", e, exc_info=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(1.0, self._interval_s - elapsed)
            time.sleep(sleep_s)
