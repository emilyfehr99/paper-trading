from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from alpaca_day_bot.config import Settings
from alpaca_day_bot.data.stream import BarBuffer, BarEvent, _to_float, _ts_utc

log = logging.getLogger("alpaca_day_bot.crypto_rest_bars")


class CryptoRestBarPoller:
    """
    Fill BarBuffer using Alpaca crypto REST bars.

    This is used for weekend / 24-7 crypto ticks where the equities IEX delay logic
    doesn't apply and the bot should not open an additional websocket.
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
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        lag_m = float(getattr(self._settings, "crypto_rest_bar_end_lag_minutes", 1.0) or 1.0)
        end = datetime.now(tz=timezone.utc) - timedelta(minutes=lag_m)
        start = (end - timedelta(days=3)) if self._wide_first_fetch else (end - timedelta(minutes=60))

        client = CryptoHistoricalDataClient(
            self._settings.apca_api_key_id,
            self._settings.apca_api_secret_key,
        )

        def chunks(xs: list[str], n: int):
            for i in range(0, len(xs), n):
                yield xs[i : i + n]

        symbols = list(self._settings.symbols)
        batch_n = 200
        out: list[BarEvent] = []

        for batch in chunks(symbols, batch_n):
            req = CryptoBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                limit=10000,
            )
            bars = client.get_crypto_bars(req)
            df = bars.df
            if df is None or getattr(df, "empty", True):
                continue
            if isinstance(df.index, pd.MultiIndex):
                for sym in batch:
                    try:
                        sdf = df.xs(sym, level=0)
                    except Exception:
                        continue
                    for ts, row in sdf.iterrows():
                        out.append(self._row_to_event(str(sym), ts, row))
            else:
                sym = str(batch[0])
                for ts, row in df.iterrows():
                    out.append(self._row_to_event(sym, ts, row))

        if out:
            self._wide_first_fetch = False
        return out

    def warm_buffer(self, *, rounds: int = 2, pause_s: float = 1.0) -> int:
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
            "crypto rest bar poller started",
            extra={"extra_json": {"interval_s": self._interval_s, "symbols": list(self._settings.symbols)}},
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
                log.warning("crypto rest bar fetch failed: %s", e, exc_info=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(1.0, self._interval_s - elapsed)
            time.sleep(sleep_s)

