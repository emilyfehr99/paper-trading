from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import logging

import websockets
from alpaca.data.enums import DataFeed
from alpaca.data.live.websocket import DataStream
from alpaca.data.live.stock import StockDataStream

from alpaca_day_bot.config import Settings
from alpaca_day_bot.ws_retry import is_connection_or_rate_limit

log = logging.getLogger("alpaca_day_bot.market_data")
_alpaca_ws_log = logging.getLogger("alpaca.data.live.websocket")


async def _datastream_run_forever_with_limit_backoff(self: DataStream) -> None:
    """
    Replaces alpaca DataStream._run_forever: stock auth ValueError (connection limit)
    tight-looped with log.exception + no sleep. We close, sleep 120s, retry.
    Patched onto DataStream so every StockDataStream instance benefits.
    """
    self._loop = asyncio.get_running_loop()
    while not any(
        v
        for k, v in self._handlers.items()
        if k not in ("cancelErrors", "corrections")
    ):
        if not self._stop_stream_queue.empty():
            self._stop_stream_queue.get(timeout=1)
            return
        await asyncio.sleep(0)
    _alpaca_ws_log.info(f"started {self._name} stream")
    self._should_run = True
    self._running = False
    while True:
        try:
            if not self._should_run:
                _alpaca_ws_log.info("{} stream stopped".format(self._name))
                return
            if not self._running:
                _alpaca_ws_log.info("starting {} websocket connection".format(self._name))
                await self._start_ws()
                await self._send_subscribe_msg()
                self._running = True
            await self._consume()
        except websockets.WebSocketException as wse:
            await self.close()
            self._running = False
            _alpaca_ws_log.warning("data websocket error, restarting connection: " + str(wse))
        except ValueError as ve:
            if "insufficient subscription" in str(ve):
                await self.close()
                self._running = False
                _alpaca_ws_log.exception("error during websocket communication: %s", ve)
                return
            if is_connection_or_rate_limit(ve):
                await self.close()
                self._running = False
                log.warning(
                    "data websocket: connection/rate limit — backing off 120s. "
                    "Kill duplicate bots; only one process per API key.",
                    extra={"extra_json": {"error": str(ve)}},
                )
                await asyncio.sleep(120.0)
                continue
            _alpaca_ws_log.exception("error during websocket communication: %s", ve)
        except Exception as e:
            _alpaca_ws_log.exception("error during websocket communication: %s", e)
        finally:
            await asyncio.sleep(0)


DataStream._run_forever = _datastream_run_forever_with_limit_backoff  # type: ignore[assignment]


@dataclass(frozen=True)
class BarEvent:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


class BarBuffer:
    def __init__(self, maxlen: int = 512) -> None:
        self._buf: dict[str, deque[BarEvent]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._last_seen_ts: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def append(self, bar: BarEvent) -> bool:
        async with self._lock:
            last = self._last_seen_ts.get(bar.symbol)
            if last is not None and bar.ts <= last:
                return False
            self._last_seen_ts[bar.symbol] = bar.ts
            self._buf[bar.symbol].append(bar)
            return True

    async def snapshot(self, symbol: str) -> list[BarEvent]:
        async with self._lock:
            return list(self._buf.get(symbol, []))

    async def latest(self, symbol: str) -> BarEvent | None:
        async with self._lock:
            dq = self._buf.get(symbol)
            if not dq:
                return None
            return dq[-1]

    async def snapshot_df(self, symbol: str):
        """
        Return a pandas DataFrame indexed by UTC timestamp with OHLCV columns.
        NaNs/infs are dropped to keep indicator pipelines stable.
        """
        bars = await self.snapshot(symbol)
        return bars_to_df(bars)

    async def snapshot_resampled_df(self, symbol: str, rule: str):
        """
        Resample OHLCV to a higher timeframe (e.g., '5min', '15min').
        """
        df = await self.snapshot_df(symbol)
        if df is None or df.empty:
            return df
        return resample_ohlcv(df, rule=rule)


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _ts_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MarketDataStreamer:
    def __init__(self, settings: Settings, buffer: BarBuffer) -> None:
        self._settings = settings
        self._buffer = buffer

    async def _on_bar(self, bar) -> None:
        # alpaca-py bar objects have fields like symbol, timestamp, open, high, low, close, volume, vwap
        evt = BarEvent(
            symbol=str(getattr(bar, "symbol")),
            ts=_ts_utc(getattr(bar, "timestamp", None)),
            open=_to_float(getattr(bar, "open", None)),
            high=_to_float(getattr(bar, "high", None)),
            low=_to_float(getattr(bar, "low", None)),
            close=_to_float(getattr(bar, "close", None)),
            volume=_to_float(getattr(bar, "volume", None)),
            vwap=(None if getattr(bar, "vwap", None) is None else _to_float(getattr(bar, "vwap"))),
        )
        await self._buffer.append(evt)

    def run_forever(self) -> None:
        # New StockDataStream + single subscribe per attempt so we never stack subscriptions
        # on a half-dead client (a common cause of Alpaca "connection limit exceeded").
        import time as _time

        backoff_s = 1.0
        while True:
            stream = StockDataStream(
                self._settings.apca_api_key_id,
                self._settings.apca_api_secret_key,
                feed=DataFeed.IEX,
            )
            try:
                stream.subscribe_bars(self._on_bar, *self._settings.symbols)
                stream.run()
            except Exception as e:
                if is_connection_or_rate_limit(e):
                    log.warning(
                        "data websocket: connection/rate limit — backing off 120s. "
                        "Stop duplicate bot processes (only one `alpaca_day_bot` per API key).",
                        extra={"extra_json": {"error": str(e)}},
                    )
                    _time.sleep(120.0)
                    backoff_s = 1.0
                else:
                    log.debug("data websocket error: %s", e, exc_info=True)
                    _time.sleep(backoff_s)
                    backoff_s = min(60.0, max(1.0, backoff_s * 2.0))
                continue


def bars_to_df(bars: Iterable[BarEvent]):
    import numpy as np
    import pandas as pd

    rows = []
    for b in bars:
        rows.append(
            {
                "ts": b.ts,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": b.vwap,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

    df = pd.DataFrame.from_records(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def resample_ohlcv(df, *, rule: str):
    """
    OHLCV resample with standard aggregations.
    """
    import numpy as np

    if df is None or df.empty:
        return df
    ohlc = df[["open", "high", "low", "close"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    vol = df[["volume"]].resample(rule).sum()
    out = ohlc.join(vol, how="inner")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

