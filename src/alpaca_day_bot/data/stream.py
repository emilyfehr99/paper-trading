from __future__ import annotations

import asyncio
import threading
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
            from alpaca_day_bot.services.telemetry import WEBSOCKET_DROPS
            WEBSOCKET_DROPS.inc()
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
                from alpaca_day_bot.services.telemetry import WEBSOCKET_DROPS
                WEBSOCKET_DROPS.inc()
                log.warning(
                     "data websocket: connection/rate limit — backing off 120s. "
                     "Kill duplicate bots; only one process per API key.",
                     extra={"extra_json": {"error": str(ve)}},
                )
                await asyncio.sleep(120.0)
                continue
            _alpaca_ws_log.exception("error during websocket communication: %s", ve)
        except Exception as e:
            from alpaca_day_bot.services.telemetry import WEBSOCKET_DROPS
            WEBSOCKET_DROPS.inc()
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
        self._lock = threading.Lock()

    def append(self, bar: BarEvent) -> bool:
        with self._lock:
            last = self._last_seen_ts.get(bar.symbol)
            if last is not None and bar.ts <= last:
                return False
            self._last_seen_ts[bar.symbol] = bar.ts
            self._buf[bar.symbol].append(bar)
            return True

    def snapshot(self, symbol: str) -> list[BarEvent]:
        with self._lock:
            return list(self._buf.get(symbol, []))

    def latest(self, symbol: str) -> BarEvent | None:
        with self._lock:
            dq = self._buf.get(symbol)
            if not dq:
                return None
            return dq[-1]

    def snapshot_df(self, symbol: str):
        """
        Return a pandas DataFrame indexed by UTC timestamp with OHLCV columns.
        NaNs/infs are dropped to keep indicator pipelines stable.
        """
        bars = self.snapshot(symbol)
        return bars_to_df(bars)

    def snapshot_resampled_df(self, symbol: str, rule: str):
        """
        Resample OHLCV to a higher timeframe (e.g., '5min', '15min').
        """
        df = self.snapshot_df(symbol)
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


@dataclass(frozen=True)
class QuoteEvent:
    symbol: str
    ts: datetime
    ask_prices: list[float] # Levels 1-10
    ask_sizes: list[float]
    bid_prices: list[float]
    bid_sizes: list[float]

class OrderBookBuffer:
    def __init__(self, maxlen: int = 100) -> None:
        self._buf: dict[str, deque[QuoteEvent]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._lock = threading.Lock()

    def append(self, quote: QuoteEvent):
        with self._lock:
            self._buf[quote.symbol].append(quote)

    def latest_features(self, symbol: str) -> list[float]:
        """
        Returns the 40 features (10 levels of Ask Price/Size, 10 levels of Bid Price/Size).
        """
        with self._lock:
            dq = self._buf.get(symbol)
            if not dq:
                return [0.0] * 40
            q = dq[-1]
            # Flatten to 40 features: 10 Ask P, 10 Ask S, 10 Bid P, 10 Bid S
            return (q.ask_prices[:10] + [0.0]*10)[:10] + \
                   (q.ask_sizes[:10] + [0.0]*10)[:10] + \
                   (q.bid_prices[:10] + [0.0]*10)[:10] + \
                   (q.bid_sizes[:10] + [0.0]*10)[:10]

class MarketDataStreamer:
    def __init__(self, settings: Settings, buffer: BarBuffer, book_buffer: OrderBookBuffer | None = None) -> None:
        self._settings = settings
        self._buffer = buffer
        self._book_buffer = book_buffer
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def _on_quote(self, quote) -> None:
        if self._book_buffer is None:
            return
        # Extract 10 levels if available (Alpaca L2 SIP)
        # Note: Alpaca quotes usually provide top level. 
        # For full 10 levels, we'd iterate through getattr(quote, 'ask_prices', []) etc.
        evt = QuoteEvent(
            symbol=str(getattr(quote, "symbol")),
            ts=_ts_utc(getattr(quote, "timestamp", None)),
            ask_prices=[_to_float(getattr(quote, "ask_price", 0.0))],
            ask_sizes=[_to_float(getattr(quote, "ask_size", 0.0))],
            bid_prices=[_to_float(getattr(quote, "bid_price", 0.0))],
            bid_sizes=[_to_float(getattr(quote, "bid_size", 0.0))],
        )
        self._book_buffer.append(evt)

    async def _on_bar(self, bar) -> None:
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
        self._buffer.append(evt)

    def run_forever(self) -> None:
        import time as _time
        from alpaca.data.live.crypto import CryptoDataStream
        from alpaca.data.live.stock import StockDataStream
        from alpaca.data.enums import DataFeed

        backoff_s = 1.0
        is_crypto = self._settings.asset_class.lower() == "crypto"
        
        while self._running:
            if is_crypto:
                stream = CryptoDataStream(
                    self._settings.apca_api_key_id,
                    self._settings.apca_api_secret_key,
                )
            else:
                stream = StockDataStream(
                    self._settings.apca_api_key_id,
                    self._settings.apca_api_secret_key,
                    feed=DataFeed.IEX,
                )
                
            try:
                # Filter symbols for crypto (must contain '/') or stocks
                valid_symbols = []
                log.info(f"Streamer checking {len(self._settings.symbols)} symbols for {self._settings.asset_class} mode...")
                for s in self._settings.symbols:
                    if is_crypto:
                        if "/" in s: valid_symbols.append(s)
                    else:
                        if "/" not in s: valid_symbols.append(s)
                
                if not valid_symbols:
                    log.warning(f"No valid {self._settings.asset_class} symbols found in: {self._settings.symbols[:10]}... Waiting...")
                    _time.sleep(30)
                    continue

                log.info(f"Subscribing to {self._settings.asset_class} bars: {valid_symbols}")
                if is_crypto:
                    stream.subscribe_bars(self._on_bar, *valid_symbols)
                else:
                    stream.subscribe_bars(self._on_bar, *valid_symbols)
                
                stream.run()
            except Exception as e:
                if not self._running:
                    break
                if is_connection_or_rate_limit(e):
                    log.warning(f"backoff {self._settings.asset_class} stream: {e}")
                    _time.sleep(120.0)
                    backoff_s = 1.0
                else:
                    log.debug("data error: %s", e)
                    _time.sleep(backoff_s)
                    backoff_s = min(60.0, max(1.0, backoff_s * 2.0))
                continue


def bars_to_df(bars: Iterable[BarEvent]):
    import numpy as np
    import pandas as pd

    rows = []
    for b in bars:
        if b is None:
            continue
        try:
            ts = getattr(b, "ts", None) or getattr(b, "timestamp", None)
            o = getattr(b, "open", None)
            h = getattr(b, "high", None)
            l = getattr(b, "low", None)
            c = getattr(b, "close", None)
            v = getattr(b, "volume", None)
            vw = getattr(b, "vwap", None)
        except Exception:
            ts = o = h = l = c = v = vw = None

        if ts is None and isinstance(b, dict):
            ts = b.get("ts") or b.get("timestamp")
            o = b.get("open")
            h = b.get("high")
            l = b.get("low")
            c = b.get("close")
            v = b.get("volume")
            vw = b.get("vwap")

        if ts is not None:
            rows.append(
                {
                    "ts": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "vwap": vw,
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

