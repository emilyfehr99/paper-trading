from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from alpaca.trading.stream import TradingStream

from alpaca_day_bot.config import Settings
from alpaca_day_bot.ws_retry import is_connection_or_rate_limit

log = logging.getLogger("alpaca_day_bot.trading_stream")


@dataclass(frozen=True)
class TradeUpdateEvent:
    event: str
    symbol: str | None
    order_id: str | None
    client_order_id: str | None
    filled_qty: float | None
    filled_avg_price: float | None
    ts: datetime
    """Full snapshot from Alpaca (order fields + event) for audit."""
    payload: dict[str, Any]


def _ts_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TradingUpdatesStreamer:
    def __init__(
        self,
        settings: Settings,
        on_update: Callable[[TradeUpdateEvent], None] | Callable[[TradeUpdateEvent], "object"],
    ) -> None:
        self._settings = settings
        self._on_update = on_update

    def _subscribe(self, stream: TradingStream) -> None:
        async def _handler(data) -> None:
            order = getattr(data, "order", None)
            payload = _build_trade_update_payload(data, order)
            evt = TradeUpdateEvent(
                event=str(getattr(data, "event", "")),
                symbol=(None if order is None else getattr(order, "symbol", None)),
                order_id=(None if order is None else str(getattr(order, "id", None))),
                client_order_id=(None if order is None else getattr(order, "client_order_id", None)),
                filled_qty=(None if order is None else _maybe_float(getattr(order, "filled_qty", None))),
                filled_avg_price=(None if order is None else _maybe_float(getattr(order, "filled_avg_price", None))),
                ts=_ts_utc(getattr(data, "timestamp", None)),
                payload=payload,
            )
            res = self._on_update(evt)
            if hasattr(res, "__await__"):
                await res  # type: ignore[misc]

        stream.subscribe_trade_updates(_handler)

    def run_forever(self) -> None:
        import time as _time

        backoff_s = 1.0
        while True:
            stream = TradingStream(
                self._settings.apca_api_key_id,
                self._settings.apca_api_secret_key,
                paper=True,
            )
            self._subscribe(stream)
            try:
                stream.run()
            except Exception as e:
                if is_connection_or_rate_limit(e):
                    log.warning(
                        "trading websocket: connection/rate limit — backing off 120s. "
                        "Stop duplicate bot processes (only one `alpaca_day_bot` per API key).",
                        extra={"extra_json": {"error": str(e)}},
                    )
                    _time.sleep(120.0)
                    backoff_s = 1.0
                else:
                    log.debug("trading websocket error: %s", e, exc_info=True)
                    _time.sleep(backoff_s)
                    backoff_s = min(60.0, max(1.0, backoff_s * 2.0))
                continue


def _maybe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _build_trade_update_payload(data, order) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stream_event": str(getattr(data, "event", "")),
        "timestamp": _ts_utc(getattr(data, "timestamp", None)).isoformat(),
    }
    if order is not None:
        out["order"] = _alpaca_order_to_dict(order)
    return out


def _alpaca_order_to_dict(order) -> dict[str, Any]:
    """Best-effort serialization of Alpaca order objects for ledger/audit."""
    keys = (
        "id",
        "client_order_id",
        "created_at",
        "updated_at",
        "submitted_at",
        "filled_at",
        "expired_at",
        "canceled_at",
        "failed_at",
        "replaced_at",
        "replaced_by",
        "replaces",
        "asset_id",
        "symbol",
        "asset_class",
        "notional",
        "qty",
        "filled_qty",
        "filled_avg_price",
        "order_class",
        "order_type",
        "type",
        "side",
        "time_in_force",
        "status",
        "limit_price",
        "stop_price",
        "trail_price",
        "trail_percent",
        "hwm",
    )
    d: dict[str, Any] = {}
    for k in keys:
        if hasattr(order, k):
            v = getattr(order, k)
            d[k] = _json_safe(v)
    return d


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)

