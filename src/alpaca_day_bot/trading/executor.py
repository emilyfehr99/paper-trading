from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from alpaca_day_bot.trading.updates import TradeUpdateEvent


@dataclass(frozen=True)
class ExecutionResult:
    submitted: bool
    reason: str
    client_order_id: str | None = None
    alpaca_order_id: str | None = None


class OrderExecutor:
    def __init__(self, trading_client: TradingClient) -> None:
        self._tc = trading_client
        self._asset_cache: dict[str, dict] = {}

    def _get_asset(self, symbol: str) -> dict | None:
        sym = (symbol or "").strip().upper()
        if not sym:
            return None
        if sym in self._asset_cache:
            return self._asset_cache[sym]
        try:
            a = self._tc.get_asset(sym)
        except Exception:
            return None
        # Normalize to plain dict-like access
        d = {}
        for k in ("shortable", "easy_to_borrow", "tradable", "marginable", "status"):
            try:
                d[k] = getattr(a, k, None)
            except Exception:
                d[k] = None
        self._asset_cache[sym] = d
        return d

    def is_shortable(self, symbol: str) -> bool:
        a = self._get_asset(symbol)
        if not a:
            # If we can't determine, don't hard-block paper trading.
            return True
        shortable = a.get("shortable")
        if shortable is False:
            return False
        # Some accounts expose easy_to_borrow; prefer it if present.
        etb = a.get("easy_to_borrow")
        if etb is False:
            return False
        return True

    def get_account_equity(self) -> float:
        acct = self._tc.get_account()
        try:
            return float(acct.equity)
        except Exception:
            return float(acct.last_equity)

    def gross_exposure_usd(self) -> float:
        positions = self._tc.get_all_positions()
        exposure = 0.0
        for p in positions:
            try:
                exposure += abs(float(p.market_value))
            except Exception:
                continue
        return exposure

    def open_positions_count(self) -> int:
        return len(self._tc.get_all_positions())

    def open_position_symbols(self) -> list[str]:
        out: list[str] = []
        try:
            for p in self._tc.get_all_positions():
                sym = str(getattr(p, "symbol", "") or "").strip().upper()
                if sym:
                    out.append(sym)
        except Exception:
            return []
        return out

    def short_positions_count(self) -> int:
        n = 0
        try:
            for p in self._tc.get_all_positions():
                try:
                    qty = float(getattr(p, "qty", 0.0) or 0.0)
                    if qty < 0:
                        n += 1
                except Exception:
                    continue
        except Exception:
            return 0
        return int(n)

    def get_position_entry_price(self, symbol: str) -> float | None:
        try:
            p = self._tc.get_open_position(symbol)
            px = getattr(p, "avg_entry_price", None)
            if px is None:
                return None
            v = float(px)
            return v if v > 0 else None
        except Exception:
            return None

    def has_position(self, symbol: str) -> bool:
        try:
            _ = self._tc.get_open_position(symbol)
            return True
        except Exception:
            return False

    def submit_bracket_buy(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        take_profit_price: float,
    ) -> ExecutionResult:
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        if stop_price <= 0 or take_profit_price <= 0:
            return ExecutionResult(False, "bad_exit_prices")

        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"

        req = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
            client_order_id=client_order_id,
        )

        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(
                False,
                f"submit_error:{e}",
                client_order_id=client_order_id,
                alpaca_order_id=None,
            )
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_entry_buy_market(self, *, symbol: str, qty: float) -> ExecutionResult:
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_entry_buy_notional_market(self, *, symbol: str, notional_usd: float) -> ExecutionResult:
        """
        Crypto-friendly market buy using notional (supports fractional qty implicitly).
        """
        if notional_usd <= 0:
            return ExecutionResult(False, "bad_notional")
        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(float(notional_usd), 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_entry_buy_limit(self, *, symbol: str, qty: float, limit_price: float) -> ExecutionResult:
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        if limit_price <= 0:
            return ExecutionResult(False, "bad_prices")
        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(float(limit_price), 2),
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_bracket_buy_limit(
        self,
        *,
        symbol: str,
        qty: float,
        limit_price: float,
        stop_price: float,
        take_profit_price: float,
    ) -> ExecutionResult:
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        if limit_price <= 0 or stop_price <= 0 or take_profit_price <= 0:
            return ExecutionResult(False, "bad_prices")

        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            limit_price=round(float(limit_price), 2),
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_bracket_short(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        take_profit_price: float,
    ) -> ExecutionResult:
        """
        Open a short position with a bracket (take-profit below, stop above).
        """
        if not self.is_shortable(symbol):
            return ExecutionResult(False, "not_shortable")
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        if stop_price <= 0 or take_profit_price <= 0:
            return ExecutionResult(False, "bad_exit_prices")

        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"

        def _make_req(tp: float) -> MarketOrderRequest:
            return MarketOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(float(tp), 2)),
                stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
                client_order_id=client_order_id,
            )

        try:
            order = self._tc.submit_order(order_data=_make_req(float(take_profit_price)))
        except Exception as e:
            # Alpaca sometimes rejects short brackets if TP is not below the actual base/entry price.
            # If we can parse base_price from the error payload, clamp TP and retry once.
            try:
                msg = str(e)
                if "take_profit.limit_price" in msg and "base_price" in msg:
                    j = json.loads(msg[msg.index("{") : msg.rindex("}") + 1])
                    base = float(j.get("base_price"))
                    # ensure TP <= base - 0.05 (extra buffer beyond the 0.01 rule)
                    tp2 = min(float(take_profit_price), base - 0.05)
                    if tp2 > 0:
                        order = self._tc.submit_order(order_data=_make_req(tp2))
                    else:
                        raise
                else:
                    raise
            except Exception:
                return ExecutionResult(
                    False,
                    f"submit_error:{e}",
                    client_order_id=client_order_id,
                    alpaca_order_id=None,
                )
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_entry_short_market(self, *, symbol: str, qty: float) -> ExecutionResult:
        if not self.is_shortable(symbol):
            return ExecutionResult(False, "not_shortable")
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_entry_short_limit(self, *, symbol: str, qty: float, limit_price: float) -> ExecutionResult:
        if not self.is_shortable(symbol):
            return ExecutionResult(False, "not_shortable")
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        if limit_price <= 0:
            return ExecutionResult(False, "bad_prices")
        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(float(limit_price), 2),
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_bracket_short_limit(
        self,
        *,
        symbol: str,
        qty: float,
        limit_price: float,
        stop_price: float,
        take_profit_price: float,
    ) -> ExecutionResult:
        """
        Open a short position with a limit-entry bracket.
        """
        if not self.is_shortable(symbol):
            return ExecutionResult(False, "not_shortable")
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        if limit_price <= 0 or stop_price <= 0 or take_profit_price <= 0:
            return ExecutionResult(False, "bad_prices")

        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            limit_price=round(float(limit_price), 2),
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def submit_exit_oco(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        take_profit_price: float,
        stop_price: float,
    ) -> ExecutionResult:
        """
        Synthetic exits using Alpaca OCO: submit TP+SL as an OCO pair.
        side: the exit side (\"sell\" for long exits, \"buy\" for short exits)
        """
        if qty <= 0:
            return ExecutionResult(False, "bad_qty")
        sym = (symbol or "").strip().upper()
        side_u = (side or "").strip().lower()
        if side_u not in ("buy", "sell"):
            return ExecutionResult(False, "bad_side")
        if take_profit_price <= 0 or stop_price <= 0:
            return ExecutionResult(False, "bad_exit_prices")

        client_order_id = f"adbot-{uuid.uuid4().hex[:16]}"
        req = MarketOrderRequest(
            symbol=sym,
            qty=int(qty),
            side=(OrderSide.BUY if side_u == "buy" else OrderSide.SELL),
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
            client_order_id=client_order_id,
        )
        try:
            order = self._tc.submit_order(order_data=req)
        except Exception as e:
            return ExecutionResult(False, f"submit_error:{e}", client_order_id=client_order_id, alpaca_order_id=None)
        oid = None
        try:
            oid = str(getattr(order, "id", None)) if order is not None else None
        except Exception:
            oid = None
        return ExecutionResult(True, "submitted", client_order_id=client_order_id, alpaca_order_id=oid)

    def close_position_market(self, symbol: str) -> ExecutionResult:
        """
        Close an open position at market via Alpaca close_position endpoint.
        This should also handle canceling/replacing bracket legs on Alpaca's side.
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return ExecutionResult(False, "empty_symbol")
        try:
            # Full close: do not pass ClosePositionRequest unless specifying qty/percentage.
            order = self._tc.close_position(sym)
            oid = None
            try:
                oid = str(getattr(order, "id", None)) if order is not None else None
            except Exception:
                oid = None
            return ExecutionResult(True, "close_submitted", client_order_id=None, alpaca_order_id=oid)
        except Exception as e:
            msg = str(e)
            # Common when bracket/OCO legs are still open: shares are held_for_orders.
            if "insufficient qty available for order" in msg or "\"held_for_orders\"" in msg:
                try:
                    self._cancel_open_orders_for_symbol(sym)
                    order = self._tc.close_position(sym)
                    oid = None
                    try:
                        oid = str(getattr(order, "id", None)) if order is not None else None
                    except Exception:
                        oid = None
                    return ExecutionResult(
                        True, "close_submitted_after_cancel", client_order_id=None, alpaca_order_id=oid
                    )
                except Exception as e2:
                    return ExecutionResult(False, f"close_error:{e2}", client_order_id=None, alpaca_order_id=None)
            return ExecutionResult(False, f"close_error:{e}", client_order_id=None, alpaca_order_id=None)

    def _cancel_open_orders_for_symbol(self, symbol: str) -> None:
        sym = (symbol or "").strip().upper()
        if not sym:
            return
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym], nested=True, limit=500)
            orders = self._tc.get_orders(req) or []
        except Exception:
            orders = []
        for o in orders:
            try:
                oid = str(getattr(o, "id", "") or "")
                if oid:
                    self._tc.cancel_order_by_id(oid)
            except Exception:
                continue

    def poll_order_fill_event(
        self,
        *,
        order_id: str,
        timeout_s: float,
        poll_s: float,
    ) -> TradeUpdateEvent | None:
        """
        REST fallback for scheduled ticks: poll an order by id and return a synthetic fill event
        if it becomes filled/partially_filled within timeout.
        """
        oid = (order_id or "").strip()
        if not oid:
            return None
        import time

        t_end = time.time() + max(1.0, float(timeout_s))
        poll = max(0.5, float(poll_s))
        last_seen = None
        while time.time() < t_end:
            try:
                o = self._tc.get_order_by_id(oid)
            except Exception:
                o = None
            if o is not None:
                try:
                    status = str(getattr(o, "status", "") or "").lower()
                    filled_qty = getattr(o, "filled_qty", None)
                    filled_avg_price = getattr(o, "filled_avg_price", None)
                    sym = getattr(o, "symbol", None)
                    client_oid = getattr(o, "client_order_id", None)
                    # order has timestamps; prefer filled_at/updated_at if present
                    ts = getattr(o, "filled_at", None) or getattr(o, "updated_at", None) or getattr(o, "submitted_at", None)
                    if status != last_seen:
                        last_seen = status
                    if status in ("filled", "partially_filled") or (
                        filled_qty not in (None, "", 0, 0.0) and filled_avg_price not in (None, "", 0, 0.0)
                    ):
                        # Convert best-effort types
                        try:
                            fq = float(filled_qty) if filled_qty is not None else None
                        except Exception:
                            fq = None
                        try:
                            fpx = float(filled_avg_price) if filled_avg_price is not None else None
                        except Exception:
                            fpx = None
                        dt = None
                        try:
                            if hasattr(ts, "tzinfo"):
                                dt = ts
                        except Exception:
                            dt = None
                        evt_ts = dt if dt is not None else datetime.now(tz=timezone.utc)
                        payload = {"source": "rest_poll", "order": {"id": oid, "status": status}}
                        return TradeUpdateEvent(
                            event=("partial_fill" if status == "partially_filled" else "fill"),
                            symbol=(None if sym is None else str(sym)),
                            order_id=str(oid),
                            client_order_id=(None if client_oid is None else str(client_oid)),
                            filled_qty=fq,
                            filled_avg_price=fpx,
                            ts=(evt_ts if evt_ts.tzinfo is not None else evt_ts.replace(tzinfo=timezone.utc)),
                            payload=payload,
                        )
                except Exception:
                    # ignore parse errors and keep polling
                    pass
            time.sleep(poll)
        return None


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

