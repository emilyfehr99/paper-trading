from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)


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

        req = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.SELL,
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

    def close_position_market(self, symbol: str) -> ExecutionResult:
        """
        Close an open position at market via Alpaca close_position endpoint.
        This should also handle canceling/replacing bracket legs on Alpaca's side.
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return ExecutionResult(False, "empty_symbol")
        try:
            _ = self._tc.close_position(sym, close_options=ClosePositionRequest())  # full position
            return ExecutionResult(True, "close_submitted", client_order_id=None, alpaca_order_id=None)
        except Exception as e:
            return ExecutionResult(False, f"close_error:{e}", client_order_id=None, alpaca_order_id=None)


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

