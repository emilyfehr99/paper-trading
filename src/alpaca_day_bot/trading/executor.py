from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
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


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

