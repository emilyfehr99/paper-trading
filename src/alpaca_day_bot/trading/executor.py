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
    def __init__(self, trading_client: TradingClient, settings: Any = None, prefix: str | None = None, ledger: Any = None) -> None:
        self._tc = trading_client
        self._asset_cache: dict[str, dict] = {}
        self._settings = settings
        self._ledger = ledger
        if prefix is not None:
            self._prefix = prefix
        elif settings is not None:
            self._prefix = getattr(settings, "client_order_id_prefix", "adbot-")
        else:
            import os
            self._prefix = os.environ.get("CLIENT_ORDER_ID_PREFIX", "adbot-")
        self._cached_open_orders = None
        self._cached_open_orders_ts = None

    def _gen_client_order_id(self, suffix: str = "") -> str:
        # Ensures client_order_id prefix is used, keeping length under 48 chars
        uid = uuid.uuid4().hex[:12]
        return f"{self._prefix}{suffix}{uid}"

    def get_owned_symbols_from_db(self) -> set[str]:
        if not self._settings:
            return set()
        import sqlite3
        from pathlib import Path
        db_path = Path(self._settings.state_dir) / "ledger.sqlite3"
        if not db_path.exists():
            return set()
        
        owned = set()
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            cursor = conn.cursor()
            # Calculate net quantity from trade_updates
            cursor.execute("""
                SELECT t.symbol, SUM(
                    CASE 
                        WHEN json_extract(t.raw_json, '$.payload.order.side') = 'buy' THEN t.filled_qty 
                        WHEN json_extract(t.raw_json, '$.payload.order.side') = 'sell' THEN -t.filled_qty 
                        ELSE 0 
                    END
                ) as net_qty
                FROM trade_updates t
                INNER JOIN (
                    SELECT order_id, MAX(rowid) as max_rowid
                    FROM trade_updates
                    GROUP BY order_id
                ) latest ON t.rowid = latest.max_rowid
                GROUP BY t.symbol
            """)
            for row in cursor.fetchall():
                sym, net_qty = row
                if net_qty is not None and abs(net_qty) > 1e-5:
                    owned.add(sym.strip().upper())
            conn.close()
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.warning(f"Error querying ledger for owned symbols: {e}")
        return owned

    def get_open_orders_cached(self) -> list:
        now = datetime.now(timezone.utc)
        if (self._cached_open_orders is not None and 
            self._cached_open_orders_ts is not None and 
            (now - self._cached_open_orders_ts).total_seconds() < 2.0):
            return self._cached_open_orders
        
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
            orders = self._tc.get_orders(req) or []
            self._cached_open_orders = orders
            self._cached_open_orders_ts = now
            return orders
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.warning(f"Error fetching open orders: {e}")
            return []

    def get_owned_symbols(self) -> set[str]:
        owned = set()
        # 1. Query DB net quantity as the source of truth
        owned_from_db = self.get_owned_symbols_from_db()
        owned.update(owned_from_db)
        
        # 2. Query open orders on Alpaca as a runtime check
        try:
            orders = self.get_open_orders_cached()
            for o in orders:
                client_oid = getattr(o, "client_order_id", None)
                if client_oid and client_oid.startswith(self._prefix):
                    owned.add(o.symbol.strip().upper())
                for leg in getattr(o, "legs", []) or []:
                    leg_client_oid = getattr(leg, "client_order_id", None)
                    if leg_client_oid and leg_client_oid.startswith(self._prefix):
                        owned.add(leg.symbol.strip().upper())
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.warning(f"Error checking open orders for owned symbols: {e}")
            
        return owned

    def pending_orders_notional_usd(self) -> float:
        """
        Calculate the estimated notional value of all open/pending entry orders
        placed by this bot.
        """
        try:
            orders = self.get_open_orders_cached()
            total_notional = 0.0
            for o in orders:
                client_oid = getattr(o, "client_order_id", None)
                # Only count entry orders placed by this bot
                if client_oid and client_oid.startswith(self._prefix):
                    qty = float(o.qty) if o.qty else 0.0
                    price = 0.0
                    if o.limit_price:
                        price = float(o.limit_price)
                    elif o.stop_price:
                        price = float(o.stop_price)
                    else:
                        # Fallback for market order: get latest price
                        sym = str(o.symbol).strip().upper()
                        try:
                            if self._settings:
                                from alpaca.data.historical import StockHistoricalDataClient
                                from alpaca.data.requests import StockLatestTradeRequest
                                data_client = StockHistoricalDataClient(
                                    self._settings.apca_api_key_id, 
                                    self._settings.apca_api_secret_key
                                )
                                res = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym))
                                if res and sym in res:
                                    price = float(res[sym].price)
                        except Exception:
                            pass
                        
                        # Database fallback if Alpaca Market Data Client fails
                        if price <= 0 and self._ledger:
                            try:
                                with self._ledger._lock:
                                    row = self._ledger._conn.execute(
                                        "SELECT features_json FROM signals WHERE symbol = ? ORDER BY ts DESC LIMIT 1;",
                                        (sym,)
                                    ).fetchone()
                                    if row:
                                        import json
                                        feat = json.loads(row[0])
                                        price = float(feat.get("close") or 0.0)
                            except Exception:
                                pass
                    total_notional += qty * price
            return total_notional
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.warning(f"Error calculating pending orders notional: {e}")
            return 0.0

    def close_all_positions(self, cancel_orders: bool = True) -> None:
        """
        Forcefully close all open positions and optionally cancel all orders for this bot.
        This is used for daily flattening or emergency risk shutdown.
        """
        from alpaca_day_bot.aetheris_bot import log
        log.info(f"Initiating close_all_positions (daily flattening / risk sweep) for prefix '{self._prefix}'...")
        
        if cancel_orders:
            try:
                log.info("Canceling our open orders...")
                # Cancel only orders that have our prefix
                req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
                orders = self._tc.get_orders(req) or []
                for o in orders:
                    client_oid = getattr(o, "client_order_id", None)
                    if client_oid and client_oid.startswith(self._prefix):
                        log.info(f"Canceling order {o.id} for {o.symbol} (client_order_id: {client_oid})")
                        try:
                            self._tc.cancel_order_by_id(o.id)
                        except Exception as e:
                            log.error(f"Error canceling order {o.id}: {e}")
            except Exception as ce:
                log.error(f"Error during itemized order cancellation: {ce}")

        try:
            positions = self.get_all_positions()
            log.info(f"Retrieved {len(positions)} owned positions for close.")
            for p in positions:
                sym = p.symbol
                log.info(f"Closing position for {sym} ({p.qty} shares)...")
                try:
                    self.close_position(sym)
                except Exception as e:
                    log.error(f"Failed to close position for {sym}: {e}")
        except Exception as e:
            log.error(f"Error during close_all_positions: {e}")

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
        positions = self.get_all_positions()
        exposure = 0.0
        for p in positions:
            try:
                exposure += abs(float(p.market_value))
            except Exception:
                continue
        # Include pending open entry orders to prevent budget cap breaches
        exposure += self.pending_orders_notional_usd()
        return exposure

    def open_positions_count(self) -> int:
        return len(self.get_all_positions())

    def open_position_symbols(self) -> list[str]:
        out: list[str] = []
        try:
            for p in self.get_all_positions():
                sym = str(getattr(p, "symbol", "") or "").strip().upper()
                if sym:
                    out.append(sym)
        except Exception:
            return []
        return out

    def get_all_positions(self):
        """Return current open positions from Alpaca that belong to this bot."""
        try:
            positions = self._tc.get_all_positions()
            owned_symbols = self.get_owned_symbols()
            return [p for p in positions if str(getattr(p, "symbol", "")).strip().upper() in owned_symbols]
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.error(f"Error fetching positions: {e}")
            return []

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
            sym = (symbol or "").strip().upper()
            if sym not in self.get_owned_symbols():
                return False
            _ = self._tc.get_open_position(sym)
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

        client_order_id = self._gen_client_order_id()

        req = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
            client_order_id=client_order_id,
            extended_hours=False,
        )

        # Record intent (pre-submit) if ledger available
        try:
            if getattr(self, "_ledger", None) is not None:
                try:
                    from datetime import datetime, timezone
                    self._ledger.record_order_intent(
                        ts=datetime.now(tz=timezone.utc),
                        symbol=symbol,
                        side="buy",
                        notional_usd=0.0,
                        stop_price=float(stop_price),
                        take_profit_price=float(take_profit_price),
                        client_order_id=client_order_id,
                        alpaca_order_id=None,
                        submitted=False,
                        reason="pre_submit_bracket_buy",
                        extra={"method": "submit_bracket_buy"},

                    )
                except Exception:
                    pass
        except Exception:
            pass

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
        client_order_id = self._gen_client_order_id()
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
        client_order_id = self._gen_client_order_id()
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
        client_order_id = self._gen_client_order_id()
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

        client_order_id = self._gen_client_order_id()
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
        # Record pre-submit intent
        try:
            if getattr(self, "_ledger", None) is not None:
                from datetime import datetime, timezone
                try:
                    self._ledger.record_order_intent(
                        ts=datetime.now(tz=timezone.utc),
                        symbol=symbol,
                        side="buy",
                        notional_usd=0.0,
                        stop_price=float(stop_price),
                        take_profit_price=float(take_profit_price),
                        client_order_id=client_order_id,
                        alpaca_order_id=None,
                        submitted=False,
                        reason="pre_submit_bracket_buy_limit",
                        extra={"method": "submit_bracket_buy_limit"},
                    )
                except Exception:
                    pass
        except Exception:
            pass
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

        client_order_id = self._gen_client_order_id()

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

        # Record pre-submit intent
        try:
            if getattr(self, "_ledger", None) is not None:
                from datetime import datetime, timezone
                try:
                    self._ledger.record_order_intent(
                        ts=datetime.now(tz=timezone.utc),
                        symbol=symbol,
                        side="sell",
                        notional_usd=0.0,
                        stop_price=float(stop_price),
                        take_profit_price=float(take_profit_price),
                        client_order_id=client_order_id,
                        alpaca_order_id=None,
                        submitted=False,
                        reason="pre_submit_bracket_short",
                        extra={"method": "submit_bracket_short"},
                    )
                except Exception:
                    pass
        except Exception:
            pass

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
        client_order_id = self._gen_client_order_id()
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
        client_order_id = self._gen_client_order_id()
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

        client_order_id = self._gen_client_order_id()
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
        # Record pre-submit intent
        try:
            if getattr(self, "_ledger", None) is not None:
                from datetime import datetime, timezone
                try:
                    self._ledger.record_order_intent(
                        ts=datetime.now(tz=timezone.utc),
                        symbol=symbol,
                        side="sell",
                        notional_usd=0.0,
                        stop_price=float(stop_price),
                        take_profit_price=float(take_profit_price),
                        client_order_id=client_order_id,
                        alpaca_order_id=None,
                        submitted=False,
                        reason="pre_submit_bracket_short_limit",
                        extra={"method": "submit_bracket_short_limit"},
                    )
                except Exception:
                    pass
        except Exception:
            pass
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

        client_order_id = self._gen_client_order_id()
        req = LimitOrderRequest(
            symbol=sym,
            qty=int(qty),
            side=(OrderSide.BUY if side_u == "buy" else OrderSide.SELL),
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OCO,
            limit_price=round(float(take_profit_price), 2),
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

    def close_position(self, symbol: str) -> ExecutionResult:
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


    def replace_stop_loss(self, order_id: str, new_stop_price: float) -> bool:
        """Replace an existing stop loss order with a new stop price."""
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            req = ReplaceOrderRequest(stop_price=round(float(new_stop_price), 2))
            self._tc.replace_order_by_id(order_id, req)
            return True
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.error(f"Error replacing stop loss {order_id}: {e}")
            return False

    def update_stop_loss_for_symbol(self, symbol: str, new_stop_price: float) -> bool:
        """Find the open stop loss order for a symbol and update its price."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True)
            orders = self._tc.get_orders(req) or []
            # Find the stop loss leg (usually has side and type stop)
            for o in orders:
                # Alpaca bracket legs are often nested or have a parent_id
                # We look for 'stop' or 'stop_limit' orders on the opposite side of the position
                if str(o.type).split(".")[-1].lower() in ("stop", "stop_limit"):
                    return self.replace_stop_loss(str(o.id), new_stop_price)
            return False
        except Exception as e:
            from alpaca_day_bot.aetheris_bot import log
            log.error(f"Error updating stop loss for {symbol}: {e}")
            return False

    def now_utc(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def _record_fill_intent(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        stop_price: float,
        take_profit_price: float,
        client_order_id: str,
        oid: str,
        live_order: Any
    ):
        try:
            if getattr(self, "_ledger", None) is not None:
                from datetime import datetime, timezone
                self._ledger.record_order_intent(
                    ts=datetime.now(tz=timezone.utc),
                    symbol=symbol,
                    side="buy" if side == OrderSide.BUY else "sell",
                    notional_usd=qty * limit_price,
                    stop_price=float(stop_price),
                    take_profit_price=float(take_profit_price),
                    client_order_id=client_order_id,
                    alpaca_order_id=oid,
                    submitted=True,
                    reason="bracket_limit_chase_filled" if stop_price > 0 else "simple_limit_chase_filled",
                    extra={
                        "method": "submit_bracket_limit_chase" if stop_price > 0 else "submit_simple_limit_chase",
                        "entry_price": float(getattr(live_order, "filled_avg_price", None) or limit_price)
                    },
                )
        except Exception:
            pass

    def submit_bracket_limit_chase(

        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        stop_price: float,
        level_price: float = 0.0,
        take_profit_price: float = 0.0,
        chase_seconds: int = 15,
    ) -> ExecutionResult:
        """
        Submits a bracket limit order and waits up to `chase_seconds` for a fill.
        """
        import time
        client_order_id = self._gen_client_order_id(suffix="chase-")
        
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            limit_price=round(float(limit_price), 2),
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_price), 2)),
            client_order_id=client_order_id,
            extended_hours=False,
        )
        
        # Record intent (pre-submit) if ledger available
        try:
            if getattr(self, "_ledger", None) is not None:
                from datetime import datetime, timezone
                from alpaca.trading.enums import OrderSide
                self._ledger.record_order_intent(
                    ts=datetime.now(tz=timezone.utc),
                    symbol=symbol,
                    side="buy" if side == OrderSide.BUY else "sell",
                    notional_usd=0.0,
                    stop_price=float(stop_price),
                    take_profit_price=float(take_profit_price),
                    client_order_id=client_order_id,
                    alpaca_order_id=None,
                    submitted=False,
                    reason="pre_submit_bracket_limit_chase",
                    extra={"method": "submit_bracket_limit_chase"},
                )
        except Exception:
            pass
            
        try:
            order = self._tc.submit_order(order_data=req)
            oid = str(getattr(order, "id", None))
        except Exception as e:
            return ExecutionResult(False, f"chase_submit_err:{e}")

        # CHASE LOOP: Wait for fill
        start_t = time.time()
        while time.time() - start_t < chase_seconds:
            try:
                live_order = self._tc.get_order_by_id(oid)
                if live_order.status == "filled":
                    from alpaca_day_bot.services.telemetry import CHASED_ORDERS
                    CHASED_ORDERS.inc()
                    self._record_fill_intent(
                        symbol, side, qty, limit_price, stop_price, take_profit_price, client_order_id, oid, live_order
                    )
                    return ExecutionResult(True, "chase_filled", alpaca_order_id=oid)

                if live_order.status in ("canceled", "expired", "rejected"):
                    from alpaca_day_bot.services.telemetry import CANCELLED_ORDERS
                    CANCELLED_ORDERS.inc()
                    return ExecutionResult(False, f"chase_order_{live_order.status}", alpaca_order_id=oid)
            except Exception:
                pass
            time.sleep(1.0)
            
        # Double check one last time before canceling to avoid race conditions!
        try:
            live_order = self._tc.get_order_by_id(oid)
            if live_order.status == "filled":
                from alpaca_day_bot.services.telemetry import CHASED_ORDERS
                CHASED_ORDERS.inc()
                self._record_fill_intent(
                    symbol, side, qty, limit_price, stop_price, take_profit_price, client_order_id, oid, live_order
                )
                return ExecutionResult(True, "chase_filled", alpaca_order_id=oid)

            if live_order.status in ("canceled", "expired", "rejected"):
                from alpaca_day_bot.services.telemetry import CANCELLED_ORDERS
                CANCELLED_ORDERS.inc()
                return ExecutionResult(False, f"chase_order_{live_order.status}", alpaca_order_id=oid)
        except Exception:
            pass

        # TIMEOUT: Cancel and return
        try:
            self._tc.cancel_order_by_id(oid)
            from alpaca_day_bot.services.telemetry import CANCELLED_ORDERS
            CANCELLED_ORDERS.inc()
        except Exception:
            pass
        return ExecutionResult(False, "chase_timeout_canceled", alpaca_order_id=oid)

    def submit_simple_limit_chase(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        chase_seconds: int = 15,
    ) -> ExecutionResult:
        """
        Submits a simple limit order (no bracket) for extended hours trading.
        """
        import time
        import uuid
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import TimeInForce

        client_order_id = self._gen_client_order_id(suffix="simple-chase-")
        
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(float(limit_price), 2),
            client_order_id=client_order_id,
            extended_hours=True,
        )
        
        try:
            order = self._tc.submit_order(order_data=req)
            oid = str(getattr(order, "id", None))
        except Exception as e:
            return ExecutionResult(False, f"simple_chase_submit_err:{e}")

        # CHASE LOOP
        start_t = time.time()
        while time.time() - start_t < chase_seconds:
            try:
                live_order = self._tc.get_order_by_id(oid)
                if live_order.status == "filled":
                    from alpaca_day_bot.services.telemetry import CHASED_ORDERS
                    CHASED_ORDERS.inc()
                    self._record_fill_intent(
                        symbol, side, qty, limit_price, 0.0, 0.0, client_order_id, oid, live_order
                    )
                    return ExecutionResult(True, "simple_chase_filled", alpaca_order_id=oid)

                if live_order.status in ("canceled", "expired", "rejected"):
                    from alpaca_day_bot.services.telemetry import CANCELLED_ORDERS
                    CANCELLED_ORDERS.inc()
                    return ExecutionResult(False, f"simple_chase_{live_order.status}", alpaca_order_id=oid)
            except Exception:
                pass
            time.sleep(1.0)
            
        # Double check one last time before canceling to avoid race conditions!
        try:
            live_order = self._tc.get_order_by_id(oid)
            if live_order.status == "filled":
                from alpaca_day_bot.services.telemetry import CHASED_ORDERS
                CHASED_ORDERS.inc()
                self._record_fill_intent(
                    symbol, side, qty, limit_price, 0.0, 0.0, client_order_id, oid, live_order
                )
                return ExecutionResult(True, "simple_chase_filled", alpaca_order_id=oid)

            if live_order.status in ("canceled", "expired", "rejected"):
                from alpaca_day_bot.services.telemetry import CANCELLED_ORDERS
                CANCELLED_ORDERS.inc()
                return ExecutionResult(False, f"simple_chase_{live_order.status}", alpaca_order_id=oid)
        except Exception:
            pass

        try:
            self._tc.cancel_order_by_id(oid)
            from alpaca_day_bot.services.telemetry import CANCELLED_ORDERS
            CANCELLED_ORDERS.inc()
        except Exception:
            pass
        return ExecutionResult(False, "simple_chase_timeout", alpaca_order_id=oid)
