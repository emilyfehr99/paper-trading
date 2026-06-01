from __future__ import annotations
import logging
import json
from datetime import datetime, timezone
from typing import Any
from alpaca_day_bot.storage.ledger import Ledger
from alpaca_day_bot.data.context import MacroContextCollector
from alpaca_day_bot.services.sentiment_engine import SentimentEngine
from alpaca_day_bot.trading.updates import TradeUpdateEvent

log = logging.getLogger("alpaca_day_bot.reviewer")

class TradeReviewer:
    """
    The 'Understand Why' Engine.
    Records post-mortem diagnostics for every closed trade to feed the Big Big Model.
    """
    def __init__(self, ledger: Ledger, macro: Any, sentiment: Any):
        self.ledger = ledger
        self.macro = macro
        self.sentiment = sentiment

    def handle_update(self, evt: TradeUpdateEvent):
        """
        Main entry point for trade updates. Filters for 'fill' events on EXIT orders.
        """
        # Always record the raw update in the ledger first
        self.ledger.record_trade_update(evt)

        if evt.event != "fill":
            return

        # We only care about fills for EXITS (where PnL is realized)
        order = evt.payload.get("order", {})
        side = order.get("side")
        qty = float(order.get("filled_qty", 0))
        price = float(order.get("filled_avg_price", 0))

        # Check if this is an exit fill
        is_exit = False
        entry_ts = None
        entry_avg_price = None
        entry_cid = None
        realized_pnl = 0.0
        signal_id = None
        signal_reason = None
        features_json = None

        try:
            with self.ledger._lock:
                cursor = self.ledger._conn.cursor()
                # 1. Calculate net quantity BEFORE this fill
                cursor.execute("""
                    SELECT SUM(
                        CASE 
                            WHEN json_extract(raw_json, '$.payload.order.side') = 'buy' THEN filled_qty 
                            WHEN json_extract(raw_json, '$.payload.order.side') = 'sell' THEN -filled_qty 
                            ELSE 0 
                        END
                    )
                    FROM trade_updates
                    WHERE symbol = ? AND ts < ? AND event = 'fill'
                """, (evt.symbol, evt.ts.isoformat()))
                row = cursor.fetchone()
                net_qty = float(row[0]) if row and row[0] is not None else 0.0

                # 2. Check if current order side matches opposing side of position
                if side == "sell" and net_qty > 0.0001:
                    is_exit = True
                    opposing_side = "buy"
                elif side == "buy" and net_qty < -0.0001:
                    is_exit = True
                    opposing_side = "sell"

                if is_exit:
                    # 3. Find the matching entry fill
                    cursor.execute("""
                        SELECT ts, filled_avg_price, client_order_id 
                        FROM trade_updates
                        WHERE symbol = ? AND ts < ? AND event = 'fill' 
                          AND json_extract(raw_json, '$.payload.order.side') = ?
                        ORDER BY ts DESC LIMIT 1
                    """, (evt.symbol, evt.ts.isoformat(), opposing_side))
                    match_row = cursor.fetchone()
                    if match_row:
                        entry_ts_str, entry_avg_price_val, entry_cid_val = match_row
                        entry_ts = datetime.fromisoformat(entry_ts_str)
                        entry_avg_price = float(entry_avg_price_val or 0.0)
                        entry_cid = entry_cid_val

                        # Calculate Realized PnL
                        if side == "sell": # Exit long
                            realized_pnl = qty * (price - entry_avg_price)
                        else: # Exit short (cover)
                            realized_pnl = qty * (entry_avg_price - price)

                        # 4. Find the matching signal
                        cursor.execute("""
                            SELECT id, reason, features_json 
                            FROM signals 
                            WHERE symbol = ? AND ts <= ? 
                            ORDER BY ts DESC LIMIT 1
                        """, (evt.symbol, entry_ts_str))
                        sig_row = cursor.fetchone()
                        if sig_row:
                            signal_id, signal_reason, features_json = sig_row
        except Exception as e:
            log.warning(f"Error calculating realized P&L and matching entry in TradeReviewer: {e}")

        # Only record review if it is indeed a closing trade (exit)
        if is_exit:
            # Capture Exit Context NOW
            context = self.macro.get_macro_context()
            sentiment = self.sentiment.get_sentiment(evt.symbol or "")
            context.update(sentiment)
            
            log.info(f"Post-Mortem Captured for {evt.symbol} Exit: VIX={context.get('vix')}, Sentiment={context.get('sentiment_score')}, PnL=${realized_pnl:+.2f}")

            self.ledger.record_executed_trade_review(
                ts_close=evt.ts,
                symbol=evt.symbol or "UNKNOWN",
                realized_pnl_usd=realized_pnl,
                qty_closed=qty,
                entry_ts=entry_ts,
                exit_ts=evt.ts,
                entry_avg_price=entry_avg_price,
                exit_price=price,
                signal_id=signal_id,
                signal_reason=signal_reason,
                features_json=features_json,
                context_json=json.dumps(context, default=str),
                raw_json=json.dumps(evt.payload, default=str)
            )
