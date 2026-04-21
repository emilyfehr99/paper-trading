from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Fill:
    ts: datetime
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    px: float
    order_id: str | None


@dataclass(frozen=True)
class RoundTrip:
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    qty: float
    entry_px: float
    exit_px: float
    direction: str  # "long" | "short"
    pnl: float


@dataclass(frozen=True)
class TradeStats:
    trades: int
    wins: int
    losses: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    expectancy: float | None
    total_pnl: float


def _parse_dt(s: str) -> datetime:
    # Stored via isoformat; tolerate Z.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def _rows_to_fills(rows: list[tuple[str, str, str]]) -> list[Fill]:
    fills: list[Fill] = []
    for ts_s, event, raw_json in rows:
        try:
            payload = json.loads(raw_json)
        except Exception:
            continue
        order = (payload.get("payload") or {}).get("order") if isinstance(payload, dict) else None
        if not isinstance(order, dict):
            continue
        sym = str(order.get("symbol") or "").strip().upper()
        side = str(order.get("side") or "").strip().lower()
        if not sym or side not in ("buy", "sell"):
            continue
        qty = order.get("filled_qty")
        px = order.get("filled_avg_price")
        try:
            qty_f = float(qty)
            px_f = float(px)
        except Exception:
            continue
        if qty_f <= 0 or px_f <= 0:
            continue
        fills.append(
            Fill(
                ts=_parse_dt(ts_s),
                symbol=sym,
                side=side,
                qty=qty_f,
                px=px_f,
                order_id=(None if order.get("id") is None else str(order.get("id"))),
            )
        )
    fills.sort(key=lambda f: f.ts)
    return fills


def reconstruct_round_trips(fills: list[Fill]) -> list[RoundTrip]:
    """
    FIFO lot matching:
    - Entry BUY opens long lots; SELL closes long lots (realized PnL = (sell - buy)*qty)
    - Entry SELL opens short lots; BUY closes short lots (realized PnL = (sell - buy)*qty)
    """
    # symbol -> list of open lots: {side, ts, qty, px}
    open_lots: dict[str, list[dict[str, Any]]] = {}
    out: list[RoundTrip] = []

    for f in fills:
        lots = open_lots.setdefault(f.symbol, [])

        def close_against(opposite_side: str) -> float:
            remaining = f.qty
            while remaining > 1e-12 and lots:
                lot = lots[0]
                if lot["side"] != opposite_side:
                    break
                take = min(float(lot["qty"]), remaining)
                lot["qty"] = float(lot["qty"]) - take
                remaining -= take

                if opposite_side == "buy":
                    # long lot: buy then sell
                    entry_px = float(lot["px"])
                    exit_px = float(f.px)
                    pnl = (exit_px - entry_px) * take
                    out.append(
                        RoundTrip(
                            symbol=f.symbol,
                            entry_ts=lot["ts"],
                            exit_ts=f.ts,
                            qty=take,
                            entry_px=entry_px,
                            exit_px=exit_px,
                            direction="long",
                            pnl=pnl,
                        )
                    )
                else:
                    # short lot: sell then buy
                    entry_px = float(lot["px"])
                    exit_px = float(f.px)
                    pnl = (entry_px - exit_px) * take
                    out.append(
                        RoundTrip(
                            symbol=f.symbol,
                            entry_ts=lot["ts"],
                            exit_ts=f.ts,
                            qty=take,
                            entry_px=entry_px,
                            exit_px=exit_px,
                            direction="short",
                            pnl=pnl,
                        )
                    )

                if float(lot["qty"]) <= 1e-12:
                    lots.pop(0)
            return remaining

        if f.side == "sell":
            # first try to close longs (open buy lots)
            rem = close_against("buy")
            if rem > 1e-12:
                lots.append({"side": "sell", "ts": f.ts, "qty": rem, "px": f.px})
        else:
            # buy: first try to close shorts (open sell lots)
            rem = close_against("sell")
            if rem > 1e-12:
                lots.append({"side": "buy", "ts": f.ts, "qty": rem, "px": f.px})

    return out


def trade_stats(round_trips: list[RoundTrip]) -> TradeStats:
    if not round_trips:
        return TradeStats(0, 0, 0, None, None, None, None, None, 0.0)
    pnls = [t.pnl for t in round_trips]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    win_n = len(wins)
    loss_n = len(losses)
    total = float(sum(pnls))
    avg_win = (sum(wins) / win_n) if win_n else None
    avg_loss = (sum(losses) / loss_n) if loss_n else None
    pf = (sum(wins) / abs(sum(losses))) if (wins and losses and abs(sum(losses)) > 1e-12) else None
    exp = total / n
    return TradeStats(
        trades=n,
        wins=win_n,
        losses=loss_n,
        win_rate=(win_n / n) if n else None,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=pf,
        expectancy=exp,
        total_pnl=total,
    )


def realized_trade_stats_for_day(db_path: str, start_iso: str, end_iso: str) -> TradeStats:
    """
    Pull fill events from trade_updates and compute realized trade stats.
    Uses trade_updates.raw_json which stores TradeUpdateEvent(asdict).
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT ts, event, raw_json
        FROM trade_updates
        WHERE ts >= ? AND ts <= ?
          AND event IN ('fill','partial_fill','filled')
        ORDER BY ts ASC
        """,
        (start_iso, end_iso),
    ).fetchall()
    conn.close()
    fills = _rows_to_fills(rows)
    rts = reconstruct_round_trips(fills)
    return trade_stats(rts)

