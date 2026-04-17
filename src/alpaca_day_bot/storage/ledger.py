from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca_day_bot.trading.updates import TradeUpdateEvent


class Ledger:
    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self._ensure_audit_file()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trade_updates (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              event TEXT NOT NULL,
              symbol TEXT,
              order_id TEXT,
              client_order_id TEXT,
              filled_qty REAL,
              filled_avg_price REAL,
              raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS order_intents (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              symbol TEXT NOT NULL,
              side TEXT,
              notional_usd REAL,
              stop_price REAL,
              take_profit_price REAL,
              client_order_id TEXT,
              alpaca_order_id TEXT,
              submitted INTEGER NOT NULL,
              reason TEXT,
              raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              equity REAL NOT NULL,
              gross_exposure REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signals (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              symbol TEXT NOT NULL,
              action TEXT NOT NULL,
              reason TEXT NOT NULL,
              features_json TEXT
            );

            CREATE TABLE IF NOT EXISTS forward_return_labels (
              signal_id INTEGER PRIMARY KEY,
              evaluated_ts TEXT NOT NULL,
              price_at_label REAL NOT NULL,
              entry_close REAL NOT NULL,
              return_pct REAL NOT NULL,
              horizon_minutes REAL NOT NULL,
              FOREIGN KEY (signal_id) REFERENCES signals(id)
            );
            """
        )
        self._conn.commit()

    def _ensure_audit_file(self) -> None:
        """Create transactions.jsonl on startup so `alpaca-watch-trades` can tail immediately."""
        p = self._path.parent / "transactions.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()

    def record_trade_update(self, evt: TradeUpdateEvent) -> None:
        payload = json.dumps(asdict(evt), default=str)
        self._conn.execute(
            """
            INSERT INTO trade_updates (ts, event, symbol, order_id, client_order_id, filled_qty, filled_avg_price, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evt.ts.isoformat(),
                evt.event,
                evt.symbol,
                evt.order_id,
                evt.client_order_id,
                evt.filled_qty,
                evt.filled_avg_price,
                payload,
            ),
        )
        self._conn.commit()
        self.append_audit_line(
            {
                "kind": "trade_update",
                "ts": evt.ts.isoformat(),
                "event": evt.event,
                "symbol": evt.symbol,
                "order_id": evt.order_id,
                "client_order_id": evt.client_order_id,
                "payload": getattr(evt, "payload", {}),
            }
        )

    def record_order_intent(
        self,
        *,
        ts: datetime,
        symbol: str,
        side: str,
        notional_usd: float,
        stop_price: float,
        take_profit_price: float,
        client_order_id: str | None,
        alpaca_order_id: str | None,
        submitted: bool,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "ts": ts.isoformat(),
            "symbol": symbol,
            "side": side,
            "notional_usd": notional_usd,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "client_order_id": client_order_id,
            "alpaca_order_id": alpaca_order_id,
            "submitted": submitted,
            "reason": reason,
            "extra": extra or {},
        }
        raw = json.dumps(row, default=str)
        self._conn.execute(
            """
            INSERT INTO order_intents (
              ts, symbol, side, notional_usd, stop_price, take_profit_price,
              client_order_id, alpaca_order_id, submitted, reason, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                symbol,
                side,
                float(notional_usd),
                float(stop_price),
                float(take_profit_price),
                client_order_id,
                alpaca_order_id,
                1 if submitted else 0,
                reason,
                raw,
            ),
        )
        self._conn.commit()
        self.append_audit_line({"kind": "order_intent", **row})

    def append_audit_line(self, obj: dict[str, Any]) -> None:
        """Append-only JSONL for every transaction-related record (easy grep / backup)."""
        audit_path = self._path.parent / "transactions.jsonl"
        line = json.dumps(obj, default=str) + "\n"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(line)

    def record_equity_snapshot(self, ts: datetime, equity: float, gross_exposure: float) -> None:
        self._conn.execute(
            """
            INSERT INTO equity_snapshots (ts, equity, gross_exposure)
            VALUES (?, ?, ?)
            """,
            (ts.isoformat(), float(equity), float(gross_exposure)),
        )
        self._conn.commit()

    def submitted_buy_stats_for_trading_date(
        self, market_day: date, tz: ZoneInfo
    ) -> dict[str, Any]:
        """Submitted BUY intents on `market_day` in `tz` (for risk rehydration between CI runs)."""
        start = datetime.combine(market_day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        start_s = start.isoformat()
        end_s = end.isoformat()
        cur = self._conn.execute(
            """
            SELECT ts, symbol FROM order_intents
            WHERE submitted = 1 AND LOWER(side) = 'buy' AND ts >= ? AND ts < ?
            ORDER BY ts
            """,
            (start_s, end_s),
        )
        rows = cur.fetchall()
        last_by_symbol: dict[str, datetime] = {}
        for ts_str, sym in rows:
            if not sym:
                continue
            ts_p = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts_p.tzinfo is None:
                ts_p = ts_p.replace(tzinfo=timezone.utc)
            last_by_symbol[str(sym)] = ts_p
        return {"count": len(rows), "last_by_symbol": last_by_symbol}

    def record_signal(
        self,
        *,
        ts: datetime,
        symbol: str,
        action: str,
        reason: str,
        features: dict[str, Any] | None = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO signals (ts, symbol, action, reason, features_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                symbol,
                action,
                reason,
                (None if features is None else json.dumps(features, default=str)),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_unlabeled_buy_signal_rows(
        self,
        *,
        market_day: date,
        tz: ZoneInfo,
        now_utc: datetime,
        min_age_minutes: float,
    ) -> list[tuple[int, str, str, str]]:
        """Returns (signal_id, ts_iso, symbol, features_json) for BUY rows needing a label."""
        start = datetime.combine(market_day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        start_s, end_s = start.isoformat(), end.isoformat()
        cur = self._conn.execute(
            """
            SELECT s.id, s.ts, s.symbol, s.features_json
            FROM signals s
            LEFT JOIN forward_return_labels f ON f.signal_id = s.id
            WHERE s.action = 'BUY'
              AND s.ts >= ? AND s.ts < ?
              AND f.signal_id IS NULL
            """,
            (start_s, end_s),
        )
        rows_out: list[tuple[int, str, str, str]] = []
        for sid, ts_s, sym, feat in cur.fetchall():
            if not feat:
                continue
            try:
                ts_p = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
            except Exception:
                continue
            if ts_p.tzinfo is None:
                ts_p = ts_p.replace(tzinfo=timezone.utc)
            age_m = (now_utc - ts_p).total_seconds() / 60.0
            if age_m < float(min_age_minutes):
                continue
            rows_out.append((int(sid), ts_s, str(sym), str(feat)))
        return rows_out

    def record_forward_return_label(
        self,
        *,
        signal_id: int,
        evaluated_ts: datetime,
        price_at_label: float,
        entry_close: float,
        return_pct: float,
        horizon_minutes: float,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO forward_return_labels
              (signal_id, evaluated_ts, price_at_label, entry_close, return_pct, horizon_minutes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(signal_id),
                evaluated_ts.isoformat(),
                float(price_at_label),
                float(entry_close),
                float(return_pct),
                float(horizon_minutes),
            ),
        )
        self._conn.commit()

