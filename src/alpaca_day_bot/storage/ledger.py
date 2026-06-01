from __future__ import annotations

import json
import sqlite3
import threading
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
        # Trading updates arrive on a background thread; allow cross-thread use and serialize with a lock.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self._ensure_audit_file()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
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
              features_json TEXT,
              explainability_json TEXT,
              context_json TEXT
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

            CREATE TABLE IF NOT EXISTS triple_barrier_labels (
              signal_id INTEGER PRIMARY KEY,
              evaluated_ts TEXT NOT NULL,
              entry_close REAL NOT NULL,
              tp_price REAL NOT NULL,
              sl_price REAL NOT NULL,
              outcome TEXT NOT NULL, -- tp | sl | timeout
              realized_return_pct REAL NOT NULL,
              horizon_minutes REAL NOT NULL,
              failure_analysis_json TEXT,
              context_json TEXT,
              FOREIGN KEY (signal_id) REFERENCES signals(id)
            );

            CREATE TABLE IF NOT EXISTS virtual_option_trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_open TEXT NOT NULL,
              ts_close TEXT,
              symbol TEXT NOT NULL,
              side TEXT NOT NULL, -- call | put
              notional_usd REAL NOT NULL,
              leverage REAL NOT NULL,
              underlying_entry REAL NOT NULL,
              underlying_exit REAL,
              pnl_usd REAL,
              meta_json TEXT
            );

            CREATE TABLE IF NOT EXISTS executed_trade_reviews (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_close TEXT NOT NULL,
              symbol TEXT NOT NULL,
              realized_pnl_usd REAL NOT NULL,
              qty_closed REAL NOT NULL,
              entry_ts TEXT,
              exit_ts TEXT,
              entry_avg_price REAL,
              exit_price REAL,
              signal_id INTEGER,
              signal_reason TEXT,
              tags_json TEXT,
              features_json TEXT,
              context_json TEXT,
              raw_json TEXT
            );
            """
            )
            self._conn.commit()

            # Dynamic migrations to handle existing sqlite databases safely
            # Ensure signals context_json and explainability_json exist
            cursor = self._conn.execute("PRAGMA table_info(signals)")
            cols = [row[1] for row in cursor.fetchall()]
            if "context_json" not in cols:
                self._conn.execute("ALTER TABLE signals ADD COLUMN context_json TEXT")
                self._conn.commit()
            if "explainability_json" not in cols:
                self._conn.execute("ALTER TABLE signals ADD COLUMN explainability_json TEXT")
                self._conn.commit()

            # Ensure triple_barrier_labels context_json and failure_analysis_json exist
            cursor = self._conn.execute("PRAGMA table_info(triple_barrier_labels)")
            cols = [row[1] for row in cursor.fetchall()]
            if "context_json" not in cols:
                self._conn.execute("ALTER TABLE triple_barrier_labels ADD COLUMN context_json TEXT")
                self._conn.commit()
            if "failure_analysis_json" not in cols:
                self._conn.execute("ALTER TABLE triple_barrier_labels ADD COLUMN failure_analysis_json TEXT")
                self._conn.commit()

            # Ensure executed_trade_reviews context_json exists
            cursor = self._conn.execute("PRAGMA table_info(executed_trade_reviews)")
            cols = [row[1] for row in cursor.fetchall()]
            if "context_json" not in cols:
                self._conn.execute("ALTER TABLE executed_trade_reviews ADD COLUMN context_json TEXT")
                self._conn.commit()

            # Ensure trade_updates slippage_pct exists
            cursor = self._conn.execute("PRAGMA table_info(trade_updates)")
            cols = [row[1] for row in cursor.fetchall()]
            if "slippage_pct" not in cols:
                self._conn.execute("ALTER TABLE trade_updates ADD COLUMN slippage_pct REAL")
                self._conn.commit()

    def _ensure_audit_file(self) -> None:
        """Create transactions.jsonl on startup so `alpaca-watch-trades` can tail immediately."""
        p = self._path.parent / "transactions.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()

    def record_trade_update(self, evt: TradeUpdateEvent) -> None:
        payload = json.dumps(asdict(evt), default=str)
        slippage_pct = None
        
        # Calculate Slippage on 'fill' events for entry orders
        if evt.event == "fill" and evt.filled_avg_price and evt.filled_avg_price > 0:
            with self._lock:
                # Find the original intent to get the signaled price
                orig = self._conn.execute(
                    "SELECT side, stop_price FROM order_intents WHERE alpaca_order_id = ? OR client_order_id = ?",
                    (evt.order_id, evt.client_order_id)
                ).fetchone()
                
                if orig:
                    side, signaled_price = orig
                    if signaled_price and signaled_price > 0:
                        # Buy slippage: (Fill - Signal) / Signal (positive is bad)
                        # Sell slippage: (Signal - Fill) / Signal (positive is bad)
                        if side.lower() == "buy":
                            slippage_pct = (evt.filled_avg_price - signaled_price) / signaled_price
                        else:
                            slippage_pct = (signaled_price - evt.filled_avg_price) / signaled_price

        with self._lock:
            self._conn.execute(
                """
            INSERT OR REPLACE INTO trade_updates (ts, event, symbol, order_id, client_order_id, filled_qty, filled_avg_price, slippage_pct, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evt.ts.isoformat(),
                    evt.event,
                    evt.symbol,
                    evt.order_id,
                    evt.client_order_id,
                    evt.filled_qty,
                    evt.filled_avg_price,
                    slippage_pct,
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
        with self._lock:
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
        with self._lock:
            self._conn.execute(
                """
            INSERT INTO equity_snapshots (ts, equity, gross_exposure)
            VALUES (?, ?, ?)
                """,
                (ts.isoformat(), float(equity), float(gross_exposure)),
            )
            self._conn.commit()

    def submitted_entry_stats_for_trading_date(
        self, market_day: date, tz: ZoneInfo
    ) -> dict[str, Any]:
        """Submitted entry intents on `market_day` in `tz` (with fallback to actual fills)."""
        start = datetime.combine(market_day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        start_s = start.isoformat()
        end_s = end.isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
            SELECT ts, symbol FROM order_intents
            WHERE submitted = 1 AND LOWER(side) IN ('buy','sell') AND ts >= ? AND ts < ?
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
            last_by_symbol[str(sym).strip().upper()] = ts_p

        # Fallback to trade_updates to catch fills that might not have a submitted=1 intent row
        try:
            with self._lock:
                cur_fills = self._conn.execute(
                    """
                SELECT ts, symbol, raw_json FROM trade_updates
                WHERE event IN ('fill', 'filled') AND ts >= ? AND ts < ?
                    """,
                    (start_s, end_s),
                )
                fill_rows = cur_fills.fetchall()
            for f_ts, f_sym, f_raw in fill_rows:
                if not f_sym:
                    continue
                try:
                    obj = json.loads(f_raw) if isinstance(f_raw, str) else f_raw
                    side = obj.get("payload", {}).get("order", {}).get("side") or obj.get("order", {}).get("side")
                    if side and side.lower() == "buy":
                        f_ts_str = f_ts.replace("Z", "+00:00")
                        if " " in f_ts_str and "+" not in f_ts_str:
                            f_ts_p = datetime.strptime(f_ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        else:
                            f_ts_p = datetime.fromisoformat(f_ts_str)
                        if f_ts_p.tzinfo is None:
                            f_ts_p = f_ts_p.replace(tzinfo=timezone.utc)
                        
                        sym_str = str(f_sym).strip().upper()
                        if sym_str not in last_by_symbol or f_ts_p > last_by_symbol[sym_str]:
                            last_by_symbol[sym_str] = f_ts_p
                except Exception:
                    pass
        except Exception:
            pass

        return {"count": len(last_by_symbol), "last_by_symbol": last_by_symbol}

    def last_submitted_entry_intents_for_trading_date(
        self, market_day: date, tz: ZoneInfo
    ) -> dict[str, dict[str, Any]]:
        """
        Return latest submitted entry intent per symbol for the day, including raw_json.
        With fallback to trade_updates fills if missing in order_intents.
        """
        start = datetime.combine(market_day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        start_s = start.isoformat()
        end_s = end.isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
            SELECT ts, symbol, side, raw_json FROM order_intents
            WHERE submitted = 1 AND LOWER(side) IN ('buy','sell') AND ts >= ? AND ts < ?
            ORDER BY ts
                """,
                (start_s, end_s),
            )
            rows = cur.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for ts_str, sym, side, raw in rows:
            if not sym:
                continue
            try:
                ts_p = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if ts_p.tzinfo is None:
                ts_p = ts_p.replace(tzinfo=timezone.utc)
            extra = {}
            try:
                j = json.loads(raw) if raw else {}
                extra = (j.get("extra") or {}) if isinstance(j, dict) else {}
            except Exception:
                extra = {}
            out[str(sym).strip().upper()] = {"ts": ts_p, "side": side, "extra": extra}

        # Fallback to trade_updates to catch fills that might not have a submitted=1 intent row
        try:
            with self._lock:
                cur_fills = self._conn.execute(
                    """
                SELECT symbol, ts, raw_json FROM trade_updates
                WHERE event IN ('fill', 'filled') AND ts >= ? AND ts < ?
                ORDER BY ts
                    """,
                    (start_s, end_s),
                )
                fill_rows = cur_fills.fetchall()
            for f_sym, f_ts, f_raw in fill_rows:
                if not f_sym:
                    continue
                try:
                    obj = json.loads(f_raw) if isinstance(f_raw, str) else f_raw
                    side = obj.get("payload", {}).get("order", {}).get("side") or obj.get("order", {}).get("side")
                    if side and side.lower() == "buy":
                        sym_str = str(f_sym).strip().upper()
                        f_ts_str = f_ts.replace("Z", "+00:00")
                        if " " in f_ts_str and "+" not in f_ts_str:
                            f_ts_p = datetime.strptime(f_ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        else:
                            f_ts_p = datetime.fromisoformat(f_ts_str)
                        if f_ts_p.tzinfo is None:
                            f_ts_p = f_ts_p.replace(tzinfo=timezone.utc)
                        
                        if sym_str not in out or f_ts_p > out[sym_str]["ts"]:
                            out[sym_str] = {
                                "ts": f_ts_p,
                                "side": side,
                                "extra": {"target_hold_minutes": 180.0}
                            }
                except Exception:
                    pass
        except Exception:
            pass

        return out

    def record_signal(
        self,
        *,
        ts: datetime,
        symbol: str,
        action: str,
        reason: str,
        features: dict[str, Any] | None = None,
        explainability: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
            INSERT INTO signals (ts, symbol, action, reason, features_json, explainability_json, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts.isoformat(),
                    symbol,
                    action,
                    reason,
                    (None if features is None else json.dumps(features, default=str)),
                    (None if explainability is None else json.dumps(explainability, default=str)),
                    (None if context is None else json.dumps(context, default=str)),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def open_virtual_option_trade(
        self,
        *,
        ts_open: datetime,
        symbol: str,
        side: str,
        notional_usd: float,
        leverage: float,
        underlying_entry: float,
        meta: dict[str, Any] | None = None,
    ) -> int:
        side_n = (side or "").strip().lower()
        if side_n not in ("call", "put"):
            side_n = "call"
        row = {
            "ts_open": ts_open.isoformat(),
            "symbol": symbol,
            "side": side_n,
            "notional_usd": float(notional_usd),
            "leverage": float(leverage),
            "underlying_entry": float(underlying_entry),
            "meta": meta or {},
        }
        with self._lock:
            cur = self._conn.execute(
                """
            INSERT INTO virtual_option_trades
              (ts_open, ts_close, symbol, side, notional_usd, leverage, underlying_entry, underlying_exit, pnl_usd, meta_json)
            VALUES (?, NULL, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (
                    ts_open.isoformat(),
                    symbol,
                    side_n,
                    float(notional_usd),
                    float(leverage),
                    float(underlying_entry),
                    json.dumps(row, default=str),
                ),
            )
            self._conn.commit()
            tid = int(cur.lastrowid)
        self.append_audit_line({"kind": "virtual_option_open", "id": tid, **row})
        return tid

    def list_open_virtual_option_trades(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
            SELECT id, ts_open, symbol, side, notional_usd, leverage, underlying_entry, meta_json
            FROM virtual_option_trades
            WHERE ts_close IS NULL
            ORDER BY ts_open ASC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for rid, ts_open, sym, side, notional, lev, uentry, meta_json in rows:
            meta = {}
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                meta = {}
            out.append(
                {
                    "id": int(rid),
                    "ts_open": str(ts_open),
                    "symbol": str(sym),
                    "side": str(side),
                    "notional_usd": float(notional),
                    "leverage": float(lev),
                    "underlying_entry": float(uentry),
                    "meta": meta,
                }
            )
        return out

    def close_virtual_option_trade(
        self,
        *,
        trade_id: int,
        ts_close: datetime,
        underlying_exit: float,
        pnl_usd: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
            UPDATE virtual_option_trades
            SET ts_close = ?, underlying_exit = ?, pnl_usd = ?
            WHERE id = ?
                """,
                (ts_close.isoformat(), float(underlying_exit), float(pnl_usd), int(trade_id)),
            )
            self._conn.commit()
        self.append_audit_line(
            {
                "kind": "virtual_option_close",
                "id": int(trade_id),
                "ts_close": ts_close.isoformat(),
                "underlying_exit": float(underlying_exit),
                "pnl_usd": float(pnl_usd),
            }
        )

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
        with self._lock:
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

    def list_unlabeled_signal_rows(
        self,
        *,
        market_day: date,
        tz: ZoneInfo,
        now_utc: datetime,
        min_age_minutes: float,
        actions: tuple[str, ...] = ("BUY", "SHORT"),
    ) -> list[tuple[int, str, str, str, str]]:
        """
        Returns (signal_id, ts_iso, symbol, action, features_json) for rows needing a label.
        """
        start = datetime.combine(market_day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        start_s, end_s = start.isoformat(), end.isoformat()
        actions_u = tuple(str(a).upper() for a in actions)
        with self._lock:
            cur = self._conn.execute(
                """
            SELECT s.id, s.ts, s.symbol, s.action, s.features_json
            FROM signals s
            LEFT JOIN forward_return_labels f ON f.signal_id = s.id
            WHERE s.action IN ({acts})
              AND s.ts >= ? AND s.ts < ?
              AND f.signal_id IS NULL
            """.format(acts=",".join(["?"] * len(actions_u))),
                (*actions_u, start_s, end_s),
            )
            rows = cur.fetchall()
        rows_out: list[tuple[int, str, str, str, str]] = []
        for sid, ts_s, sym, act, feat in rows:
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
            rows_out.append((int(sid), ts_s, str(sym), str(act), str(feat)))
        return rows_out

    def list_unlabeled_signal_rows_for_triple_barrier(
        self,
        *,
        market_day: date,
        tz: ZoneInfo,
        now_utc: datetime,
        min_age_minutes: float,
        actions: tuple[str, ...] = ("BUY", "SHORT"),
    ) -> list[tuple[int, str, str, str, str]]:
        """
        Returns (signal_id, ts_iso, symbol, action, features_json) for rows needing a triple-barrier label.
        Uses triple_barrier_labels presence (not forward_return_labels).
        """
        start = datetime.combine(market_day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        start_s, end_s = start.isoformat(), end.isoformat()
        actions_u = tuple(str(a).upper() for a in actions)
        with self._lock:
            cur = self._conn.execute(
                """
            SELECT s.id, s.ts, s.symbol, s.action, s.features_json
            FROM signals s
            LEFT JOIN triple_barrier_labels tb ON tb.signal_id = s.id
            WHERE s.action IN ({acts})
              AND s.ts >= ? AND s.ts < ?
              AND tb.signal_id IS NULL
            """.format(acts=",".join(["?"] * len(actions_u))),
                (*actions_u, start_s, end_s),
            )
            rows = cur.fetchall()
        rows_out: list[tuple[int, str, str, str, str]] = []
        for sid, ts_s, sym, act, feat in rows:
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
            rows_out.append((int(sid), ts_s, str(sym), str(act), str(feat)))
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
        with self._lock:
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

    def record_triple_barrier_label(
        self,
        *,
        signal_id: int,
        evaluated_ts: datetime,
        entry_close: float,
        tp_price: float,
        sl_price: float,
        outcome: str,
        realized_return_pct: float,
        horizon_minutes: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        out = (outcome or "").strip().lower()
        if out not in ("tp", "sl", "timeout"):
            out = "timeout"
        with self._lock:
            self._conn.execute(
                """
            INSERT OR REPLACE INTO triple_barrier_labels
              (signal_id, evaluated_ts, entry_close, tp_price, sl_price, outcome, realized_return_pct, horizon_minutes, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(signal_id),
                    evaluated_ts.isoformat(),
                    float(entry_close),
                    float(tp_price),
                    float(sl_price),
                    str(out),
                    float(realized_return_pct),
                    float(horizon_minutes),
                    (None if context is None else json.dumps(context, default=str)),
                ),
            )
            self._conn.commit()
    def record_executed_trade_review(
        self,
        *,
        ts_close: datetime,
        symbol: str,
        realized_pnl_usd: float,
        qty_closed: float,
        entry_ts: datetime | None = None,
        exit_ts: datetime | None = None,
        entry_avg_price: float | None = None,
        exit_price: float | None = None,
        signal_id: int | None = None,
        signal_reason: str | None = None,
        tags_json: str | None = None,
        features_json: str | None = None,
        context_json: str | None = None,
        raw_json: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
            INSERT INTO executed_trade_reviews (
              ts_close, symbol, realized_pnl_usd, qty_closed, entry_ts, exit_ts,
              entry_avg_price, exit_price, signal_id, signal_reason, tags_json,
              features_json, context_json, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_close.isoformat(),
                    symbol,
                    float(realized_pnl_usd),
                    float(qty_closed),
                    entry_ts.isoformat() if entry_ts else None,
                    exit_ts.isoformat() if exit_ts else None,
                    float(entry_avg_price) if entry_avg_price else None,
                    float(exit_price) if exit_price else None,
                    int(signal_id) if signal_id else None,
                    signal_reason,
                    tags_json,
                    features_json,
                    context_json,
                    raw_json or "{}",
                ),
            )
            self._conn.commit()
