from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class SimLedger:
    """
    Lightweight SQLite sink for simulated/backtest outputs (separate from live paper ledger).

    Purpose:
    - persist per-signal feature snapshots (for training on signal-time features)
    - persist simulated trades (for PnL distribution / labeling / diagnostics)
    """

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sim_signals (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              market TEXT NOT NULL, -- equity | crypto
              symbol TEXT NOT NULL,
              action TEXT NOT NULL,
              reason TEXT NOT NULL,
              features_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sim_signals_ts ON sim_signals(ts);
            CREATE INDEX IF NOT EXISTS idx_sim_signals_symbol_ts ON sim_signals(symbol, ts);

            CREATE TABLE IF NOT EXISTS sim_trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              sim_signal_id INTEGER, -- links to sim_signals.id when available
              market TEXT NOT NULL, -- equity | crypto
              symbol TEXT NOT NULL,
              side TEXT NOT NULL, -- long | short
              entry_ts TEXT NOT NULL,
              exit_ts TEXT NOT NULL,
              entry_price REAL NOT NULL,
              exit_price REAL NOT NULL,
              qty REAL NOT NULL,
              pnl REAL NOT NULL,
              pnl_r REAL,
              risk_r REAL,
              hold_minutes REAL,
              entry_cost_usd REAL,
              exit_cost_usd REAL,
              meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sim_trades_entry_ts ON sim_trades(entry_ts);
            CREATE INDEX IF NOT EXISTS idx_sim_trades_symbol_entry_ts ON sim_trades(symbol, entry_ts);
            CREATE INDEX IF NOT EXISTS idx_sim_trades_sim_signal_id ON sim_trades(sim_signal_id);
            """
        )
        self._conn.commit()

    def record_signal(
        self,
        *,
        ts: datetime,
        market: str,
        symbol: str,
        action: str,
        reason: str,
        features: dict[str, Any] | None = None,
    ) -> int:
        feat_json = None
        if isinstance(features, dict):
            feat_json = json.dumps(features, default=str)
        cur = self._conn.execute(
            """
            INSERT INTO sim_signals (ts, market, symbol, action, reason, features_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts.isoformat(), str(market), str(symbol), str(action), str(reason), feat_json),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_trade(self, *, market: str, side: str, trade: Any, meta: dict[str, Any] | None = None) -> None:
        """
        Accepts either:
        - a dataclass with backtest-like fields (e.g. backtest.Trade), OR
        - a dict containing the same keys.
        """
        if is_dataclass(trade):
            row = asdict(trade)
        elif isinstance(trade, dict):
            row = dict(trade)
        else:
            raise TypeError("trade must be a dataclass or dict")

        meta_obj = meta or {}
        meta_json = json.dumps(meta_obj, default=str)
        sim_signal_id = None
        try:
            if "sim_signal_id" in meta_obj and meta_obj["sim_signal_id"] is not None:
                sim_signal_id = int(meta_obj["sim_signal_id"])
        except Exception:
            sim_signal_id = None
        self._conn.execute(
            """
            INSERT INTO sim_trades (
              sim_signal_id, market, symbol, side, entry_ts, exit_ts,
              entry_price, exit_price, qty, pnl, pnl_r, risk_r,
              hold_minutes, entry_cost_usd, exit_cost_usd, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sim_signal_id,
                str(market),
                str(row.get("symbol")),
                str(side),
                str(row.get("entry_ts")),
                str(row.get("exit_ts")),
                float(row.get("entry_price")),
                float(row.get("exit_price")),
                float(row.get("qty")),
                float(row.get("pnl")),
                (None if row.get("pnl_r") is None else float(row.get("pnl_r"))),
                (None if row.get("risk_r") is None else float(row.get("risk_r"))),
                (None if row.get("hold_minutes") is None else float(row.get("hold_minutes"))),
                (None if row.get("entry_cost_usd") is None else float(row.get("entry_cost_usd"))),
                (None if row.get("exit_cost_usd") is None else float(row.get("exit_cost_usd"))),
                meta_json,
            ),
        )
        self._conn.commit()

