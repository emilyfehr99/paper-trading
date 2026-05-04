from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _table_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [r[1] for r in rows]  # name


def _ensure_schema(dst: sqlite3.Connection) -> None:
    # Reuse the schema from SimLedger; keep in sync with storage/sim_ledger.py
    dst.executescript(
        """
        CREATE TABLE IF NOT EXISTS sim_signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          market TEXT NOT NULL,
          symbol TEXT NOT NULL,
          action TEXT NOT NULL,
          reason TEXT NOT NULL,
          features_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sim_signals_ts ON sim_signals(ts);
        CREATE INDEX IF NOT EXISTS idx_sim_signals_symbol_ts ON sim_signals(symbol, ts);

        CREATE TABLE IF NOT EXISTS sim_trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          sim_signal_id INTEGER,
          market TEXT NOT NULL,
          symbol TEXT NOT NULL,
          side TEXT NOT NULL,
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
    dst.commit()


def _next_id(dst: sqlite3.Connection, table: str) -> int:
    try:
        r = dst.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table};").fetchone()
        return int(r[0] or 0) + 1
    except Exception:
        return 1


def merge(*, inputs: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(str(out))
    dst.execute("PRAGMA journal_mode=WAL;")
    dst.execute("PRAGMA synchronous=NORMAL;")
    _ensure_schema(dst)

    for inp in inputs:
        if not inp.exists():
            continue
        src = sqlite3.connect(str(inp))
        try:
            src_cols_sig = _table_cols(src, "sim_signals")
            src_cols_tr = _table_cols(src, "sim_trades")
        except Exception:
            src.close()
            continue

        # Build ID offsets so shard IDs don't collide and trade links remain valid.
        sig_offset = _next_id(dst, "sim_signals") - 1
        tr_offset = _next_id(dst, "sim_trades") - 1

        # Copy signals
        sig_common = [c for c in ("id", "ts", "market", "symbol", "action", "reason", "features_json") if c in src_cols_sig]
        if "id" in sig_common:
            rows = src.execute(
                "SELECT id, ts, market, symbol, action, reason, features_json FROM sim_signals ORDER BY id ASC"
            ).fetchall()
            dst.executemany(
                "INSERT INTO sim_signals (id, ts, market, symbol, action, reason, features_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (int(r[0]) + sig_offset, r[1], r[2], r[3], r[4], r[5], r[6])
                    for r in rows
                ],
            )
        else:
            rows = src.execute(
                "SELECT ts, market, symbol, action, reason, features_json FROM sim_signals ORDER BY ts ASC"
            ).fetchall()
            dst.executemany(
                "INSERT INTO sim_signals (ts, market, symbol, action, reason, features_json) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

        # Copy trades
        has_sig_link = "sim_signal_id" in src_cols_tr
        if "id" in src_cols_tr:
            if has_sig_link:
                rows = src.execute(
                    """
                    SELECT id, sim_signal_id, market, symbol, side, entry_ts, exit_ts,
                           entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                           entry_cost_usd, exit_cost_usd, meta_json
                    FROM sim_trades
                    ORDER BY id ASC
                    """
                ).fetchall()
                dst.executemany(
                    """
                    INSERT INTO sim_trades (
                      id, sim_signal_id, market, symbol, side, entry_ts, exit_ts,
                      entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                      entry_cost_usd, exit_cost_usd, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            int(r[0]) + tr_offset,
                            (None if r[1] is None else int(r[1]) + sig_offset),
                            r[2],
                            r[3],
                            r[4],
                            r[5],
                            r[6],
                            r[7],
                            r[8],
                            r[9],
                            r[10],
                            r[11],
                            r[12],
                            r[13],
                            r[14],
                            r[15],
                            r[16],
                        )
                        for r in rows
                    ],
                )
            else:
                rows = src.execute(
                    """
                    SELECT id, market, symbol, side, entry_ts, exit_ts,
                           entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                           entry_cost_usd, exit_cost_usd, meta_json
                    FROM sim_trades
                    ORDER BY id ASC
                    """
                ).fetchall()
                dst.executemany(
                    """
                    INSERT INTO sim_trades (
                      id, market, symbol, side, entry_ts, exit_ts,
                      entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                      entry_cost_usd, exit_cost_usd, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(int(r[0]) + tr_offset, *r[1:]) for r in rows],
                )
        else:
            # Let SQLite assign IDs; adjust sim_signal_id if present.
            if has_sig_link:
                rows = src.execute(
                    """
                    SELECT sim_signal_id, market, symbol, side, entry_ts, exit_ts,
                           entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                           entry_cost_usd, exit_cost_usd, meta_json
                    FROM sim_trades
                    ORDER BY entry_ts ASC
                    """
                ).fetchall()
                dst.executemany(
                    """
                    INSERT INTO sim_trades (
                      sim_signal_id, market, symbol, side, entry_ts, exit_ts,
                      entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                      entry_cost_usd, exit_cost_usd, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            (None if r[0] is None else int(r[0]) + sig_offset),
                            *r[1:],
                        )
                        for r in rows
                    ],
                )
            else:
                rows = src.execute(
                    """
                    SELECT market, symbol, side, entry_ts, exit_ts,
                           entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                           entry_cost_usd, exit_cost_usd, meta_json
                    FROM sim_trades
                    ORDER BY entry_ts ASC
                    """
                ).fetchall()
                dst.executemany(
                    """
                    INSERT INTO sim_trades (
                      market, symbol, side, entry_ts, exit_ts,
                      entry_price, exit_price, qty, pnl, pnl_r, risk_r, hold_minutes,
                      entry_cost_usd, exit_cost_usd, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        dst.commit()
        src.close()

    dst.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output sim_rollup sqlite path")
    ap.add_argument("--in", dest="inputs", required=True, nargs="+", help="Input shard sqlite paths")
    args = ap.parse_args()
    merge(inputs=[Path(p) for p in args.inputs], out=Path(args.out))


if __name__ == "__main__":
    main()

