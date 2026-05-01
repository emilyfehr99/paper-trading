from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _table_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows]


def _max_id(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0


def _copy_with_id_offset(*, src: sqlite3.Connection, dst: sqlite3.Connection, table: str, id_offset: int) -> int:
    if table not in (r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()):
        return 0
    cols = _table_cols(src, table)
    if not cols:
        return 0
    dst_cols = _table_cols(dst, table)
    if not dst_cols:
        return 0

    # If src lacks id, copy common columns and let dst assign ids.
    if "id" not in cols or "id" not in dst_cols:
        common = [c for c in cols if c in dst_cols and c != "id"]
        if not common:
            return 0
        rows = src.execute(f"SELECT {', '.join(common)} FROM {table}").fetchall()
        if not rows:
            return 0
        ph = ", ".join(["?"] * len(common))
        dst.executemany(f"INSERT INTO {table} ({', '.join(common)}) VALUES ({ph})", rows)
        return len(rows)

    cols_wo_id = [c for c in cols if c != "id" and c in dst_cols]
    if not cols_wo_id:
        return 0
    rows = src.execute(f"SELECT id, {', '.join(cols_wo_id)} FROM {table}").fetchall()
    if not rows:
        return 0
    ph = ", ".join(["?"] * (1 + len(cols_wo_id)))
    ins = f"INSERT INTO {table} (id, {', '.join(cols_wo_id)}) VALUES ({ph})"
    out = []
    for rid, *rest in rows:
        out.append([int(rid) + int(id_offset), *rest])
    dst.executemany(ins, out)
    return len(out)


def _copy_labels_with_signal_offset(*, src: sqlite3.Connection, dst: sqlite3.Connection, table: str, sig_off: int) -> int:
    if table not in (r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()):
        return 0
    cols = _table_cols(src, table)
    dst_cols = _table_cols(dst, table)
    if "signal_id" not in cols or "signal_id" not in dst_cols:
        return 0
    common = [c for c in cols if c in dst_cols]
    rows = src.execute(f"SELECT {', '.join(common)} FROM {table}").fetchall()
    if not rows:
        return 0
    ph = ", ".join(["?"] * len(common))
    ins = f"INSERT OR REPLACE INTO {table} ({', '.join(common)}) VALUES ({ph})"
    out = []
    for r in rows:
        d = dict(zip(common, r))
        d["signal_id"] = int(d["signal_id"]) + int(sig_off)
        out.append([d[c] for c in common])
    dst.executemany(ins, out)
    return len(out)


def ensure_schema(dst_db: Path) -> None:
    from alpaca_day_bot.storage.ledger import Ledger

    ld = Ledger(str(dst_db))
    ld.close()


def rollup(*, src_db: Path, dst_db: Path) -> dict[str, int]:
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    if not dst_db.exists():
        ensure_schema(dst_db)

    src = sqlite3.connect(str(src_db))
    src.execute("PRAGMA query_only=ON;")
    dst = sqlite3.connect(str(dst_db))
    dst.execute("PRAGMA journal_mode=WAL;")
    dst.execute("PRAGMA synchronous=NORMAL;")

    sig_off = _max_id(dst, "signals")
    out = {
        "signals": 0,
        "forward_return_labels": 0,
        "triple_barrier_labels": 0,
        "order_intents": 0,
        "trade_updates": 0,
        "equity_snapshots": 0,
        "virtual_option_trades": 0,
    }

    dst.execute("BEGIN;")
    try:
        out["signals"] = _copy_with_id_offset(src=src, dst=dst, table="signals", id_offset=sig_off)
        out["order_intents"] = _copy_with_id_offset(
            src=src, dst=dst, table="order_intents", id_offset=_max_id(dst, "order_intents")
        )
        out["trade_updates"] = _copy_with_id_offset(
            src=src, dst=dst, table="trade_updates", id_offset=_max_id(dst, "trade_updates")
        )
        out["equity_snapshots"] = _copy_with_id_offset(
            src=src, dst=dst, table="equity_snapshots", id_offset=_max_id(dst, "equity_snapshots")
        )
        out["virtual_option_trades"] = _copy_with_id_offset(
            src=src, dst=dst, table="virtual_option_trades", id_offset=_max_id(dst, "virtual_option_trades")
        )
        out["forward_return_labels"] = _copy_labels_with_signal_offset(
            src=src, dst=dst, table="forward_return_labels", sig_off=sig_off
        )
        out["triple_barrier_labels"] = _copy_labels_with_signal_offset(
            src=src, dst=dst, table="triple_barrier_labels", sig_off=sig_off
        )
        dst.commit()
    except Exception:
        dst.rollback()
        raise
    finally:
        src.close()
        dst.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source ledger sqlite path")
    ap.add_argument("--dst", required=True, help="Destination rollup sqlite path")
    args = ap.parse_args()
    res = rollup(src_db=Path(args.src), dst_db=Path(args.dst))
    print(res)


if __name__ == "__main__":
    main()

