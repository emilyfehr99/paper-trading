from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RollupHealth:
    db_path: str
    signals_buy: int
    signals_short: int
    tb_labels_buy: int
    tb_labels_short: int
    executed_round_trips_total: int
    executed_round_trips_buy: int
    executed_round_trips_short: int
    model_buy_status: str | None
    model_short_status: str | None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def _model_status_line(meta_json_path: Path) -> str | None:
    if not meta_json_path.exists():
        return None
    try:
        j = json.loads(meta_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    if bool(j.get("skipped")):
        return f"skipped:{j.get('skip_reason','n/a')} n={j.get('n_labeled','n/a')}"
    prov = j.get("provider", "n/a")
    ds = j.get("dataset_kind", "n/a")
    return f"trained:{prov} ({ds})"


def rollup_health(*, rollup_db_path: str, state_dir: str = "state") -> RollupHealth | None:
    p = Path(rollup_db_path)
    if not p.exists():
        return None

    conn = sqlite3.connect(str(p))
    try:
        if not _table_exists(conn, "signals") or not _table_exists(conn, "trade_updates"):
            return None

        signals_buy = _count(conn, "SELECT COUNT(1) FROM signals WHERE action='BUY'")
        signals_short = _count(conn, "SELECT COUNT(1) FROM signals WHERE action='SHORT'")

        tb_buy = 0
        tb_short = 0
        if _table_exists(conn, "triple_barrier_labels"):
            tb_buy = _count(
                conn,
                """
                SELECT COUNT(1)
                FROM triple_barrier_labels tb
                JOIN signals s ON s.id = tb.signal_id
                WHERE s.action='BUY'
                """,
            )
            tb_short = _count(
                conn,
                """
                SELECT COUNT(1)
                FROM triple_barrier_labels tb
                JOIN signals s ON s.id = tb.signal_id
                WHERE s.action='SHORT'
                """,
            )

        # Executed round trips (FIFO matching), split by direction.
        try:
            from alpaca_day_bot.reporting.trades import _rows_to_fills, reconstruct_round_trips

            rows = conn.execute(
                "SELECT ts, event, raw_json FROM trade_updates WHERE event IN ('fill','partial_fill','filled') ORDER BY ts"
            ).fetchall()
            fills = _rows_to_fills(rows)
            rts = reconstruct_round_trips(fills)
            rt_total = int(len(rts))
            rt_long = int(sum(1 for rt in rts if rt.direction == "long"))
            rt_short = int(sum(1 for rt in rts if rt.direction == "short"))
        except Exception:
            rt_total, rt_long, rt_short = 0, 0, 0

        sd = Path(state_dir)
        buy_status = _model_status_line(sd / "models" / "latest_buy.json")
        short_status = _model_status_line(sd / "models" / "latest_short.json")

        return RollupHealth(
            db_path=str(p),
            signals_buy=int(signals_buy),
            signals_short=int(signals_short),
            tb_labels_buy=int(tb_buy),
            tb_labels_short=int(tb_short),
            executed_round_trips_total=int(rt_total),
            executed_round_trips_buy=int(rt_long),
            executed_round_trips_short=int(rt_short),
            model_buy_status=buy_status,
            model_short_status=short_status,
        )
    finally:
        conn.close()

