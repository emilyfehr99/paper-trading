from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class VirtualOptionsDayStats:
    n_closed: int
    total_pnl_usd: float


def virtual_options_stats_for_day(db_path: str, day: date) -> VirtualOptionsDayStats:
    conn = sqlite3.connect(db_path)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()
    row = conn.execute(
        """
        SELECT
          COUNT(1) AS n,
          COALESCE(SUM(CASE WHEN pnl_usd IS NULL THEN 0 ELSE pnl_usd END), 0.0) AS pnl
        FROM virtual_option_trades
        WHERE ts_close IS NOT NULL
          AND ts_close BETWEEN ? AND ?
        """,
        (start, end),
    ).fetchone()
    conn.close()
    n = int(row[0] or 0) if row else 0
    pnl = float(row[1] or 0.0) if row else 0.0
    return VirtualOptionsDayStats(n_closed=n, total_pnl_usd=pnl)

