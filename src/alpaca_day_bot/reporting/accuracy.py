from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ForwardAccuracySummary:
    """Directional hit rate for BUY signals once forward-return labels exist."""

    labeled_count: int
    directional_hits: int
    mean_return_pct: float | None
    median_return_pct: float | None
    note: str


def forward_accuracy_for_calendar_day(
    db_path: str,
    day: date,
    *,
    market_tz: str,
) -> ForwardAccuracySummary | None:
    """
    Labels are joined to signals where the BUY occurred on `day` in `market_tz`.
    This measures close→close proxy accuracy at label time, not bracket fill outcomes.
    """
    tz = ZoneInfo(market_tz)
    start = datetime.combine(day, time(0, 0, 0), tzinfo=tz).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    start_s, end_s = start.isoformat(), end.isoformat()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT f.return_pct
        FROM forward_return_labels f
        JOIN signals s ON s.id = f.signal_id
        WHERE s.action = 'BUY'
          AND s.ts >= ? AND s.ts < ?
        """,
        (start_s, end_s),
    ).fetchall()
    conn.close()

    if not rows:
        return None

    rets = [float(r[0]) for r in rows]
    hits = sum(1 for r in rets if r > 0)
    n = len(rets)
    mean = sum(rets) / n
    sorted_r = sorted(rets)
    med = sorted_r[n // 2] if n % 2 == 1 else 0.5 * (sorted_r[n // 2 - 1] + sorted_r[n // 2])

    note = (
        "Forward return = (label_price − signal_close) / signal_close after a minimum age; "
        "not the same as realized bracket PnL."
    )
    return ForwardAccuracySummary(
        labeled_count=n,
        directional_hits=hits,
        mean_return_pct=mean,
        median_return_pct=med,
        note=note,
    )
