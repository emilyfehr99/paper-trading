"""Ledger backlog listers include signals from prior calendar days."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca_day_bot.storage.ledger import Ledger


def test_backlog_forward_labels_spans_prior_days(tmp_path):
    db = tmp_path / "l.sqlite3"
    lg = Ledger(str(db))
    utc = ZoneInfo("UTC")
    old = datetime(2023, 6, 1, 16, 0, tzinfo=utc)
    lg.record_signal(ts=old, symbol="SPY", action="BUY", reason="long_momo", features={"close": 100.0})
    today = datetime(2024, 1, 10, 16, 0, tzinfo=utc)
    lg.record_signal(ts=today, symbol="SPY", action="BUY", reason="long_momo", features={"close": 101.0})
    now = today + timedelta(hours=2)
    # Per-day lister for "today" misses 2023-06-01 signal.
    day_rows = lg.list_unlabeled_signal_rows(
        market_day=date(2024, 1, 10),
        tz=utc,
        now_utc=now,
        min_age_minutes=0.0,
        actions=("BUY",),
    )
    assert len(day_rows) == 1
    bl = lg.list_unlabeled_signal_rows_backlog(
        now_utc=now,
        min_age_minutes=0.0,
        actions=("BUY",),
        limit=50,
    )
    assert len(bl) == 2
    lg.close()
