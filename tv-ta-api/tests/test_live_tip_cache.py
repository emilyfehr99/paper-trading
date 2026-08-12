"""Live tip cache bypass + tip response schema helpers."""

from __future__ import annotations

import time

from app.main import (
    _cached_last_bar_open_unix,
    _expected_last_closed_bar_open_unix,
    _is_live_history_request,
    _live_cache_stale,
)
from app.models import TipResponse


def test_is_live_history_request_near_now():
    now = int(time.time())
    assert _is_live_history_request(None) is True
    assert _is_live_history_request(now + 60) is True
    assert _is_live_history_request(now - 3600) is True
    assert _is_live_history_request(now - 2 * 86400) is False


def test_expected_last_closed_bar_open_unix_5m():
    # 12:07:30 → forming 12:05–12:10 → last closed open 12:00
    # Use a fixed unix so the assertion is deterministic.
    # 2026-08-03 17:07:30 UTC = epoch floored to 300s.
    now_ts = 1754240850.0  # arbitrary; verify consistency of math
    expected = _expected_last_closed_bar_open_unix("5", now_ts=now_ts)
    bm = 300
    slot = int(now_ts // bm) * bm
    if now_ts < slot + bm:
        slot -= bm
    assert expected == slot


def test_live_cache_stale_when_tip_behind():
    now = time.time()
    expected = _expected_last_closed_bar_open_unix("5", now_ts=now)
    stale = {
        "bar_count": 10,
        "points": [{"t": expected - 300, "c": 1.0}],
    }
    fresh = {
        "bar_count": 10,
        "points": [{"t": expected, "c": 1.0}],
    }
    assert _live_cache_stale(stale, "5", end_ts=None) is True
    assert _live_cache_stale(fresh, "5", end_ts=None) is False
    # Completed historical range — do not bypass on tip age.
    assert _live_cache_stale(stale, "5", end_ts=int(now) - 2 * 86400) is False


def test_cached_last_bar_open_handles_ms():
    assert _cached_last_bar_open_unix({"points": [{"t": 1_700_000_000_000}]}) == 1_700_000_000
    assert _cached_last_bar_open_unix({"points": []}) is None


def test_tip_response_schema():
    tip = TipResponse(
        symbol="ES=F",
        resolution="5",
        bar_count=1,
        tip_open_ts=1_700_000_000,
        tip_close_ts=1_700_000_300,
        close=5000.0,
        points=[{"t": 1_700_000_000, "o": 1, "h": 1, "l": 1, "c": 5000.0, "indicators": {}}],
    )
    assert tip.tip_close_ts == tip.tip_open_ts + 300
    assert tip.bar_count == 1
