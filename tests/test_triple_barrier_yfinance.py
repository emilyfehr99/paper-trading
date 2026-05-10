"""Shared triple-barrier bar walk (live bot + yfinance backfill)."""
from __future__ import annotations

import pandas as pd

from alpaca_day_bot.tools.triple_barrier_yfinance import (
    realized_return_from_last_close,
    triple_barrier_outcome_from_bars,
)


def _df(rows: list[tuple]) -> pd.DataFrame:
    idx, hi, lo, cl = zip(*rows)
    return pd.DataFrame({"high": hi, "low": lo, "close": cl}, index=pd.DatetimeIndex(idx, tz="UTC"))


def test_buy_tp_first():
    dfx = _df(
        [
            ("2024-01-01 15:00:00+00:00", 100.0, 99.0, 99.5),
            ("2024-01-01 15:05:00+00:00", 102.5, 99.5, 102.0),
        ]
    )
    out, px = triple_barrier_outcome_from_bars(dfx, act="BUY", entry_f=100.0, tp_f=102.0, sl_f=98.0)
    assert out == "tp"
    assert px == 102.0


def test_buy_sl_first():
    dfx = _df(
        [
            ("2024-01-01 15:00:00+00:00", 100.0, 99.0, 99.5),
            ("2024-01-01 15:05:00+00:00", 100.5, 97.5, 98.0),
        ]
    )
    out, px = triple_barrier_outcome_from_bars(dfx, act="BUY", entry_f=100.0, tp_f=102.0, sl_f=98.0)
    assert out == "sl"
    assert px == 98.0


def test_short_tp_first():
    dfx = _df(
        [
            ("2024-01-01 15:00:00+00:00", 101.0, 100.0, 100.5),
            ("2024-01-01 15:05:00+00:00", 100.0, 97.0, 97.5),
        ]
    )
    out, px = triple_barrier_outcome_from_bars(dfx, act="SHORT", entry_f=100.0, tp_f=98.0, sl_f=102.0)
    assert out == "tp"
    assert px == 97.5


def test_timeout_neither_hit():
    dfx = _df(
        [
            ("2024-01-01 15:00:00+00:00", 100.5, 99.5, 100.0),
            ("2024-01-01 15:05:00+00:00", 100.5, 99.5, 100.1),
        ]
    )
    out, px = triple_barrier_outcome_from_bars(dfx, act="BUY", entry_f=100.0, tp_f=102.0, sl_f=98.0)
    assert out == "timeout"
    assert px == 100.1


def test_realized_return_buy():
    r = realized_return_from_last_close(act="BUY", entry_f=100.0, px_at_eval=101.0)
    assert abs(r - 0.01) < 1e-9
