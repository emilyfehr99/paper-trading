"""OHLCV integrity checks — empty / corrupt frames are errors, not success."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# CME equity-index micros are CT-session — never use UTC calendar dates for
# weekday floors (Sun 17:00 CT open → end_ts can land on Mon UTC and inflate
# min_expected → false sparse_ohlcv 502 on an otherwise good Sunday reopen).
_FUTURES_CAL_TZ = ZoneInfo("America/Chicago")


class OhlcvIntegrityError(Exception):
    """Raised when TVTA would otherwise return empty or corrupt bars."""


def _as_futures_date(ts: int) -> date:
    return datetime.fromtimestamp(int(ts), tz=_FUTURES_CAL_TZ).date()


def _weekday_span_days(start: date, end: date) -> int:
    """Count Mon–Fri calendar days in [start, end] inclusive (futures approx)."""
    if end < start:
        return 0
    n = 0
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


def expect_futures_bars(start_ts: int | None, end_ts: int | None) -> bool:
    """True when the request covers at least one weekday — empty is not allowed."""
    if start_ts is None or end_ts is None:
        return False
    start = _as_futures_date(start_ts)
    end = _as_futures_date(end_ts)
    return _weekday_span_days(start, end) >= 1


def min_expected_5m_bars(start_ts: int, end_ts: int) -> int:
    """Conservative floor: ~120 5m bars per weekday (TV continuous can be thinner than full Globex)."""
    start = _as_futures_date(start_ts)
    end = _as_futures_date(end_ts)
    weekdays = _weekday_span_days(start, end)
    return max(50, weekdays * 120)


def validate_ohlcv_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    resolution: str = "5",
    enforce_sparse: bool = True,
) -> pd.DataFrame:
    """
    Ensure OHLCV is real and usable.

    Raises OhlcvIntegrityError on empty-when-expected or corrupt OHLC.

    ``enforce_sparse=False`` for live / near-now tip windows — weekday*120 floors
    false-positive overnight (2026-08-10: seed_days=2 into early Mon Globex →
    ``ohlcv_integrity`` 502 storms while tip was healthy).
    """
    if df is None or df.empty:
        if expect_futures_bars(start_ts, end_ts):
            raise OhlcvIntegrityError(
                f"empty_ohlcv symbol={symbol} resolution={resolution} "
                f"start_ts={start_ts} end_ts={end_ts} — TV returned 0 bars for a "
                f"weekday-spanning range (treat as fetch failure, not success)"
            )
        raise OhlcvIntegrityError(f"empty_ohlcv symbol={symbol} resolution={resolution}")

    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close"]
    missing = [k for k in need if k not in cols]
    if missing:
        # Title-case variants
        cols = {c: c for c in df.columns}
        for k in need:
            if k.title() in df.columns:
                cols[k] = k.title()
            elif k.capitalize() in df.columns:
                cols[k] = k.capitalize()
            elif k in df.columns:
                cols[k] = k
        missing = [k for k in need if k not in cols]
        if missing:
            raise OhlcvIntegrityError(f"missing_columns {missing} symbol={symbol}")

    o = df[cols["open"]].astype(float)
    h = df[cols["high"]].astype(float)
    l = df[cols["low"]].astype(float)
    c = df[cols["close"]].astype(float)

    if o.isna().any() or h.isna().any() or l.isna().any() or c.isna().any():
        raise OhlcvIntegrityError(f"nan_ohlc symbol={symbol} n={len(df)}")

    if (h < l).any() or (c > h).any() or (c < l).any() or (o > h).any() or (o < l).any():
        bad = int(((h < l) | (c > h) | (c < l) | (o > h) | (o < l)).sum())
        raise OhlcvIntegrityError(f"ohlc_inconsistent symbol={symbol} bad_rows={bad}")

    if (c <= 0).any() or (h <= 0).any():
        raise OhlcvIntegrityError(f"non_positive_price symbol={symbol}")

    if (
        enforce_sparse
        and start_ts is not None
        and end_ts is not None
        and str(resolution) in {"1", "5", "15", "30", "60"}
    ):
        floor = min_expected_5m_bars(start_ts, end_ts)
        # Scale floor for coarser bars.
        minutes = int(resolution) if str(resolution).isdigit() else 5
        floor = max(20, int(floor * (5 / max(minutes, 1))))
        if len(df) < floor:
            raise OhlcvIntegrityError(
                f"sparse_ohlcv symbol={symbol} bars={len(df)} min_expected={floor} "
                f"start_ts={start_ts} end_ts={end_ts}"
            )

    # Index must be datetime and sorted unique.
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    if not idx.is_monotonic_increasing:
        df = df.copy()
        df.index = idx
        df = df.sort_index()
        idx = pd.DatetimeIndex(df.index)
    if idx.has_duplicates:
        df = df[~idx.duplicated(keep="last")]

    return df
