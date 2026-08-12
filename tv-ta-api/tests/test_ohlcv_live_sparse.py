"""Live history must not hard-fail sparse density floor."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from app.ohlcv_validate import OhlcvIntegrityError, validate_ohlcv_frame


def _frame(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-08-10 00:00:00", periods=n, freq="5min", tz="UTC")
    px = 5000.0
    return pd.DataFrame(
        {
            "Open": [px] * n,
            "High": [px + 1] * n,
            "Low": [px - 1] * n,
            "Close": [px] * n,
        },
        index=idx,
    )


def test_live_sparse_waived_when_enforce_sparse_false():
    now = int(time.time())
    start = now - 3 * 86400
    end = now + 3600
    # Under weekday*120 floor this would raise if enforce_sparse=True.
    df = validate_ohlcv_frame(
        _frame(60),
        symbol="NQ=F",
        start_ts=start,
        end_ts=end,
        resolution="5",
        enforce_sparse=False,
    )
    assert len(df) == 60


def test_historical_sparse_still_enforced():
    start = 1_700_000_000
    end = start + 3 * 86400
    with pytest.raises(OhlcvIntegrityError, match="sparse_ohlcv"):
        validate_ohlcv_frame(
            _frame(60),
            symbol="NQ=F",
            start_ts=start,
            end_ts=end,
            resolution="5",
            enforce_sparse=True,
        )
