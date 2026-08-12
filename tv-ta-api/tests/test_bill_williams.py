"""Unit tests for Williams Alligator + MACD histogram (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.bill_williams import macd_histogram_latest, williams_alligator_latest
from app.indicators import compute_series


def _ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    idx = pd.date_range("2026-06-01", periods=n, freq="min")
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        },
        index=idx,
    )


def test_alligator_returns_finite_lines():
    out = williams_alligator_latest(_ohlcv())
    for k in ("alligator_jaw", "alligator_teeth", "alligator_lips"):
        assert k in out
        assert out[k] == out[k]  # not nan after warmup


def test_macd_histogram_has_tv_bar_fields():
    out = macd_histogram_latest(_ohlcv())
    assert "macd_hist" in out
    assert out["macd_hist"] == out["macd_hist"]
    assert out["macd_bar_green"] in (0.0, 1.0)
    assert out["macd_cross_green"] in (0.0, 1.0)
    assert out["macd_hist_fading"] in (0.0, 1.0)


def test_compute_series_macd():
    df = _ohlcv()
    pts = compute_series(df, indicator="macd", period=14, count=10)
    assert len(pts) == 10
    for p in pts:
        assert "macd" in p
        assert "signal" in p
        assert "hist" in p
        assert isinstance(p["t"], int)


def test_compute_series_alligator():
    df = _ohlcv()
    pts = compute_series(df, indicator="alligator", period=14, count=10)
    assert len(pts) == 10
    for p in pts:
        assert "jaw" in p
        assert "teeth" in p
        assert "lips" in p
        assert isinstance(p["t"], int)

