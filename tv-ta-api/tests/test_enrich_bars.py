"""Historical enrich_bars endpoint (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ta_enrich import bars_to_dataframe, enrich_ohlcv_frame, frame_to_points, indicators_at_timestamp


def _bars(n: int = 80) -> list[dict]:
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0.1, 0.2, n))
    out = []
    base_t = 1717401600
    for i, c in enumerate(close):
        out.append(
            {
                "t": base_t + i * 60,
                "o": float(c - 0.05),
                "h": float(c + 0.1),
                "l": float(c - 0.1),
                "c": float(c),
                "v": 10000.0,
            }
        )
    return out


def test_enrich_frame_produces_hist_per_bar():
    df = bars_to_dataframe(_bars())
    en = enrich_ohlcv_frame(df, alligator_mode="williams")
    assert "macd_hist" in en.columns
    pts = frame_to_points(en)
    assert len(pts) == len(en)
    assert "macd_hist" in pts[-1]["indicators"]
    assert pts[-1]["c"] == float(en["close"].iloc[-1])
    assert pts[-1]["o"] == float(en["open"].iloc[-1])


def test_indicators_at_timestamp():
    df = bars_to_dataframe(_bars())
    en = enrich_ohlcv_frame(df)
    mid_ts = int(pd.Timestamp(en.index[len(en) // 2]).timestamp())
    ts, ind = indicators_at_timestamp(en, mid_ts)
    assert ts <= mid_ts
    assert "macd_hist" in ind


def test_enrich_frame_exports_macd_line_crosses():
    close = 100 + np.cumsum(np.linspace(-0.5, 0.8, 120))
    bars = []
    base_t = 1717401600
    for i, c in enumerate(close):
        bars.append(
            {
                "t": base_t + i * 900,
                "o": float(c - 0.05),
                "h": float(c + 0.1),
                "l": float(c - 0.1),
                "c": float(c),
                "v": 10000.0,
            }
        )
    en = enrich_ohlcv_frame(bars_to_dataframe(bars), alligator_mode="williams")
    assert "macd_line_cross_green" in en.columns
    assert "macd_line_cross_red" in en.columns
    pts = frame_to_points(en)
    assert "macd_cross_green" in pts[-1]["indicators"]
