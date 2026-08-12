"""TradingView provider symbol mapping and range helper tests."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.tradingview_provider import (
    _estimate_bars_for_range,
    _merge_frames,
    is_futures_like_symbol,
    normalize_symbol_for_tradingview,
)


def test_mnq_yfinance_to_tv():
    assert normalize_symbol_for_tradingview("MNQ=F") == "CME_MINI:MNQ1!"


def test_mes_bot_ticker_to_tv():
    assert normalize_symbol_for_tradingview("MES1") == "CME_MINI:MES1!"


def test_es_nq_rty_map_to_cme_mini():
    assert normalize_symbol_for_tradingview("ES=F") == "CME_MINI:ES1!"
    assert normalize_symbol_for_tradingview("NQ1") == "CME_MINI:NQ1!"
    assert normalize_symbol_for_tradingview("RTY=F") == "CME_MINI:RTY1!"


def test_futures_detection():
    assert is_futures_like_symbol("MNQ=F")
    assert is_futures_like_symbol("CME_MINI:MNQ1!")
    assert not is_futures_like_symbol("NASDAQ:AAPL")


def test_merge_frames_dedupes():
    idx = pd.date_range("2026-03-01", periods=3, freq="15min", tz="UTC")
    a = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    b = pd.DataFrame({"close": [9.0]}, index=[idx[-1]])
    out = _merge_frames([a, b])
    assert len(out) == 3
    assert float(out.iloc[-1]["close"]) == 9.0


def test_pick_single_frame_prefers_fewer_jumps():
    from app.tradingview_provider import _ohlcv_jump_count, _pick_single_ohlcv_frame

    idx = pd.date_range("2026-06-01", periods=5, freq="5min", tz="UTC")
    clean = pd.DataFrame({"close": [30000.0, 30001.0, 30002.0, 30001.5, 30003.0]}, index=idx)
    dirty = clean.copy()
    dirty.iloc[2, 0] = 30350.0
    assert _ohlcv_jump_count(dirty) > _ohlcv_jump_count(clean)
    picked = _pick_single_ohlcv_frame([("dirty", dirty), ("clean", clean)])
    assert float(picked.iloc[2]["close"]) == 30002.0


def test_estimate_bars_for_range_15m():
    start = datetime(2026, 3, 1, tzinfo=UTC)
    end = datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC)
    est = _estimate_bars_for_range(start, end, "15")
    assert 1500 <= est <= 20000
