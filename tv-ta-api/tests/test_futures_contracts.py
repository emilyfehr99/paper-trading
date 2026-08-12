"""Futures contract segment resolution tests."""

from datetime import UTC, date, datetime, time

from app.futures_contracts import contract_segments_for_range


def test_mnq_february_2026_uses_march_contract():
    start = datetime(2026, 2, 1, tzinfo=UTC)
    end = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)
    segs = contract_segments_for_range("MNQ=F", start, end)
    assert len(segs) >= 1
    assert any(s.tv_symbol == "CME_MINI:MNQH2026" for s in segs)


def test_mes_february_2026_uses_march_contract():
    start = datetime(2026, 2, 1, tzinfo=UTC)
    end = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)
    segs = contract_segments_for_range("MES1", start, end)
    assert any(s.tv_symbol == "CME_MINI:MESH2026" for s in segs)


def test_es_named_contract_uses_cme_mini():
    start = datetime(2024, 6, 1, tzinfo=UTC)
    end = datetime(2024, 6, 15, 23, 59, 59, tzinfo=UTC)
    segs = contract_segments_for_range("ES=F", start, end)
    assert any(s.tv_symbol.startswith("CME_MINI:ES") for s in segs)


def test_march_2026_spans_roll():
    start = datetime.combine(date(2026, 3, 1), time.min, tzinfo=UTC)
    end = datetime.combine(date(2026, 3, 31), time(23, 59, 59), tzinfo=UTC)
    segs = contract_segments_for_range("MNQ=F", start, end)
    symbols = {s.tv_symbol for s in segs}
    assert "CME_MINI:MNQH2026" in symbols
    assert "CME_MINI:MNQM2026" in symbols
