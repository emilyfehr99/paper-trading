"""Training vs inference feature parity and bundle guards."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from alpaca_day_bot.ml.dataset import (
    FEATURE_VECTOR_ID,
    LIVE_INFERENCE_PATH_SIGNAL_FEATURES,
    flatten_signal_features,
)
from alpaca_day_bot.ml.infer import _flatten_feature_dict, live_inference_path_ok


def _anchor() -> datetime:
    return datetime(2024, 6, 15, 14, 5, tzinfo=timezone.utc)


def test_flatten_matches_infer_when_ts_present():
    anchor = _anchor()
    feat = {
        "close": 100.0,
        "rsi_14": 45.0,
        "macd": 0.1,
        "macd_signal": 0.05,
        "atr": 1.2,
        "volume_ratio": 1.1,
        "news": {"ok": True, "articles": []},
        "taapi": {"rsi_1m": 50.0, "rsi_15m": 48.0, "macd_1m": 0.0, "macd_signal_1m": 0.0},
        "ts": anchor.isoformat(),
        "reason": "crypto_macd_alligator_momo",
        "action": "BUY",
    }
    a = flatten_signal_features(
        feat,
        reason="crypto_macd_alligator_momo",
        action="BUY",
        anchor_ts=anchor,
    )
    b = _flatten_feature_dict(feat)
    assert set(a.keys()) == set(b.keys())
    for k in sorted(a.keys()):
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float) and (va != va or vb != vb):
            continue
        assert va == pytest.approx(vb, rel=1e-9, abs=1e-9) or va == vb


def test_crypto_setup_one_hot():
    anchor = _anchor()
    x = flatten_signal_features(
        {"close": 1.0, "rsi_14": 50.0},
        reason="crypto_macd_alligator_momo",
        action="BUY",
        anchor_ts=anchor,
    )
    assert x["setup_crypto_macd_alligator_momo"] == 1.0
    assert x["setup_long_momo"] == 0.0


def test_taapi_present_only_when_finite_values():
    anchor = _anchor()
    empty = flatten_signal_features(
        {"close": 1.0, "taapi": {"rsi_1m": float("nan"), "rsi_15m": float("nan"), "macd_1m": float("nan"), "macd_signal_1m": float("nan")}},
        reason="long_momo",
        action="BUY",
        anchor_ts=anchor,
    )
    assert empty["taapi_present"] == 0.0
    some = flatten_signal_features(
        {"close": 1.0, "taapi": {"rsi_1m": 55.0, "rsi_15m": float("nan"), "macd_1m": float("nan"), "macd_signal_1m": float("nan")}},
        reason="long_momo",
        action="BUY",
        anchor_ts=anchor,
    )
    assert some["taapi_present"] == 1.0


def test_live_inference_path_ok():
    assert live_inference_path_ok(None) == (True, None)
    assert live_inference_path_ok({}) == (True, None)
    assert live_inference_path_ok({"live_inference_path": LIVE_INFERENCE_PATH_SIGNAL_FEATURES}) == (True, None)
    assert live_inference_path_ok({"live_inference_path": "signal_features"}) == (True, None)
    ok, err = live_inference_path_ok({"live_inference_path": "executed_context"})
    assert ok is False and err == "unsupported_live_inference_path"
    ok2, err2 = live_inference_path_ok({"live_inference_path": "unknown_mode"})
    assert ok2 is False and err2 == "unsupported_live_inference_path"


def test_feature_vector_constant():
    assert FEATURE_VECTOR_ID == "signal_flatten_v1"
    assert LIVE_INFERENCE_PATH_SIGNAL_FEATURES == "signal_features"
