"""Bot-side ML bundle filtering (mirrors main._accept_ml_bundle_for_live logic)."""
from __future__ import annotations

from types import SimpleNamespace

from alpaca_day_bot.main import _accept_ml_bundle_for_live


def test_rejects_executed_context_path():
    b = {"model": object(), "meta": {"live_inference_path": "executed_context", "dataset_kind": "signals"}}
    out = _accept_ml_bundle_for_live(b, SimpleNamespace(ml_inference_disallow_executed_dataset=False), "t")
    assert out is None


def test_disallow_executed_dataset_setting():
    b = {
        "model": object(),
        "meta": {"dataset_kind": "executed_trades:long", "live_inference_path": "signal_features"},
    }
    out = _accept_ml_bundle_for_live(
        b,
        SimpleNamespace(ml_inference_disallow_executed_dataset=True),
        "t",
    )
    assert out is None
    out2 = _accept_ml_bundle_for_live(
        b,
        SimpleNamespace(ml_inference_disallow_executed_dataset=False),
        "t",
    )
    assert out2 is b


def test_accepts_legacy_missing_inference_path():
    b = {"model": object(), "meta": {"dataset_kind": "signals"}}
    out = _accept_ml_bundle_for_live(b, SimpleNamespace(ml_inference_disallow_executed_dataset=False), "t")
    assert out is b


def test_rejects_unknown_feature_vector_id():
    b = {
        "model": object(),
        "meta": {"dataset_kind": "signals", "feature_vector_id": "signal_flatten_v0"},
    }
    assert _accept_ml_bundle_for_live(b, SimpleNamespace(ml_inference_disallow_executed_dataset=False), "t") is None
