from __future__ import annotations

from alpaca_day_bot.ml.infer import feature_vector_id_ok, predict_proba


def test_feature_vector_rejects_unknown_id():
    ok, err = feature_vector_id_ok({"feature_vector_id": "signal_flatten_future_proto"})
    assert ok is False and err == "unsupported_feature_vector_id"


def test_feature_vector_accepts_v1():
    assert feature_vector_id_ok({"feature_vector_id": "signal_flatten_v1"}) == (True, None)


def test_feature_vector_legacy_missing_key():
    assert feature_vector_id_ok({"dataset_kind": "signals"}) == (True, None)


def test_predict_proba_rejects_unknown_feature_vector():
    bundle = {
        "model": _DummyClf(),
        "meta": {
            "task": "classification",
            "feature_columns": ["close", "rsi_14"],
            "feature_vector_id": "signal_flatten_v0",
        },
    }
    md = predict_proba(model_bundle=bundle, features={"close": 1.0, "rsi_14": 50.0, "ts": "2024-01-01T12:00:00+00:00"})
    assert md.ok is False and md.error == "unsupported_feature_vector_id"


class _DummyClf:
    def predict_proba(self, X):  # noqa: ANN001
        import numpy as np

        return np.array([[0.5, 0.5]])
