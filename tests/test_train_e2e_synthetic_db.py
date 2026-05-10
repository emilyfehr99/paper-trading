"""End-to-end train_and_save on a tiny synthetic ledger (CI-safe)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca_day_bot.ml.train import train_and_save
from alpaca_day_bot.storage.ledger import Ledger


def test_train_and_save_fits_classifier_on_synthetic_signals(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite3"
    lg = Ledger(str(db))
    base = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    feat: dict = {
        "close": 100.0,
        "rsi_14": 48.0,
        "htf_rsi": 50.0,
        "ema": 99.5,
        "ema_9": 99.0,
        "ema_21": 99.2,
        "ema_9_21_bias": 0.01,
        "alligator_jaw": 98.0,
        "alligator_teeth": 98.5,
        "alligator_lips": 99.0,
        "alligator_trend_up": 1.0,
        "alligator_trend_down": 0.0,
        "macd": 0.1,
        "macd_signal": 0.05,
        "vwap": 99.8,
        "atr": 1.0,
        "atr_avg": 1.05,
        "atr_monthly": 1.2,
        "volume_ratio": 1.0,
        "htf_ok_long": True,
        "htf_ok_short": False,
        "news": {
            "ok": True,
            "articles": [
                {
                    "created_at": "2024-01-02T14:00:00+00:00",
                    "sentiment_score": 0.1,
                    "headline": "test",
                    "summary": "",
                    "provider": "alpaca",
                }
            ],
        },
        "taapi": {"rsi_1m": 50.0, "rsi_15m": 49.0, "macd_1m": 0.0, "macd_signal_1m": 0.0},
        "tp_price": 102.0,
        "sl_price": 98.5,
    }
    for i in range(30):
        ts = base + timedelta(hours=i)
        sid = lg.record_signal(ts=ts, symbol="SPY", action="BUY", reason="long_momo", features=feat)
        # Same convention as live bot: return_pct is (exit-entry)/entry (fraction, not bps).
        ret = 0.012 if (i % 2 == 0) else -0.012
        px = 100.0 * (1.0 + ret)
        lg.record_forward_return_label(
            signal_id=sid,
            evaluated_ts=ts + timedelta(minutes=30),
            price_at_label=float(px),
            entry_close=100.0,
            return_pct=float(ret),
            horizon_minutes=60.0,
        )
    lg.close()

    out = tmp_path / "model.joblib"
    meta = train_and_save(
        db_path=str(db),
        out_path=str(out),
        min_horizon_minutes=1.0,
        min_rows=12,
        min_class_count=5,
        action="BUY",
        dataset_source="signals",
        target_mode="binary",
    )
    assert meta.get("skipped") is not True, meta
    assert out.is_file()
    assert meta.get("feature_vector_id") == "signal_flatten_v1"
    assert meta.get("live_inference_path") == "signal_features"
