from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class ModelDecision:
    ok: bool
    provider: str | None
    proba: float | None
    error: str | None = None


def _flatten_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the per-signal features_json dict into a flat numeric dict that matches training columns.
    Keep this in sync with ml.dataset.build_signal_label_dataset().
    """
    feat = features if isinstance(features, dict) else {}
    news = feat.get("news") if isinstance(feat.get("news"), dict) else None
    taapi = feat.get("taapi") if isinstance(feat.get("taapi"), dict) else None

    # minimal inline feature extraction (avoid importing ml.dataset to keep runtime small)
    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    x: dict[str, Any] = {
        "close": f(feat.get("close")),
        "rsi_14": f(feat.get("rsi_14")),
        "htf_rsi": f(feat.get("htf_rsi")),
        "ema": f(feat.get("ema")),
        "macd": f(feat.get("macd")),
        "macd_signal": f(feat.get("macd_signal")),
        "vwap": f(feat.get("vwap")),
        "volume_ratio": f(feat.get("volume_ratio")),
        "atr": f(feat.get("atr")),
        "atr_avg": f(feat.get("atr_avg")),
        "htf_ok_long": 1.0 if bool(feat.get("htf_ok_long")) else 0.0,
        "htf_ok_short": 1.0 if bool(feat.get("htf_ok_short")) else 0.0,
        "is_buy": 1.0,
    }

    # News
    if isinstance(news, dict) and isinstance(news.get("articles"), list):
        arts = [a for a in news.get("articles", []) if isinstance(a, dict)]
        x["news_ok"] = 1.0 if bool(news.get("ok")) else 0.0
        x["news_count"] = float(len(arts))
        x["news_recency_min"] = float("nan")
        sent = []
        src_counts = {"alpaca": 0, "alphavantage": 0, "google_rss": 0, "tickertick": 0}
        for a in arts:
            prov = (a.get("provider") or "").strip().lower()
            if prov in src_counts:
                src_counts[prov] += 1
            s = a.get("sentiment_score")
            if s is not None:
                try:
                    sent.append(float(s))
                except Exception:
                    pass
        x["news_sent_mean"] = float(sum(sent) / len(sent)) if sent else float("nan")
        x["news_sent_present"] = 1.0 if sent else 0.0
        x["news_src_alpaca"] = float(src_counts["alpaca"])
        x["news_src_alphavantage"] = float(src_counts["alphavantage"])
        x["news_src_google_rss"] = float(src_counts["google_rss"])
        x["news_src_tickertick"] = float(src_counts["tickertick"])
    else:
        x.update(
            {
                "news_ok": 0.0,
                "news_count": 0.0,
                "news_recency_min": float("nan"),
                "news_sent_mean": float("nan"),
                "news_sent_present": 0.0,
                "news_src_alpaca": 0.0,
                "news_src_alphavantage": 0.0,
                "news_src_google_rss": 0.0,
                "news_src_tickertick": 0.0,
            }
        )

    # TAAPI
    if isinstance(taapi, dict):
        x["taapi_rsi_1m"] = f(taapi.get("rsi_1m"))
        x["taapi_rsi_15m"] = f(taapi.get("rsi_15m"))
        x["taapi_macd_1m"] = f(taapi.get("macd_1m"))
        x["taapi_macd_signal_1m"] = f(taapi.get("macd_signal_1m"))
        x["taapi_present"] = 1.0
    else:
        x.update(
            {
                "taapi_rsi_1m": float("nan"),
                "taapi_rsi_15m": float("nan"),
                "taapi_macd_1m": float("nan"),
                "taapi_macd_signal_1m": float("nan"),
                "taapi_present": 0.0,
            }
        )

    return x


def load_model(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = joblib.load(p)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def predict_proba(*, model_bundle: dict[str, Any], features: dict[str, Any]) -> ModelDecision:
    try:
        model = model_bundle.get("model")
        meta = model_bundle.get("meta") or {}
        provider = str(meta.get("provider")) if isinstance(meta, dict) else None
        cols = meta.get("feature_columns") if isinstance(meta, dict) else None
        x = _flatten_feature_dict(features)
        X = pd.DataFrame([x])
        if isinstance(cols, list) and cols:
            # align to training columns
            for c in cols:
                if c not in X.columns:
                    X[c] = float("nan")
            X = X[cols]
        p = float(model.predict_proba(X)[:, 1][0])
        return ModelDecision(ok=True, provider=provider, proba=p, error=None)
    except Exception as e:
        return ModelDecision(ok=False, provider=None, proba=None, error=str(e)[:200])

