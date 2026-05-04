from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelDecision:
    ok: bool
    provider: str | None
    proba: float | None
    error: str | None = None
    task: str = "classification"
    regression_pred: float | None = None


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
        "ema_9": f(feat.get("ema_9")),
        "ema_21": f(feat.get("ema_21")),
        "ema_9_21_bias": f(feat.get("ema_9_21_bias")),
        "alligator_jaw": f(feat.get("alligator_jaw")),
        "alligator_teeth": f(feat.get("alligator_teeth")),
        "alligator_lips": f(feat.get("alligator_lips")),
        "alligator_trend_up": f(feat.get("alligator_trend_up")),
        "alligator_trend_down": f(feat.get("alligator_trend_down")),
        "macd": f(feat.get("macd")),
        "macd_signal": f(feat.get("macd_signal")),
        "vwap": f(feat.get("vwap")),
        "volume_ratio": f(feat.get("volume_ratio")),
        "atr": f(feat.get("atr")),
        "atr_avg": f(feat.get("atr_avg")),
        "atr_monthly": f(feat.get("atr_monthly")),
        "htf_ok_long": 1.0 if bool(feat.get("htf_ok_long")) else 0.0,
        "htf_ok_short": 1.0 if bool(feat.get("htf_ok_short")) else 0.0,
        "is_buy": 1.0,
    }
    try:
        act = (feat.get("action") or feat.get("signal_action") or "").strip().upper()
        if act in ("SHORT", "SELL"):
            x["is_buy"] = 0.0
    except Exception:
        pass

    # Time-of-day context (if ts captured in features; else NaN)
    try:
        ts_s = feat.get("ts") or feat.get("signal_ts")  # not currently set; ok
        dt = None
        if isinstance(ts_s, str) and ts_s:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        if dt is None:
            raise ValueError("no_ts")
        x["hour_utc"] = float(dt.hour)
        x["minute_utc"] = float(dt.minute)
        x["dow_utc"] = float(dt.weekday())
    except Exception:
        x["hour_utc"] = float("nan")
        x["minute_utc"] = float("nan")
        x["dow_utc"] = float("nan")

    # Setup type (reason)
    rs = (feat.get("reason") or "").strip().lower()
    x["setup_long_pullback"] = 1.0 if rs == "long_rsi_macd_vwap_volume" else 0.0
    x["setup_long_momo"] = 1.0 if rs == "long_momo" else 0.0
    x["setup_crypto_macd_alligator_momo"] = 1.0 if rs == "crypto_macd_alligator_momo" else 0.0
    x["setup_short_pullback"] = 1.0 if rs == "short_rsi_macd_vwap_volume" else 0.0
    x["setup_short_momo"] = 1.0 if rs == "short_momo" else 0.0
    x["setup_short_rsi_overbought_fade"] = 1.0 if rs == "short_rsi_overbought_fade" else 0.0
    x["setup_short_bb_upper_fade"] = 1.0 if rs == "short_bb_upper_fade" else 0.0
    x["setup_short_break_retest"] = 1.0 if rs == "short_break_retest" else 0.0

    # News
    if isinstance(news, dict) and isinstance(news.get("articles"), list):
        arts = [a for a in news.get("articles", []) if isinstance(a, dict)]
        x["news_ok"] = 1.0 if bool(news.get("ok")) else 0.0
        x["news_count"] = float(len(arts))
        x["news_recency_min"] = float("nan")
        x["news_sent_wmean"] = float("nan")
        x["news_event_risk"] = 0.0
        sent = []
        sent_w = []
        src_counts = {"alpaca": 0, "alphavantage": 0, "google_rss": 0, "tickertick": 0}
        risk_words = (
            "earnings",
            "offering",
            "secondary",
            "sec ",
            "investigation",
            "lawsuit",
            "downgrade",
            "upgrade",
            "guidance",
            "merger",
            "acquisition",
            "halt",
            "bankruptcy",
        )
        for a in arts:
            prov = (a.get("provider") or "").strip().lower()
            if prov in src_counts:
                src_counts[prov] += 1
            txt = f"{a.get('headline') or ''} {a.get('summary') or ''}".strip().lower()
            if txt and any(w in txt for w in risk_words):
                x["news_event_risk"] = 1.0
            s = a.get("sentiment_score")
            if s is not None:
                try:
                    sv = float(s)
                    sent.append(sv)
                    # weight by recency if possible
                    w = 1.0
                    created = a.get("created_at")
                    if isinstance(created, str) and created:
                        try:
                            from datetime import datetime, timezone

                            dtc = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            if dtc.tzinfo is None:
                                dtc = dtc.replace(tzinfo=timezone.utc)
                            # use the signal timestamp if available (dt) else don't weight
                            if "hour_utc" in x and isinstance(dt, datetime):
                                age_min = max(0.0, (dt - dtc).total_seconds() / 60.0)
                                w = 1.0 / (1.0 + (age_min / 60.0))
                        except Exception:
                            w = 1.0
                    sent_w.append((sv, w))
                except Exception:
                    pass
        x["news_sent_mean"] = float(sum(sent) / len(sent)) if sent else float("nan")
        x["news_sent_wmean"] = (
            float(sum(v * w for v, w in sent_w) / sum(w for _v, w in sent_w)) if sent_w else float("nan")
        )
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
                "news_sent_wmean": float("nan"),
                "news_sent_present": 0.0,
                "news_event_risk": 0.0,
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
        if not isinstance(model_bundle, dict):
            return ModelDecision(
                ok=False,
                provider=None,
                proba=None,
                error="invalid_model_bundle",
                task="classification",
                regression_pred=None,
            )
        model = model_bundle.get("model")
        if model is None:
            return ModelDecision(
                ok=False,
                provider=None,
                proba=None,
                error="missing_model",
                task="classification",
                regression_pred=None,
            )
        meta = model_bundle.get("meta") or {}
        provider = str(meta.get("provider")) if isinstance(meta, dict) else None
        task = str(meta.get("task") or "classification").strip().lower()
        cols = meta.get("feature_columns") if isinstance(meta, dict) else None
        if isinstance(cols, list) and len(cols) == 0:
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error="empty_feature_columns",
                task=str(task),
                regression_pred=None,
            )
        x = _flatten_feature_dict(features)
        X = pd.DataFrame([x])
        if isinstance(cols, list) and cols:
            # align to training columns
            for c in cols:
                if c not in X.columns:
                    X[c] = float("nan")
            X = X[cols]
            if int(X.shape[1]) != int(len(cols)):
                return ModelDecision(
                    ok=False,
                    provider=provider,
                    proba=None,
                    error="feature_column_count_mismatch",
                    task=str(task),
                    regression_pred=None,
                )
        if task == "regression":
            if not hasattr(model, "predict"):
                return ModelDecision(
                    ok=False,
                    provider=provider,
                    proba=None,
                    error="missing_predict",
                    task="regression",
                    regression_pred=None,
                )
            raw_pred = model.predict(X)
            if raw_pred is None:
                return ModelDecision(
                    ok=False,
                    provider=provider,
                    proba=None,
                    error="missing_regression_pred",
                    task="regression",
                    regression_pred=None,
                )
            flat = np.asarray(raw_pred, dtype=float).ravel()
            if flat.size != 1:
                return ModelDecision(
                    ok=False,
                    provider=provider,
                    proba=None,
                    error="invalid_regression_shape",
                    task="regression",
                    regression_pred=None,
                )
            pred_raw = float(flat[0])
            if not math.isfinite(pred_raw):
                return ModelDecision(
                    ok=False,
                    provider=provider,
                    proba=None,
                    error="non_finite_regression_pred",
                    task="regression",
                    regression_pred=None,
                )
            return ModelDecision(
                ok=True,
                provider=provider,
                proba=None,
                error=None,
                task="regression",
                regression_pred=pred_raw,
            )
        if not hasattr(model, "predict_proba"):
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error="missing_predict_proba",
                task="classification",
                regression_pred=None,
            )
        pr = model.predict_proba(X)
        if pr is None or not hasattr(pr, "shape") or len(pr.shape) != 2 or pr.shape[1] < 2:
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error="invalid_proba_shape",
                task="classification",
                regression_pred=None,
            )
        rowp = np.asarray(pr[0], dtype=float).ravel()
        try:
            sm = float(np.sum(rowp))
        except Exception:
            sm = float("nan")
        if not math.isfinite(sm) or abs(sm - 1.0) > 0.03:
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error="invalid_proba_mass",
                task="classification",
                regression_pred=None,
            )
        p_raw = float(pr[0, 1])
        if not math.isfinite(p_raw):
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error="non_finite_proba",
                task="classification",
                regression_pred=None,
            )
        eps = 1e-6
        p = min(1.0 - eps, max(eps, p_raw))
        return ModelDecision(ok=True, provider=provider, proba=p, error=None, task="classification", regression_pred=None)
    except Exception as e:
        return ModelDecision(
            ok=False,
            provider=None,
            proba=None,
            error=str(e)[:200],
            task="classification",
            regression_pred=None,
        )

