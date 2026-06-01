from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime

import joblib
import pandas as pd


@dataclass(frozen=True)
class ModelDecision:
    ok: bool
    provider: str | None
    proba: float | None
    error: str | None = None
    explainability: dict[str, float] | None = None
    threshold: float | None = None


def _taapi_features(taapi: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(taapi, dict):
        return {
            "taapi_rsi_1m": float("nan"),
            "taapi_rsi_15m": float("nan"),
            "taapi_macd_1m": float("nan"),
            "taapi_macd_signal_1m": float("nan"),
            "taapi_present": 0.0,
        }
    
    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")
            
    rsi1 = f(taapi.get("rsi_1m"))
    rsi15 = f(taapi.get("rsi_15m"))
    macd = f(taapi.get("macd_1m"))
    macds = f(taapi.get("macd_signal_1m"))
    
    import math
    present = 1.0 if any(math.isfinite(v) for v in (rsi1, rsi15, macd, macds)) else 0.0
    return {
        "taapi_rsi_1m": rsi1,
        "taapi_rsi_15m": rsi15,
        "taapi_macd_1m": macd,
        "taapi_macd_signal_1m": macds,
        "taapi_present": present,
    }


def _flatten_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the per-signal features_json dict into a flat numeric dict that matches training columns.
    Keep this in sync with ml.dataset.build_signal_label_dataset().
    """
    import math
    feat = features if isinstance(features, dict) else {}

    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    price = f(feat.get("close"))
    vwap = f(feat.get("vwap"))
    atr = f(feat.get("atr"))
    avg50 = f(feat.get("avg_50"))
    if math.isnan(avg50) or avg50 <= 0:
        ema21 = f(feat.get("ema_21"))
        alligator = f(feat.get("alligator_jaw"))
        if not math.isnan(ema21) and ema21 > 0:
            avg50 = ema21
        elif not math.isnan(alligator) and alligator > 0:
            avg50 = alligator
        else:
            avg50 = price

    # Volatility normalization scale to prevent momentum compression bias
    norm_scale = atr if (atr and atr > 0) else (price * 0.01 if price > 0 else 1.0)

    vwap_ratio = (price - vwap) / norm_scale if (price and vwap and norm_scale > 0) else 0.0
    avg50_ratio = (price - avg50) / norm_scale if (price and avg50 and norm_scale > 0) else 0.0

    # Extract order book level 0 details if available
    ob = feat.get("order_book", {})
    ask_price = f(ob.get("val_0"))
    ask_size = f(ob.get("val_1"))
    bid_price = f(ob.get("val_2"))
    bid_size = f(ob.get("val_3"))
    
    spread_ratio = (ask_price - bid_price) / price if (price > 0 and ask_price > 0 and bid_price > 0) else 0.0
    obi = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size and ask_size and (bid_size + ask_size) > 0) else 0.0
    
    # Multi-timeframe divergences
    htf_rsi = f(feat.get("htf_rsi"))
    rsi_1m = f(feat.get("rsi_14"))
    rsi_divergence = rsi_1m - htf_rsi if (math.isfinite(rsi_1m) and math.isfinite(htf_rsi)) else 0.0
    
    macd_val = f(feat.get("macd"))
    macd_sig = f(feat.get("macd_signal"))
    macd_convergence = macd_val / macd_sig if (macd_val and macd_sig and abs(macd_sig) > 0) else 0.0

    vol_sma = f(feat.get("vol_sma_20"))
    if not vol_sma or math.isnan(vol_sma) or vol_sma <= 0:
        vol_sma = f(feat.get("volume"))
    if not vol_sma or math.isnan(vol_sma) or vol_sma <= 0:
        vol_sma = 1.0

    obv_val = f(feat.get("obv", 0.0))
    obv_ema_val = f(feat.get("obv_ema", 0.0))
    vol_imb_val = f(feat.get("volume_imbalance", 0.0))
    supertrend_val = f(feat.get("supertrend", 0.0))
    supertrend_norm = (supertrend_val - price) / price if (price > 0 and supertrend_val > 0) else 0.0

    def normalize_band(val_raw):
        val = f(val_raw)
        if not val or math.isnan(val):
            return 0.0
        if abs(val) < 2.0:
            # Already normalized (close - band) / close. Negate to get (band - close) / close.
            return -val
        return (val - price) / price if price > 0 else 0.0

    x: dict[str, Any] = {
        "vwap_dist_ratio": vwap_ratio,
        "rsi_14": rsi_1m,
        "macd_line_ratio": f(feat.get("macd_line", 0.0)) / norm_scale if norm_scale > 0 else 0.0,
        "macd_signal_ratio": f(feat.get("macd_signal", 0.0)) / norm_scale if norm_scale > 0 else 0.0,
        "macd_hist_ratio": f(feat.get("macd_hist", 0.0)) / norm_scale if norm_scale > 0 else 0.0,
        "alligator_jaw_ratio": (price - f(feat.get("alligator_jaw", 0.0))) / norm_scale if norm_scale > 0 else 0.0,
        "alligator_teeth_ratio": (price - f(feat.get("alligator_teeth", 0.0))) / norm_scale if norm_scale > 0 else 0.0,
        "alligator_lips_ratio": (price - f(feat.get("alligator_lips", 0.0))) / norm_scale if norm_scale > 0 else 0.0,
        "alligator_convergence_index": f(feat.get("alligator_convergence_index", 0.0)),
        "atr_norm": norm_scale / price if price > 0 else 0.0,
        "avg_50_ratio": avg50_ratio,
        "momentum_pct": f(feat.get("momentum_pct")) if feat.get("momentum_pct") is not None and not math.isnan(f(feat.get("momentum_pct"))) else ((price - avg50) / avg50 if (price and avg50 and avg50 > 0) else 0.0),
        
        # Deep Intelligence Features
        "news_sentiment_score": f(feat.get("news_sentiment_score", 0.0)),
        "fed_liquidity_momentum": f(feat.get("fed_liquidity_momentum", 0.0)),
        "sector_dispersion_factor": f(feat.get("sector_dispersion_factor", 0.0)),
        "moc_imbalance_shares": f(feat.get("moc_imbalance_shares", 0.0)),
        "minutes_until_earnings_announcement": f(feat.get("minutes_until_earnings_announcement", 1440.0)),
        
        # Engineered Microstructure & Divergence Features
        "spread_ratio": spread_ratio,
        "order_book_imbalance": obi,
        "rsi_divergence": rsi_divergence,
        "macd_convergence_ratio": macd_convergence,
        
        # Pillar 2 & Pillar 3
        "spy_ret_5m": f(feat.get("spy_ret_5m")),
        "qqq_ret_5m": f(feat.get("qqq_ret_5m")),
        "vix_roc_5m": f(feat.get("vix_roc_5m")),
        "spy_trend_1h": f(feat.get("spy_trend_1h")),
        "spy_trend_4h": f(feat.get("spy_trend_4h")),
        "qqq_trend_1h": f(feat.get("qqq_trend_1h")),
        "qqq_trend_4h": f(feat.get("qqq_trend_4h")),

        # Primary live strategy rules indicators
        "ar1_rho": f(feat.get("ar1_rho", 0.0)),
        "z_score_vwap": f(feat.get("z_score_vwap", 0.0)),
        "willr": f(feat.get("willr", -50.0)),
        "stoch_k": f(feat.get("stoch_k", 50.0)),
        "stoch_d": f(feat.get("stoch_d", 50.0)),
        "obv": obv_val / vol_sma,
        "obv_ema": obv_ema_val / vol_sma,
        "cmf": f(feat.get("cmf", 0.0)),
        "supertrend": supertrend_norm,
        "supertrend_dir": f(feat.get("supertrend_dir", 0.0)),
        "rvol": f(feat.get("rvol", 1.0)),
        "ret_1m": f(feat.get("ret_1m", 0.0)),
        "ret_5m": f(feat.get("ret_5m", 0.0)),
        "ret_15m": f(feat.get("ret_15m", 0.0)),
        "vwap_slope": f(feat.get("vwap_slope", 0.0)),
        "minutes_since_open": f(feat.get("minutes_since_open", 0.0)),
        "minutes_until_close": f(feat.get("minutes_until_close", 0.0)),
        # New features
        "vwap_band_1": normalize_band(feat.get("vwap_band_1")),
        "vwap_band_2": normalize_band(feat.get("vwap_band_2")),
        "vwap_band_3": normalize_band(feat.get("vwap_band_3")),
        "vwap_band_neg1": normalize_band(feat.get("vwap_band_neg1")),
        "vwap_band_neg2": normalize_band(feat.get("vwap_band_neg2")),
        "vwap_band_neg3": normalize_band(feat.get("vwap_band_neg3")),
        "volume_imbalance": vol_imb_val / vol_sma,
        "volume_imbalance_norm": f(feat.get("volume_imbalance_norm", 0.0)),
        "spy_realized_vol_1h": f(feat.get("spy_realized_vol_1h", 0.0)),
        "qqq_realized_vol_1h": f(feat.get("qqq_realized_vol_1h", 0.0)),
        "regime": feat.get("regime", "neutral"),
    }
    
    # Get TAAPI features (nested JSON)
    taapi_feats = _taapi_features(feat.get("taapi"))
    x.update(taapi_feats)
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
        x = _flatten_feature_dict(features)
        regime = x.get("regime", "neutral")
        
        # Dual-model regime-switching routing
        if "model_trend" in model_bundle and "model_mr" in model_bundle:
            if regime == "trend":
                model = model_bundle.get("model_trend")
                scaler = model_bundle.get("scaler_trend")
                threshold = model_bundle.get("threshold_trend", 0.50)
                provider = "regime_switching_trend"
            else:
                model = model_bundle.get("model_mr")
                scaler = model_bundle.get("scaler_mr")
                threshold = model_bundle.get("threshold_mr", 0.50)
                provider = "regime_switching_mr"
        else:
            model = model_bundle.get("model")
            scaler = model_bundle.get("scaler")
            meta = model_bundle.get("meta") or {}
            threshold = model_bundle.get("threshold") or meta.get("recommended_min_proba") or meta.get("threshold") or 0.50
            provider = model_bundle.get("provider") or meta.get("provider") or "trinity_ensemble_v1"
            
        cols = model_bundle.get("feature_columns")
        meta = model_bundle.get("meta") or {}
        if not cols and isinstance(meta, dict):
            cols = meta.get("feature_columns")
            
        if provider:
            provider = str(provider)
            
        # DEBUG: Write features to a file for inspection
        try:
            with open("/Users/emilyfehr8/CascadeProjects/alpaca-paper-day-bot/scratch/ml_features_debug.json", "w") as f_debug:
                # Remove non-serializable elements if any
                serializable_x = {k: v for k, v in x.items() if not isinstance(v, (pd.Timestamp, datetime))}
                json.dump(serializable_x, f_debug, indent=2)
        except Exception:
            pass

        # Prepare features (drop regime since it is not a model feature)
        x_features = {k: v for k, v in x.items() if k != "regime"}
        X = pd.DataFrame([x_features])
        if isinstance(cols, list) and cols:
            # align to training columns with fuzzy fallback to flattened features
            mapped = []
            missing = []
            def _simplify(name: str) -> str:
                return ''.join(ch for ch in str(name).lower() if ch.isalnum())
            for c in cols:
                if c in X.columns:
                    continue
                # try to find a best candidate from flattened features `x`
                cand = None
                sc = _simplify(c)
                for k in x.keys():
                    if _simplify(k) == sc or sc in _simplify(k) or _simplify(k) in sc:
                        cand = k
                        break
                if cand:
                    try:
                        X[c] = x.get(cand, 0.0)
                        mapped.append((c, cand))
                    except Exception:
                        X[c] = 0.0
                        missing.append(c)
                else:
                    X[c] = 0.0
                    missing.append(c)
            X = X[cols]
            try:
                import logging
                logging.getLogger(__name__).warning("Feature alignment: mapped %d cols, missing %d cols", len(mapped), len(missing))
            except Exception:
                pass
        
        # --- NORMALIZATION (Apply saved scaler) ---
        scaler_to_use = scaler
        X_final = X.fillna(0) # Final safety fill
        if scaler_to_use:
            try:
                X_final = pd.DataFrame(scaler_to_use.transform(X_final), columns=X_final.columns)
            except Exception:
                pass # Fallback to unscaled if shape mismatch

        p = float(model.predict_proba(X_final)[:, 1][0])

        # allow runtime override of decision threshold via env var
        try:
            import os as _os
            override = _os.getenv("MODEL_DECISION_THRESHOLD_OVERRIDE")
            if override:
                try:
                    threshold = float(override)
                except Exception:
                    pass
        except Exception:
            pass

        # --- EXPLAINABILITY: Why did the model pick this? ---
        explainability = {}
        try:
            target_model = model
            if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
                target_model = model.calibrated_classifiers_[0].estimator
            
            if hasattr(target_model, "final_estimator_"):
                target_model = target_model.final_estimator_

            # We use the top 10 most influential features for this specific prediction
            if hasattr(target_model, "coef_"): # LogisticRegression
                coefs = target_model.coef_[0]
                contributions = {c: float(X[c].iloc[0] * coefs[i]) for i, c in enumerate(cols)}
                explainability = dict(sorted(contributions.items(), key=lambda x: -abs(x[1]))[:10])
            elif hasattr(target_model, "feature_importances_"): # RF / LGBM
                importances = target_model.feature_importances_
                contributions = {c: float(X[c].iloc[0] * importances[i]) for i, c in enumerate(cols)}
                explainability = dict(sorted(contributions.items(), key=lambda x: -abs(x[1]))[:10])
        except Exception:
            pass

        return ModelDecision(ok=True, provider=provider, proba=p, explainability=explainability, error=None, threshold=threshold)
    except Exception as e:
        return ModelDecision(ok=False, provider=None, proba=None, error=str(e)[:200], threshold=None)

