from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from alpaca_day_bot.ml.dataset import (
    LIVE_INFERENCE_PATH_SIGNAL_FEATURES,
    SUPPORTED_FEATURE_VECTOR_IDS,
    _parse_iso_dt,
    flatten_signal_features,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDecision:
    ok: bool
    provider: str | None
    proba: float | None
    error: str | None = None
    task: str = "classification"
    regression_pred: float | None = None


def _infer_anchor_ts(features: dict[str, Any]) -> datetime:
    """
    Anchor for UTC time columns + news recency. Prefer explicit ``ts`` / ``signal_ts`` from
    the live feature dict; otherwise use UTC wall clock so columns stay finite (matches
    training when DB signal time is the anchor).
    """
    feat = features if isinstance(features, dict) else {}
    ts_s = feat.get("ts") or feat.get("signal_ts")
    dt = _parse_iso_dt(ts_s) if isinstance(ts_s, str) else None
    if dt is not None:
        return dt
    if os.environ.get("ML_INFER_LOG_ANCHOR_FALLBACK", "").strip() in ("1", "true", "yes"):
        _log.info("ml_infer_anchor_fallback_utc missing ts/signal_ts in feature dict")
    else:
        _log.debug("ml_infer_anchor_fallback_utc missing ts/signal_ts in feature dict")
    return datetime.now(tz=timezone.utc)


def _flatten_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the per-signal ``features_json`` dict into a flat numeric dict for inference.

    Delegates to :func:`alpaca_day_bot.ml.dataset.flatten_signal_features` so training and
    live scoring cannot drift.
    """
    feat = features if isinstance(features, dict) else {}
    act = str(feat.get("action") or feat.get("signal_action") or "BUY")
    reason = str(feat.get("reason") or "")
    anchor = _infer_anchor_ts(feat)
    return flatten_signal_features(feat, reason=reason, action=act, anchor_ts=anchor)


def feature_vector_id_ok(meta: dict[str, Any] | None) -> tuple[bool, str | None]:
    """
    If ``feature_vector_id`` is present on the artifact, it must be in the supported set.
    Missing key = legacy bundle (allowed).
    """
    if not isinstance(meta, dict):
        return True, None
    fvid = meta.get("feature_vector_id")
    if fvid is None or str(fvid).strip() == "":
        return True, None
    if str(fvid) not in SUPPORTED_FEATURE_VECTOR_IDS:
        return False, "unsupported_feature_vector_id"
    return True, None


def live_inference_path_ok(meta: dict[str, Any] | None) -> tuple[bool, str | None]:
    """
    Return (False, code) if this artifact cannot be scored from live signal features alone.

    Legacy bundles omit ``live_inference_path`` and are treated as compatible.
    """
    if not isinstance(meta, dict):
        return True, None
    lip = meta.get("live_inference_path")
    if lip is None:
        return True, None
    lip_s = str(lip).strip()
    if lip_s == "executed_context":
        return False, "unsupported_live_inference_path"
    if lip_s and lip_s != str(LIVE_INFERENCE_PATH_SIGNAL_FEATURES):
        return False, "unsupported_live_inference_path"
    return True, None


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
        ok_path, path_err = live_inference_path_ok(meta if isinstance(meta, dict) else None)
        if not ok_path:
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error=path_err or "unsupported_live_inference_path",
                task=str(task),
                regression_pred=None,
            )
        ok_fv, fv_err = feature_vector_id_ok(meta if isinstance(meta, dict) else None)
        if not ok_fv:
            return ModelDecision(
                ok=False,
                provider=provider,
                proba=None,
                error=fv_err or "unsupported_feature_vector_id",
                task=str(task),
                regression_pred=None,
            )
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
