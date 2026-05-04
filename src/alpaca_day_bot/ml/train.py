from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)

from alpaca_day_bot.ml.dataset import build_signal_label_dataset
from alpaca_day_bot.ml.executed_dataset import build_executed_trade_dataset
from alpaca_day_bot.ml.sim_dataset import build_sim_trade_dataset


@dataclass(frozen=True)
class TrainMetrics:
    n: int
    pos_rate: float
    auc: float | None
    acc: float


def _safe_auc(y_true, y_score) -> float | None:
    try:
        if len(set(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _safe_average_precision(y_true, y_score) -> float | None:
    try:
        if len(set(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def _precision_at_k(y_true, y_score, k_frac: float) -> float | None:
    try:
        n = len(y_true)
        if n <= 0:
            return None
        k = max(1, int(n * float(k_frac)))
        idx = np.argsort(-np.asarray(y_score))[:k]
        yt = np.asarray(y_true)[idx]
        return float(np.mean(yt))
    except Exception:
        return None


def _drop_low_variance_numeric_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    min_non_na: int = 12,
    var_eps: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove numeric columns that are (near) constant on the train split — they add no signal
    but increase overfitting risk for tree models.
    """
    drop: list[str] = []
    for col in list(X_train.columns):
        if col not in X_train.columns:
            continue
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            continue
        s = pd.to_numeric(X_train[col], errors="coerce")
        if int(s.notna().sum()) < int(min_non_na):
            continue
        try:
            v = float(s.var(skipna=True))
        except Exception:
            v = float("nan")
        if not np.isfinite(v) or v <= float(var_eps):
            drop.append(str(col))
    if not drop:
        return X_train, X_test, []
    Xt2 = X_train.drop(columns=drop, errors="ignore").copy()
    Xe2 = X_test.drop(columns=drop, errors="ignore").copy()
    return Xt2, Xe2, drop


def _training_quality_mask(X: pd.DataFrame) -> pd.Series:
    """
    Drop rows where too many core technicals are missing (reduces noisy / incomplete labels).
    """
    candidates = ("close", "rsi_14", "macd", "macd_signal", "atr", "volume_ratio")
    present = [c for c in candidates if c in X.columns]
    if len(present) < 2:
        return pd.Series(True, index=X.index)
    sub = X[present].apply(pd.to_numeric, errors="coerce")
    need = min(3, len(present))
    return sub.notna().sum(axis=1) >= int(need)


def _cv_from_y_series(y: pd.Series) -> int:
    counts: dict[int, int] = {}
    try:
        for v in list(y):
            iv = int(v)
            counts[iv] = counts.get(iv, 0) + 1
    except Exception:
        return 0
    min_class = min(counts.values()) if counts else 0
    return min(3, int(min_class)) if min_class else 0


def _clip_proba_1d(p: np.ndarray) -> np.ndarray:
    """Match inference bounds: stable Brier / logs and comparable thresholds."""
    x = np.asarray(p, dtype=float).reshape(-1)
    x = np.where(np.isfinite(x), x, 0.5)
    eps = 1e-6
    return np.clip(x, eps, 1.0 - eps)


def _fit_calibrated_pipeline(
    pipe: Pipeline,
    X,
    y,
    *,
    cv_n: int,
    cal_method: str,
) -> Pipeline | CalibratedClassifierCV:
    p = clone(pipe)
    if cv_n >= 2:
        try:
            n_fit = int(len(X))
        except Exception:
            n_fit = 0
        # Fewer disjoint calibrators when n is modest — less variance than a large ensemble.
        cal_ensemble: bool | str = False if n_fit < 380 else "auto"
        cal = CalibratedClassifierCV(
            p,
            method=str(cal_method),
            cv=int(cv_n),
            ensemble=cal_ensemble,
            n_jobs=1,
        )
        cal.fit(X, y)
        return cal
    p.fit(X, y)
    return p


def train_and_save(
    *,
    db_path: str,
    out_path: str,
    min_horizon_minutes: float = 15.0,
    min_rows: int = 50,
    action: str = "BUY",  # BUY | SHORT
    min_class_count: int = 30,
    dataset_source: str = "auto",  # auto | executed | signals | sim
    target_mode: str = "binary",
    min_edge_bps: float = 10.0,
) -> dict:
    act = (action or "").strip().upper() or "BUY"
    if act not in ("BUY", "SHORT"):
        act = "BUY"

    tm = (target_mode or "binary").strip().lower()
    if tm not in ("binary", "beat_fee_bps", "regression_r", "regression_return_pct"):
        tm = "binary"

    src = (dataset_source or "auto").strip().lower()
    if src not in ("auto", "executed", "signals", "sim"):
        src = "auto"

    ds_kind = "signals"
    X = None
    y = None
    meta = None

    if src == "sim":
        ds = build_sim_trade_dataset(
            db_path=db_path,
            actions=(act,),
            target_mode=tm,
            min_edge_bps=float(min_edge_bps),
        )
        X, y, meta = ds.X, ds.y, ds.meta
        ds_kind = "sim_trades"
    elif src == "signals":
        ds = build_signal_label_dataset(
            db_path=db_path,
            min_horizon_minutes=min_horizon_minutes,
            actions=(act,),
            target_mode=tm,
            min_edge_bps=float(min_edge_bps),
        )
        X, y, meta = ds.X, ds.y, ds.meta
        ds_kind = "signals"
    else:
        # Prefer executed-trade dataset (real fills) when enough trades exist.
        exec_ds = None
        if src in ("auto", "executed"):
            try:
                want_dir = "long" if act == "BUY" else "short"
                exec_ds = build_executed_trade_dataset(
                    db_path=db_path,
                    min_trades=max(10, int(min_rows)),
                    direction=want_dir,
                    target_mode=tm,
                    min_edge_bps=float(min_edge_bps),
                )
            except Exception:
                exec_ds = None

        if exec_ds is not None:
            X, y, meta = exec_ds.X, exec_ds.y, exec_ds.meta
            ds_kind = f"executed_trades:{'long' if act == 'BUY' else 'short'}"
        else:
            # Fall back to triple-barrier / forward-return labels from signals.
            sig_tm = "regression_return_pct" if tm == "regression_r" else tm
            ds = build_signal_label_dataset(
                db_path=db_path,
                min_horizon_minutes=min_horizon_minutes,
                actions=(act,),
                target_mode=sig_tm,
                min_edge_bps=float(min_edge_bps),
            )
            X, y, meta = ds.X, ds.y, ds.meta
            ds_kind = "signals"

    # Drop label rows with sparse core features (improves signal-to-noise vs training on NaN-heavy rows).
    try:
        qm = _training_quality_mask(X)
        if int(qm.sum()) >= int(min_rows):
            X = X.loc[qm].copy()
            y = y.loc[qm].copy()
            meta = meta.loc[qm].copy()
    except Exception:
        pass

    if len(X) < int(min_rows):
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "db_path": db_path,
            "min_horizon_minutes": float(min_horizon_minutes),
            "skipped": True,
            "skip_reason": "not_enough_labeled_rows",
            "n_labeled": int(len(X)),
            "min_required": int(min_rows),
            "dataset_kind": ds_kind,
            "action": act,
            "target_mode": tm,
        }
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    is_regression = tm in ("regression_r", "regression_return_pct")

    # Refuse weak datasets: class balance (classification) or near-constant targets (regression).
    if not is_regression:
        try:
            y_int = [int(v) for v in list(y)]
            n_pos = int(sum(1 for v in y_int if v == 1))
            n_neg = int(sum(1 for v in y_int if v == 0))
        except Exception:
            n_pos, n_neg = 0, 0
        if min(n_pos, n_neg) < int(min_class_count):
            outp = Path(out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "db_path": db_path,
                "min_horizon_minutes": float(min_horizon_minutes),
                "skipped": True,
                "skip_reason": "insufficient_class_balance",
                "n_labeled": int(len(X)),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "min_class_count": int(min_class_count),
                "dataset_kind": ds_kind,
                "action": act,
                "target_mode": tm,
            }
            (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
    else:
        ys = pd.to_numeric(y, errors="coerce")
        if int(ys.notna().sum()) < int(min_rows):
            outp = Path(out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "db_path": db_path,
                "min_horizon_minutes": float(min_horizon_minutes),
                "skipped": True,
                "skip_reason": "regression_insufficient_rows",
                "n_labeled": int(len(X)),
                "min_required": int(min_rows),
                "dataset_kind": ds_kind,
                "action": act,
                "target_mode": tm,
            }
            (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        if float(ys.std(skipna=True)) < 1e-10:
            outp = Path(out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "db_path": db_path,
                "min_horizon_minutes": float(min_horizon_minutes),
                "skipped": True,
                "skip_reason": "regression_near_constant_target",
                "n_labeled": int(len(X)),
                "dataset_kind": ds_kind,
                "action": act,
                "target_mode": tm,
            }
            (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload

    # Chronological split (reduce leakage): last 25% as test, with embargo to reduce overlapping labels.
    n = len(X)
    cut = max(10, int(n * 0.75))
    # Wider dead zone reduces leakage when labels span multiple bars (e.g. 15m TB horizons).
    try:
        ts = pd.to_datetime(meta["ts"], utc=True, errors="coerce")
    except Exception:
        ts = None

    cut_ts = None
    if ts is not None and len(ts) > cut:
        try:
            cut_ts = ts.iloc[cut]
            if getattr(cut_ts, "tzinfo", None) is None:
                cut_ts = None
        except Exception:
            cut_ts = None

    purge_minutes = 240.0
    if cut_ts is not None and ts is not None:
        gap = pd.Timedelta(minutes=float(purge_minutes))
        train_mask = ts < (cut_ts - gap)
        test_mask = ts > (cut_ts + gap)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    else:
        # Fallback: row-embargo (slightly wider vs row count when timestamps unavailable).
        embargo = min(80, max(8, int(n * 0.03)))
        cut2 = min(n, cut + embargo)
        X_train, y_train = X.iloc[:cut], y.iloc[:cut]
        X_test, y_test = X.iloc[cut2:], y.iloc[cut2:]

    if len(X_test) < 20:
        # If too small after purging, fall back to a simple chronological split (no dead zone).
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    X_train, X_test, dropped_var_cols = _drop_low_variance_numeric_columns(X_train, X_test)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if int(X_train.shape[1]) <= 0 or len(X_train) < int(min_rows):
        payload = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "db_path": db_path,
            "min_horizon_minutes": float(min_horizon_minutes),
            "skipped": True,
            "skip_reason": "insufficient_rows_or_features_after_column_pruning",
            "n_train": int(len(X_train)),
            "n_features": int(X_train.shape[1]),
            "min_required": int(min_rows),
            "dataset_kind": ds_kind,
            "action": act,
            "target_mode": tm,
            "dropped_constant_features": dropped_var_cols,
        }
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    if is_regression:
        y_train_f = pd.to_numeric(y_train, errors="coerce").astype(float)
        y_test_f = pd.to_numeric(y_test, errors="coerce").astype(float)
        reg_kw: dict = {
            "max_depth": 10,
            "max_iter": 750,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "random_state": 42,
        }
        if int(len(X_train)) >= 120:
            reg_kw = {
                **reg_kw,
                "early_stopping": True,
                "validation_fraction": 0.12,
                "n_iter_no_change": 35,
            }
        reg = HistGradientBoostingRegressor(**reg_kw)
        reg.fit(X_train, y_train_f)
        pred_te = reg.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test_f, pred_te))) if len(y_test_f) else float("nan")
        pred_tr = reg.predict(X_train)
        y_cls_tr = (y_train_f.values > 0.0).astype(int)
        y_cls_te = (y_test_f.values > 0.0).astype(int)
        best_thr = float(np.median(pred_tr)) if len(pred_tr) else 0.0
        best_f1_gate = -1.0
        best_prec_gate = 0.0
        try:
            qu = np.unique(np.quantile(pred_tr, np.linspace(0.05, 0.95, 29)))
            for thr in qu:
                if not np.isfinite(thr):
                    continue
                pred_bin = (pred_tr >= float(thr)).astype(int)
                if int(np.sum(pred_bin)) < 5:
                    continue
                f1g = float(f1_score(y_cls_tr, pred_bin, zero_division=0))
                pre_g = float(precision_score(y_cls_tr, pred_bin, zero_division=0))
                if f1g > best_f1_gate or (f1g == best_f1_gate and pre_g > best_prec_gate):
                    best_f1_gate = f1g
                    best_prec_gate = pre_g
                    best_thr = float(thr)
        except Exception:
            pass
        f1_test_at_thr = float(
            f1_score(y_cls_te, (pred_te >= float(best_thr)).astype(int), zero_division=0)
        )

        fi = None
        try:
            if hasattr(reg, "feature_importances_"):
                fi = sorted(
                    [
                        {"feature": c, "importance": float(v)}
                        for c, v in zip(list(X_train.columns), list(reg.feature_importances_))
                    ],
                    key=lambda r: -r["importance"],
                )[:30]
        except Exception:
            fi = None

        payload = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "db_path": db_path,
            "min_horizon_minutes": float(min_horizon_minutes),
            "provider": "hist_gbr",
            "dataset_kind": ds_kind,
            "action": act,
            "target_mode": tm,
            "min_edge_bps": float(min_edge_bps),
            "task": "regression",
            "metrics": {"rmse": rmse, "n": int(len(y_test_f))},
            "extra_metrics": {
                "gate_f1_positive_return_train_select": float(best_f1_gate),
                "gate_precision_positive_return_train_select": float(best_prec_gate),
                "gate_f1_positive_return_test_applied": float(f1_test_at_thr),
                "hist_gbr_early_stopping": bool(reg_kw.get("early_stopping", False)),
            },
            "recommended_regression_min": float(best_thr),
            "recommended_min_proba": None,
            "feature_columns": list(X_train.columns),
            "rows_seen": int(len(meta)),
            "explainability": fi,
            "dropped_constant_features": dropped_var_cols,
        }
        joblib.dump({"model": reg, "meta": payload}, outp)
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Holdout must contain both classes for meaningful ranking metrics (AUC / PR-AUC).
    try:
        if len(y_test) >= 8:
            yt_set = {int(v) for v in list(y_test)}
            if len(yt_set) < 2:
                payload = {
                    "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                    "db_path": db_path,
                    "min_horizon_minutes": float(min_horizon_minutes),
                    "skipped": True,
                    "skip_reason": "test_set_single_class",
                    "n_test": int(len(y_test)),
                    "dataset_kind": ds_kind,
                    "action": act,
                    "target_mode": tm,
                    "dropped_constant_features": dropped_var_cols,
                }
                (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
    except Exception:
        pass

    # If training has only one class, fall back to a constant-probability baseline.
    try:
        yuniq = set(int(v) for v in list(y_train))
    except Exception:
        yuniq = set()
    if len(yuniq) < 2:
        const = int(list(yuniq)[0]) if yuniq else 0
        dummy = DummyClassifier(strategy="constant", constant=const)
        dummy.fit(X_train, y_train)
        provider = "dummy"
        model = dummy
        metrics = TrainMetrics(
            n=int(len(y_test)),
            pos_rate=float(np.mean(y_test)) if len(y_test) else float(const),
            auc=None,
            acc=float(accuracy_score(y_test, (dummy.predict_proba(X_test)[:, 1] >= 0.5).astype(int)))
            if len(y_test)
            else 1.0,
        )
        payload = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "db_path": db_path,
            "min_horizon_minutes": float(min_horizon_minutes),
            "provider": provider,
            "dataset_kind": ds_kind,
            "action": act,
            "target_mode": tm,
            "task": "classification",
            "metrics": asdict(metrics),
            "extra_metrics": {},
            "recommended_min_proba": 0.5,
            "feature_columns": list(X_train.columns),
            "rows_seen": int(len(meta)),
            "explainability": None,
            "dropped_constant_features": dropped_var_cols,
        }
        joblib.dump({"model": model, "meta": payload}, outp)
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # ---- Unfitted classifier templates (clone + fit per stage) ----
    logreg = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=1,
                    class_weight="balanced",
                    C=0.25,
                    penalty="l2",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=20,
                    max_samples=0.85,
                    ccp_alpha=1e-4,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    lgbm_pipe: Pipeline | None = None
    try:
        import lightgbm as lgb

        lgbm_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    lgb.LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        colsample_bynode=0.78,
                        random_state=42,
                        class_weight="balanced",
                        min_child_samples=44,
                        reg_alpha=0.1,
                        reg_lambda=0.15,
                        extra_trees=True,
                        min_gain_to_split=0.02,
                    ),
                ),
            ]
        )
    except Exception:
        lgbm_pipe = None

    hgb_params: dict = {
        "max_depth": 7,
        "max_iter": 700,
        "learning_rate": 0.06,
        "l2_regularization": 1.0,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    # Early stopping needs enough rows for internal val split + folds inside calibration CV.
    if int(len(X_train)) >= 130:
        hgb_params = {
            **hgb_params,
            "early_stopping": True,
            "validation_fraction": 0.12,
            "n_iter_no_change": 30,
        }
    hgb_clf = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(**hgb_params)),
        ]
    )

    counts: dict[int, int] = {}
    try:
        for v in list(y_train):
            iv = int(v)
            counts[iv] = counts.get(iv, 0) + 1
    except Exception:
        counts = {}
    min_class = min(counts.values()) if counts else 0
    cv_full = min(3, int(min_class)) if min_class else 0
    cal_method = "sigmoid" if (len(X_train) < 450 or int(min_class) < 45) else "isotonic"

    use_inner = bool(len(X_train) >= 220)
    cut_sel = max(45, int(len(X_train) * 0.72))
    X_tr_src, y_tr_src = X_train, y_train
    X_va, y_va = None, None
    if use_inner and (len(X_train) - cut_sel) >= 35:
        X_tr_src = X_train.iloc[:cut_sel]
        y_tr_src = y_train.iloc[:cut_sel]
        X_va = X_train.iloc[cut_sel:]
        y_va = y_train.iloc[cut_sel:]
        try:
            if len({int(v) for v in list(y_tr_src)}) < 2 or len({int(v) for v in list(y_va)}) < 2:
                use_inner = False
                X_tr_src, y_tr_src = X_train, y_train
                X_va, y_va = None, None
        except Exception:
            use_inner = False
            X_tr_src, y_tr_src = X_train, y_train
            X_va, y_va = None, None
    else:
        use_inner = False

    cv_src = _cv_from_y_series(y_tr_src) if use_inner else cv_full

    base_model = _fit_calibrated_pipeline(logreg, X_tr_src, y_tr_src, cv_n=cv_src, cal_method=cal_method)
    rf_model = _fit_calibrated_pipeline(rf, X_tr_src, y_tr_src, cv_n=cv_src, cal_method=cal_method)
    lgbm_model_f = (
        _fit_calibrated_pipeline(lgbm_pipe, X_tr_src, y_tr_src, cv_n=cv_src, cal_method=cal_method)
        if lgbm_pipe is not None
        else None
    )
    hgb_model = _fit_calibrated_pipeline(hgb_clf, X_tr_src, y_tr_src, cv_n=cv_src, cal_method=cal_method)

    def eval_model(m, Xev=None, yev=None):
        Xh = X_test if Xev is None else Xev
        yh = y_test if yev is None else yev
        proba = _clip_proba_1d(m.predict_proba(Xh)[:, 1])
        pred = (proba >= 0.5).astype(int)
        tm = TrainMetrics(
            n=int(len(yh)),
            pos_rate=float(np.mean(yh)),
            auc=_safe_auc(yh, proba),
            acc=float(accuracy_score(yh, pred)),
        )
        cal = {}
        try:
            cal["brier"] = float(brier_score_loss(yh, proba))
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                msk = (proba >= lo) & (proba < hi)
                if int(np.sum(msk)) <= 0:
                    continue
                rows.append(
                    {
                        "bin": f"{lo:.1f}-{min(1.0, hi):.1f}",
                        "n": int(np.sum(msk)),
                        "p_mean": float(np.mean(proba[msk])),
                        "hit_rate": float(np.mean(np.asarray(yh)[msk])),
                    }
                )
            cal["calibration_bins"] = rows
        except Exception:
            cal = {}
        ap = _safe_average_precision(yh, proba)
        bac = None
        try:
            y_arr = np.asarray(yh, dtype=int)
            if len(np.unique(y_arr)) >= 2:
                bac = float(balanced_accuracy_score(y_arr, pred))
        except Exception:
            bac = None
        extra = {
            "average_precision": ap,
            "balanced_accuracy_at_0p5": bac,
            "precision_at_10pct": _precision_at_k(yh, proba, 0.10),
            "precision_at_15pct": _precision_at_k(yh, proba, 0.15),
            "precision_at_20pct": _precision_at_k(yh, proba, 0.20),
            "precision_at_30pct": _precision_at_k(yh, proba, 0.30),
            **cal,
        }
        return tm, extra, proba

    Xev, yev = (X_va, y_va) if use_inner else (None, None)
    m_log, extra_log, proba_log = eval_model(base_model, Xev, yev)
    best = ("logreg", base_model, m_log, extra_log, proba_log)

    def model_score(m: TrainMetrics, extra: dict) -> float:
        """
        Blend ROC-AUC, PR-AUC, Brier, and balanced accuracy at 0.5; fall back when pieces missing.
        """
        ap = extra.get("average_precision") if isinstance(extra, dict) else None
        br = extra.get("brier") if isinstance(extra, dict) else None
        bac = extra.get("balanced_accuracy_at_0p5") if isinstance(extra, dict) else None
        auc = m.auc
        if auc is not None and ap is not None and br is not None and np.isfinite(float(br)):
            s = 0.49 * float(auc) + 0.30 * float(ap) - 0.13 * float(br)
            if bac is not None and np.isfinite(float(bac)):
                s += 0.08 * float(bac)
            return s
        if auc is not None and ap is not None:
            return 0.62 * float(auc) + 0.38 * float(ap)
        if auc is not None:
            return float(auc)
        if ap is not None:
            return float(ap)
        return float(m.acc)

    m_rf, extra_rf, proba_rf = eval_model(rf_model, Xev, yev)
    if model_score(m_rf, extra_rf) >= model_score(best[2], best[3]):
        best = ("rf", rf_model, m_rf, extra_rf, proba_rf)
    if lgbm_model_f is not None:
        m_lgb, extra_lgb, proba_lgb = eval_model(lgbm_model_f, Xev, yev)
        if model_score(m_lgb, extra_lgb) >= model_score(best[2], best[3]):
            best = ("lgbm", lgbm_model_f, m_lgb, extra_lgb, proba_lgb)

    m_hgb, extra_hgb, proba_hgb = eval_model(hgb_model, Xev, yev)
    if model_score(m_hgb, extra_hgb) >= model_score(best[2], best[3]):
        best = ("hist_gbc", hgb_model, m_hgb, extra_hgb, proba_hgb)

    if use_inner:
        templates: dict[str, Pipeline] = {"logreg": logreg, "rf": rf, "hist_gbc": hgb_clf}
        if lgbm_pipe is not None:
            templates["lgbm"] = lgbm_pipe
        winner = str(best[0])
        model = _fit_calibrated_pipeline(templates[winner], X_train, y_train, cv_n=cv_full, cal_method=cal_method)
        provider = winner
        metrics, extra_metrics, proba_best = eval_model(model)
    else:
        provider, model, metrics, extra_metrics, proba_best = best[0], best[1], best[2], best[3], best[4]

    outp.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(extra_metrics, dict):
        extra_metrics = {
            **extra_metrics,
            "calibration_method": str(cal_method),
            "model_selection_inner_val": bool(use_inner),
            "model_selection_score_blend": "0.49*auc+0.30*ap-0.13*brier+0.08*balacc_when_all_finite_else_legacy",
            "calibration_ensemble": "false_lt_380_rows_else_auto",
            "hist_gbc_early_stopping": bool(hgb_params.get("early_stopping", False)),
            "logreg_standard_scaled": True,
        }

    # Threshold sweep on TRAIN scores only (avoids optimistic threshold fit on the same test used for AUC).
    proba_tr = _clip_proba_1d(model.predict_proba(X_train)[:, 1])
    best_thr = 0.55
    best_f1_tr = -1.0
    best_thr_precision_tr = 0.0
    best_thr_recall_tr = -1.0
    n_tr_thr = int(len(y_train))
    min_pred_per_class = max(8, int(0.02 * n_tr_thr))
    thr_grid = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

    def _threshold_sweep(require_min_support: bool) -> tuple[float, float, float, float]:
        b_thr, b_f1, b_prec, b_rec = 0.55, -1.0, 0.0, -1.0
        for thr in thr_grid:
            pred = (proba_tr >= thr).astype(int)
            if require_min_support:
                n_pos = int(np.sum(pred))
                n_neg = int(n_tr_thr - n_pos)
                if n_pos < min_pred_per_class or n_neg < min_pred_per_class:
                    continue
            f1_thr = float(f1_score(y_train, pred, zero_division=0))
            prec_thr = float(precision_score(y_train, pred, zero_division=0))
            rec_thr = float(recall_score(y_train, pred, zero_division=0))
            better = f1_thr > b_f1
            better |= f1_thr == b_f1 and prec_thr > b_prec
            better |= f1_thr == b_f1 and prec_thr == b_prec and rec_thr > b_rec
            if better:
                b_f1, b_prec, b_rec, b_thr = f1_thr, prec_thr, rec_thr, float(thr)
        return b_thr, b_f1, b_prec, b_rec

    thr_used_min_sup = True
    try:
        best_thr, best_f1_tr, best_thr_precision_tr, best_thr_recall_tr = _threshold_sweep(True)
        if best_f1_tr < 0.0:
            thr_used_min_sup = False
            best_thr, best_f1_tr, best_thr_precision_tr, best_thr_recall_tr = _threshold_sweep(False)
    except Exception:
        thr_used_min_sup = False
        pass
    try:
        f1_te_applied = float(f1_score(y_test, (proba_best >= float(best_thr)).astype(int), zero_division=0))
    except Exception:
        f1_te_applied = 0.0
    if isinstance(extra_metrics, dict):
        extra_metrics = {
            **extra_metrics,
            "recommended_threshold_f1_train_select": float(best_f1_tr),
            "recommended_threshold_precision_train_select": float(best_thr_precision_tr),
            "recommended_threshold_recall_train_select": float(best_thr_recall_tr),
            "recommended_threshold_f1_test_applied": float(f1_te_applied),
            "threshold_min_pred_per_class": int(min_pred_per_class),
            "threshold_min_support_constraint_applied": bool(thr_used_min_sup),
        }

    # Save simple explainability artifacts
    feature_importance = None
    try:
        clf = None
        if provider in ("lgbm", "hist_gbc"):
            # CalibratedClassifierCV -> base pipeline; uncalibrated hist_gbc is a bare Pipeline.
            est0 = getattr(model, "calibrated_classifiers_", None)
            if isinstance(est0, list) and est0:
                base = getattr(est0[0], "estimator", None)
                if base is not None and hasattr(base, "named_steps"):
                    clf = base.named_steps.get("clf")
            if clf is None and isinstance(model, Pipeline):
                clf = model.named_steps.get("clf")
            if clf is not None and hasattr(clf, "feature_importances_"):
                fi = list(getattr(clf, "feature_importances_"))
                feature_importance = sorted(
                    [{"feature": c, "importance": float(v)} for c, v in zip(list(X_train.columns), fi)],
                    key=lambda r: -r["importance"],
                )[:30]
        else:
            # LogisticRegression coefficients (approx; calibrated wrapper)
            est0 = getattr(model, "calibrated_classifiers_", [None])[0]
            base = getattr(est0, "estimator", None)
            clf_lr = None
            if base is not None and hasattr(base, "named_steps"):
                clf_lr = base.named_steps.get("clf")
            if clf_lr is not None and hasattr(clf_lr, "coef_"):
                coefs = list(clf_lr.coef_[0])
                pairs = [{"feature": c, "coef": float(v)} for c, v in zip(list(X_train.columns), coefs)]
                feature_importance = {
                    "top_positive": sorted(pairs, key=lambda r: -r["coef"])[:15],
                    "top_negative": sorted(pairs, key=lambda r: r["coef"])[:15],
                }
    except Exception:
        feature_importance = None

    payload = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "db_path": db_path,
        "min_horizon_minutes": float(min_horizon_minutes),
        "provider": provider,
        "dataset_kind": ds_kind,
        "action": act,
        "target_mode": tm,
        "min_edge_bps": float(min_edge_bps),
        "task": "classification",
        "metrics": asdict(metrics),
        "extra_metrics": extra_metrics,
        "recommended_min_proba": best_thr,
        "recommended_threshold_metric": "f1_then_precision_on_train",
        "recommended_threshold_f1_test": float(f1_te_applied),
        "feature_columns": list(X_train.columns),
        "rows_seen": int(len(meta)),
        "explainability": feature_importance,
        "dropped_constant_features": dropped_var_cols,
    }
    joblib.dump({"model": model, "meta": payload}, outp)
    (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to ledger.sqlite3")
    ap.add_argument("--out", required=True, help="Output model artifact path (joblib)")
    ap.add_argument("--action", default="BUY", help="Which side to train: BUY or SHORT")
    ap.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "executed", "signals", "sim"],
        help="Dataset source: auto (prefer executed), executed, signals (triple-barrier), or sim (sim_rollup).",
    )
    ap.add_argument("--min-horizon-minutes", type=float, default=15.0)
    ap.add_argument("--min-rows", type=int, default=200, help="Minimum labeled rows required to train")
    ap.add_argument(
        "--min-class-count",
        type=int,
        default=30,
        help="Minimum examples required for BOTH classes (pos/neg) to train (avoid fitting noise).",
    )
    ap.add_argument(
        "--target-mode",
        default="binary",
        choices=["binary", "beat_fee_bps", "regression_r", "regression_return_pct"],
        help="Label/target for supervised training.",
    )
    ap.add_argument(
        "--min-edge-bps",
        type=float,
        default=10.0,
        help="For beat_fee_bps: require PnL / return above this many basis points (friction proxy).",
    )
    args = ap.parse_args()

    meta = train_and_save(
        db_path=args.db,
        out_path=args.out,
        min_horizon_minutes=float(args.min_horizon_minutes),
        min_rows=int(args.min_rows),
        action=str(args.action),
        min_class_count=int(args.min_class_count),
        dataset_source=str(args.dataset),
        target_mode=str(args.target_mode),
        min_edge_bps=float(args.min_edge_bps),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

