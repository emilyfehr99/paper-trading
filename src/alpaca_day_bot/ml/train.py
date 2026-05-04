from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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


def train_and_save(
    *,
    db_path: str,
    out_path: str,
    min_horizon_minutes: float = 15.0,
    min_rows: int = 50,
    action: str = "BUY",  # BUY | SHORT
    min_class_count: int = 30,
    dataset_source: str = "auto",  # auto | executed | signals | sim
) -> dict:
    act = (action or "").strip().upper() or "BUY"
    if act not in ("BUY", "SHORT"):
        act = "BUY"

    src = (dataset_source or "auto").strip().lower()
    if src not in ("auto", "executed", "signals", "sim"):
        src = "auto"

    ds_kind = "signals"
    X = None
    y = None
    meta = None

    if src == "sim":
        ds = build_sim_trade_dataset(db_path=db_path, actions=(act,))
        X, y, meta = ds.X, ds.y, ds.meta
        ds_kind = "sim_trades"
    elif src == "signals":
        ds = build_signal_label_dataset(db_path=db_path, min_horizon_minutes=min_horizon_minutes, actions=(act,))
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
                )
            except Exception:
                exec_ds = None

        if exec_ds is not None:
            X, y, meta = exec_ds.X, exec_ds.y, exec_ds.meta
            ds_kind = f"executed_trades:{'long' if act == 'BUY' else 'short'}"
        else:
            # Fall back to triple-barrier / forward-return labels from signals.
            ds = build_signal_label_dataset(db_path=db_path, min_horizon_minutes=min_horizon_minutes, actions=(act,))
            X, y, meta = ds.X, ds.y, ds.meta
            ds_kind = "signals"

    if len(X) < int(min_rows):
        # In early deployment we may have 0–few labeled rows. Treat this as a
        # successful no-op so scheduled training doesn't fail the workflow.
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
        }
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Refuse to train on one-class or tiny-class datasets (noise fitting hurts accuracy).
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
        }
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Chronological split (reduce leakage): last 25% as test, with a small embargo to reduce overlap.
    n = len(X)
    cut = max(10, int(n * 0.75))
    # Time-series purging: enforce a 60-minute "dead zone" around the split timestamp.
    try:
        import pandas as pd

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

    purge_minutes = 60.0
    if cut_ts is not None and ts is not None:
        gap = pd.Timedelta(minutes=float(purge_minutes))
        train_mask = ts < (cut_ts - gap)
        test_mask = ts > (cut_ts + gap)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    else:
        # Fallback: row-embargo.
        embargo = min(50, max(5, int(n * 0.02)))
        cut2 = min(n, cut + embargo)
        X_train, y_train = X.iloc[:cut], y.iloc[:cut]
        X_test, y_test = X.iloc[cut2:], y.iloc[cut2:]

    if len(X_test) < 20:
        # If too small after purging, fall back to a simple chronological split (no dead zone).
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]

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
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "db_path": db_path,
            "min_horizon_minutes": float(min_horizon_minutes),
            "provider": provider,
            "dataset_kind": ds_kind,
            "action": act,
            "metrics": asdict(metrics),
            "extra_metrics": {},
            "recommended_min_proba": 0.5,
            "feature_columns": list(X.columns),
            "rows_seen": int(len(meta)),
            "explainability": None,
        }
        joblib.dump({"model": model, "meta": payload}, outp)
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Baseline calibrated logistic regression
    logreg = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Meta-labeling model (nonlinear): RandomForest on success/fail history
    rf = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    # Calibration needs enough samples per class. For very small datasets, skip calibration.
    counts: dict[int, int] = {}
    try:
        for v in list(y_train):
            iv = int(v)
            counts[iv] = counts.get(iv, 0) + 1
    except Exception:
        counts = {}
    min_class = min(counts.values()) if counts else 0
    cv = min(3, int(min_class)) if min_class else 0
    if cv >= 2:
        cal = CalibratedClassifierCV(logreg, method="isotonic", cv=cv)
        cal.fit(X_train, y_train)
        base_model = cal
    else:
        logreg.fit(X_train, y_train)
        base_model = logreg

    # RF calibration only if enough samples per class; else fit raw.
    if cv >= 2:
        rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv=cv)
        rf_cal.fit(X_train, y_train)
        rf_model = rf_cal
    else:
        rf.fit(X_train, y_train)
        rf_model = rf

    # LightGBM (if available)
    lgbm_model = None
    lgbm_cal = None
    try:
        import lightgbm as lgb

        lgbm_model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight="balanced",
        )
        lgbm_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("clf", lgbm_model),
            ]
        )
        if cv >= 2:
            lgbm_cal = CalibratedClassifierCV(lgbm_pipe, method="isotonic", cv=cv)
            lgbm_cal.fit(X_train, y_train)
        else:
            lgbm_pipe.fit(X_train, y_train)
            lgbm_cal = lgbm_pipe
    except Exception:
        lgbm_cal = None

    def eval_model(m):
        proba = m.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tm = TrainMetrics(
            n=int(len(y_test)),
            pos_rate=float(np.mean(y_test)),
            auc=_safe_auc(y_test, proba),
            acc=float(accuracy_score(y_test, pred)),
        )
        cal = {}
        try:
            cal["brier"] = float(brier_score_loss(y_test, proba))
            # 5 bucket calibration summary: avg p vs empirical hit-rate
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
                        "hit_rate": float(np.mean(np.asarray(y_test)[msk])),
                    }
                )
            cal["calibration_bins"] = rows
        except Exception:
            cal = {}
        ap = _safe_average_precision(y_test, proba)
        extra = {
            "average_precision": ap,
            "precision_at_10pct": _precision_at_k(y_test, proba, 0.10),
            "precision_at_20pct": _precision_at_k(y_test, proba, 0.20),
            "precision_at_30pct": _precision_at_k(y_test, proba, 0.30),
            **cal,
        }
        return tm, extra, proba

    m_log, extra_log, proba_log = eval_model(base_model)
    best = ("logreg", base_model, m_log, extra_log, proba_log)

    def model_score(m: TrainMetrics, extra: dict) -> float:
        """Prefer ROC-AUC; if undefined (degenerate labels), use PR-AUC then accuracy."""
        if m.auc is not None:
            return float(m.auc)
        ap = extra.get("average_precision") if isinstance(extra, dict) else None
        if ap is not None:
            return float(ap)
        return float(m.acc)

    m_rf, extra_rf, proba_rf = eval_model(rf_model)
    if model_score(m_rf, extra_rf) >= model_score(best[2], best[3]):
        best = ("rf", rf_model, m_rf, extra_rf, proba_rf)
    if lgbm_cal is not None:
        m_lgb, extra_lgb, proba_lgb = eval_model(lgbm_cal)
        if model_score(m_lgb, extra_lgb) >= model_score(best[2], best[3]):
            best = ("lgbm", lgbm_cal, m_lgb, extra_lgb, proba_lgb)

    provider, model, metrics, extra_metrics, proba_best = best
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Threshold sweep for operational gating: prefer F1 (class imbalance), then precision.
    best_thr = 0.55
    best_f1 = -1.0
    best_thr_precision = 0.0
    try:
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            pred = (proba_best >= thr).astype(int)
            f1_thr = float(f1_score(y_test, pred, zero_division=0))
            prec_thr = float(precision_score(y_test, pred, zero_division=0))
            if f1_thr > best_f1 or (f1_thr == best_f1 and prec_thr > best_thr_precision):
                best_f1 = f1_thr
                best_thr_precision = prec_thr
                best_thr = float(thr)
    except Exception:
        pass

    # Save simple explainability artifacts
    feature_importance = None
    try:
        if provider == "lgbm":
            # CalibratedClassifierCV -> base_estimator pipeline in cv estimators; use the first fitted one.
            est0 = getattr(model, "calibrated_classifiers_", [None])[0]
            base = getattr(est0, "estimator", None)
            clf = None
            if base is not None and hasattr(base, "named_steps"):
                clf = base.named_steps.get("clf")
            if clf is not None and hasattr(clf, "feature_importances_"):
                fi = list(getattr(clf, "feature_importances_"))
                feature_importance = sorted(
                    [{"feature": c, "importance": float(v)} for c, v in zip(list(X.columns), fi)],
                    key=lambda r: -r["importance"],
                )[:30]
        else:
            # LogisticRegression coefficients (approx; calibrated wrapper)
            est0 = getattr(model, "calibrated_classifiers_", [None])[0]
            base = getattr(est0, "estimator", None)
            clf = None
            if base is not None and hasattr(base, "named_steps"):
                clf = base.named_steps.get("clf")
            if clf is not None and hasattr(clf, "coef_"):
                coefs = list(clf.coef_[0])
                pairs = [{"feature": c, "coef": float(v)} for c, v in zip(list(X.columns), coefs)]
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
        "metrics": asdict(metrics),
        "extra_metrics": extra_metrics,
        "recommended_min_proba": best_thr,
        "recommended_threshold_metric": "f1_then_precision",
        "recommended_threshold_f1_test": best_f1,
        "feature_columns": list(X.columns),
        "rows_seen": int(len(meta)),
        "explainability": feature_importance,
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
    args = ap.parse_args()

    meta = train_and_save(
        db_path=args.db,
        out_path=args.out,
        min_horizon_minutes=float(args.min_horizon_minutes),
        min_rows=int(args.min_rows),
        action=str(args.action),
        min_class_count=int(args.min_class_count),
        dataset_source=str(args.dataset),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

