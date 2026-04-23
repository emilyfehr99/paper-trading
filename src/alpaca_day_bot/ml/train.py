from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from alpaca_day_bot.ml.dataset import build_signal_label_dataset


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
) -> dict:
    ds = build_signal_label_dataset(
        db_path=db_path, min_horizon_minutes=min_horizon_minutes, actions=("BUY", "SHORT")
    )
    X = ds.X
    y = ds.y
    meta = ds.meta

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
        }
        (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Chronological split (reduce leakage): last 25% as test, with a small embargo to reduce overlap.
    n = len(X)
    cut = max(10, int(n * 0.75))
    embargo = min(50, max(5, int(n * 0.02)))
    cut2 = min(n, cut + embargo)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_test, y_test = X.iloc[cut2:], y.iloc[cut2:]
    if len(X_test) < 20:
        # If too small after embargo, fall back to no embargo.
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    # Baseline calibrated logistic regression
    logreg = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
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
        extra = {
            "precision_at_10pct": _precision_at_k(y_test, proba, 0.10),
            "precision_at_20pct": _precision_at_k(y_test, proba, 0.20),
            "precision_at_30pct": _precision_at_k(y_test, proba, 0.30),
            **cal,
        }
        return tm, extra, proba

    m_log, extra_log, proba_log = eval_model(base_model)
    best = ("logreg", base_model, m_log, extra_log, proba_log)
    if lgbm_cal is not None:
        m_lgb, extra_lgb, proba_lgb = eval_model(lgbm_cal)
        # choose by AUC when available, otherwise accuracy
        def score(m: TrainMetrics) -> float:
            return (m.auc if m.auc is not None else m.acc)

        if score(m_lgb) >= score(m_log):
            best = ("lgbm", lgbm_cal, m_lgb, extra_lgb, proba_lgb)

    provider, model, metrics, extra_metrics, proba_best = best
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Simple threshold sweep for operational use (pick threshold maximizing accuracy on test).
    best_thr = 0.55
    best_acc = -1.0
    try:
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            pred = (proba_best >= thr).astype(int)
            acc_thr = float(accuracy_score(y_test, pred))
            if acc_thr >= best_acc:
                best_acc = acc_thr
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
        "metrics": asdict(metrics),
        "extra_metrics": extra_metrics,
        "recommended_min_proba": best_thr,
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
    ap.add_argument("--min-horizon-minutes", type=float, default=15.0)
    ap.add_argument("--min-rows", type=int, default=50, help="Minimum labeled rows required to train")
    args = ap.parse_args()

    meta = train_and_save(
        db_path=args.db,
        out_path=args.out,
        min_horizon_minutes=float(args.min_horizon_minutes),
        min_rows=int(args.min_rows),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

