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
from sklearn.metrics import accuracy_score, roc_auc_score
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


def train_and_save(*, db_path: str, out_path: str, min_horizon_minutes: float = 15.0) -> dict:
    ds = build_signal_label_dataset(db_path=db_path, min_horizon_minutes=min_horizon_minutes, actions=("BUY",))
    X = ds.X
    y = ds.y
    meta = ds.meta

    if len(X) < 50:
        raise SystemExit(f"Not enough labeled rows to train (n={len(X)}).")

    # Chronological split (reduce leakage): last 25% as test.
    n = len(X)
    cut = max(10, int(n * 0.75))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    # Baseline calibrated logistic regression
    logreg = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )
    cal = CalibratedClassifierCV(logreg, method="isotonic", cv=3)
    cal.fit(X_train, y_train)

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
        lgbm_cal = CalibratedClassifierCV(lgbm_pipe, method="isotonic", cv=3)
        lgbm_cal.fit(X_train, y_train)
    except Exception:
        lgbm_cal = None

    def eval_model(m):
        proba = m.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return TrainMetrics(
            n=int(len(y_test)),
            pos_rate=float(np.mean(y_test)),
            auc=_safe_auc(y_test, proba),
            acc=float(accuracy_score(y_test, pred)),
        )

    m_log = eval_model(cal)
    best = ("logreg", cal, m_log)
    if lgbm_cal is not None:
        m_lgb = eval_model(lgbm_cal)
        # choose by AUC when available, otherwise accuracy
        def score(m: TrainMetrics) -> float:
            return (m.auc if m.auc is not None else m.acc)

        if score(m_lgb) >= score(m_log):
            best = ("lgbm", lgbm_cal, m_lgb)

    provider, model, metrics = best
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Simple threshold sweep for operational use (pick threshold maximizing accuracy on test).
    best_thr = 0.55
    best_acc = -1.0
    try:
        proba_best = model.predict_proba(X_test)[:, 1]
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            pred = (proba_best >= thr).astype(int)
            acc_thr = float(accuracy_score(y_test, pred))
            if acc_thr >= best_acc:
                best_acc = acc_thr
                best_thr = float(thr)
    except Exception:
        pass

    payload = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "db_path": db_path,
        "min_horizon_minutes": float(min_horizon_minutes),
        "provider": provider,
        "metrics": asdict(metrics),
        "recommended_min_proba": best_thr,
        "feature_columns": list(X.columns),
        "rows_seen": int(len(meta)),
    }
    joblib.dump({"model": model, "meta": payload}, outp)
    (outp.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to ledger.sqlite3")
    ap.add_argument("--out", required=True, help="Output model artifact path (joblib)")
    ap.add_argument("--min-horizon-minutes", type=float, default=15.0)
    args = ap.parse_args()

    meta = train_and_save(db_path=args.db, out_path=args.out, min_horizon_minutes=float(args.min_horizon_minutes))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

