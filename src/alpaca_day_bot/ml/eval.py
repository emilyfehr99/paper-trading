from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from alpaca_day_bot.ml.dataset import build_signal_label_dataset


@dataclass(frozen=True)
class WalkForwardFold:
    train_n: int
    test_n: int
    test_acc: float | None


def quick_walk_forward_eval(*, db_path: str, min_horizon_minutes: float = 15.0) -> list[WalkForwardFold]:
    """
    Lightweight walk-forward on the labeled signals dataset (not bar backtests).
    Uses logistic regression (fast) to provide an early signal about model viability.
    """
    ds = build_signal_label_dataset(db_path=db_path, min_horizon_minutes=min_horizon_minutes, actions=("BUY",))
    X, y, meta = ds.X, ds.y, ds.meta
    if len(X) < 100:
        return [WalkForwardFold(train_n=int(len(X)), test_n=0, test_acc=None)]

    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3)

    # 3 folds time-ordered (60/20, 70/20, 80/20)
    n = len(X)
    folds = []
    for frac in (0.60, 0.70, 0.80):
        cut = int(n * frac)
        test_end = min(n, cut + int(n * 0.20))
        X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
        X_te, y_te = X.iloc[cut:test_end], y.iloc[cut:test_end]
        if len(X_te) < 20 or len(set(y_tr)) < 2:
            folds.append(WalkForwardFold(train_n=int(len(X_tr)), test_n=int(len(X_te)), test_acc=None))
            continue
        cal.fit(X_tr, y_tr)
        proba = cal.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        folds.append(
            WalkForwardFold(
                train_n=int(len(X_tr)),
                test_n=int(len(X_te)),
                test_acc=float(accuracy_score(y_te, pred)),
            )
        )
    return folds

