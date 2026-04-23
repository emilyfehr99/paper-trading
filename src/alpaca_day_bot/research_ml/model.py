from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetaModel:
    primary: object
    safety: object
    feature_columns: list[str]


def _tb_to_trend_label(y_tb: pd.Series) -> pd.Series:
    """
    Primary model predicts direction (trend):
    - 1 for tp
    - 0 for sl/timeout
    """
    y = (y_tb.astype(int) == 1).astype(int)
    y.name = "y_trend"
    return y


def train_meta_labeling(
    X: pd.DataFrame,
    y_tb: pd.Series,
    *,
    safety_threshold: float = 0.65,
    random_state: int = 42,
) -> tuple[MetaModel, pd.Series]:
    """
    Meta-labeling:
    - Primary model (RandomForest) predicts trend direction.
    - Secondary model (XGBoost) predicts whether the primary prediction is correct.
    - Returns (trained bundle, safety_proba_series aligned to X).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("xgboost is required for the safety filter") from e

    X = X.copy()
    y_tb = y_tb.reindex(X.index)

    y_trend = _tb_to_trend_label(y_tb)

    # Primary
    primary = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    primary.fit(X, y_trend)
    p1 = pd.Series(primary.predict_proba(X)[:, 1], index=X.index, name="p_primary")
    pred1 = (p1 >= 0.5).astype(int)

    # Safety label = primary correctness
    y_safe = (pred1 == y_trend.astype(int)).astype(int)

    # Safety features = original features + primary probability
    X2 = X.copy()
    X2["p_primary"] = p1

    safety = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=int(random_state),
                    n_jobs=1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    safety.fit(X2, y_safe)
    p_safe = pd.Series(safety.predict_proba(X2)[:, 1], index=X.index, name="p_safe")

    bundle = MetaModel(primary=primary, safety=safety, feature_columns=list(X.columns))
    # trading rule: only “trade” when p_safe > threshold AND primary predicts up
    trade_mask = (p_safe >= float(safety_threshold)) & (pred1 == 1)
    trade_mask.name = "trade_mask"
    return bundle, trade_mask


def predict_trade_mask(
    bundle: MetaModel,
    X: pd.DataFrame,
    *,
    safety_threshold: float = 0.65,
) -> pd.Series:
    X = X.copy()
    # align columns
    for c in bundle.feature_columns:
        if c not in X.columns:
            X[c] = np.nan
    X = X[bundle.feature_columns]

    p1 = pd.Series(bundle.primary.predict_proba(X)[:, 1], index=X.index)
    pred1 = (p1 >= 0.5).astype(int)
    X2 = X.copy()
    X2["p_primary"] = p1
    p_safe = pd.Series(bundle.safety.predict_proba(X2)[:, 1], index=X.index)
    trade_mask = (p_safe >= float(safety_threshold)) & (pred1 == 1)
    trade_mask.name = "trade_mask"
    return trade_mask

