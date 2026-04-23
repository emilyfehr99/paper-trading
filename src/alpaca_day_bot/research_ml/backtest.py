from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpaca_day_bot.research_ml.labeling import build_dataset
from alpaca_day_bot.research_ml.model import MetaModel, predict_trade_mask, train_meta_labeling


@dataclass(frozen=True)
class WalkForwardResult:
    train_rows: int
    test_rows: int
    trades: int
    total_return: float | None
    sharpe: float | None
    # Accuracy on the hidden holdout (direction = TP vs not-TP)
    primary_acc: float | None
    # Among bars the safety filter allows ("trades"), how often primary is correct
    trade_acc: float | None
    # Fraction of holdout bars allowed to trade (coverage)
    trade_coverage: float | None


def _vectorbt_portfolio_from_signals(close: pd.Series, entries: pd.Series) -> WalkForwardResult:
    try:
        import vectorbt as vbt
    except Exception as e:
        raise RuntimeError("vectorbt is required for research backtest") from e

    # Simplest backtest: long-only, enter on True, exit after 1 bar (daytrading-like).
    # This is intentionally conservative and stable; we can refine later.
    entries = entries.astype(bool).reindex(close.index).fillna(False)
    exits = entries.shift(1).fillna(False)  # hold 1 bar

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        freq="1T",
        fees=0.0005,  # 5 bps
        slippage=0.0005,
        init_cash=1.0,
        direction="longonly",
    )
    stats = pf.stats()
    tr = float(stats.get("Total Return [%]", np.nan)) / 100.0
    sh = float(stats.get("Sharpe Ratio", np.nan))
    return WalkForwardResult(
        train_rows=0,
        test_rows=int(len(close)),
        trades=int(stats.get("Total Trades", 0)),
        total_return=None if not np.isfinite(tr) else tr,
        sharpe=None if not np.isfinite(sh) else sh,
        primary_acc=None,
        trade_acc=None,
        trade_coverage=None,
    )


def walk_forward_vectorbt(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    max_bars: int = 5,
    safety_threshold: float = 0.65,
) -> tuple[MetaModel, WalkForwardResult]:
    """
    Walk-forward with a fixed 70/30 chronological split:
    - Train on first 70%
    - Evaluate on last 30% (hidden holdout)
    """
    ds = build_dataset(df, tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars)
    X = ds.X
    y_tb = ds.y_tb

    # align and drop NaNs for training stability
    ok = X.notna().all(axis=1) & y_tb.notna()
    X = X[ok]
    y_tb = y_tb[ok]

    n = len(X)
    if n < 200:
        raise ValueError(f"Not enough rows for walk-forward (n={n}); use more history or higher timeframe.")

    cut = int(n * float(train_frac))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y_tb.iloc[:cut], y_tb.iloc[cut:]

    bundle, _mask_tr = train_meta_labeling(X_tr, y_tr, safety_threshold=safety_threshold)
    mask_te = predict_trade_mask(bundle, X_te, safety_threshold=safety_threshold)

    # Holdout accuracy projection
    y_te_dir = (y_te.astype(int) == 1).astype(int)  # TP=1 else 0
    p1 = pd.Series(bundle.primary.predict_proba(X_te)[:, 1], index=X_te.index)
    pred1 = (p1 >= 0.5).astype(int)
    primary_acc = float((pred1 == y_te_dir).mean()) if len(y_te_dir) else None

    trade_cov = float(mask_te.mean()) if len(mask_te) else None
    trade_acc = None
    if bool(mask_te.any()):
        trade_acc = float((pred1[mask_te] == y_te_dir[mask_te]).mean())

    close_te = df.loc[mask_te.index, "close"].astype(float)
    res_pf = _vectorbt_portfolio_from_signals(close_te, mask_te)
    res = WalkForwardResult(
        train_rows=int(len(X_tr)),
        test_rows=int(len(X_te)),
        trades=res_pf.trades,
        total_return=res_pf.total_return,
        sharpe=res_pf.sharpe,
        primary_acc=primary_acc,
        trade_acc=trade_acc,
        trade_coverage=trade_cov,
    )
    return bundle, res

