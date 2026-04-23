from __future__ import annotations

from dataclasses import dataclass

from alpaca_day_bot.ml.executed_dataset import build_executed_trade_dataset


@dataclass(frozen=True)
class ExecutedMlSummary:
    n: int
    win_rate: float | None
    total_pnl: float | None


def executed_ml_summary(db_path: str) -> ExecutedMlSummary | None:
    ds = build_executed_trade_dataset(db_path=db_path, min_trades=1)
    if ds is None or ds.meta is None or ds.meta.empty:
        return None
    try:
        n = int(len(ds.meta))
        pnls = [float(x) for x in list(ds.meta["pnl"]) if x is not None]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / len(pnls)) if pnls else None
        total = float(sum(pnls)) if pnls else None
        return ExecutedMlSummary(n=n, win_rate=win_rate, total_pnl=total)
    except Exception:
        return None

