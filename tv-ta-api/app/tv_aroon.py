"""TradingView Pine Aroon — exact ``f_aroon`` from MES_MNQ_5m_Research_v3.pine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bars_since_extreme(arr: np.ndarray, *, find_max: bool) -> float:
    if len(arr) == 0:
        return float("nan")
    idx = int(np.argmax(arr)) if find_max else int(np.argmin(arr))
    return float(len(arr) - 1 - idx)


def tv_aroon(high: pd.Series, low: pd.Series, *, length: int = 14) -> pd.DataFrame:
    h = high.astype(float)
    l = low.astype(float)
    win = int(length) + 1

    def _roll_extreme(s: pd.Series, *, find_max: bool) -> pd.Series:
        return s.rolling(win, min_periods=1).apply(
            lambda x: _bars_since_extreme(x, find_max=find_max),
            raw=True,
        )

    bars_since_high = _roll_extreme(h, find_max=True)
    bars_since_low = _roll_extreme(l, find_max=False)
    highestbars = -bars_since_high
    lowestbars = -bars_since_low
    up = 100.0 * (length + highestbars) / float(length)
    dn = 100.0 * (length + lowestbars) / float(length)

    out = pd.DataFrame(index=h.index)
    out["aroon_up"] = up
    out["aroon_down"] = dn
    out["aroon_osc"] = up - dn
    return out
