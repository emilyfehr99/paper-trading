from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpaca_day_bot.data.stream import BarEvent


@dataclass(frozen=True)
class FeatureVector:
    symbol: str
    close: float
    mom_5: float
    mom_15: float
    vol_15: float
    atr_14: float
    rsi_14: float


def compute_features(symbol: str, bars: list[BarEvent]) -> FeatureVector | None:
    # Need enough bars for 15-min window + ATR/RSI.
    if len(bars) < 20:
        return None

    closes = np.asarray([b.close for b in bars], dtype=float)
    highs = np.asarray([b.high for b in bars], dtype=float)
    lows = np.asarray([b.low for b in bars], dtype=float)

    close = float(closes[-1])

    def _mom(k: int) -> float:
        if len(closes) <= k or closes[-k - 1] == 0:
            return 0.0
        return float(closes[-1] / closes[-k - 1] - 1.0)

    mom_5 = _mom(5)
    mom_15 = _mom(15)

    # Volatility proxy: std of 1-bar log returns over last 15 bars
    rets = np.diff(np.log(np.clip(closes[-16:], 1e-12, None)))
    vol_15 = float(np.std(rets)) if rets.size else 0.0

    # ATR(14)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr_14 = float(np.mean(tr[-14:])) if tr.size >= 14 else float(np.mean(tr))

    # RSI(14) (simple average version)
    deltas = np.diff(closes)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    if gains.size >= 14:
        avg_gain = float(np.mean(gains[-14:]))
        avg_loss = float(np.mean(losses[-14:]))
    else:
        avg_gain = float(np.mean(gains)) if gains.size else 0.0
        avg_loss = float(np.mean(losses)) if losses.size else 0.0
    if avg_loss <= 1e-12:
        rsi_14 = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_14 = float(100.0 - (100.0 / (1.0 + rs)))

    return FeatureVector(
        symbol=symbol,
        close=close,
        mom_5=mom_5,
        mom_15=mom_15,
        vol_15=vol_15,
        atr_14=atr_14,
        rsi_14=rsi_14,
    )

