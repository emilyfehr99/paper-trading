"""ML label helpers: binary PnL, fee-aware binary, and continuous regression targets."""

from __future__ import annotations

import math
from typing import Any


def binary_win(pnl: float) -> int:
    return 1 if float(pnl) > 0.0 else 0


def beat_fee_binary(*, pnl: float, notional_abs: float, min_edge_bps: float, buffer: float = 1.0) -> int:
    """y=1 iff realized PnL clears a simple bps-of-notional friction floor."""
    n = abs(float(notional_abs))
    if n <= 1e-12:
        return binary_win(pnl)
    edge_usd = n * (float(min_edge_bps) / 10000.0) * float(buffer)
    return 1 if float(pnl) > float(edge_usd) else 0


def return_pct_beats_fee(return_pct: float, min_edge_bps: float) -> int:
    """
    return_pct is in *percent points* (e.g. 0.15 means +0.15% return).
    min_edge_bps in basis points (10 bps => require >= +0.10 percent points).
    """
    return 1 if float(return_pct) >= (float(min_edge_bps) / 100.0) else 0


def signal_binary_from_tb_or_return(
    *,
    tb_outcome: str | None,
    return_pct: float,
    min_edge_bps: float | None = None,
    beat_fee: bool = False,
) -> int:
    """Label for signal rows: TB outcome when present; else return vs (optional) fee floor."""
    tb = (tb_outcome or "").strip().lower() if isinstance(tb_outcome, str) else ""
    if tb in ("tp", "sl", "timeout"):
        if beat_fee and tb == "timeout":
            if min_edge_bps is None:
                return 1 if float(return_pct) > 0.0 else 0
            return return_pct_beats_fee(return_pct, float(min_edge_bps))
        return 1 if tb == "tp" else 0
    if beat_fee:
        if min_edge_bps is None:
            return 1 if float(return_pct) > 0.0 else 0
        return return_pct_beats_fee(return_pct, float(min_edge_bps))
    return 1 if float(return_pct) > 0.0 else 0


def risk_usd_from_signal_features(
    feat: dict[str, Any],
    *,
    entry_px: float,
    qty: float,
    direction: str,
) -> float:
    """
    Risk $ for R-multiple: prefer bracket SL distance * qty; else 2*ATR*qty; else 0.5% of notional.
    direction: 'long' | 'short'
    """
    ep = float(entry_px)
    q = abs(float(qty))
    d = (direction or "").strip().lower()
    sl_raw = feat.get("sl_price")
    try:
        sl = float(sl_raw) if sl_raw is not None else float("nan")
        if math.isfinite(sl) and math.isfinite(ep) and q > 0:
            if d == "long" and sl < ep:
                return max(1e-6, (ep - sl) * q)
            if d == "short" and sl > ep:
                return max(1e-6, (sl - ep) * q)
    except Exception:
        pass
    atr_raw = feat.get("atr")
    try:
        atr = float(atr_raw)
        if math.isfinite(atr) and atr > 0 and q > 0:
            return max(1e-6, 2.0 * atr * q)
    except Exception:
        pass
    return max(1e-6, abs(ep * q) * 0.005)


def regression_r_multiple(pnl: float, feat: dict[str, Any], *, entry_px: float, qty: float, direction: str) -> float:
    r_usd = risk_usd_from_signal_features(feat, entry_px=entry_px, qty=qty, direction=direction)
    return float(pnl) / float(r_usd)


def regression_return_pct_from_trade(pnl: float, notional_abs: float) -> float:
    n = abs(float(notional_abs))
    if n <= 1e-12:
        return 0.0
    return float(pnl) / n * 100.0
