from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from alpaca_day_bot.reporting.trades import Fill, _rows_to_fills, reconstruct_round_trips
from alpaca_day_bot.ml.infer import _flatten_feature_dict


@dataclass(frozen=True)
class ExecutedDatasetResult:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # symbol, entry_ts, exit_ts, direction, pnl


def _parse_iso_dt(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _load_signals(conn: sqlite3.Connection) -> list[tuple[datetime, str, str, str]]:
    rows = conn.execute(
        """
        SELECT ts, symbol, reason, features_json
        FROM signals
        ORDER BY ts ASC
        """
    ).fetchall()
    out: list[tuple[datetime, str, str, str]] = []
    for ts_s, sym, reason, feat_json in rows:
        ts = _parse_iso_dt(ts_s) or datetime.now(tz=timezone.utc)
        out.append((ts, str(sym), str(reason or ""), str(feat_json or "")))
    return out


def _nearest_signal_before(
    signals: list[tuple[datetime, str, str, str]],
    *,
    symbol: str,
    ts: datetime,
    max_age_minutes: float = 30.0,
) -> dict[str, Any] | None:
    """
    Find most recent signal for symbol at/just before ts within max_age_minutes.
    signals is sorted by ts asc.
    """
    sym_u = (symbol or "").strip().upper()
    if not sym_u:
        return None
    best = None
    for s_ts, s_sym, s_reason, s_feat_json in reversed(signals):
        if s_sym.strip().upper() != sym_u:
            continue
        if s_ts > ts:
            continue
        age_m = (ts - s_ts).total_seconds() / 60.0
        if age_m > float(max_age_minutes):
            return None
        try:
            feat = json.loads(s_feat_json) if s_feat_json else {}
        except Exception:
            feat = {}
        if not isinstance(feat, dict):
            feat = {}
        feat["reason"] = s_reason
        feat["signal_ts"] = s_ts.isoformat()
        return feat
    return best


def build_executed_trade_dataset(
    *,
    db_path: str,
    max_signal_age_minutes: float = 30.0,
    min_trades: int = 5,
    direction: str | None = None,  # long | short | None
) -> ExecutedDatasetResult | None:
    """
    Build dataset from EXECUTED fills:
      trade_updates -> fills -> FIFO round trips -> label by PnL sign
    Then attach features from the most recent signal before the entry timestamp.
    """
    conn = sqlite3.connect(db_path)
    tu = conn.execute(
        """
        SELECT ts, event, raw_json
        FROM trade_updates
        WHERE event IN ('fill','partial_fill','filled')
        ORDER BY ts ASC
        """
    ).fetchall()
    fills = _rows_to_fills(tu)
    rts = reconstruct_round_trips(fills)
    if len(rts) < int(min_trades):
        conn.close()
        return None

    signals = _load_signals(conn)
    conn.close()

    X_rows: list[dict[str, Any]] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    want_dir = (direction or "").strip().lower() or None
    for rt in rts:
        if want_dir in ("long", "short") and str(rt.direction).strip().lower() != want_dir:
            continue
        feat = _nearest_signal_before(
            signals,
            symbol=rt.symbol,
            ts=rt.entry_ts,
            max_age_minutes=float(max_signal_age_minutes),
        )
        if feat is None:
            continue
        x = _flatten_feature_dict(feat)
        # Add executed-trade context
        x["exec_direction_long"] = 1.0 if rt.direction == "long" else 0.0
        x["exec_qty"] = float(rt.qty)
        x["exec_entry_px"] = float(rt.entry_px)
        x["exec_exit_px"] = float(rt.exit_px)

        X_rows.append(x)
        y_rows.append(1 if float(rt.pnl) > 0 else 0)
        meta_rows.append(
            {
                # Align with signal/sim meta so train_and_save() time-purge uses entry time.
                "ts": rt.entry_ts.isoformat(),
                "symbol": rt.symbol,
                "entry_ts": rt.entry_ts.isoformat(),
                "exit_ts": rt.exit_ts.isoformat(),
                "direction": rt.direction,
                "qty": float(rt.qty),
                "entry_px": float(rt.entry_px),
                "exit_px": float(rt.exit_px),
                "pnl": float(rt.pnl),
            }
        )

    if len(X_rows) < int(min_trades):
        return None

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, dtype=int, name="y")
    meta = pd.DataFrame(meta_rows)
    return ExecutedDatasetResult(X=X, y=y, meta=meta)

