"""
Offline backfill: write ``triple_barrier_labels`` for signals missing TB rows, using yfinance OHLC.

Requires ``features_json`` with ``close``, ``tp_price``, and ``sl_price`` (same as live labeling).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca_day_bot.research_ml.data_manager import fetch_ohlcv_yfinance
from alpaca_day_bot.storage.ledger import Ledger
from alpaca_day_bot.tools.triple_barrier_yfinance import realized_return_from_last_close, triple_barrier_outcome_from_bars


def _parse_ts(ts_s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _label_one_tb(
    *,
    ledger: Ledger,
    signal_id: int,
    ts_s: str,
    sym: str,
    action: str,
    feat_json: str,
    now_utc: datetime,
    min_age_minutes: float,
    interval: str,
    max_lookback_days: int,
) -> bool:
    ts_sig = _parse_ts(ts_s)
    if ts_sig is None:
        return False
    age_m = (now_utc - ts_sig).total_seconds() / 60.0
    if age_m < float(min_age_minutes):
        return False

    try:
        feat = json.loads(feat_json) if feat_json else {}
    except Exception:
        return False
    if not isinstance(feat, dict):
        return False

    entry = feat.get("close")
    tp = feat.get("tp_price")
    sl = feat.get("sl_price")
    if entry is None or tp is None or sl is None:
        return False
    try:
        entry_f = float(entry)
        tp_f = float(tp)
        sl_f = float(sl)
    except Exception:
        return False
    if entry_f <= 0 or tp_f <= 0 or sl_f <= 0:
        return False

    end = min(now_utc, ts_sig + timedelta(days=max(1, int(max_lookback_days))))
    ohlc = fetch_ohlcv_yfinance(symbol=str(sym), start=ts_sig, end=end, interval=interval)
    df = ohlc.df
    if df is None or getattr(df, "empty", True):
        return False

    dfx = df.copy()
    dfx_full = dfx
    try:
        dfx.index = pd.to_datetime(dfx.index, utc=True, errors="coerce")
        dfx = dfx.sort_index()
        dfx = dfx[dfx.index >= ts_sig]
        if dfx is None or getattr(dfx, "empty", True):
            dfx = dfx_full
            dfx.index = pd.to_datetime(dfx.index, utc=True, errors="coerce")
            dfx = dfx.sort_index()
    except Exception:
        dfx = dfx_full

    if dfx is None or getattr(dfx, "empty", True):
        return False

    act = (action or "").strip().upper()
    outcome, px_at_eval = triple_barrier_outcome_from_bars(
        dfx,
        act=act,
        entry_f=entry_f,
        tp_f=tp_f,
        sl_f=sl_f,
    )
    if not isinstance(px_at_eval, float) or px_at_eval != px_at_eval:
        return False
    realized_ret = realized_return_from_last_close(act=act, entry_f=entry_f, px_at_eval=px_at_eval)

    ledger.record_triple_barrier_label(
        signal_id=int(signal_id),
        evaluated_ts=now_utc,
        entry_close=entry_f,
        tp_price=tp_f,
        sl_price=sl_f,
        outcome=outcome,
        realized_return_pct=float(realized_ret),
        horizon_minutes=float(age_m),
    )
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Backfill triple_barrier_labels via yfinance (offline).")
    ap.add_argument("--db", required=True, help="Path to ledger.sqlite3")
    ap.add_argument("--limit", type=int, default=200, help="Max signals to attempt")
    ap.add_argument("--min-age-minutes", type=float, default=15.0)
    ap.add_argument("--interval", default="5m", help="yfinance interval (1m/5m/15m/...)")
    ap.add_argument("--max-lookback-days", type=int, default=14, help="Download window from signal ts")
    ap.add_argument("--dry-run", action="store_true", help="List work only; do not write labels")
    args = ap.parse_args(argv)

    ledger = Ledger(str(args.db))
    now_utc = datetime.now(tz=timezone.utc)
    pending = ledger.list_unlabeled_signal_rows_for_triple_barrier_backlog(
        now_utc=now_utc,
        min_age_minutes=float(args.min_age_minutes),
        actions=("BUY", "SHORT"),
        limit=max(1, int(args.limit)),
    )
    n_ok = 0
    for signal_id, ts_s, sym, action, feat_json in pending:
        if bool(args.dry_run):
            print("dry_run_tb", signal_id, sym, ts_s, action)
            continue
        if _label_one_tb(
            ledger=ledger,
            signal_id=int(signal_id),
            ts_s=str(ts_s),
            sym=str(sym),
            action=str(action),
            feat_json=str(feat_json),
            now_utc=now_utc,
            min_age_minutes=float(args.min_age_minutes),
            interval=str(args.interval),
            max_lookback_days=int(args.max_lookback_days),
        ):
            n_ok += 1
    ledger.close()
    if not bool(args.dry_run):
        print(f"backfill_triple_barrier_yf: labeled {n_ok} / {len(pending)} attempted")


if __name__ == "__main__":
    main()
