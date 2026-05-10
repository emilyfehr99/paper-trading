"""
Offline backfill: write ``forward_return_labels`` for signals missing labels, using yfinance OHLC.

Use when the live bot had no bar buffer for old dates (per-day labeling never ran).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from alpaca_day_bot.research_ml.data_manager import fetch_ohlcv_yfinance
from alpaca_day_bot.storage.ledger import Ledger


def _parse_ts(ts_s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _label_one(
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
    try:
        entry_f = float(feat.get("close"))
    except Exception:
        return False
    if entry_f <= 0:
        return False

    end = min(now_utc, ts_sig + timedelta(days=7))
    ohlc = fetch_ohlcv_yfinance(symbol=str(sym), start=ts_sig, end=end, interval=interval)
    df = ohlc.df
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return False
    try:
        now_px = float(df["close"].iloc[-1])
    except Exception:
        return False

    act_u = (action or "").strip().upper()
    if act_u == "SHORT":
        ret = (entry_f - now_px) / entry_f
    else:
        ret = (now_px - entry_f) / entry_f
    ledger.record_forward_return_label(
        signal_id=signal_id,
        evaluated_ts=now_utc,
        price_at_label=now_px,
        entry_close=entry_f,
        return_pct=float(ret),
        horizon_minutes=float(age_m),
    )
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Backfill forward_return_labels via yfinance (offline).")
    ap.add_argument("--db", required=True, help="Path to ledger.sqlite3")
    ap.add_argument("--limit", type=int, default=200, help="Max signals to attempt")
    ap.add_argument("--min-age-minutes", type=float, default=15.0)
    ap.add_argument("--interval", default="5m", help="yfinance interval (1m/5m/15m/...)")
    ap.add_argument("--dry-run", action="store_true", help="List work only; do not write labels")
    args = ap.parse_args(argv)

    ledger = Ledger(str(args.db))
    now_utc = datetime.now(tz=timezone.utc)
    pending = ledger.list_unlabeled_signal_rows_backlog(
        now_utc=now_utc,
        min_age_minutes=float(args.min_age_minutes),
        actions=("BUY", "SHORT"),
        limit=max(1, int(args.limit)),
    )
    n_ok = 0
    for signal_id, ts_s, sym, action, feat_json in pending:
        if bool(args.dry_run):
            print("dry_run", signal_id, sym, ts_s, action)
            continue
        if _label_one(
            ledger=ledger,
            signal_id=int(signal_id),
            ts_s=str(ts_s),
            sym=str(sym),
            action=str(action),
            feat_json=str(feat_json),
            now_utc=now_utc,
            min_age_minutes=float(args.min_age_minutes),
            interval=str(args.interval),
        ):
            n_ok += 1
    ledger.close()
    if not bool(args.dry_run):
        print(f"backfill_forward_labels_yf: labeled {n_ok} / {len(pending)} attempted")


if __name__ == "__main__":
    main()
