from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import sqlite3

from alpaca_day_bot.backtest import run_backtest
from alpaca_day_bot.data.historical_crypto import fetch_crypto_minute_bars
from alpaca_day_bot.data.historical_equity import fetch_equity_minute_bars
from alpaca_day_bot.storage.sim_ledger import SimLedger
from alpaca_day_bot.universe import build_liquid_universe, build_master_universe_assets


def _parse_ymd(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # interpret as date in UTC midnight
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "equity", "both"], default="both")
    ap.add_argument("--start", help="YYYY-MM-DD (UTC)", default=None)
    ap.add_argument("--end", help="YYYY-MM-DD (UTC, exclusive-ish)", default=None)
    ap.add_argument("--sim-db", default="state/ledgers/sim_rollup.sqlite3")
    ap.add_argument("--cache-dir", default="state/mock_cache")
    ap.add_argument(
        "--crypto-symbols",
        default="BTC/USD,ETH/USD,SOL/USD,XRP/USD,DOGE/USD,ADA/USD,LTC/USD,AVAX/USD",
    )
    ap.add_argument("--equity-max-symbols", type=int, default=500)
    ap.add_argument("--equity-min-avg-dollar-vol", type=float, default=10_000_000.0)
    ap.add_argument("--equity-max-price", type=float, default=500.0)
    ap.add_argument("--starting-equity", type=float, default=100000.0)
    ap.add_argument("--risk-per-trade-pct", type=float, default=0.0025)
    ap.add_argument("--max-gross-exposure-pct", type=float, default=0.35)
    ap.add_argument("--stop-loss-atr-mult", type=float, default=1.5)
    ap.add_argument("--take-profit-r-mult", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--commission-bps", type=float, default=0.5)
    ap.add_argument("--report-path", default="reports/mock_backtest_latest.md")
    args = ap.parse_args()

    # Default: last 365 days ending yesterday (Winnipeg clock), but expressed in UTC.
    if args.start and args.end:
        start = _parse_ymd(args.start)
        end = _parse_ymd(args.end)
    else:
        win = ZoneInfo("America/Winnipeg")
        now = datetime.now(win)
        end_local = datetime.combine(now.date(), datetime.min.time(), tzinfo=win)
        start_local = end_local - timedelta(days=365)
        start = start_local.astimezone(timezone.utc)
        end = end_local.astimezone(timezone.utc)

    api_key = (Path(".").resolve() and None)  # keep mypy quiet in this repo
    import os

    api_key = os.environ.get("APCA_API_KEY_ID", "").strip()
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment")

    sim = SimLedger(args.sim_db)

    def run_crypto() -> None:
        symbols = [s.strip() for s in str(args.crypto_symbols).split(",") if s.strip()]
        cache_dir = str(Path(args.cache_dir) / "crypto")
        bars_by_symbol, _metas = fetch_crypto_minute_bars(
            api_key=api_key,
            api_secret=api_secret,
            symbols=symbols,
            start=start,
            end=end,
            cache_dir=cache_dir,
        )

        def on_signal(ts: datetime, sym: str, sig) -> int | None:
            if sig is None or str(sig.action).upper() != "BUY":
                return
            return sim.record_signal(
                ts=ts,
                market="crypto",
                symbol=sym,
                action=str(sig.action),
                reason=str(sig.reason),
                features=(sig.features if isinstance(sig.features, dict) else None),
            )

        def on_trade(tr, sim_signal_id=None) -> None:
            sim.record_trade(
                market="crypto",
                side="long",
                trade=tr,
                meta={"source": "run_backtest", "sim_signal_id": sim_signal_id},
            )

        run_backtest(
            bars_by_symbol=bars_by_symbol,
            starting_equity=float(args.starting_equity),
            risk_per_trade_pct=float(args.risk_per_trade_pct),
            max_gross_exposure_pct=float(args.max_gross_exposure_pct),
            stop_loss_atr_mult=float(args.stop_loss_atr_mult),
            take_profit_r_mult=float(args.take_profit_r_mult),
            slippage_bps=float(args.slippage_bps),
            commission_bps=float(args.commission_bps),
            open_delay_minutes=0,
            market_context_filter=False,
            strategy_params={"signal_timeframe": "15m", "enable_shorts": False},
            on_signal=on_signal,
            on_trade=on_trade,
        )

    def run_equity() -> None:
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        master_path = str(cache_root / "equity_master_universe.json")
        master = build_master_universe_assets(
            apca_api_key_id=api_key,
            apca_api_secret_key=api_secret,
            out_path=master_path,
            max_symbols=10000,
            require_shortable=False,
        )
        candidates = list(master.get("symbols") or [])

        liquid_path = str(cache_root / "equity_liquid_universe.json")
        uni = build_liquid_universe(
            apca_api_key_id=api_key,
            apca_api_secret_key=api_secret,
            out_path=liquid_path,
            candidate_symbols=candidates,
            max_symbols=int(args.equity_max_symbols),
            lookback_days=20,
            min_price=1.0,
            max_price=float(args.equity_max_price),
            min_avg_dollar_vol=float(args.equity_min_avg_dollar_vol),
        )
        symbols = list(uni.selected or [])

        eq_cache_dir = str(cache_root / "equity_1m")
        for sym in symbols:
            df, _meta = fetch_equity_minute_bars(
                api_key=api_key,
                api_secret=api_secret,
                symbol=sym,
                start=start,
                end=end,
                cache_dir=eq_cache_dir,
            )
            if df is None or df.empty:
                continue

            def on_signal(ts: datetime, _sym: str, sig) -> int | None:
                if sig is None or str(sig.action).upper() != "BUY":
                    return
                return sim.record_signal(
                    ts=ts,
                    market="equity",
                    symbol=_sym,
                    action=str(sig.action),
                    reason=str(sig.reason),
                    features=(sig.features if isinstance(sig.features, dict) else None),
                )

            def on_trade(tr, sim_signal_id=None) -> None:
                sim.record_trade(
                    market="equity",
                    side="long",
                    trade=tr,
                    meta={"source": "run_backtest", "sim_signal_id": sim_signal_id},
                )

            run_backtest(
                bars_by_symbol={sym: df},
                starting_equity=float(args.starting_equity),
                risk_per_trade_pct=float(args.risk_per_trade_pct),
                max_gross_exposure_pct=float(args.max_gross_exposure_pct),
                stop_loss_atr_mult=float(args.stop_loss_atr_mult),
                take_profit_r_mult=float(args.take_profit_r_mult),
                slippage_bps=float(args.slippage_bps),
                commission_bps=float(args.commission_bps),
                open_delay_minutes=5,
                market_context_filter=False,
                strategy_params={"signal_timeframe": "15m", "enable_shorts": False},
                on_signal=on_signal,
                on_trade=on_trade,
            )

    m = str(args.market).strip().lower()
    if m in ("crypto", "both"):
        run_crypto()
    if m in ("equity", "both"):
        run_equity()

    sim.close()
    # Write a short markdown summary to reports/
    try:
        rp = Path(args.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(args.sim_db))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sim_signals")
        n_sig = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT COUNT(*) FROM sim_trades")
        n_tr = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT market, COUNT(*) FROM sim_signals GROUP BY market")
        sig_by_mkt = {str(m): int(c) for (m, c) in cur.fetchall() or []}
        cur.execute("SELECT market, COUNT(*) FROM sim_trades GROUP BY market")
        tr_by_mkt = {str(m): int(c) for (m, c) in cur.fetchall() or []}
        cur.execute("SELECT market, COALESCE(SUM(pnl),0.0) FROM sim_trades GROUP BY market")
        pnl_by_mkt = {str(m): float(p) for (m, p) in cur.fetchall() or []}
        cur.execute("SELECT COALESCE(SUM(pnl),0.0), COALESCE(AVG(pnl),0.0) FROM sim_trades")
        pnl_sum, pnl_avg = cur.fetchone() or (0.0, 0.0)
        conn.close()

        lines = []
        lines.append("# Mock backtest (dataset build)")
        lines.append("")
        lines.append(f"- **window_utc**: `{start.isoformat()}` → `{end.isoformat()}`")
        lines.append(f"- **sim_db**: `{args.sim_db}`")
        lines.append(f"- **markets**: `{args.market}`")
        lines.append("")
        lines.append("## Totals")
        lines.append(f"- **sim_signals**: {n_sig}")
        lines.append(f"- **sim_trades**: {n_tr}")
        lines.append(f"- **pnl_sum**: {float(pnl_sum):.2f}")
        lines.append(f"- **pnl_avg**: {float(pnl_avg):.4f}")
        lines.append("")
        lines.append("## By market")
        for mk in sorted(set(sig_by_mkt) | set(tr_by_mkt) | set(pnl_by_mkt)):
            lines.append(
                f"- **{mk}**: signals={sig_by_mkt.get(mk,0)} trades={tr_by_mkt.get(mk,0)} pnl_sum={pnl_by_mkt.get(mk,0.0):.2f}"
            )
        lines.append("")
        rp.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()

