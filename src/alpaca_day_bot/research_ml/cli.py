from __future__ import annotations

import argparse
import json

from alpaca_day_bot.research_ml.backtest import walk_forward_vectorbt
from alpaca_day_bot.research_ml.data_manager import fetch_ohlcv_yfinance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    ap.add_argument("--start", required=True, help="Start date/time (ISO or YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date/time (ISO or YYYY-MM-DD)")
    ap.add_argument("--interval", default="15m", help="yfinance interval (1m/5m/15m/1h/1d)")
    ap.add_argument("--tp", type=float, default=0.02, help="Take-profit percent for triple-barrier")
    ap.add_argument("--sl", type=float, default=0.01, help="Stop-loss percent for triple-barrier")
    ap.add_argument("--max-bars", type=int, default=5, help="Max bars (time limit) for triple-barrier")
    ap.add_argument("--safety-threshold", type=float, default=0.65, help="Meta-label safety filter threshold")
    args = ap.parse_args()

    o = fetch_ohlcv_yfinance(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval)
    df = o.df
    if df is None or df.empty:
        raise SystemExit("No OHLCV returned from yfinance")

    bundle, res = walk_forward_vectorbt(
        df,
        tp_pct=float(args.tp),
        sl_pct=float(args.sl),
        max_bars=int(args.max_bars),
        safety_threshold=float(args.safety_threshold),
    )

    payload = {
        "symbol": args.symbol,
        "interval": args.interval,
        "train_rows": res.train_rows,
        "test_rows": res.test_rows,
        "trades": res.trades,
        "total_return": res.total_return,
        "sharpe": res.sharpe,
        "primary_acc_holdout": res.primary_acc,
        "trade_acc_holdout": res.trade_acc,
        "trade_coverage_holdout": res.trade_coverage,
        "feature_columns": bundle.feature_columns,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

