from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.screeners import DEFAULT_DAYTRADE_SYMBOLS, rank_top_daytrading  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", default="1D", help="1,5,15,30,60,120,240,1D,1W")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--metric", default="daytrade_score", help="daytrade_score|dollar_volume|volatility")
    p.add_argument(
        "--symbols",
        default="",
        help="CSV TradingView symbols (optional). If empty, uses DEFAULT_DAYTRADE_SYMBOLS.",
    )
    p.add_argument("--out", default="out/top_daytrading.json")
    args = p.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols.strip()
        else list(DEFAULT_DAYTRADE_SYMBOLS)
    )

    payload = rank_top_daytrading(
        symbols=symbols,
        resolution=args.resolution,  # type: ignore[arg-type]
        limit=args.limit,
        metric=args.metric,  # type: ignore[arg-type]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

