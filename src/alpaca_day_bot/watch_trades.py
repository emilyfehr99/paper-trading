"""Follow state/transactions.jsonl (append-only audit) for a live trade feed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _format_audit(obj: dict) -> str:
    kind = obj.get("kind", "?")
    ts = obj.get("ts", "")
    if kind == "trade_update":
        ev = obj.get("event", "")
        sym = obj.get("symbol") or "?"
        oid = obj.get("order_id") or "-"
        p = obj.get("payload") or {}
        o = p.get("order") or {}
        side = o.get("side", "")
        qty = o.get("filled_qty")
        px = o.get("filled_avg_price")
        st = o.get("status", "")
        parts = [f"[{ts}]", "TRADE", ev, sym]
        if side:
            parts.append(f"side={side}")
        if qty is not None:
            parts.append(f"qty={qty}")
        if px is not None:
            parts.append(f"@ {px}")
        if st:
            parts.append(f"status={st}")
        parts.append(f"order_id={oid}")
        return " ".join(str(x) for x in parts)
    if kind == "order_intent":
        sym = obj.get("symbol", "?")
        ok = obj.get("submitted")
        mark = "SUBMIT" if ok else "REJECT"
        n = obj.get("notional_usd")
        reason = obj.get("reason") or ""
        oid = obj.get("alpaca_order_id") or "-"
        return f"[{ts}] ORDER {mark} {sym} notional={n} alpaca_id={oid} {reason}".rstrip()
    return json.dumps(obj, default=str)


def _follow(path: Path) -> None:
    while not path.exists():
        print(f"Waiting for {path} ...", file=sys.stderr, flush=True)
        time.sleep(1.0)
    with path.open("r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.25)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(line, flush=True)
                continue
            print(_format_audit(obj), flush=True)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(description="Live tail of transactions.jsonl (orders + trade updates).")
    p.add_argument(
        "--state-dir",
        default=os.environ.get("STATE_DIR", "state"),
        help="Directory with ledger.sqlite3 and transactions.jsonl (default: env STATE_DIR or ./state).",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print each JSONL line as-is (for piping to jq).",
    )
    args = p.parse_args()
    path = Path(args.state_dir).resolve() / "transactions.jsonl"
    if args.raw:

        def _follow_raw() -> None:
            while not path.exists():
                print(f"Waiting for {path} ...", file=sys.stderr, flush=True)
                time.sleep(1.0)
            with path.open("r", encoding="utf-8") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.25)
                        continue
                    print(line.rstrip(), flush=True)

        _follow_raw()
    else:
        _follow(path)


if __name__ == "__main__":
    main()
