from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("alpaca_day_bot.universe")


@dataclass(frozen=True)
class UniverseBuildResult:
    asof_utc: str
    lookback_days: int
    total_assets_seen: int
    bars_symbols: int
    selected: list[str]
    rejected_counts: dict[str, int]


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def build_liquid_universe(
    *,
    apca_api_key_id: str,
    apca_api_secret_key: str,
    out_path: str,
    max_symbols: int,
    lookback_days: int,
    min_price: float,
    min_avg_dollar_vol: float,
    batch_size: int = 200,
) -> UniverseBuildResult:
    """
    Free + efficient universe approximation:
    - Enumerate tradable US equities from trading API
    - Pull recent *daily* bars in batches (much cheaper than intraday bars)
    - Rank by average dollar volume (close * volume)
    - Persist list to JSON for use by scheduled ticks
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import ScreenerClient, StockHistoricalDataClient
    from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    t0 = datetime.now(tz=timezone.utc)
    lookback = max(5, int(lookback_days))
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback * 2)  # calendar padding for weekends/holidays

    # 1) Candidate symbols via Alpaca Screener endpoints (does NOT require /v2/assets)
    sc = ScreenerClient(apca_api_key_id, apca_api_secret_key)
    candidates: list[str] = []
    rejects = {"no_candidates": 0, "no_bars": 0, "low_price": 0, "low_dollar_vol": 0}

    top_n = max(50, min(int(max_symbols) * 3, 1000))
    try:
        ma = sc.get_most_actives(MostActivesRequest(top=top_n))
        for row in getattr(ma, "most_actives", []) or []:
            s = str(getattr(row, "symbol", "")).strip().upper()
            if s:
                candidates.append(s)
    except Exception as e:
        log.warning("universe most_actives failed err=%s", e)

    try:
        mv = sc.get_market_movers(MarketMoversRequest(top=min(200, top_n)))
        for row in (getattr(mv, "gainers", []) or []) + (getattr(mv, "losers", []) or []):
            s = str(getattr(row, "symbol", "")).strip().upper()
            if s:
                candidates.append(s)
    except Exception as e:
        log.warning("universe movers failed err=%s", e)

    symbols = sorted(set(candidates))
    total_assets_seen = len(symbols)
    if not symbols:
        rejects["no_candidates"] = 1
        payload = {
            "generated_at_utc": t0.isoformat(),
            "lookback_days": int(lookback),
            "max_symbols": int(max_symbols),
            "min_price": float(min_price),
            "min_avg_dollar_vol": float(min_avg_dollar_vol),
            "total_assets_seen": int(total_assets_seen),
            "bars_symbols": 0,
            "rejected_counts": rejects,
            "symbols": [],
            "notes": [
                "Screener returned no candidates; falling back to configured SYMBOLS.",
                "If this persists, your market-data entitlements may not include screener endpoints.",
            ],
        }
        _write_json(Path(out_path), payload)
        return UniverseBuildResult(
            asof_utc=t0.isoformat(),
            lookback_days=int(lookback),
            total_assets_seen=int(total_assets_seen),
            bars_symbols=0,
            selected=[],
            rejected_counts=rejects,
        )

    # 2) Daily bars in batches
    data_client = StockHistoricalDataClient(apca_api_key_id, apca_api_secret_key)
    scored: list[tuple[str, float, float]] = []  # (sym, avg_dollar_vol, last_close)

    def chunks(xs: list[str], n: int):
        for i in range(0, len(xs), n):
            yield xs[i : i + n]

    bars_symbols = 0
    for batch in chunks(symbols, max(1, int(batch_size))):
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
            end=datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc),
            feed=DataFeed.IEX,
        )
        try:
            bars = data_client.get_stock_bars(req)
            df = bars.df
        except Exception as e:
            log.warning("universe daily bars batch failed n=%s err=%s", len(batch), e)
            continue

        if df is None or getattr(df, "empty", True):
            continue

        if not isinstance(df.index, pd.MultiIndex):
            continue

        for sym in batch:
            try:
                sdf = df.xs(sym, level=0).copy()
            except Exception:
                rejects["no_bars"] += 1
                continue
            if sdf is None or sdf.empty:
                rejects["no_bars"] += 1
                continue
            bars_symbols += 1
            try:
                sdf = sdf.sort_index()
                last_close = float(sdf["close"].iloc[-1])
                if last_close < float(min_price):
                    rejects["low_price"] += 1
                    continue
                dv = (sdf["close"].astype(float) * sdf["volume"].astype(float)).dropna()
                if dv.empty:
                    rejects["no_bars"] += 1
                    continue
                avg_dv = float(dv.tail(lookback).mean())
                if avg_dv < float(min_avg_dollar_vol):
                    rejects["low_dollar_vol"] += 1
                    continue
                scored.append((sym, avg_dv, last_close))
            except Exception:
                continue

    scored.sort(key=lambda r: (-r[1], r[0]))
    top = [s for (s, _, _) in scored[: max(1, int(max_symbols))]]

    payload = {
        "generated_at_utc": t0.isoformat(),
        "lookback_days": int(lookback),
        "max_symbols": int(max_symbols),
        "min_price": float(min_price),
        "min_avg_dollar_vol": float(min_avg_dollar_vol),
        "total_assets_seen": int(total_assets_seen),
        "bars_symbols": int(bars_symbols),
        "rejected_counts": rejects,
        "symbols": top,
        "notes": [
            "Universe is ranked by average daily dollar volume using IEX daily bars.",
            "This is a liquidity filter to keep intraday scanning efficient; it is not an edge by itself.",
        ],
    }
    _write_json(Path(out_path), payload)

    return UniverseBuildResult(
        asof_utc=t0.isoformat(),
        lookback_days=int(lookback),
        total_assets_seen=int(total_assets_seen),
        bars_symbols=int(bars_symbols),
        selected=top,
        rejected_counts=rejects,
    )


def load_universe_symbols(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        syms = data.get("symbols") or []
        out = [str(s).strip().upper() for s in syms if str(s).strip()]
        return [s for s in out if s]
    except Exception:
        return []

