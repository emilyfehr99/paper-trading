from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HistoricalFetchMeta:
    symbol: str
    start_utc: str
    end_utc: str
    rows: int
    source: str


def _norm_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_equity_minute_bars(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: str | None = None,
    chunk_minutes: int = 9000,
) -> tuple[pd.DataFrame, HistoricalFetchMeta]:
    """
    Fetch 1-minute equity bars from Alpaca (IEX feed by default) in time chunks.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    s0 = _norm_utc(start)
    e0 = _norm_utc(end)
    sym = str(symbol).strip().upper()
    if not sym or e0 <= s0:
        return pd.DataFrame(), HistoricalFetchMeta(symbol=sym, start_utc=s0.isoformat(), end_utc=e0.isoformat(), rows=0, source="none")

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"equity_1m_{sym}_{s0.date().isoformat()}_{e0.date().isoformat()}.pkl"
        meta_file = cache_path / f"equity_1m_{sym}_{s0.date().isoformat()}_{e0.date().isoformat()}.meta.json"
        if cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                if isinstance(df, pd.DataFrame):
                    rows = int(len(df))
                    return (
                        df,
                        HistoricalFetchMeta(
                            symbol=sym,
                            start_utc=str(json.loads(meta_file.read_text()).get("start_utc", s0.isoformat()))
                            if meta_file.exists()
                            else s0.isoformat(),
                            end_utc=str(json.loads(meta_file.read_text()).get("end_utc", e0.isoformat()))
                            if meta_file.exists()
                            else e0.isoformat(),
                            rows=rows,
                            source="alpaca_cache",
                        ),
                    )
            except Exception:
                pass

    client = StockHistoricalDataClient(api_key, api_secret)
    pieces: list[pd.DataFrame] = []
    t = s0
    step = timedelta(minutes=int(chunk_minutes))
    while t < e0:
        t2 = min(e0, t + step)
        req = StockBarsRequest(
            symbol_or_symbols=[sym],
            timeframe=TimeFrame.Minute,
            start=t,
            end=t2,
            limit=10000,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        df = getattr(bars, "df", None)
        if df is None or getattr(df, "empty", True):
            t = t2
            continue
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(sym, level=0)
            except Exception:
                t = t2
                continue
        df = df.copy()
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass
        df = df.sort_index()
        keep = [c for c in ("open", "high", "low", "close", "volume", "vwap") if c in df.columns]
        df = df[keep]
        pieces.append(df)
        t = t2

    if pieces:
        df_all = pd.concat(pieces, axis=0)
        df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
    else:
        df_all = pd.DataFrame()

    meta = HistoricalFetchMeta(symbol=sym, start_utc=s0.isoformat(), end_utc=e0.isoformat(), rows=int(len(df_all)), source="alpaca_equity")

    if cache_path is not None:
        try:
            df_all.to_pickle(cache_file)
            meta_file.write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")
        except Exception:
            pass

    return df_all, meta

