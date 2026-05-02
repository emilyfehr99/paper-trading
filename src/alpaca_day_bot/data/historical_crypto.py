from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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


def fetch_crypto_minute_bars(
    *,
    api_key: str,
    api_secret: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    cache_dir: str | None = None,
    chunk_minutes: int = 9000,
) -> tuple[dict[str, pd.DataFrame], list[HistoricalFetchMeta]]:
    """
    Fetch 1-minute crypto bars from Alpaca in time chunks (API limits apply).

    Returns:
    - dict: symbol -> DataFrame indexed by UTC timestamp with columns open/high/low/close/volume (+vwap if present)
    - list of per-symbol fetch metadata
    """
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    s0 = _norm_utc(start)
    e0 = _norm_utc(end)
    if e0 <= s0:
        return {}, []

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    client = CryptoHistoricalDataClient(api_key, api_secret)
    out: dict[str, pd.DataFrame] = {}
    metas: list[HistoricalFetchMeta] = []

    for sym in symbols:
        cache_file = None
        if cache_path is not None:
            safe = sym.replace("/", "_").replace(":", "_")
            cache_file = cache_path / f"crypto_1m_{safe}_{s0.date().isoformat()}_{e0.date().isoformat()}.pkl"
            meta_file = cache_path / f"crypto_1m_{safe}_{s0.date().isoformat()}_{e0.date().isoformat()}.meta.json"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[sym] = df
                        if meta_file.exists():
                            try:
                                m = json.loads(meta_file.read_text(encoding="utf-8"))
                                metas.append(
                                    HistoricalFetchMeta(
                                        symbol=sym,
                                        start_utc=str(m.get("start_utc", s0.isoformat())),
                                        end_utc=str(m.get("end_utc", e0.isoformat())),
                                        rows=int(m.get("rows", len(df))),
                                        source=str(m.get("source", "alpaca_cache")),
                                    )
                                )
                            except Exception:
                                metas.append(
                                    HistoricalFetchMeta(
                                        symbol=sym,
                                        start_utc=s0.isoformat(),
                                        end_utc=e0.isoformat(),
                                        rows=int(len(df)),
                                        source="alpaca_cache",
                                    )
                                )
                        else:
                            metas.append(
                                HistoricalFetchMeta(
                                    symbol=sym,
                                    start_utc=s0.isoformat(),
                                    end_utc=e0.isoformat(),
                                    rows=int(len(df)),
                                    source="alpaca_cache",
                                )
                            )
                        continue
                except Exception:
                    pass

        pieces: list[pd.DataFrame] = []
        t = s0
        step = timedelta(minutes=int(chunk_minutes))
        while t < e0:
            t2 = min(e0, t + step)
            req = CryptoBarsRequest(
                symbol_or_symbols=[sym],
                timeframe=TimeFrame.Minute,
                start=t,
                end=t2,
                limit=10000,
            )
            bars = client.get_crypto_bars(req)
            df = getattr(bars, "df", None)
            if df is None or getattr(df, "empty", True):
                t = t2
                continue
            # Normalize: sometimes MultiIndex (symbol, timestamp)
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

        out[sym] = df_all
        metas.append(
            HistoricalFetchMeta(
                symbol=sym,
                start_utc=s0.isoformat(),
                end_utc=e0.isoformat(),
                rows=int(len(df_all)),
                source="alpaca_crypto",
            )
        )

        if cache_file is not None:
            try:
                df_all.to_pickle(cache_file)
                meta_file.write_text(
                    json.dumps(
                        {
                            "symbol": sym,
                            "start_utc": s0.isoformat(),
                            "end_utc": e0.isoformat(),
                            "rows": int(len(df_all)),
                            "source": "alpaca_crypto",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

    return out, metas

