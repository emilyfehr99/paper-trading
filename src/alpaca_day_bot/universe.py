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


def build_master_universe_assets(
    *,
    apca_api_key_id: str,
    apca_api_secret_key: str,
    out_path: str,
    asset_class: str = "equity",
    max_symbols: int = 5000,
    allowed_exchanges: tuple[str, ...] = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"),
    require_marginable: bool = True,
    require_tradable: bool = True,
    require_shortable: bool = False,
) -> dict:
    """
    Build a broad, cached symbol list using Alpaca Trading "assets" endpoint.
    Supports both 'equity' and 'crypto' asset classes.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass

    t0 = datetime.now(tz=timezone.utc)
    tc = TradingClient(apca_api_key_id, apca_api_secret_key, paper=True)

    # Map string to Alpaca Enum
    ac_enum = AssetClass.US_EQUITY
    if asset_class.lower() == "crypto":
        ac_enum = AssetClass.CRYPTO

    try:
        from alpaca.trading.requests import GetAssetsRequest
        req = GetAssetsRequest(asset_class=ac_enum)
        assets = tc.get_all_assets(req)
    except Exception as e:
        payload = {
            "generated_at_utc": t0.isoformat(),
            "source": "alpaca_assets",
            "asset_class": asset_class,
            "error": str(e),
            "total_assets_seen": 0,
            "symbols": [],
        }
        _write_json(Path(out_path), payload)
        return payload

    symbols: list[str] = []
    rejected = {
        "not_target_asset_class": 0,
        "inactive": 0,
        "not_tradable": 0,
        "not_marginable": 0,
        "not_shortable": 0,
        "bad_exchange": 0,
        "no_symbol": 0,
    }

    for a in assets or []:
        sym = str(getattr(a, "symbol", "") or "").strip().upper()
        if not sym:
            rejected["no_symbol"] += 1
            continue

        # Strictly enforce asset class
        if getattr(a, "asset_class", None) != ac_enum:
             rejected["not_target_asset_class"] += 1
             continue

        # Handle status
        status_str = str(getattr(a, "status", "active")).split(".")[-1].upper()
        if status_str != "ACTIVE":
            rejected["inactive"] += 1
            continue
            
        # Exchange filter (only for equities)
        if asset_class.lower() == "equity":
            ex_str = str(getattr(a, "exchange", "")).split(".")[-1].upper()
            if allowed_exchanges and ex_str and ex_str not in set(allowed_exchanges):
                rejected["bad_exchange"] += 1
                continue

        if require_tradable and (getattr(a, "tradable", True) is False):
            rejected["not_tradable"] += 1
            continue
            
        # Margin/Short filters only apply to equities
        if asset_class.lower() == "equity":
            if require_marginable and (getattr(a, "marginable", True) is False):
                rejected["not_marginable"] += 1
                continue
            if require_shortable and (getattr(a, "shortable", True) is False):
                rejected["not_shortable"] += 1
                continue

        symbols.append(sym)

    # Deduplicate; do NOT sort alphabetically here — order is preserved from
    # Alpaca assets endpoint which has no meaningful order bias.
    symbols = list(dict.fromkeys(symbols))  # deduplicate, preserve insertion order
    if int(max_symbols) > 0:
        symbols = symbols[: int(max_symbols)]

    payload = {
        "generated_at_utc": t0.isoformat(),
        "source": "alpaca_assets",
        "allowed_exchanges": list(allowed_exchanges),
        "require_tradable": bool(require_tradable),
        "require_marginable": bool(require_marginable),
        "require_shortable": bool(require_shortable),
        "max_symbols": int(max_symbols),
        "total_assets_seen": len(assets or []),
        "rejected_counts": rejected,
        "symbols": symbols,
        "notes": [
            "This is a broad cached universe; per-tick scanning should use a smaller liquid subset.",
        ],
    }
    _write_json(Path(out_path), payload)
    return payload


def intraday_prefilter_symbols(
    *,
    apca_api_key_id: str,
    apca_api_secret_key: str,
    method: str = "movers_actives",
    max_symbols: int = 400,
) -> list[str]:
    """
    Cheap per-tick prefilter to avoid scanning thousands of symbols.
    Uses Alpaca screener endpoints (most-actives + movers) which are capped but fast.
    """
    m = (method or "movers_actives").strip().lower()
    if m != "movers_actives":
        m = "movers_actives"

    try:
        from alpaca.data.historical import ScreenerClient
        from alpaca.data.requests import MarketMoversRequest, MostActivesRequest
    except Exception:
        return []

    sc = ScreenerClient(apca_api_key_id, apca_api_secret_key)
    out: list[str] = []

    # Screener API caps:
    # - most-actives: top <= 100
    # - movers: top <= 50
    top_n = max(50, min(int(max_symbols), 100))
    try:
        ma = sc.get_most_actives(MostActivesRequest(top=top_n))
        for row in getattr(ma, "most_actives", []) or []:
            s = str(getattr(row, "symbol", "") or "").strip().upper()
            if s:
                out.append(s)
    except Exception:
        pass

    try:
        mv = sc.get_market_movers(MarketMoversRequest(top=min(50, top_n)))
        for row in (getattr(mv, "gainers", []) or []) + (getattr(mv, "losers", []) or []):
            s = str(getattr(row, "symbol", "") or "").strip().upper()
            if s:
                out.append(s)
    except Exception:
        pass

    # Deduplicate/preserve order; keep under max_symbols.
    seen = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
        if len(uniq) >= int(max_symbols):
            break
    return uniq


def build_liquid_universe(
    *,
    apca_api_key_id: str,
    apca_api_secret_key: str,
    out_path: str,
    asset_class: str = "equity",
    candidate_symbols: list[str] | None = None,
    max_symbols: int,
    lookback_days: int,
    min_price: float,
    max_price: float | None = None,
    min_avg_dollar_vol: float,
    batch_size: int = 200,
) -> UniverseBuildResult:
    """
    Rank symbols by liquidity (average dollar volume).
    Supports both 'equity' and 'crypto' asset classes.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import ScreenerClient, StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    is_crypto = asset_class.lower() == "crypto"
    t0 = datetime.now(tz=timezone.utc)
    lookback = max(5, int(lookback_days))
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback * 2)

    # 1) Candidate symbols
    rejects = {"no_candidates": 0, "no_bars": 0, "low_price": 0, "high_price": 0, "low_dollar_vol": 0}
    symbols: list[str] = []
    
    # --- INSTITUTIONAL DISCOVERY INJECTION ---
    discovered_syms = []
    if not is_crypto:
        from alpaca_day_bot.data.discovery import DiscoveryEngine
        discovery = DiscoveryEngine()
        discovered_syms = discovery.build_expanded_universe()
        log.info(f"Augmenting universe with {len(discovered_syms)} discovered momentum symbols.")
    
    if candidate_symbols:
        symbols = list({str(s).strip().upper() for s in candidate_symbols if str(s).strip()} | set(discovered_syms))
    else:
        # Alpaca Screeners + Discovery
        candidates: list[str] = list(discovered_syms)
        if not is_crypto:
            sc = ScreenerClient(apca_api_key_id, apca_api_secret_key)
            top_n = max(50, min(int(max_symbols) * 3, 100))
            try:
                ma = sc.get_most_actives(MostActivesRequest(top=top_n))
                for row in getattr(ma, "most_actives", []) or []:
                    s = str(getattr(row, "symbol", "")).strip().upper()
                    if s: candidates.append(s)
            except Exception: pass
            
            try:
                mv = sc.get_market_movers(MarketMoversRequest(top=min(50, top_n)))
                for row in (getattr(mv, "gainers", []) or []) + (getattr(mv, "losers", []) or []):
                    s = str(getattr(row, "symbol", "")).strip().upper()
                    if s: candidates.append(s)
            except Exception: pass
        # Deduplicate but preserve screener/discovery ranking order (already ranked by activity)
        seen_c: set[str] = set()
        deduped: list[str] = []
        for s in candidates:
            if s not in seen_c:
                seen_c.add(s)
                deduped.append(s)
        symbols = deduped

    total_assets_seen = len(symbols)
    if not symbols:
        rejects["no_candidates"] = 1
        # Final payload and return ... (abbreviated for chunk)
        return UniverseBuildResult(asof_utc=t0.isoformat(), lookback_days=int(lookback), 
                                  total_assets_seen=int(total_assets_seen), bars_symbols=0, 
                                  selected=[], rejected_counts=rejects)

    # 2) Daily bars in batches
    if is_crypto:
        data_client = CryptoHistoricalDataClient(apca_api_key_id, apca_api_secret_key)
    else:
        data_client = StockHistoricalDataClient(apca_api_key_id, apca_api_secret_key)
    
    scored: list[tuple[str, float, float]] = []

    def chunks(xs: list[str], n: int):
        for i in range(0, len(xs), n):
            yield xs[i : i + n]

    bars_symbols = 0
    for batch in chunks(symbols, max(1, int(batch_size))):
        if is_crypto:
            req = CryptoBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
                end=datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc),
            )
        else:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
                end=datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc),
                feed=DataFeed.IEX,
            )
            
        try:
            if is_crypto:
                bars = data_client.get_crypto_bars(req)
            else:
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
                if max_price is not None and float(max_price) > 0 and last_close > float(max_price):
                    rejects["high_price"] += 1
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
        "max_price": (None if (max_price is None) else float(max_price)),
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


def filter_universe_symbols_by_max_price(
    *,
    symbols: list[str],
    max_price: float,
    apca_api_key_id: str,
    apca_api_secret_key: str,
) -> list[str]:
    """
    Defensive filter in case a universe file was built with a different max_price.
    Uses daily IEX bars (cheap) to drop symbols whose last close is above max_price.
    """
    if not symbols:
        return []
    if max_price <= 0:
        return list(symbols)
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    data_client = StockHistoricalDataClient(apca_api_key_id, apca_api_secret_key)
    end = datetime.now(tz=timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=14)
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    try:
        bars = data_client.get_stock_bars(req)
        df = bars.df
    except Exception:
        return list(symbols)
    if df is None or getattr(df, "empty", True) or not isinstance(df.index, pd.MultiIndex):
        return list(symbols)

    out = []
    for sym in symbols:
        try:
            sdf = df.xs(sym, level=0).sort_index()
            last_close = float(sdf["close"].iloc[-1])
            if last_close <= float(max_price):
                out.append(sym)
        except Exception:
            continue
    return out

