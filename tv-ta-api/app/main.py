from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .available_indicators import available_indicators_catalog
from .cache import Cache
from .data_provider import Resolution, fetch_ohlcv, normalize_resolution
from .indicators import compute_latest, compute_series, parse_indicator_list
from .models import (
    BatchRequest,
    BatchResponse,
    EnrichBarsRequest,
    EnrichBarsResponse,
    HistoryBulkJobResponse,
    HistoryBulkRequest,
    HistoryResponse,
    IndicatorsAtRequest,
    IndicatorsResponse,
    OhlcvBar,
    SeriesResponse,
    SignalResponse,
    TipResponse,
    TopDaytradingRequest,
    TopDaytradingResponse,
)
from .ta_enrich import (
    bars_to_dataframe,
    enrich_ohlcv_frame,
    frame_to_points,
    indicators_at_timestamp,
)
from .screeners import DEFAULT_DAYTRADE_SYMBOLS, rank_top_daytrading
from .settings import settings
from .signals import build_trade_plan

app = FastAPI(title="TradingView TA API", version="0.1.0")
cache = Cache(ttl_seconds=settings.cache_ttl_seconds, redis_url=settings.redis_url)

# Single-flight tip fetches per symbol (4 bots stampeding the same tip window).
# Event-based: do NOT hold the lock across TradingView I/O (that froze all stacks).
_tip_inflight: dict[str, threading.Event] = {}
_tip_inflight_guard = threading.Lock()
_tip_locks: dict[str, threading.Lock] = {}
_tip_locks_guard = threading.Lock()


def _tip_lock_for(symbol: str) -> threading.Lock:
    with _tip_locks_guard:
        lock = _tip_locks.get(symbol)
        if lock is None:
            lock = threading.Lock()
            _tip_locks[symbol] = lock
        return lock


def _resolution_minutes(resolution: str) -> int:
    r = str(resolution or "5").strip().lower()
    mapping = {
        "1": 1,
        "1m": 1,
        "5": 5,
        "5m": 5,
        "15": 15,
        "15m": 15,
        "30": 30,
        "30m": 30,
        "60": 60,
        "1h": 60,
        "120": 120,
        "2h": 120,
        "240": 240,
        "4h": 240,
    }
    return int(mapping.get(r, 5))


def _is_live_history_request(end_ts: int | None) -> bool:
    """True when request is for a live / near-now tip window (not completed history)."""
    now = int(time.time())
    if end_ts is None:
        return True
    # end within the last day or still in the future → treat as live tip pull
    return int(end_ts) >= now - 86400


def _expected_last_closed_bar_open_unix(
    resolution: str, *, now_ts: float | None = None
) -> int:
    """Unix UTC open of the last fully closed bar (mirrors watch_replay_cutoff open)."""
    bm = max(1, _resolution_minutes(resolution))
    now = float(now_ts if now_ts is not None else time.time())
    # Floor to bar open of current forming slot, then step back if still forming.
    slot = int(now // (bm * 60)) * (bm * 60)
    bar_close = slot + bm * 60
    if now < bar_close:
        slot -= bm * 60
    return int(slot)


def _cached_last_bar_open_unix(cached: dict) -> int | None:
    pts = cached.get("points") or []
    if not pts:
        return None
    t = pts[-1].get("t")
    if t is None:
        return None
    t = int(t)
    if t > 1_000_000_000_000:
        t = t // 1000
    return t


def _live_cache_stale(cached: dict, resolution: str, end_ts: int | None) -> bool:
    """Bypass cache when live tip is behind the expected last closed bar."""
    if not _is_live_history_request(end_ts):
        return False
    expected = _expected_last_closed_bar_open_unix(resolution)
    got = _cached_last_bar_open_unix(cached)
    if got is None:
        return True
    return int(got) < int(expected)


@app.on_event("startup")
async def _bump_threadpool_for_4stack_load() -> None:
    """Sync /history routes share anyio's default threadpool (~40).

    Four bots × multi-symbol history can exhaust it so /health (also sync)
    stalls → supervisor false-unhealthy → ensure recycle death spiral.
    Keep /health async (below) and enlarge the pool for OHLCV work.
    """
    try:
        import anyio

        limiter = anyio.to_thread.current_default_thread_limiter()
        # Cap high enough for 4 stacks without unbounded TV websocket fan-out
        # (fetch_ohlcv also has its own semaphore).
        limiter.total_tokens = max(int(limiter.total_tokens), 64)
    except Exception:
        pass


def _cache_key(prefix: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()}"


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness only — must never wait on the sync threadpool / TV fetches."""
    return {"status": "ok"}


@app.get("/health/tradingview")
async def health_tradingview() -> JSONResponse:
    """Verify tvkit can pull live bars (auth + WebSocket path)."""
    from .tvkit_health import probe_tradingview_auth

    result = await probe_tradingview_auth()
    status = 200 if result.get("ok") else 503
    return JSONResponse(status_code=status, content=result)


@app.get("/api/ta/available-indicators")
def available_indicators(
    q: str | None = Query(None, description="Optional substring filter"),
    include_docs: bool = Query(False),
    include_signatures: bool = Query(False),
    limit: int = Query(500, ge=1, le=5000),
) -> dict:
    items = available_indicators_catalog()
    if q:
        qn = q.strip().lower()
        items = [i for i in items if qn in i["name"].lower()]

    items = items[:limit]
    if not include_docs:
        for i in items:
            i.pop("doc", None)
    if not include_signatures:
        for i in items:
            i.pop("signature", None)

    return {"count": len(items), "items": items}


@app.get("/api/ta/indicators", response_model=IndicatorsResponse)
def get_indicators(
    symbol: str,
    indicators: str = Query(..., description="CSV: rsi,sma,macd,bbands or rsi:14,sma:50,..."),
    resolution: Resolution = "1D",
    count: int = Query(300, ge=50, le=5000, description="Bars to fetch before computing latest values"),
) -> IndicatorsResponse:
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    specs = parse_indicator_list(indicators)
    key = _cache_key(
        "indicators",
        {"symbol": symbol, "indicators": indicators, "resolution": resolution, "count": count},
    )
    cached = cache.get_json(key)
    if cached:
        return IndicatorsResponse(**cached)

    bars = fetch_ohlcv(symbol=symbol, resolution=resolution, count=count, extra_bars=200)
    df = bars.df
    if df.empty:
        return IndicatorsResponse(symbol=symbol, timestamp=0, indicators={})

    latest_ts = int(pd.Timestamp(df.index[-1]).timestamp())
    values = compute_latest(df, specs=specs)
    payload = {"symbol": symbol, "timestamp": latest_ts, "indicators": values}
    cache.set_json(key, payload)
    return IndicatorsResponse(**payload)


@app.get("/api/ta/series", response_model=SeriesResponse)
def get_series(
    symbol: str,
    indicator: str,
    period: int = Query(14, ge=2, le=500),
    resolution: Resolution = "1D",
    count: int = Query(100, ge=10, le=5000),
) -> SeriesResponse:
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    key = _cache_key(
        "series",
        {
            "symbol": symbol,
            "indicator": indicator,
            "period": period,
            "resolution": resolution,
            "count": count,
        },
    )
    cached = cache.get_json(key)
    if cached:
        return SeriesResponse(**cached)

    bars = fetch_ohlcv(symbol=symbol, resolution=resolution, count=max(count, period) + 50, extra_bars=200)
    df = bars.df
    if df.empty:
        return SeriesResponse(symbol=symbol, indicator=indicator, period=period, resolution=resolution, points=[])

    try:
        pts = compute_series(df, indicator=indicator, period=period, count=count)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    payload = {
        "symbol": symbol,
        "indicator": indicator,
        "period": period,
        "resolution": resolution,
        "points": pts,
    }
    cache.set_json(key, payload)
    return SeriesResponse(**payload)


@app.post("/api/ta/batch", response_model=BatchResponse)
def batch(req: BatchRequest) -> BatchResponse:
    try:
        req.resolution = normalize_resolution(req.resolution)  # type: ignore[assignment]
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    results: list[IndicatorsResponse] = []
    for item in req.items:
        indicators_csv = ",".join(item.indicators)
        specs = parse_indicator_list(indicators_csv)
        key = _cache_key(
            "batch",
            {
                "symbol": item.symbol,
                "indicators": indicators_csv,
                "resolution": req.resolution,
                "count": req.count,
                "as_of": req.as_of,
            },
        )
        cached = cache.get_json(key)
        if cached:
            results.append(IndicatorsResponse(**cached))
            continue

        bars = fetch_ohlcv(
            symbol=item.symbol, resolution=req.resolution, count=req.count, extra_bars=200
        )
        df = bars.df
        if df.empty:
            payload = {"symbol": item.symbol, "timestamp": 0, "indicators": {}}
            cache.set_json(key, payload)
            results.append(IndicatorsResponse(**payload))
            continue

        if req.as_of is not None:
            latest_ts, values = indicators_at_timestamp(df, int(req.as_of))
            if not values:
                sub = df[df.index <= pd.to_datetime(req.as_of, unit="s", utc=True)]
                if not sub.empty:
                    values = compute_latest(sub, specs=specs)
                    latest_ts = int(pd.Timestamp(sub.index[-1]).timestamp())
        else:
            latest_ts = int(pd.Timestamp(df.index[-1]).timestamp())
            values = compute_latest(df, specs=specs)
        if not df.empty:
            last = df.iloc[-1]
            values.setdefault("close", float(last["close"]))
            values.setdefault("high", float(last["high"]))
            values.setdefault("low", float(last["low"]))
            if "volume" in df.columns:
                values.setdefault("volume", float(last.get("volume") or 0))
        payload = {"symbol": item.symbol, "timestamp": latest_ts, "indicators": values}
        cache.set_json(key, payload)
        results.append(IndicatorsResponse(**payload))
    return BatchResponse(results=results)


@app.post("/api/ta/enrich_bars", response_model=EnrichBarsResponse)
def enrich_bars(req: EnrichBarsRequest) -> EnrichBarsResponse | JSONResponse:
    """
    Backtest path: client uploads historical OHLCV (e.g. Alpaca 1m bars).
    Returns MACD histogram + Williams Alligator (and helpers) for every bar.
    """
    try:
        resolution = normalize_resolution(req.resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    raw_bars = [b.model_dump() for b in req.bars]
    key = _cache_key(
        "enrich_bars",
        {
            "symbol": req.symbol,
            "resolution": resolution,
            "mode": req.alligator_mode,
            "n": len(raw_bars),
            "tail": raw_bars[-3:] if raw_bars else [],
        },
    )
    cached = cache.get_json(key)
    if cached:
        return EnrichBarsResponse(**cached)

    df_in = bars_to_dataframe(raw_bars)
    if df_in.empty or len(df_in) < 3:
        return JSONResponse(status_code=400, content={"error": "need >= 3 bars"})

    mode = (req.alligator_mode or "williams").strip().lower()
    if mode not in ("williams", "sma_shift"):
        return JSONResponse(status_code=400, content={"error": "alligator_mode must be williams|sma_shift"})

    enriched = enrich_ohlcv_frame(df_in, alligator_mode=mode)
    if enriched.empty:
        return JSONResponse(status_code=400, content={"error": "enrichment failed"})

    payload = {
        "symbol": req.symbol,
        "resolution": resolution,
        "bar_count": len(enriched),
        "points": frame_to_points(enriched),
    }
    cache.set_json(key, payload)
    return EnrichBarsResponse(**payload)


def _job_to_response(job: dict) -> HistoryBulkJobResponse:
    payload = {k: job.get(k) for k in HistoryBulkJobResponse.model_fields}
    # Queued jobs may omit newer keys — coerce int defaults.
    for key in (
        "segments_total",
        "segments_done",
        "segments_cached",
        "empty_segments",
        "bar_count",
    ):
        if payload.get(key) is None:
            payload[key] = 0
    if payload.get("elapsed_sec") is None:
        payload["elapsed_sec"] = 0.0
    if payload.get("limiter") is None:
        payload["limiter"] = {}
    return HistoryBulkJobResponse(**payload)


@app.post("/api/ta/history_bulk", response_model=HistoryBulkJobResponse)
async def ta_history_bulk(req: HistoryBulkRequest) -> HistoryBulkJobResponse | JSONResponse:
    """
    Fetch years of OHLCV+indicators without returning a giant JSON body.

    Uses weekly segments, adaptive TV pacing (avoids hard rate-limits), and a disk
    segment cache so repeats never re-hit TradingView. Result is a parquet path.
    """
    from .history_bulk import BulkJobRequest, start_bulk_job, wait_job

    try:
        resolution = normalize_resolution(req.resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    job_req = BulkJobRequest(
        symbol=req.symbol,
        resolution=resolution,
        start_ts=int(req.start_ts),
        end_ts=int(req.end_ts),
        alligator_mode=(req.alligator_mode or "williams").strip().lower(),
        concurrency=req.concurrency,
        use_cache=bool(req.use_cache),
        write_parquet=bool(req.write_parquet),
        out_dir=req.out_dir,
        chunk_days=req.chunk_days,
        fast_continuous=bool(req.fast_continuous),
    )
    job = await start_bulk_job(job_req)
    if req.wait:
        job = await wait_job(job["job_id"], timeout_sec=float(req.timeout_sec))
    return _job_to_response(job)


@app.get("/api/ta/history_bulk/{job_id}", response_model=HistoryBulkJobResponse)
def ta_history_bulk_status(job_id: str) -> HistoryBulkJobResponse | JSONResponse:
    from .history_bulk import get_job

    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": f"unknown job_id {job_id}"})
    return _job_to_response(job)


@app.get("/api/ta/history_bulk")
def ta_history_bulk_list(limit: int = Query(20, ge=1, le=100)) -> dict:
    from .history_bulk import list_jobs

    return {"jobs": list_jobs(limit=limit)}


@app.get("/api/ta/history", response_model=HistoryResponse)
def ta_history(
    symbol: str,
    resolution: Resolution = "5",
    count: int = Query(500, ge=50, le=5000),
    start_ts: int | None = None,
    end_ts: int | None = None,
    alligator_mode: str = Query("williams"),
) -> HistoryResponse | JSONResponse:
    """
    Historical indicator series from provider OHLCV (yfinance).
    For backtest parity with Alpaca, prefer POST /api/ta/enrich_bars with your bars.
    """
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    key = _cache_key(
        "history",
        {
            "symbol": symbol,
            "resolution": resolution,
            "count": count,
            "start": start_ts,
            "end": end_ts,
            "mode": alligator_mode,
        },
    )
    from .ohlcv_validate import OhlcvIntegrityError, expect_futures_bars, validate_ohlcv_frame

    cached = cache.get_json(key)
    if cached and int(cached.get("bar_count") or 0) > 0:
        # Live tip: never serve a cached series whose last bar trails the
        # expected last closed open (post-:05 CACHE_TTL reuse bug).
        if _live_cache_stale(cached, resolution, end_ts):
            cached = None
        else:
            try:
                pts = cached.get("points") or []
                if not pts and expect_futures_bars(start_ts, end_ts):
                    cached = None
                else:
                    return HistoryResponse(**cached)
            except Exception:
                cached = None
    if cached is not None and int(cached.get("bar_count") or 0) == 0:
        if expect_futures_bars(start_ts, end_ts):
            pass  # empty cache hit — refetch
        else:
            return HistoryResponse(**cached)
    elif cached is not None and not _is_live_history_request(end_ts):
        # Non-live non-empty already returned above; leftover empty handled.
        return HistoryResponse(**cached)

    bars = fetch_ohlcv(
        symbol=symbol,
        resolution=resolution,
        count=count,
        extra_bars=200,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    df = bars.df
    if df.empty:
        # Never cache empty history — transient tvkit misses were poisoning gap backfill.
        if expect_futures_bars(start_ts, end_ts):
            return JSONResponse(
                status_code=502,
                content={
                    "error": "empty_ohlcv",
                    "detail": (
                        f"TV returned 0 bars for {symbol} resolution={resolution} "
                        f"start_ts={start_ts} end_ts={end_ts}. Treat as fetch failure."
                    ),
                    "symbol": symbol,
                    "resolution": resolution,
                    "bar_count": 0,
                },
            )
        return HistoryResponse(symbol=symbol, resolution=resolution, bar_count=0, points=[])

    if start_ts is not None:
        df = df[df.index >= pd.to_datetime(start_ts, unit="s", utc=True)]
    if end_ts is not None:
        df = df[df.index <= pd.to_datetime(end_ts, unit="s", utc=True)]

    try:
        # Live tip windows: keep empty/NaN/OHLC corruption checks, skip density
        # floor (overnight seed_days=2 under Globex was false-positiving 502).
        df = validate_ohlcv_frame(
            df,
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            resolution=resolution,
            enforce_sparse=not _is_live_history_request(end_ts),
        )
    except OhlcvIntegrityError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": "ohlcv_integrity",
                "detail": str(exc),
                "symbol": symbol,
                "resolution": resolution,
                "bar_count": int(len(bars.df)),
            },
        )

    mode = (alligator_mode or "williams").strip().lower()
    enriched = enrich_ohlcv_frame(df, alligator_mode=mode)
    if enriched.empty:
        return JSONResponse(
            status_code=502,
            content={
                "error": "enrich_failed",
                "detail": f"enrichment produced 0 bars for {symbol}",
                "symbol": symbol,
                "resolution": resolution,
            },
        )
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "bar_count": len(enriched),
        "points": frame_to_points(enriched),
    }
    hist_ttl = None
    if end_ts is not None and end_ts < int(time.time()) - 900:
        hist_ttl = settings.historical_cache_ttl_seconds
    elif _is_live_history_request(end_ts):
        hist_ttl = int(settings.live_tip_cache_ttl_seconds)
    cache.set_json(key, payload, ttl_seconds=hist_ttl)
    return HistoryResponse(**payload)


@app.get("/api/ta/tip", response_model=TipResponse)
def ta_tip(
    symbol: str,
    resolution: Resolution = "5",
    count: int = Query(80, ge=50, le=250),
    alligator_mode: str = Query("williams"),
) -> TipResponse | JSONResponse:
    """Lightweight live tip — last N bars only (no multi-day seed).

    Single-flight per symbol so 4-stack bots share one TV pull per tip window.
    Leader does the fetch without holding a lock across TradingView I/O.
    """
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    bm = _resolution_minutes(resolution)
    flight_key = f"{symbol}|{resolution}|{count}|{alligator_mode}"
    key = _cache_key(
        "tip",
        {"symbol": symbol, "resolution": resolution, "count": count, "mode": alligator_mode},
    )

    cached = cache.get_json(key)
    if cached and int(cached.get("bar_count") or 0) > 0:
        if not _live_cache_stale(cached, resolution, end_ts=None):
            return TipResponse(**cached)

    leader = False
    wait_ev: threading.Event | None = None
    with _tip_inflight_guard:
        existing = _tip_inflight.get(flight_key)
        if existing is None:
            wait_ev = threading.Event()
            _tip_inflight[flight_key] = wait_ev
            leader = True
        else:
            wait_ev = existing

    if not leader:
        # Wait for leader (bounded) then serve cache / fail soft.
        wait_ev.wait(timeout=50.0)
        cached = cache.get_json(key)
        if cached and int(cached.get("bar_count") or 0) > 0:
            return TipResponse(**cached)
        return JSONResponse(
            status_code=503,
            content={
                "error": "tip_inflight_timeout",
                "detail": f"tip single-flight timed out for {symbol}",
                "symbol": symbol,
                "resolution": resolution,
            },
        )

    try:
        bars = fetch_ohlcv(
            symbol=symbol,
            resolution=resolution,
            count=count,
            extra_bars=20,
            start_ts=None,
            end_ts=None,
        )
        df = bars.df
        if df.empty:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "empty_ohlcv",
                    "detail": f"TV tip returned 0 bars for {symbol}",
                    "symbol": symbol,
                    "resolution": resolution,
                    "bar_count": 0,
                },
            )
        mode = (alligator_mode or "williams").strip().lower()
        enriched = enrich_ohlcv_frame(df, alligator_mode=mode)
        if enriched.empty:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "enrich_failed",
                    "detail": f"tip enrichment produced 0 bars for {symbol}",
                    "symbol": symbol,
                    "resolution": resolution,
                },
            )
        points = frame_to_points(enriched)
        tip_open = None
        tip_close = None
        close_px = None
        if points:
            tip_open = int(points[-1].get("t") or 0)
            if tip_open > 1_000_000_000_000:
                tip_open = tip_open // 1000
            tip_close = tip_open + bm * 60
            close_px = points[-1].get("c")
            try:
                close_px = float(close_px) if close_px is not None else None
            except Exception:
                close_px = None
        payload = {
            "symbol": symbol,
            "resolution": resolution,
            "bar_count": len(points),
            "tip_open_ts": tip_open,
            "tip_close_ts": tip_close,
            "close": close_px,
            "points": points,
        }
        cache.set_json(
            key, payload, ttl_seconds=int(settings.live_tip_cache_ttl_seconds)
        )
        return TipResponse(**payload)
    finally:
        with _tip_inflight_guard:
            _tip_inflight.pop(flight_key, None)
        if wait_ev is not None:
            wait_ev.set()


@app.post("/api/ta/indicators_at", response_model=IndicatorsResponse)
def indicators_at(req: IndicatorsAtRequest) -> IndicatorsResponse | JSONResponse:
    """Point-in-time indicators (backtest bar timestamp). Uses supplied bars or yfinance."""
    try:
        resolution = normalize_resolution(req.resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if req.use_supplied_bars and req.bars:
        df = bars_to_dataframe([b.model_dump() for b in req.bars])
        mode = (req.alligator_mode or "sma_shift").strip().lower()
        if mode not in ("williams", "sma_shift"):
            return JSONResponse(
                status_code=400,
                content={"error": "alligator_mode must be williams|sma_shift"},
            )
        enriched = enrich_ohlcv_frame(df, alligator_mode=mode)
        ts, ind = indicators_at_timestamp(enriched, req.as_of)
        return IndicatorsResponse(symbol=req.symbol, timestamp=ts, indicators=ind)

    bars = fetch_ohlcv(symbol=req.symbol, resolution=resolution, count=req.count, extra_bars=200)
    df = bars.df
    if df.empty:
        return IndicatorsResponse(symbol=req.symbol, timestamp=0, indicators={})
    ts, ind = indicators_at_timestamp(df, req.as_of)
    if not ind:
        specs = parse_indicator_list(",".join(req.indicators))
        sub = df[df.index <= pd.to_datetime(req.as_of, unit="s", utc=True)]
        if sub.empty:
            return IndicatorsResponse(symbol=req.symbol, timestamp=0, indicators={})
        ind = compute_latest(sub, specs=specs)
        ts = int(pd.Timestamp(sub.index[-1]).timestamp())
    return IndicatorsResponse(symbol=req.symbol, timestamp=ts, indicators=ind)


def _top_daytrading_rank(
    *,
    limit: int,
    resolution: Resolution,
    metric: str,
    symbol_list: list[str],
    max_price: float | None,
) -> TopDaytradingResponse | JSONResponse:
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    metric_norm = metric.strip().lower()
    if metric_norm not in {"daytrade_score", "dollar_volume", "volatility"}:
        return JSONResponse(status_code=400, content={"error": f"Unsupported metric: {metric}"})  # type: ignore[return-value]

    if len(symbol_list) > 10000:
        symbol_list = symbol_list[:10000]

    key = _cache_key(
        "top-daytrading",
        {"limit": limit, "resolution": resolution, "metric": metric_norm, "symbols": symbol_list, "max_price": max_price},
    )
    cached = cache.get_json(key)
    if cached:
        return TopDaytradingResponse(**cached)

    payload = rank_top_daytrading(
        symbols=symbol_list,
        resolution=resolution,
        limit=limit,
        metric=metric_norm,  # type: ignore[arg-type]
        max_price=max_price,
    )
    cache.set_json(key, payload)
    return TopDaytradingResponse(**payload)


@app.get("/api/ta/top-daytrading", response_model=TopDaytradingResponse)
def top_daytrading(
    limit: int = Query(20, ge=1, le=10000),
    resolution: Resolution = "1D",
    metric: str = Query("daytrade_score", description="daytrade_score | dollar_volume | volatility"),
    symbols: str | None = Query(None, description="Optional CSV list of symbols (TradingView style)"),
    max_price: float | None = Query(150.0, gt=0, description="Filter: last close <= max_price"),
) -> TopDaytradingResponse:
    symbol_list = (
        [s.strip() for s in symbols.split(",") if s.strip()]
        if symbols
        else list(DEFAULT_DAYTRADE_SYMBOLS)
    )
    return _top_daytrading_rank(
        limit=limit,
        resolution=resolution,
        metric=metric,
        symbol_list=symbol_list,
        max_price=max_price,
    )  # type: ignore[return-value]


@app.post("/api/ta/top-daytrading", response_model=TopDaytradingResponse)
def top_daytrading_post(req: TopDaytradingRequest) -> TopDaytradingResponse:
    symbol_list = list(req.symbols) if req.symbols else list(DEFAULT_DAYTRADE_SYMBOLS)
    return _top_daytrading_rank(
        limit=req.limit,
        resolution=req.resolution,
        metric=req.metric,
        symbol_list=symbol_list,
        max_price=req.max_price,
    )  # type: ignore[return-value]


@app.get("/api/ta/signal", response_model=SignalResponse)
def signal(
    symbol: str,
    resolution: Resolution = "15",
    count: int = Query(300, ge=80, le=5000),
    atr_mult: float = Query(1.5, gt=0, le=10),
    rr: float = Query(2.0, gt=0, le=10),
) -> SignalResponse:
    """
    Educational, rules-based signal (NOT financial advice).
    Returns a bias + risk levels derived from TA indicators and ATR sizing.
    """
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    key = _cache_key(
        "signal",
        {"symbol": symbol, "resolution": resolution, "count": count, "atr_mult": atr_mult, "rr": rr},
    )
    cached = cache.get_json(key)
    if cached:
        return SignalResponse(**cached)

    bars = fetch_ohlcv(symbol=symbol, resolution=resolution, count=count, extra_bars=200)
    df = bars.df
    if df.empty:
        payload = {
            "symbol": symbol,
            "timestamp": 0,
            "resolution": resolution,
            "indicators": {},
            "bias": "neutral",
            "reason": "no_data",
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "option_bias": "none",
        }
        cache.set_json(key, payload)
        return SignalResponse(**payload)

    latest_ts = int(pd.Timestamp(df.index[-1]).timestamp())
    indicators_map, plan = build_trade_plan(df, atr_mult=atr_mult, rr=rr)

    payload = {
        "symbol": symbol,
        "timestamp": latest_ts,
        "resolution": resolution,
        "indicators": indicators_map,
        "bias": plan.bias,
        "reason": plan.reason,
        "entry": plan.entry,
        "stop_loss": plan.stop_loss,
        "take_profit": plan.take_profit,
        "option_bias": plan.option_bias,
    }
    cache.set_json(key, payload)
    return SignalResponse(**payload)


@app.websocket("/ws/ta")
async def ws_ta(
    websocket: WebSocket,
    symbol: str,
    indicators: str,
    resolution: Resolution = "1D",
    count: int = 300,
    push_interval_seconds: Annotated[int, Query(ge=1, le=60)] = settings.ws_push_interval_seconds,
) -> None:
    """Live push. Must not call sync TV fetch on the event loop — that froze /health
    under 4-stack + alert WS and triggered ensure recycle stampedes.
    """
    await websocket.accept()
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1003)
        return

    # Wake / exit monitors only need a short tip; cap so WS never pulls 500+ bars.
    fetch_count = max(50, min(int(count), 120))
    try:
        while True:
            specs = parse_indicator_list(indicators)
            bars = await asyncio.to_thread(
                fetch_ohlcv,
                symbol=symbol,
                resolution=resolution,
                count=fetch_count,
                extra_bars=0,
            )
            df = bars.df
            if df.empty:
                payload = {"symbol": symbol, "timestamp": 0, "indicators": {}}
            else:
                latest_ts = int(pd.Timestamp(df.index[-1]).timestamp())
                values = compute_latest(df, specs=specs)
                payload = {"symbol": symbol, "timestamp": latest_ts, "indicators": values}

            await websocket.send_json(payload)
            await asyncio.sleep(push_interval_seconds)
    except WebSocketDisconnect:
        return

