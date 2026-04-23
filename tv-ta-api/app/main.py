from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .available_indicators import available_indicators_catalog
from .cache import Cache
from .data_provider import Resolution, fetch_ohlcv, normalize_resolution
from .indicators import compute_latest, compute_series, parse_indicator_list
from .models import BatchRequest, BatchResponse, IndicatorsResponse, SeriesResponse, SignalResponse, TopDaytradingResponse
from .screeners import DEFAULT_DAYTRADE_SYMBOLS, rank_top_daytrading
from .settings import settings
from .signals import build_trade_plan

app = FastAPI(title="TradingView TA API", version="0.1.0")
cache = Cache(ttl_seconds=settings.cache_ttl_seconds, redis_url=settings.redis_url)


def _cache_key(prefix: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
            {"symbol": item.symbol, "indicators": indicators_csv, "resolution": req.resolution},
        )
        cached = cache.get_json(key)
        if cached:
            results.append(IndicatorsResponse(**cached))
            continue

        bars = fetch_ohlcv(symbol=item.symbol, resolution=req.resolution, count=300, extra_bars=200)
        df = bars.df
        if df.empty:
            payload = {"symbol": item.symbol, "timestamp": 0, "indicators": {}}
            cache.set_json(key, payload)
            results.append(IndicatorsResponse(**payload))
            continue

        latest_ts = int(pd.Timestamp(df.index[-1]).timestamp())
        values = compute_latest(df, specs=specs)
        payload = {"symbol": item.symbol, "timestamp": latest_ts, "indicators": values}
        cache.set_json(key, payload)
        results.append(IndicatorsResponse(**payload))
    return BatchResponse(results=results)


@app.get("/api/ta/top-daytrading", response_model=TopDaytradingResponse)
def top_daytrading(
    limit: int = Query(20, ge=1, le=200),
    resolution: Resolution = "1D",
    metric: str = Query("daytrade_score", description="daytrade_score | dollar_volume | volatility"),
    symbols: str | None = Query(None, description="Optional CSV list of symbols (TradingView style)"),
    max_price: float | None = Query(None, gt=0, description="Filter: last close <= max_price"),
) -> TopDaytradingResponse:
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})  # type: ignore[return-value]

    metric_norm = metric.strip().lower()
    if metric_norm not in {"daytrade_score", "dollar_volume", "volatility"}:
        return JSONResponse(status_code=400, content={"error": f"Unsupported metric: {metric}"})  # type: ignore[return-value]

    symbol_list = (
        [s.strip() for s in symbols.split(",") if s.strip()]
        if symbols
        else list(DEFAULT_DAYTRADE_SYMBOLS)
    )
    # guardrails for demo provider limits
    if len(symbol_list) > 200:
        symbol_list = symbol_list[:200]

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
    await websocket.accept()
    try:
        resolution = normalize_resolution(resolution)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1003)
        return

    try:
        while True:
            specs = parse_indicator_list(indicators)
            bars = fetch_ohlcv(symbol=symbol, resolution=resolution, count=count, extra_bars=200)
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

