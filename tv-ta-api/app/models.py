from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .data_provider import Resolution


class IndicatorsResponse(BaseModel):
    symbol: str
    timestamp: int
    indicators: dict[str, float]


class SeriesResponse(BaseModel):
    symbol: str
    indicator: str
    period: int
    resolution: Resolution
    points: list[dict[str, Any]]  # {t:int, v:float}


class BatchItem(BaseModel):
    symbol: str
    indicators: list[str] = Field(min_length=1)


class BatchRequest(BaseModel):
    resolution: Resolution = "1D"
    items: list[BatchItem] = Field(min_length=1)
    count: int = Field(default=300, ge=50, le=5000)
    as_of: int | None = Field(
        default=None,
        description="Optional unix UTC — snapshot indicators at or before this time (yfinance bars)",
    )


class BatchResponse(BaseModel):
    results: list[IndicatorsResponse]


class OhlcvBar(BaseModel):
    t: int = Field(description="Bar open time, unix seconds UTC")
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


class EnrichBarsRequest(BaseModel):
    """Backtest: send historical Alpaca/yfinance OHLCV; receive per-bar indicators."""

    symbol: str
    resolution: Resolution = "1"
    bars: list[OhlcvBar] = Field(min_length=3, max_length=8000)
    alligator_mode: str = Field(
        default="williams",
        description="williams (TVTA) or sma_shift (legacy bot strategy lines)",
    )


class EnrichBarsPoint(BaseModel):
    t: int
    o: float | None = None
    h: float | None = None
    l: float | None = None
    c: float | None = None
    v: float | None = None
    indicators: dict[str, float]


class EnrichBarsResponse(BaseModel):
    symbol: str
    resolution: Resolution
    bar_count: int
    points: list[EnrichBarsPoint]


class IndicatorsAtRequest(BaseModel):
    symbol: str
    resolution: Resolution = "5"
    as_of: int = Field(description="Unix seconds UTC — indicators as of this bar")
    indicators: list[str] = Field(default_factory=lambda: ["rsi:14", "macd", "alligator"])
    count: int = Field(default=500, ge=50, le=5000)
    use_supplied_bars: bool = False
    bars: list[OhlcvBar] | None = None
    alligator_mode: str = Field(
        default="sma_shift",
        description="williams (TVTA) or sma_shift (legacy bot strategy lines)",
    )


class HistoryRequest(BaseModel):
    symbol: str
    resolution: Resolution = "5"
    count: int = Field(default=500, ge=50, le=5000)
    start_ts: int | None = None
    end_ts: int | None = None
    alligator_mode: str = "williams"


class HistoryResponse(BaseModel):
    symbol: str
    resolution: Resolution
    bar_count: int
    points: list[EnrichBarsPoint]


class TipResponse(BaseModel):
    """Lightweight live tip — last N bars only (no multi-day seed)."""

    symbol: str
    resolution: Resolution
    bar_count: int
    tip_open_ts: int | None = Field(
        default=None, description="Last bar open unix UTC (history ``t``)"
    )
    tip_close_ts: int | None = Field(
        default=None,
        description="Close-keyed tip = tip_open_ts + resolution minutes",
    )
    close: float | None = None
    points: list[EnrichBarsPoint] = Field(default_factory=list)


class HistoryBulkRequest(BaseModel):
    """Multi-year / large-range history job (writes parquet; does not return 400k JSON points)."""

    symbol: str
    resolution: Resolution = "5"
    start_ts: int = Field(description="Unix seconds UTC inclusive start")
    end_ts: int = Field(description="Unix seconds UTC inclusive end")
    alligator_mode: str = "williams"
    concurrency: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Override starting concurrency (adaptive pacing still applies)",
    )
    use_cache: bool = Field(default=True, description="Reuse disk-cached weekly segments")
    write_parquet: bool = True
    out_dir: str | None = Field(
        default=None,
        description="Optional output directory for job parquet (default: state/tvta_bulk_cache/jobs)",
    )
    chunk_days: int | None = Field(
        default=None,
        ge=7,
        le=62,
        description="Calendar days per TV fetch (default 28). Larger = fewer round-trips.",
    )
    fast_continuous: bool = Field(
        default=True,
        description="Try continuous-contract fetch first before named-contract fallback",
    )
    wait: bool = Field(
        default=False,
        description="If true, block until job completes (or timeout_sec)",
    )
    timeout_sec: float = Field(default=600.0, ge=30.0, le=7200.0)


class HistoryBulkJobResponse(BaseModel):
    job_id: str
    status: str
    symbol: str | None = None
    resolution: str | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    created_at: float | None = None
    updated_at: float | None = None
    segments_total: int = 0
    segments_done: int = 0
    segments_cached: int = 0
    empty_segments: int = 0
    bar_count: int = 0
    elapsed_sec: float = 0.0
    parquet_path: str | None = None
    error: str | None = None
    warning: str | None = None
    serial_tv: bool | None = None
    limiter: dict[str, Any] = Field(default_factory=dict)
    preview_points: list[dict[str, Any]] | None = None


class TopDaytradingRequest(BaseModel):
    limit: int = Field(20, ge=1, le=10000)
    resolution: Resolution = "1D"
    metric: str = Field("daytrade_score", description="daytrade_score | dollar_volume | volatility")
    symbols: list[str] | None = Field(
        default=None,
        description="Optional TradingView-style symbols; omit for server default pool",
    )
    max_price: float | None = Field(default=150.0, gt=0)


class TopDaytradingResult(BaseModel):
    symbol: str
    score: float | None = None
    dollar_volume_5: float | None = None
    vol_ann: float | None = None
    last_close: float | None = None


class TopDaytradingResponse(BaseModel):
    timestamp: int
    resolution: Resolution
    metric: str
    results: list[TopDaytradingResult]


class SignalResponse(BaseModel):
    symbol: str
    timestamp: int
    resolution: Resolution
    indicators: dict[str, float]
    bias: str
    reason: str
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    option_bias: str

