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


class BatchResponse(BaseModel):
    results: list[IndicatorsResponse]


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

