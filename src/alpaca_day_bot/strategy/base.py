from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    action: str  # "BUY" or "HOLD"
    reason: str
    features: dict[str, Any] | None = None


class BaseStrategy(ABC):
    @abstractmethod
    def decide(self, *, symbol: str, df_1m, df_15m) -> StrategySignal | None:
        raise NotImplementedError

