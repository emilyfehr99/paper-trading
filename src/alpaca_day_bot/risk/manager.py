from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from alpaca_day_bot.storage.ledger import Ledger


@dataclass(frozen=True)
class RiskDecision:
    allow: bool
    reason: str
    notional_usd: float = 0.0
    qty: float = 0.0
    stop_distance: float = 0.0


class RiskManager:
    def __init__(
        self,
        *,
        max_gross_exposure_pct: float,
        max_positions: int,
        max_trades_per_day: int,
        max_daily_loss_pct: float,
        risk_per_trade_pct: float,
        per_symbol_cooldown_s: int = 600,
        daily_profit_target_usd: float = 0.0,
    ) -> None:
        self._max_gross_exposure_pct = max_gross_exposure_pct
        self._max_positions = max_positions
        self._max_trades_per_day = max_trades_per_day
        self._max_daily_loss_pct = max_daily_loss_pct
        self._risk_per_trade_pct = risk_per_trade_pct
        self._cooldown_s = per_symbol_cooldown_s
        self._daily_profit_target_usd = float(daily_profit_target_usd)

        self._trading_day: date | None = None
        self._start_equity: float | None = None
        self._trades_today: int = 0
        self._last_trade_ts_by_symbol: dict[str, datetime] = {}

    def reset_day_if_needed(self, trading_date: date, equity: float) -> None:
        """Reset daily counters on market session date (not UTC midnight)."""
        if self._trading_day != trading_date:
            self._trading_day = trading_date
            self._start_equity = equity
            self._trades_today = 0
            self._last_trade_ts_by_symbol = {}
        elif self._start_equity is None:
            self._start_equity = equity

    def rehydrate_from_ledger(self, ledger: Ledger, trading_date: date, tz: ZoneInfo) -> None:
        """Restore same-day trade count / cooldowns from SQLite (GitHub Actions ticks)."""
        stats = ledger.submitted_buy_stats_for_trading_date(trading_date, tz)
        self._trading_day = trading_date
        self._trades_today = int(stats["count"])
        self._last_trade_ts_by_symbol = dict(stats["last_by_symbol"])

    def daily_loss_breached(self, equity: float) -> bool:
        if self._start_equity is None:
            return False
        return equity <= self._start_equity * (1.0 - self._max_daily_loss_pct)

    def daily_profit_target_reached(self, equity: float) -> bool:
        if self._daily_profit_target_usd <= 0:
            return False
        if self._start_equity is None:
            return False
        return (equity - self._start_equity) >= self._daily_profit_target_usd

    def can_trade_symbol(self, symbol: str, now_utc: datetime) -> bool:
        last = self._last_trade_ts_by_symbol.get(symbol)
        if last is None:
            return True
        return (now_utc - last).total_seconds() >= self._cooldown_s

    def register_trade(self, symbol: str, now_utc: datetime) -> None:
        self._trades_today += 1
        self._last_trade_ts_by_symbol[symbol] = now_utc

    def decide_entry(
        self,
        *,
        symbol: str,
        equity: float,
        gross_exposure_usd: float,
        open_positions: int,
        now_utc: datetime,
        trading_date: date,
        price: float,
        stop_distance: float,
    ) -> RiskDecision:
        self.reset_day_if_needed(trading_date, equity)

        if self.daily_loss_breached(equity):
            return RiskDecision(False, "daily_loss_limit")

        if self.daily_profit_target_reached(equity):
            return RiskDecision(False, "daily_profit_target")

        if self._trades_today >= self._max_trades_per_day:
            return RiskDecision(False, "max_trades_per_day")

        if open_positions >= self._max_positions:
            return RiskDecision(False, "max_positions")

        if not self.can_trade_symbol(symbol, now_utc):
            return RiskDecision(False, "symbol_cooldown")

        max_gross = equity * self._max_gross_exposure_pct
        if gross_exposure_usd >= max_gross:
            return RiskDecision(False, "max_gross_exposure")

        # Position sizing by risk: risk_per_trade% of equity / stop_distance gives qty.
        risk_budget = equity * self._risk_per_trade_pct
        if stop_distance <= 1e-8:
            return RiskDecision(False, "bad_stop_distance")
        if price <= 0:
            return RiskDecision(False, "bad_price")

        remaining = max(0.0, max_gross - gross_exposure_usd)
        qty = risk_budget / stop_distance
        notional = qty * price

        # Cap notional to remaining gross; allow fractional sizing.
        if notional > remaining and remaining > 0:
            qty = remaining / price
            notional = qty * price

        if notional <= 5.0:
            return RiskDecision(False, "too_small_notional")

        return RiskDecision(True, "ok", notional_usd=notional, qty=qty, stop_distance=stop_distance)


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

