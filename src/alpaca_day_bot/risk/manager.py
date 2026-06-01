from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

log = logging.getLogger("alpaca_day_bot")

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
        max_notional_per_trade_usd: float = 0.0,
        per_symbol_cooldown_s: int = 600,
        daily_profit_target_usd: float = 0.0,
    ) -> None:
        self._max_gross_exposure_pct = max_gross_exposure_pct
        self._max_positions = max_positions
        self._max_trades_per_day = max_trades_per_day
        self._max_daily_loss_pct = max_daily_loss_pct
        self._risk_per_trade_pct = risk_per_trade_pct
        self._max_notional_per_trade_usd = float(max_notional_per_trade_usd)
        self._cooldown_s = per_symbol_cooldown_s
        self._daily_profit_target_usd = float(daily_profit_target_usd)
        
        # --- ADVANCED RISK: FRACTIONAL KELLY CRITERION ---
        # Formula: f* = (bp - q) / b
        # b = Win/Loss Ratio (2.0)
        # p = Win Probability (0.55)
        # q = Loss Probability (0.45)
        b, p = 2.0, 0.55
        q = 1.0 - p
        kelly_f = (b * p - q) / b # 0.325
        # Fractional Kelly (1/10th for professional safety)
        self._kelly_risk_pct = kelly_f * 0.10 # ~0.0325 (3.25%)
        
        # Override risk_per_trade with Kelly if it's more optimal
        if self._risk_per_trade_pct <= 0.01: # If user didn't set a high custom risk
             self._risk_per_trade_pct = self._kelly_risk_pct
             log.info(f"[bold cyan]Kelly Criterion Active:[/bold cyan] Risking {self._risk_per_trade_pct*100:.2f}% per trade.")

        self._trading_day: date | None = None
        self._start_acct_equity: float | None = None
        self._start_virtual_equity: float | None = None
        self._start_equity: float | None = None
        self._trades_today: int = 0
        self._last_trade_ts_by_symbol: dict[str, datetime] = {}

    def reset_day_if_needed(self, trading_date: date, equity: float, virtual_equity: float | None = None) -> None:
        """Reset daily counters on market session date (not UTC midnight)."""
        if virtual_equity is None:
            virtual_equity = equity

        if self._trading_day != trading_date:
            self._trading_day = trading_date
            self._start_acct_equity = equity
            self._start_virtual_equity = virtual_equity
            self._start_equity = virtual_equity
            self._trades_today = 0
            self._last_trade_ts_by_symbol = {}
        else:
            if self._start_acct_equity is None:
                self._start_acct_equity = equity
            if self._start_virtual_equity is None:
                self._start_virtual_equity = virtual_equity
                self._start_equity = virtual_equity

    def rehydrate_from_ledger(self, ledger: Ledger, trading_date: date, tz: ZoneInfo) -> None:
        """Restore same-day trade count / cooldowns from SQLite (GitHub Actions ticks)."""
        stats = ledger.submitted_entry_stats_for_trading_date(trading_date, tz)
        self._trading_day = trading_date
        self._trades_today = int(stats["count"])
        self._last_trade_ts_by_symbol = dict(stats["last_by_symbol"])

    def daily_loss_breached(self, equity: float) -> bool:
        if self._max_daily_loss_pct <= 0:
            return False
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
        
    def get_recent_trades(self, count: int = 10) -> list:
        """Get recent trades for adaptive threshold calculation."""
        # Track recent trades with P&L for adaptive threshold logic
        if not hasattr(self, '_recent_trades'):
            self._recent_trades = []
        return self._recent_trades[-count:] if self._recent_trades else []
    
    def add_trade_result(self, pnl: float) -> None:
        """Add a completed trade result for adaptive threshold calculation."""
        if not hasattr(self, '_recent_trades'):
            self._recent_trades = []
        from dataclasses import dataclass
        @dataclass
        class TradeResult:
            pnl: float
            ts: datetime
        self._recent_trades.append(TradeResult(pnl=pnl, ts=datetime.now(timezone.utc)))
        # Keep only last 20 trades
        if len(self._recent_trades) > 20:
            self._recent_trades = self._recent_trades[-20:]

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
        ml_proba: float | None = None, # Dynamic Conviction
    ) -> RiskDecision:
        self.reset_day_if_needed(trading_date, equity)

        # --- DYNAMIC POSITION SIZING (Kelly Criterion) ---
        risk_pct = self._risk_per_trade_pct
        if ml_proba is not None:
            # Kelly Formula: f* = (bp - q) / b
            # We assume b (Win/Loss) = 2.0 based on TP/SL ratio (2.5:1)
            b = 2.0 
            p = ml_proba
            q = 1.0 - p
            raw_kelly = (b * p - q) / b
            # Fractional Kelly (1/5th for aggressive but safe growth) floored at 0.0
            dynamic_risk = max(0.0, raw_kelly * 0.20) 
            risk_pct = min(0.05, dynamic_risk) # Cap at 5% total equity risk
            log.info(f"Dynamic Kelly Scaling: {symbol} @ {ml_proba:.2f} -> Risk: {risk_pct*100:.2f}%")

        if self.daily_loss_breached(equity):
            return RiskDecision(False, "daily_loss_limit")

        if self.daily_profit_target_reached(equity):
            return RiskDecision(False, "daily_profit_target")

        if self._max_trades_per_day > 0 and self._trades_today >= self._max_trades_per_day:
            return RiskDecision(False, "max_trades_per_day")

        if self._max_positions > 0 and open_positions >= self._max_positions:
            return RiskDecision(False, "max_positions")

        if not self.can_trade_symbol(symbol, now_utc):
            return RiskDecision(False, "symbol_cooldown")

        max_gross = None
        if self._max_gross_exposure_pct > 0:
            max_gross = equity * self._max_gross_exposure_pct
            # log.debug(f"RISK CHECK: Exposure={gross_exposure_usd:.2f}, Max={max_gross:.2f}")
            if gross_exposure_usd >= max_gross:
                log.warning(f"Risk Blocked {symbol}: max_gross_exposure (Current: {gross_exposure_usd:.2f}, Max: {max_gross:.2f})")
                return RiskDecision(False, "max_gross_exposure")

        # Position sizing by risk: risk_pct of equity / stop_distance gives qty.
        risk_budget = equity * risk_pct
        if stop_distance <= 1e-8:
            return RiskDecision(False, "bad_stop_distance")
        if price <= 0:
            return RiskDecision(False, "bad_price")

        remaining = float("inf") if max_gross is None else max(0.0, max_gross - gross_exposure_usd)
        
        # Position sizing by risk budget:
        # If we risk $20 (budget) and our stop loss is $1 away, we buy 20 shares.
        qty = risk_budget / stop_distance if stop_distance > 0 else 0.0
        notional = qty * price

        # Hard cap per-trade notional (paper realism / user budget).
        if self._max_notional_per_trade_usd > 0:
            notional = min(float(notional), float(self._max_notional_per_trade_usd))
            qty = notional / price if price > 0 else 0.0

        # Cap notional to remaining gross; allow fractional sizing.
        if remaining != float("inf") and notional > remaining and remaining > 0:
            qty = remaining / price
            notional = qty * price

        if notional <= 5.0:
            return RiskDecision(False, "too_small_notional")

        return RiskDecision(True, "ok", notional_usd=notional, qty=qty, stop_distance=stop_distance)


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

