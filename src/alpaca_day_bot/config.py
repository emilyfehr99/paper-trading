from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    apca_api_key_id: str = Field(alias="APCA_API_KEY_ID")
    apca_api_secret_key: str = Field(alias="APCA_API_SECRET_KEY")

    # Universe + data
    symbols: list[str] = Field(
        default_factory=lambda: [
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "GOOGL",
            "AMD",
        ],
        alias="SYMBOLS",
        description="Comma-separated symbols or JSON list.",
    )
    bar_timeframe: str = Field(default="1Min", alias="BAR_TIMEFRAME")
    # "rest" = poll 1m bars via HTTP (default; avoids Alpaca market-data websocket limit with trading stream).
    # "websocket" = StockDataStream live bars (needs spare MD websocket capacity).
    market_data_mode: str = Field(default="rest", alias="MARKET_DATA_MODE")
    rest_bar_poll_interval_s: float = Field(default=15.0, alias="REST_BAR_POLL_INTERVAL_S")
    # IEX (free) bars are delayed vs real time; requesting `end=now` often returns SIP / permission errors.
    rest_bar_end_lag_minutes: float = Field(default=18.0, alias="REST_BAR_END_LAG_MINUTES")

    # Run modes
    observe_only: bool = Field(default=False, alias="OBSERVE_ONLY")

    # Trading hours (US equities). We keep it simple and rely on Alpaca clock in code too.
    market_tz: str = Field(default="America/New_York", alias="MARKET_TZ")
    trade_start: time = Field(default=time(9, 35), alias="TRADE_START")
    trade_end: time = Field(default=time(15, 50), alias="TRADE_END")

    # Risk limits
    starting_equity_usd: float = Field(default=1000.0, alias="STARTING_EQUITY_USD")
    max_gross_exposure_pct: float = Field(default=0.5, alias="MAX_GROSS_EXPOSURE_PCT")
    max_positions: int = Field(default=5, alias="MAX_POSITIONS")
    max_trades_per_day: int = Field(default=20, alias="MAX_TRADES_PER_DAY")
    max_daily_loss_pct: float = Field(default=0.03, alias="MAX_DAILY_LOSS_PCT")
    risk_per_trade_pct: float = Field(default=0.005, alias="RISK_PER_TRADE_PCT")
    per_symbol_cooldown_s: int = Field(default=600, alias="PER_SYMBOL_COOLDOWN_S")
    # Stop opening new trades once equity is up this much vs session start (0 = off).
    daily_profit_target_usd: float = Field(default=100.0, alias="DAILY_PROFIT_TARGET_USD")

    # Realism (modeled; does not affect Alpaca fills)
    slippage_bps: float = Field(default=1.0, alias="SLIPPAGE_BPS")
    commission_bps: float = Field(default=0.5, alias="COMMISSION_BPS")
    spread_proxy_k: float = Field(default=0.10, alias="SPREAD_PROXY_K")
    spread_proxy_min: float = Field(default=0.01, alias="SPREAD_PROXY_MIN")

    # Guardrails
    open_delay_minutes: int = Field(default=5, alias="OPEN_DELAY_MINUTES")
    market_context_filter: bool = Field(default=False, alias="MARKET_CONTEXT_FILTER")
    spy_5m_rsi_min: float = Field(default=35.0, alias="SPY_5M_RSI_MIN")

    # Strategy (V1 rules) — higher defaults = more candidate entries (also more risk).
    rsi_pullback_max: float = Field(default=42.0, alias="RSI_PULLBACK_MAX")
    volume_confirm_mult: float = Field(default=1.0, alias="VOLUME_CONFIRM_MULT")
    htf_rsi_min: float = Field(default=45.0, alias="HTF_RSI_MIN")
    atr_regime_max_mult: float = Field(default=2.5, alias="ATR_REGIME_MAX_MULT")

    # Exits (bracket-like)
    stop_loss_atr_mult: float = Field(default=1.5, alias="STOP_LOSS_ATR_MULT")
    take_profit_r_mult: float = Field(default=1.5, alias="TAKE_PROFIT_R_MULT")

    # Live diagnostics: periodic log of closest-to-BUY symbols (see signal_scan in logs).
    signal_scan_interval_s: float = Field(default=60.0, alias="SIGNAL_SCAN_INTERVAL_S")

    # Storage
    state_dir: str = Field(default="state", alias="STATE_DIR")
    reports_dir: str = Field(default="reports", alias="REPORTS_DIR")
    # Optional path to day_trade_recommendations_*.json; default tries reports/day_trade_recommendations_latest.json
    recommendations_json: str | None = Field(default=None, alias="RECOMMENDATIONS_JSON")

    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.market_tz)


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

