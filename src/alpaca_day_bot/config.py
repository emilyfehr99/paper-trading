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
    bar_buffer_maxlen: int = Field(default=3000, alias="BAR_BUFFER_MAXLEN")

    # Market-wide universe (free + efficient): build a liquid universe daily and trade/scan that.
    universe_enabled: bool = Field(default=False, alias="UNIVERSE_ENABLED")
    universe_max_symbols: int = Field(default=750, alias="UNIVERSE_MAX_SYMBOLS")
    universe_lookback_days: int = Field(default=20, alias="UNIVERSE_LOOKBACK_DAYS")
    universe_min_price: float = Field(default=2.0, alias="UNIVERSE_MIN_PRICE")
    universe_max_price: float = Field(default=20.0, alias="UNIVERSE_MAX_PRICE")
    universe_min_avg_dollar_vol: float = Field(default=20_000_000.0, alias="UNIVERSE_MIN_AVG_DOLLAR_VOL")

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
    max_notional_per_trade_usd: float = Field(default=0.0, alias="MAX_NOTIONAL_PER_TRADE_USD")  # 0 disables
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

    # Scheduled-tick fill confirmation (REST polling) to avoid missing websocket fills after process exit.
    fill_confirm_enabled: bool = Field(default=True, alias="FILL_CONFIRM_ENABLED")
    fill_confirm_timeout_s: float = Field(default=60.0, alias="FILL_CONFIRM_TIMEOUT_S")
    fill_confirm_poll_s: float = Field(default=3.0, alias="FILL_CONFIRM_POLL_S")

    # Strategy (V1 rules) — higher defaults = more candidate entries (also more risk).
    rsi_pullback_max: float = Field(default=42.0, alias="RSI_PULLBACK_MAX")
    volume_confirm_mult: float = Field(default=1.0, alias="VOLUME_CONFIRM_MULT")
    htf_rsi_min: float = Field(default=45.0, alias="HTF_RSI_MIN")
    atr_regime_max_mult: float = Field(default=2.5, alias="ATR_REGIME_MAX_MULT")
    aggressive_mode: bool = Field(default=False, alias="AGGRESSIVE_MODE")
    indicator_provider: str = Field(default="local", alias="INDICATOR_PROVIDER")  # local | taapi | tvta
    taapi_secret: str | None = Field(default=None, alias="TAAPI_SECRET")
    taapi_confirm_on_trade: bool = Field(default=True, alias="TAAPI_CONFIRM_ON_TRADE")
    # If TAAPI is rate-limited / forbidden, do not block trades (still record taapi in features).
    taapi_fail_open: bool = Field(default=True, alias="TAAPI_FAIL_OPEN")
    # Your TV TA API (FastAPI) base URL, e.g. http://127.0.0.1:8000
    tvta_api_base_url: str | None = Field(default=None, alias="TVTA_API_BASE_URL")
    # Prefix used to convert symbols into TradingView style, e.g. NYSE:NIO. Empty => raw symbol.
    tvta_symbol_prefix: str = Field(default="NYSE", alias="TVTA_SYMBOL_PREFIX")

    # ML model layer (trained from ledger labels)
    model_enabled: bool = Field(default=False, alias="MODEL_ENABLED")
    model_min_proba: float = Field(default=0.55, alias="MODEL_MIN_PROBA")
    top_n_per_tick: int = Field(default=2, alias="TOP_N_PER_TICK")
    model_path: str = Field(default="state/models/latest.joblib", alias="MODEL_PATH")

    # Exits (bracket-like)
    stop_loss_atr_mult: float = Field(default=1.5, alias="STOP_LOSS_ATR_MULT")
    take_profit_r_mult: float = Field(default=1.5, alias="TAKE_PROFIT_R_MULT")

    # Smarter exits (in addition to TP/SL brackets)
    max_hold_minutes: float = Field(default=0.0, alias="MAX_HOLD_MINUTES")  # 0 disables time-exit
    model_exit_enabled: bool = Field(default=False, alias="MODEL_EXIT_ENABLED")
    model_exit_min_proba: float = Field(default=0.45, alias="MODEL_EXIT_MIN_PROBA")
    dynamic_hold_enabled: bool = Field(default=True, alias="DYNAMIC_HOLD_ENABLED")
    base_hold_minutes: float = Field(default=45.0, alias="BASE_HOLD_MINUTES")
    min_hold_minutes: float = Field(default=10.0, alias="MIN_HOLD_MINUTES")
    max_hold_minutes_dynamic: float = Field(default=90.0, alias="MAX_HOLD_MINUTES_DYNAMIC")
    hold_atr_ratio_weight: float = Field(default=-15.0, alias="HOLD_ATR_RATIO_WEIGHT")
    hold_model_proba_weight: float = Field(default=20.0, alias="HOLD_MODEL_PROBA_WEIGHT")

    # Shorts (paper): enable short entries and separate HTF gate.
    enable_shorts: bool = Field(default=False, alias="ENABLE_SHORTS")
    htf_rsi_max_short: float = Field(default=55.0, alias="HTF_RSI_MAX_SHORT")
    rsi_rebound_min_short: float = Field(default=58.0, alias="RSI_REBOUND_MIN_SHORT")

    # Live diagnostics: periodic log of closest-to-BUY symbols (see signal_scan in logs).
    signal_scan_interval_s: float = Field(default=60.0, alias="SIGNAL_SCAN_INTERVAL_S")

    # News: alpaca (default) | alphavantage | google_rss | tickertick | both | combo
    news_provider: str = Field(default="alpaca", alias="NEWS_PROVIDER")
    alphavantage_api_key: str | None = Field(default=None, alias="ALPHAVANTAGE_API_KEY")

    # Alpaca Market Data news (plan-dependent). Gate modes: off | log_only | skip_if_any | skip_if_busy
    news_fetch_enabled: bool = Field(default=True, alias="NEWS_FETCH_ENABLED")
    news_lookback_hours: float = Field(default=6.0, alias="NEWS_LOOKBACK_HOURS")
    news_limit: int = Field(default=8, alias="NEWS_LIMIT")
    news_gate_mode: str = Field(default="log_only", alias="NEWS_GATE_MODE")
    news_busy_min_articles: int = Field(default=5, alias="NEWS_BUSY_MIN_ARTICLES")
    # If true, block entries when news_event_risk==1 (earnings/offering/halt/etc).
    news_block_on_event_risk: bool = Field(default=True, alias="NEWS_BLOCK_ON_EVENT_RISK")

    # Signal "accuracy": label BUY rows with forward return vs signal-time close after min wall-clock age.
    signal_accuracy_enabled: bool = Field(default=True, alias="SIGNAL_ACCURACY_ENABLED")
    signal_accuracy_min_age_minutes: float = Field(default=15.0, alias="SIGNAL_ACCURACY_MIN_AGE_MINUTES")

    # Robustness: smaller cost grid / walk-forward / sweep (GitHub Actions sets true by default in workflow).
    robustness_light: bool = Field(default=False, alias="ROBUSTNESS_LIGHT")

    # Storage
    state_dir: str = Field(default="state", alias="STATE_DIR")
    reports_dir: str = Field(default="reports", alias="REPORTS_DIR")
    # Optional path to day_trade_recommendations_*.json; default tries reports/day_trade_recommendations_latest.json
    recommendations_json: str | None = Field(default=None, alias="RECOMMENDATIONS_JSON")

    # Order type
    # market = market-entry bracket; limit = limit-entry bracket (still uses TP/SL legs).
    entry_order_type: str = Field(default="market", alias="ENTRY_ORDER_TYPE")
    # For limit entries, move limit away from last price by this many bps.
    # BUY uses (1 - bps/10_000), SHORT uses (1 + bps/10_000). 0 = use last price.
    limit_entry_offset_bps: float = Field(default=0.0, alias="LIMIT_ENTRY_OFFSET_BPS")

    # Accuracy guards / gating
    # Avoid stacking highly correlated positions (0 disables).
    max_corr_with_open_positions: float = Field(default=0.0, alias="MAX_CORR_WITH_OPEN_POSITIONS")
    corr_lookback_bars_1m: int = Field(default=60, alias="CORR_LOOKBACK_BARS_1M")

    # Model gating by regime (optional JSON mapping). Example:
    # {"trend_low_vol":0.55,"chop_high_vol":0.70}
    model_min_proba_by_regime_json: str = Field(default="", alias="MODEL_MIN_PROBA_BY_REGIME_JSON")

    # Model exit controls
    model_exit_min_hold_minutes: float = Field(default=5.0, alias="MODEL_EXIT_MIN_HOLD_MINUTES")

    # Synthetic exits: submit entry without bracket, then attach OCO TP/SL as separate orders.
    synthetic_exits_enabled: bool = Field(default=False, alias="SYNTHETIC_EXITS_ENABLED")

    # Virtual options (mock calls/puts; NOT sent to Alpaca)
    sim_options_enabled: bool = Field(default=False, alias="SIM_OPTIONS_ENABLED")
    # Risk: at worst you lose the notional (like long premium). Profit scales with leverage * underlying move.
    sim_options_leverage: float = Field(default=6.0, alias="SIM_OPTIONS_LEVERAGE")
    # If 0, reuses MAX_NOTIONAL_PER_TRADE_USD; otherwise per-virtual-option cap.
    sim_options_notional_usd: float = Field(default=0.0, alias="SIM_OPTIONS_NOTIONAL_USD")

    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.market_tz)


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

