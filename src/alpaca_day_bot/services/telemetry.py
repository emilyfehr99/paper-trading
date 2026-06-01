import logging
from prometheus_client import Counter, Gauge, Histogram, start_http_server

log = logging.getLogger("alpaca_day_bot.telemetry")

# --- PROMETHEUS METRICS SPECIFICATION ---

# Operational Latency & Telemetry
EXECUTION_LATENCY = Histogram(
    "aetheris_execution_latency_seconds",
    "Time taken to execute trades on Alpaca in seconds"
)

WEBSOCKET_DROPS = Counter(
    "aetheris_websocket_drops",
    "Total count of micro-drops or connection disconnects"
)

# Order Operations Metrics
CHASED_ORDERS = Counter(
    "aetheris_chased_orders",
    "Total count of order chase events executed by limit chase loop"
)

CANCELLED_ORDERS = Counter(
    "aetheris_cancelled_orders",
    "Total count of order cancel events triggered"
)

# Risk & Mathematical Diagnostics
KELLY_RISK_PCT = Gauge(
    "aetheris_kelly_risk_pct",
    "Current Kelly risk multiplier mapped to the active symbol",
    ["symbol"]
)

GARCH_VOLATILITY = Gauge(
    "aetheris_garch_volatility",
    "Rolling statistical GARCH predicted volatility mapped to the active symbol",
    ["symbol"]
)

ESTIMATED_EQUITY = Gauge(
    "aetheris_estimated_equity_usd",
    "Total estimated portfolio value/equity in USD"
)

DAILY_LOSS_PCT = Gauge(
    "aetheris_daily_loss_pct",
    "Trailing daily loss percentage relative to starting equity"
)

CALIBRATION_DRIFT = Gauge(
    "aetheris_calibration_drift",
    "Absolute deviation tracking error between Platt meta-learner win probability (p) and realized trade outcomes",
    ["symbol"]
)

PROCESS_MEMORY_MB = Gauge(
    "aetheris_process_memory_mb",
    "Current RSS memory utilization of the Aetheris bot process in Megabytes"
)

# Server State Trackers
_METRICS_SERVER_STARTED = False

def start_metrics_server(port: int = 8000) -> bool:
    """
    Launch a local Prometheus metrics server inside a separate thread to serve telemetry.
    """
    global _METRICS_SERVER_STARTED
    if _METRICS_SERVER_STARTED:
        return True
    try:
        start_http_server(port)
        log.info(f"Prometheus Metrics Server successfully launched at http://localhost:{port}/metrics")
        _METRICS_SERVER_STARTED = True
        return True
    except Exception as e:
        log.error(f"Failed to launch Prometheus Metrics Server: {e}", exc_info=True)
        return False
