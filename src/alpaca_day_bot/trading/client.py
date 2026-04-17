from __future__ import annotations

from alpaca.trading.client import TradingClient

from alpaca_day_bot.config import Settings


def make_trading_client(settings: Settings) -> TradingClient:
    # Non-negotiable: paper-only.
    return TradingClient(
        api_key=settings.apca_api_key_id,
        secret_key=settings.apca_api_secret_key,
        paper=True,
    )

