from __future__ import annotations

from typing import TYPE_CHECKING

from alpaca_day_bot.config import Settings

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient


def make_trading_client(settings: Settings) -> TradingClient:
    # Non-negotiable: paper-only.
    from alpaca.trading.client import TradingClient

    return TradingClient(
        api_key=settings.apca_api_key_id,
        secret_key=settings.apca_api_secret_key,
        paper=True,
    )

