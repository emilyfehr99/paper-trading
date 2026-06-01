import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo
from alpaca_day_bot.data.stream import OrderBookBuffer

log = logging.getLogger("alpaca_day_bot.moc")

class MocImbalanceService:
    def __init__(self):
        self.ny_tz = ZoneInfo("America/New_York")

    def get_moc_imbalance_shares(self, symbol: str, current_time: datetime, book_buffer: OrderBookBuffer) -> float:
        """
        Calculates the net order book imbalance past 3:30 PM EST (15:30) as a proxy for Market-On-Close (MOC) imbalances.
        Returns:
            - Net imbalance shares (Positive = Buyer Imbalance, Negative = Seller Imbalance).
            - Returns 0.0 if prior to 3:30 PM EST or if order book data is missing.
        """
        # Convert current time to America/New_York to inspect market hours precisely
        ny_time = current_time.astimezone(self.ny_tz)
        
        # MOC imbalance calculation starts only after 3:30 PM EST
        if ny_time.time() < time(15, 30):
            return 0.0
            
        try:
            with book_buffer._lock:
                dq = book_buffer._buf.get(symbol)
                if not dq or len(dq) == 0:
                    return 0.0
                
                latest_quote = dq[-1]
                
                # Sum the sizes across the top 10 levels
                total_bids = sum(latest_quote.bid_sizes[:10])
                total_asks = sum(latest_quote.ask_sizes[:10])
                
                # Net imbalance in shares
                net_imbalance = total_bids - total_asks
                
                log.debug(f"MOC Imbalance Check ({symbol}): Bids={total_bids}, Asks={total_asks} -> Net={net_imbalance}")
                return float(net_imbalance)
        except Exception as e:
            log.warning(f"Error calculating MOC imbalance for {symbol}: {e}")
            
        return 0.0
