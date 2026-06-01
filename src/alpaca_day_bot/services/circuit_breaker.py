import logging
import sys
import time
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import ClosePositionRequest

log = logging.getLogger("CircuitBreaker")

class CircuitBreaker:
    """
    Independent safety service that monitors account health 
    and can force-close all positions (Kill Switch).
    """
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)
        self.max_daily_loss_pct = 5.0 # Institutional default

    def check_health(self):
        """
        Scans for catastrophic conditions.
        """
        try:
            acc = self.client.get_account()
            equity = float(acc.equity)
            last_equity = float(acc.last_equity)
            
            pnl_pct = (equity - last_equity) / last_equity * 100.0
            
            if pnl_pct <= -self.max_daily_loss_pct:
                log.critical(f"Circuit Breaker Tripped! Daily Loss: {pnl_pct:.2%}. FLATTENING PORTFOLIO.")
                self.flatten_all()
                return False
            
            return True
        except Exception as e:
            log.error(f"Health Check Failed: {e}")
            return False

    def flatten_all(self):
        """
        The 'Red Button' - Closes all active positions immediately.
        """
        log.warning("FLATTENING ALL POSITIONS...")
        try:
            # 1. Cancel all open orders
            self.client.cancel_orders()
            # 2. Close all positions
            self.client.close_all_positions(cancel_orders=True)
            log.info("Portfolio successfully flattened.")
        except Exception as e:
            log.error(f"Flattening failed: {e}")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(".env.stocks")
    
    cb = CircuitBreaker(
        api_key=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY")
    )
    
    # Manual command line usage
    if len(sys.argv) > 1 and sys.argv[1] == "--kill":
        cb.flatten_all()
    else:
        print("Usage: python3 -m alpaca_day_bot.services.circuit_breaker --kill")
