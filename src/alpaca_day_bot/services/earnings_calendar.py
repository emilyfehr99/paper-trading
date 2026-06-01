import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("alpaca_day_bot.services.earnings_calendar")

class EarningsCalendarService:
    """
    Tracks minutes until next corporate earnings release to protect against massive gap risk.
    """
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        self.calendar_path = Path(state_dir) / "earnings_calendar.json"
        self._cache = {}
        self._load_calendar()

    def _load_calendar(self):
        """
        Loads the earnings calendar JSON database.
        """
        if not self.calendar_path.exists():
            # Create a clean default calendar skeleton to avoid I/O crashes
            os.makedirs(os.path.dirname(self.calendar_path), exist_ok=True)
            with open(self.calendar_path, "w") as f:
                json.dump({"updated_at": datetime.now(timezone.utc).isoformat(), "schedules": {}}, f, indent=4)
        
        try:
            with open(self.calendar_path, "r") as f:
                data = json.load(f)
                self._cache = data.get("schedules", {})
                log.info(f"Earnings Calendar Database successfully loaded with {len(self._cache)} schedules.")
        except Exception as e:
            log.error(f"Failed to load earnings calendar database: {e}")
            self._cache = {}

    def get_minutes_until_earnings(self, symbol: str) -> float:
        """
        Calculates the remaining minutes until next corporate earnings event for the symbol.
        Returns a very large float (e.g. 999999.0) if no upcoming earnings are scheduled.
        """
        earning_time_str = self._cache.get(symbol)
        if not earning_time_str:
            return 999999.0 # Safe default: no known upcoming earnings
            
        try:
            earning_dt = datetime.fromisoformat(earning_time_str)
            # Ensure earning_dt is timezone-aware
            if earning_dt.tzinfo is None:
                earning_dt = earning_dt.replace(tzinfo=timezone.utc)
                
            now = datetime.now(timezone.utc)
            delta = earning_dt - now
            minutes = delta.total_seconds() / 60.0
            
            # If earnings are in the past, they shouldn't block future setups
            if minutes < 0:
                return 999999.0
                
            return float(minutes)
        except Exception as e:
            log.error(f"Failed parsing earnings timestamp '{earning_time_str}' for {symbol}: {e}")
            return 999999.0
