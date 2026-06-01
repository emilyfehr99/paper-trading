import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("alpaca_day_bot.services.macro_calendar")

class MacroCalendarService:
    """
    Tracks minutes until or since scheduled macroeconomic announcements (FOMC, CPI, PPI).
    Blocks new order submissions 30 minutes prior to and 15 minutes following scheduled events.
    """
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        self.calendar_path = Path(state_dir) / "macro_calendar.json"
        self._schedules = []
        self._load_calendar()

    def _load_calendar(self):
        """
        Loads the macroeconomic calendar schedules.
        """
        if not self.calendar_path.exists():
            try:
                os.makedirs(os.path.dirname(self.calendar_path), exist_ok=True)
                # Seed with clean structural skeleton and mock upcoming high-impact schedule
                default_data = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "announcements": [
                        {"event": "FOMC Interest Rate Decision", "timestamp": "2026-05-20T14:00:00Z"},
                        {"event": "CPI Inflation Report", "timestamp": "2026-06-10T08:30:00Z"}
                    ]
                }
                with open(self.calendar_path, "w") as f:
                    json.dump(default_data, f, indent=4)
            except Exception as e:
                log.error(f"Failed to create default macro calendar: {e}")
        
        try:
            with open(self.calendar_path, "r") as f:
                data = json.load(f)
                self._schedules = data.get("announcements", [])
                log.info(f"Macroeconomic Calendar successfully loaded with {len(self._schedules)} scheduled announcements.")
        except Exception as e:
            log.error(f"Failed to load macroeconomic calendar: {e}")
            self._schedules = []

    def get_minutes_to_nearest_macro_event(self) -> tuple[float, float, str]:
        """
        Calculates minutes until and minutes since the nearest high-impact macroeconomic event.
        Returns:
            minutes_until: float (positive value if event is in the future)
            minutes_since: float (positive value if event is in the past)
            event_name: str
        """
        now = datetime.now(timezone.utc)
        nearest_until = 999999.0
        nearest_since = 999999.0
        nearest_event = "None"
        
        for announcement in self._schedules:
            ts_str = announcement.get("timestamp")
            event_name = announcement.get("event", "Macro Event")
            if not ts_str:
                continue
            try:
                event_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                delta = event_dt - now
                diff_minutes = delta.total_seconds() / 60.0
                
                if diff_minutes >= 0:
                    # Future event
                    if diff_minutes < nearest_until:
                        nearest_until = diff_minutes
                        nearest_event = event_name
                else:
                    # Past event
                    abs_since = abs(diff_minutes)
                    if abs_since < nearest_since:
                        nearest_since = abs_since
                        nearest_event = event_name
            except Exception as e:
                log.error(f"Error parsing macro timestamp '{ts_str}': {e}")
                
        return nearest_until, nearest_since, nearest_event

    def is_lockout_active(self) -> tuple[bool, str]:
        """
        Returns (True, reason) if we are inside the pre-announcement 30-minute lockout
        or post-announcement 15-minute cool-down window.
        """
        until, since, event = self.get_minutes_to_nearest_macro_event()
        
        if until < 30.0:
            return True, f"Pre-event lockout active: {event} occurs in {until:.1f} minutes (< 30m)."
        if since < 15.0:
            return True, f"Post-event cooldown active: {event} occurred {since:.1f} minutes ago (< 15m)."
            
        return False, ""
