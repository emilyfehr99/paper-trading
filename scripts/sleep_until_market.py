#!/usr/bin/env python3
import os
import sys
import argparse
import time
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo
from rich.console import Console

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpaca_day_bot.config import load_settings

console = Console()

def get_next_session(settings) -> tuple[datetime, bool]:
    """
    Returns (next_session_start_datetime, is_currently_in_session).
    All calculations are done in the market timezone.
    """
    tz = ZoneInfo(settings.market_tz or "America/Chicago")
    start_val = settings.trade_start
    end_val = settings.trade_end

    if isinstance(start_val, str):
        start_time = datetime.strptime(start_val, "%H:%M:%S").time()
    else:
        start_time = start_val

    if isinstance(end_val, str):
        end_time = datetime.strptime(end_val, "%H:%M:%S").time()
    else:
        end_time = end_val

    now = datetime.now(tz)
    d = now.date()

    # Loop to find the next valid session day/time
    for i in range(10): # look up to 10 days ahead
        candidate_date = d + timedelta(days=i)
        
        # We only trade on weekdays (Mon-Fri)
        if candidate_date.weekday() >= 5:
            continue
            
        session_start = datetime.combine(candidate_date, start_time, tzinfo=tz)
        session_end = datetime.combine(candidate_date, end_time, tzinfo=tz)
        
        if now < session_end:
            # This session is either running or in the future
            if now >= session_start:
                # We are currently in the market session!
                return session_start, True
            else:
                # The session starts in the future (could be today or a future weekday)
                return session_start, False
                
    # Fallback to tomorrow if loop fails (should not happen)
    fallback = datetime.combine(d + timedelta(days=1), start_time, tzinfo=tz)
    return fallback, False

def main():
    parser = argparse.ArgumentParser(description="Pre-Session Blocking Sleep Script")
    parser.add_argument("--env", type=str, default=".env.stocks_trial", help="Env to load configuration from")
    args = parser.parse_args()

    os.environ["ENV_FILE"] = args.env
    settings = load_settings(args.env)

    tz = ZoneInfo(settings.market_tz or "America/Chicago")
    
    console.print(f"[bold cyan][PRE-SESSION][/bold cyan] Checking market session for [yellow]{args.env}[/yellow] ({settings.market_tz})...")

    # Keep looping and checking until we are in a session or the start time is reached
    while True:
        # 1. Try Alpaca Clock API first for holiday & weekend accuracy
        try:
            from alpaca.trading.client import TradingClient
            tc = TradingClient(settings.apca_api_key_id, settings.apca_api_secret_key, paper=True)
            clock = tc.get_clock()
            if clock.is_open:
                console.print("[bold green][PRE-SESSION] Alpaca Clock: Market is OPEN. Exiting sleep script to start trading bot.[/bold green]")
                sys.exit(0)
            else:
                now_utc = datetime.now(timezone.utc)
                seconds_to_sleep = (clock.next_open - now_utc).total_seconds()
                if seconds_to_sleep > 0:
                    hours = seconds_to_sleep / 3600.0
                    console.print(
                        f"[bold yellow][PRE-SESSION] Alpaca Clock: Market is CLOSED (Holiday/Weekend).[/bold yellow]\n"
                        f"Next session starts at [cyan]{clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}[/cyan].\n"
                        f"Sleeping for [bold yellow]{seconds_to_sleep:,.1f}[/bold yellow] seconds ([bold yellow]{hours:.2f}[/bold yellow] hours)..."
                    )
                    chunk = min(seconds_to_sleep, 600.0)
                    time.sleep(chunk)
                    continue
        except Exception as e:
            console.print(f"[bold red][PRE-SESSION] Alpaca Clock API failed: {e}. Falling back to local calculations.[/bold red]")

        # 2. Local Fallback (if Alpaca API fails or is offline)
        next_start, in_session = get_next_session(settings)
        if in_session:
            console.print("[bold green][PRE-SESSION][/bold green] Market is currently open/active! Exiting sleep script to start the trading bot.")
            sys.exit(0)

        now = datetime.now(tz)
        seconds_to_sleep = (next_start - now).total_seconds()
        
        if seconds_to_sleep <= 0:
            console.print("[bold green][PRE-SESSION][/bold green] Sleep target reached/passed. Exiting sleep script to start the trading bot.")
            sys.exit(0)

        hours = seconds_to_sleep / 3600.0
        console.print(
            f"[dim][PRE-SESSION][/dim] Current time: [cyan]{now.strftime('%Y-%m-%d %H:%M:%S %Z')}[/cyan] | "
            f"Next session starts at [cyan]{next_start.strftime('%Y-%m-%d %H:%M:%S %Z')}[/cyan]. "
            f"Sleeping for [bold yellow]{seconds_to_sleep:,.1f}[/bold yellow] seconds ([bold yellow]{hours:.2f}[/bold yellow] hours)..."
        )
        
        chunk = min(seconds_to_sleep, 600.0)
        time.sleep(chunk)

if __name__ == "__main__":
    main()
