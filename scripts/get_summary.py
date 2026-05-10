import os
import pytz
from datetime import datetime
from alpaca_day_bot.reporting.report import daily_summary

def main():
    # Use the project's local directory structure
    db_path = "state/ledger.sqlite3"
    tz = pytz.timezone('America/Chicago')
    today = datetime.now(tz).date()
    
    try:
        s = daily_summary(db_path, today)
        pnl = s.pnl if s.pnl is not None else 0.0
        pct = s.pnl_pct * 100 if s.pnl_pct is not None else 0.0
        max_gross = s.max_gross if s.max_gross is not None else 0.0
        
        print("\n" + "="*30)
        print(f" TODAY'S UPDATE: {today}")
        print("="*30)
        print(f" Net P&L:    {pnl:+.2f} USD ({pct:+.2f}%)")
        print(f" Trades:     {s.trades}")
        print(f" Max Gross:  ${max_gross:,.2f}")
        print("="*30 + "\n")
    except Exception as e:
        print(f"Error reading summary: {e}")

if __name__ == "__main__":
    main()
