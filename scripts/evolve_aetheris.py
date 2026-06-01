#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from rich.console import Console

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpaca_day_bot.config import load_settings
from alpaca_day_bot.storage.ledger import Ledger

console = Console()

def evolve():
    parser = argparse.ArgumentParser(description="Aetheris Stock Evolution Loop")
    # Singular focus on Stocks
    parser.add_argument("--env", type=str, default=".env.stocks", help="Env to evolve")
    args = parser.parse_args()

    os.environ["ENV_FILE"] = args.env
    settings = load_settings(args.env)

    # Skip evolution if the market is currently active to avoid blocking trading startup
    try:
        sys.path.append(str(Path(__file__).parent))
        from sleep_until_market import get_next_session
        _, in_session = get_next_session(settings)
        if in_session:
            console.print("[bold yellow][EVOLUTION] Market is currently active! Skipping evolution/retraining to allow instant bot startup.[/bold yellow]")
            # Silent Notification File to satisfy launch scripts
            with open("state/LATEST_EVOLUTION_COMPLETE.txt", "w") as f:
                f.write(f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}\n")
                f.write(f"PRECISION: SKIPPED (MARKET ACTIVE)\n")
                f.write(f"STATUS: READY FOR TRADING\n")
            return
    except Exception as e:
        console.print(f"[bold red]Failed to check market session status: {e}[/bold red]")

    console.print(f"[bold cyan]Starting Continuous Evolution for {args.env} (STOCKS ONLY)...[/bold cyan]")

    # 1. LABELING MISTAKES
    console.print("[yellow]Phase 1: Identifying After-Market mistakes...[/yellow]")
    os.system(f"{sys.executable} scripts/label_signals.py --env {args.env}")

    # 2. RETRAINING
    console.print("[yellow]Phase 2: Updating Sniper weights based on new Stock data...[/yellow]")
    # Run trainer and capture output to extract precision
    import subprocess
    cmd = f".venv/bin/python3 scripts/train_sniper.py --env {args.env} --save"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    console.print(result.stdout)

    # 3. RECORDING EVOLUTION STATS
    precision = "N/A"
    for line in result.stdout.split('\n'):
        if "Precision at Threshold:" in line and "MEAN_REVERSION" in line:
            parts = line.split("Precision at Threshold:")
            if len(parts) > 1:
                precision = parts[1].split(',')[0].strip()
        elif "Precision at Threshold:" in line and precision == "N/A":
            parts = line.split("Precision at Threshold:")
            if len(parts) > 1:
                precision = parts[1].split(',')[0].strip()
    
    stats_path = Path(settings.state_dir) / "evolution_stats.csv"
    with open(stats_path, "a") as f:
        if stats_path.stat().st_size == 0:
            f.write("timestamp,env,precision\n")
        f.write(f"{datetime.now(timezone.utc).isoformat()},{args.env},{precision}\n")

    console.print(f"[bold green]Stock Evolution Complete. New Precision: {precision}. Saved to {stats_path}[/bold green]")

    # Silent Notification File
    with open("state/LATEST_EVOLUTION_COMPLETE.txt", "w") as f:
        f.write(f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"PRECISION: {precision}\n")
        f.write(f"STATUS: READY FOR MONDAY OPEN\n")

if __name__ == "__main__":
    evolve()
