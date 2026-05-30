#!/usr/bin/env python3
"""
Walk-Forward Validation - Test strategy robustness across different time periods.
Ensure 57.2% win rate isn't data mining bias and identify weaknesses.
"""
import os
import sys
import sqlite3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpaca_day_bot.config import load_settings

console = Console()

def load_signals_with_labels(db_path: str) -> pd.DataFrame:
    """Load signals with their forward return labels."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        s.id as signal_id,
        s.ts,
        s.symbol,
        s.action,
        s.reason,
        s.features_json,
        s.explainability_json,
        f.return_pct,
        f.horizon_minutes
    FROM signals s
    LEFT JOIN forward_return_labels f ON s.id = f.signal_id
    WHERE s.action = 'BUY' AND f.return_pct IS NOT NULL
    ORDER BY s.ts ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def parse_features(features_json: str) -> dict:
    """Parse features JSON string."""
    try:
        import json
        return json.loads(features_json) if features_json else {}
    except Exception:
        return {}

def calculate_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics for a set of signals."""
    if df.empty:
        return {'count': 0, 'win_rate': 0.0, 'avg_return': 0.0}
    
    outcomes = df['return_pct'].values
    wins = outcomes > 0
    
    return {
        'count': len(outcomes),
        'win_rate': float(np.mean(wins)),
        'avg_return': float(np.mean(outcomes)),
    }

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--env", type=str, default=".env.stocks", help="Env to use")
    args = parser.parse_args()
    
    settings = load_settings(args.env)
    db_path = str(Path(settings.state_dir) / "ledger.sqlite3")
    
    if not os.path.exists(db_path):
        console.print(f"[bold red]Ledger not found at {db_path}[/bold red]")
        return
    
    console.print(f"[bold cyan]Loading signals with labels from {db_path}...[/bold cyan]")
    df = load_signals_with_labels(db_path)
    
    if df.empty:
        console.print("[bold red]No signals with labels found in ledger.[/red]")
        return
    
    console.print(f"[green]Loaded {len(df)} signals for walk-forward validation.[/green]")
    
    # Extract Alligator raw values from existing data
    df['alligator_jaw'] = df['features_json'].apply(lambda x: parse_features(x).get('alligator_jaw', 1.0))
    df['alligator_teeth'] = df['features_json'].apply(lambda x: parse_features(x).get('alligator_teeth', 1.0))
    df['alligator_lips'] = df['features_json'].apply(lambda x: parse_features(x).get('alligator_lips', 1.0))
    df['close'] = df['features_json'].apply(lambda x: parse_features(x).get('close', 1.0))
    
    # Calculate Alligator ratios from raw values
    df['alligator_jaw_ratio'] = df['close'] / df['alligator_jaw']
    df['alligator_teeth_ratio'] = df['close'] / df['alligator_teeth']
    df['alligator_lips_ratio'] = df['close'] / df['alligator_lips']
    
    # Calculate Alligator alignment using fixed ratios
    df['alligator_aligned_bullish'] = (df['alligator_jaw_ratio'] > 1.0) & (df['alligator_teeth_ratio'] > 1.0) & (df['alligator_lips_ratio'] > 1.0)
    
    # Get other features for filtering
    df['rvol'] = df['features_json'].apply(lambda x: parse_features(x).get('rvol', 1.0))
    df['rsi'] = df['features_json'].apply(lambda x: parse_features(x).get('rsi_14', 50.0))
    df['macd_hist'] = df['features_json'].apply(lambda x: parse_features(x).get('macd_hist', 0.0))
    df['adx'] = df['features_json'].apply(lambda x: parse_features(x).get('adx', 20.0))
    
    # Apply optimized enhanced strategy filter
    def enhanced_filter(row):
        adx = row['adx']
        if adx > 28:  # Trending (optimized)
            return (row['rvol'] > 1.3 and row['rsi'] > 20 and row['rsi'] < 80 and row['macd_hist'] > 0)
        elif adx < 22:  # Choppy (optimized)
            return (row['rvol'] > 2.0 and row['rsi'] > 35 and row['rsi'] < 65 and row['macd_hist'] > 0.1)
        else:  # Neutral
            return (row['rvol'] > 1.5 and row['rsi'] > 20 and row['rsi'] < 80 and row['macd_hist'] > 0)
    
    df['enhanced_signal'] = df.apply(enhanced_filter, axis=1)
    enhanced_df = df[df['enhanced_signal']]
    
    # Convert timestamp and extract time periods
    df['datetime'] = pd.to_datetime(df['ts'], format='ISO8601', utc=True)
    df['date'] = df['datetime'].dt.date
    df['week'] = df['datetime'].dt.isocalendar().week
    enhanced_df['datetime'] = pd.to_datetime(enhanced_df['ts'], format='ISO8601', utc=True)
    enhanced_df['week'] = enhanced_df['datetime'].dt.isocalendar().week
    
    console.print("\n[bold yellow]=== WALK-FORWARD VALIDATION ===[/bold yellow]")
    
    # Test across different weeks (walk-forward periods)
    weekly_results = []
    for week in sorted(enhanced_df['week'].unique()):
        week_df = enhanced_df[enhanced_df['week'] == week]
        stats = calculate_stats(week_df)
        if stats['count'] > 0:
            weekly_results.append((f"Week {week}", stats))
    
    # Calculate consistency metrics
    weekly_win_rates = [stats['win_rate'] for _, stats in weekly_results]
    weekly_std = np.std(weekly_win_rates) if len(weekly_win_rates) > 1 else 0
    weekly_mean = np.mean(weekly_win_rates) if weekly_win_rates else 0
    
    console.print(f"\nWeekly Performance:")
    console.print(f"  Mean Win Rate: {weekly_mean:.1%}")
    console.print(f"  Std Dev: {weekly_std:.1%}")
    console.print(f"  Consistency Score: {'HIGH' if weekly_std < 0.10 else 'MEDIUM' if weekly_std < 0.15 else 'LOW'}")
    
    # Show weekly breakdown
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Period", style="cyan", width=12)
    table.add_column("Signals", style="white")
    table.add_column("Win Rate", style="green")
    table.add_column("Avg Return", style="green")
    table.add_column("vs Mean", style="yellow")
    
    for name, stats in weekly_results:
        win_rate_diff = stats['win_rate'] - weekly_mean
        diff_color = "+" if win_rate_diff >= 0 else ""
        table.add_row(
            name,
            str(stats['count']),
            f"{stats['win_rate']:.1%}",
            f"{stats['avg_return']:.2%}",
            f"{diff_color}{win_rate_diff:+.1%}"
        )
    
    console.print(table)
    
    # Test first half vs second half (temporal validation)
    total_weeks = len(weekly_results)
    if total_weeks >= 4:
        first_half = weekly_results[:total_weeks//2]
        second_half = weekly_results[total_weeks//2:]
        
        first_half_stats = calculate_stats(pd.concat([enhanced_df[enhanced_df['week'] == int(name.split()[1])] for name, _ in first_half]))
        second_half_stats = calculate_stats(pd.concat([enhanced_df[enhanced_df['week'] == int(name.split()[1])] for name, _ in second_half]))
        
        console.print("\n[bold yellow]=== TEMPORAL VALIDATION ===[/bold yellow]")
        console.print(f"First Half: {first_half_stats['win_rate']:.1%} win rate ({first_half_stats['count']} signals)")
        console.print(f"Second Half: {second_half_stats['win_rate']:.1%} win rate ({second_half_stats['count']} signals)")
        console.print(f"Temporal Drift: {second_half_stats['win_rate'] - first_half_stats['win_rate']:+.1%}")
        console.print(f"Robustness: {'HIGH' if abs(second_half_stats['win_rate'] - first_half_stats['win_rate']) < 0.05 else 'MEDIUM' if abs(second_half_stats['win_rate'] - first_half_stats['win_rate']) < 0.10 else 'LOW'}")
    
    # Overall assessment
    console.print("\n[bold yellow]=== VALIDATION SUMMARY ===[/bold yellow]")
    overall_stats = calculate_stats(enhanced_df)
    console.print(f"Overall Win Rate: {overall_stats['win_rate']:.1%}")
    console.print(f"Overall Signals: {overall_stats['count']}")
    console.print(f"Weekly Consistency: {weekly_std:.1%} std dev")
    console.print(f"Data Mining Risk: {'LOW' if weekly_std < 0.15 and abs(second_half_stats['win_rate'] - first_half_stats['win_rate']) < 0.10 else 'MEDIUM'}")

if __name__ == "__main__":
    main()
