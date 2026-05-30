#!/usr/bin/env python3
"""
Time-Based Optimization - Analyze hourly performance patterns.
Identify best trading hours to improve win rate by 2-3%.
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
    parser = argparse.ArgumentParser(description="Time-Based Optimization")
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
    
    console.print(f"[green]Loaded {len(df)} signals for time-based optimization.[/green]")
    
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
    
    # Apply enhanced strategy filter
    def enhanced_filter(row):
        adx = row['adx']
        if adx > 25:  # Trending
            return (row['rvol'] > 1.3 and row['rsi'] > 25 and row['rsi'] < 75 and row['macd_hist'] > 0)
        elif adx < 20:  # Choppy
            return (row['rvol'] > 2.0 and row['rsi'] > 35 and row['rsi'] < 65 and row['macd_hist'] > 0.1)
        else:  # Neutral
            return (row['rvol'] > 1.5 and row['rsi'] > 30 and row['rsi'] < 70 and row['macd_hist'] > 0)
    
    df['enhanced_signal'] = df.apply(enhanced_filter, axis=1)
    enhanced_df = df[df['enhanced_signal']]
    
    # Extract hour from timestamp
    df['hour'] = pd.to_datetime(df['ts'], format='ISO8601', utc=True).dt.hour
    enhanced_df['hour'] = pd.to_datetime(enhanced_df['ts'], format='ISO8601', utc=True).dt.hour
    
    console.print("\n[bold yellow]=== HOURLY PERFORMANCE ANALYSIS ===[/bold yellow]")
    
    # Analyze hourly performance for enhanced strategy
    hourly_results = []
    for hour in sorted(enhanced_df['hour'].unique()):
        hour_df = enhanced_df[enhanced_df['hour'] == hour]
        stats = calculate_stats(hour_df)
        if stats['count'] > 0:
            hourly_results.append((f"{hour}:00", stats))
    
    # Sort by win rate
    hourly_results.sort(key=lambda x: x[1]['win_rate'], reverse=True)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Hour", style="cyan", width=10)
    table.add_column("Signals", style="white")
    table.add_column("Win Rate", style="green")
    table.add_column("Avg Return", style="green")
    table.add_column("vs Overall", style="yellow")
    
    overall_stats = calculate_stats(enhanced_df)
    
    for hour, stats in hourly_results:
        win_rate_diff = stats['win_rate'] - overall_stats['win_rate']
        diff_color = "+" if win_rate_diff >= 0 else ""
        table.add_row(
            hour,
            str(stats['count']),
            f"{stats['win_rate']:.1%}",
            f"{stats['avg_return']:.2%}",
            f"{diff_color}{win_rate_diff:+.1%}"
        )
    
    console.print(table)
    
    console.print(f"\nOverall Enhanced Strategy: {overall_stats['win_rate']:.1%} win rate")
    
    # Identify best hours
    best_hours = [hour for hour, stats in hourly_results if stats['win_rate'] > overall_stats['win_rate'] + 0.05]
    console.print(f"\nBest Performing Hours (+5% above overall): {', '.join(best_hours) if best_hours else 'None'}")
    
    # Test filtering to best hours
    if best_hours:
        best_hours_int = [int(hour.split(':')[0]) for hour in best_hours]
        best_hours_df = enhanced_df[enhanced_df['hour'].isin(best_hours_int)]
        best_hours_stats = calculate_stats(best_hours_df)
        
        console.print(f"\nFiltered to Best Hours Only:")
        console.print(f"  Win Rate: {best_hours_stats['win_rate']:.1%} ({best_hours_stats['win_rate'] - overall_stats['win_rate']:+.1%} improvement)")
        console.print(f"  Signals: {best_hours_stats['count']} ({best_hours_stats['count'] / overall_stats['count'] * 100:.1f}% of total)")

if __name__ == "__main__":
    main()
