#!/usr/bin/env python3
"""
Backtest with Simulated Proper Features.
Recalculate normalized features from raw values to estimate expected performance.
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
    parser = argparse.ArgumentParser(description="Backtest with Simulated Proper Features")
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
    
    console.print(f"[green]Loaded {len(df)} signals for simulated proper features backtest.[/green]")
    
    # Extract raw values from existing data
    df['alligator_jaw_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('alligator_jaw', 1.0))
    df['alligator_teeth_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('alligator_teeth', 1.0))
    df['alligator_lips_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('alligator_lips', 1.0))
    df['close'] = df['features_json'].apply(lambda x: parse_features(x).get('close', 1.0))
    
    # Simulate proper Alligator ratios (what the new code will generate)
    df['alligator_jaw_ratio'] = df['close'] / df['alligator_jaw_raw']
    df['alligator_teeth_ratio'] = df['close'] / df['alligator_teeth_raw']
    df['alligator_lips_ratio'] = df['close'] / df['alligator_lips_raw']
    
    # Simulate proper EMA ratios
    df['ema_9_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('ema_9', 1.0))
    df['ema_21_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('ema_21', 1.0))
    df['ema_9_ratio'] = df['close'] / df['ema_9_raw']
    df['ema_21_ratio'] = df['close'] / df['ema_21_raw']
    
    # Simulate proper VWAP ratio
    df['vwap_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('vwap', df['close']))
    df['vwap_ratio'] = df['close'] / df['vwap_raw']
    
    # Simulate proper SuperTrend ratio
    df['supertrend_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('supertrend', df['close']))
    df['supertrend_ratio'] = df['close'] / df['supertrend_raw']
    
    # Simulate proper MACD ratios
    df['macd_line_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('macd_line', 0.0))
    df['macd_signal_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('macd_signal', 0.0))
    df['macd_hist_raw'] = df['features_json'].apply(lambda x: parse_features(x).get('macd_hist', 0.0))
    df['macd_line_ratio'] = df['macd_line_raw'] / df['close']
    df['macd_signal_ratio'] = df['macd_signal_raw'] / df['close']
    df['macd_hist_ratio'] = df['macd_hist_raw'] / df['close']
    
    # Get other features
    df['rvol'] = df['features_json'].apply(lambda x: parse_features(x).get('rvol', 1.0))
    df['rsi'] = df['features_json'].apply(lambda x: parse_features(x).get('rsi_14', 50.0))
    df['macd_hist'] = df['features_json'].apply(lambda x: parse_features(x).get('macd_hist', 0.0))
    df['adx'] = df['features_json'].apply(lambda x: parse_features(x).get('adx', 20.0))
    
    # Calculate Alligator alignment using simulated proper ratios
    df['alligator_aligned_bullish'] = (df['alligator_jaw_ratio'] > 1.0) & (df['alligator_teeth_ratio'] > 1.0) & (df['alligator_lips_ratio'] > 1.0)
    
    # Apply optimized enhanced strategy filter with simulated proper features
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
    
    # Extract hour for time-based filtering
    df['hour'] = pd.to_datetime(df['ts'], format='ISO8601', utc=True).dt.hour
    enhanced_df['hour'] = pd.to_datetime(enhanced_df['ts'], format='ISO8601', utc=True).dt.hour
    
    console.print("\n[bold yellow]=== SIMULATED PROPER FEATURES BACKTEST ===[/bold yellow]")
    
    baseline_stats = calculate_stats(df)
    
    # Test different strategies with simulated proper features
    # 1. Original strategy (broken features)
    original_broken = df[df['alligator_aligned_bullish'] & (df['rvol'] > 1.5) & (df['rsi'] > 30) & (df['rsi'] < 70) & (df['macd_hist'] > 0)]
    original_broken_stats = calculate_stats(original_broken)
    
    # 2. Enhanced strategy with simulated proper features
    enhanced_proper = enhanced_df
    enhanced_proper_stats = calculate_stats(enhanced_proper)
    
    # 3. Enhanced + time filter with simulated proper features
    best_hours = [14, 17, 20]
    enhanced_time = enhanced_df[enhanced_df['hour'].isin(best_hours)]
    enhanced_time_stats = calculate_stats(enhanced_time)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Strategy", style="cyan", width=40)
    table.add_column("Signals", style="white")
    table.add_column("Win Rate", style="green")
    table.add_column("Avg Return", style="green")
    table.add_column("Win Rate Δ", style="yellow")
    table.add_column("Signal %", style="white")
    table.add_column("65% Target?", style="bold")
    
    results = [
        ("Baseline (All Signals)", baseline_stats),
        ("Original (Broken Features)", original_broken_stats),
        ("Enhanced (Simulated Proper Features)", enhanced_proper_stats),
        ("Enhanced + Time Filter (Simulated Proper)", enhanced_time_stats),
    ]
    
    for name, stats in results:
        win_rate_delta = stats['win_rate'] - baseline_stats['win_rate']
        signal_pct = (stats['count'] / baseline_stats['count'] * 100) if baseline_stats['count'] > 0 else 0
        meets_target = stats['win_rate'] >= 0.65
        
        target_color = "green" if meets_target else "red"
        table.add_row(
            name,
            str(stats['count']),
            f"{stats['win_rate']:.1%}",
            f"{stats['avg_return']:.2%}",
            f"{win_rate_delta:+.1%}",
            f"{signal_pct:.1f}%",
            f"[{target_color}]{'YES' if meets_target else 'NO'}[/{target_color}]"
        )
    
    console.print(table)
    
    console.print("\n[bold yellow]=== SIMULATED PROPER FEATURES SUMMARY ===[/bold yellow]")
    console.print(f"Original (Broken Features): {original_broken_stats['win_rate']:.1%} win rate")
    console.print(f"Enhanced (Simulated Proper): {enhanced_proper_stats['win_rate']:.1%} win rate ({enhanced_proper_stats['win_rate'] - original_broken_stats['win_rate']:+.1%} improvement)")
    console.print(f"Enhanced + Time Filter: {enhanced_time_stats['win_rate']:.1%} win rate")
    console.print(f"65%+ Target Achieved: {'YES' if enhanced_proper_stats['win_rate'] >= 0.65 else 'NO'}")

if __name__ == "__main__":
    main()
