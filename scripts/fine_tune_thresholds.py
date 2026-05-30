#!/usr/bin/env python3
"""
Fine-Tune Threshold Values - Optimize ADX, RVOL, RSI ranges.
Test different parameter combinations to improve win rate by 1-2%.
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
from itertools import product

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
    parser = argparse.ArgumentParser(description="Fine-Tune Threshold Values")
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
    
    console.print(f"[green]Loaded {len(df)} signals for threshold fine-tuning.[/green]")
    
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
    
    console.print("\n[bold yellow]=== THRESHOLD FINE-TUNING ===[/bold yellow]")
    
    # Test different ADX thresholds for regime detection
    adx_thresholds = [(20, 25), (22, 28), (18, 24), (23, 27)]
    adx_results = []
    
    for choppy_thresh, trend_thresh in adx_thresholds:
        def adx_filter(row):
            adx = row['adx']
            if adx > trend_thresh:  # Trending
                return (row['rvol'] > 1.3 and row['rsi'] > 25 and row['rsi'] < 75 and row['macd_hist'] > 0)
            elif adx < choppy_thresh:  # Choppy
                return (row['rvol'] > 2.0 and row['rsi'] > 35 and row['rsi'] < 65 and row['macd_hist'] > 0.1)
            else:  # Neutral
                return (row['rvol'] > 1.5 and row['rsi'] > 30 and row['rsi'] < 70 and row['macd_hist'] > 0)
        
        test_df = df[df.apply(adx_filter, axis=1)]
        stats = calculate_stats(test_df)
        adx_results.append((f"ADX {choppy_thresh}/{trend_thresh}", stats))
    
    # Test different RVOL thresholds
    rvol_thresholds = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    rvol_results = []
    
    for rvol_thresh in rvol_thresholds:
        test_df = df[df['alligator_aligned_bullish'] & (df['rvol'] > rvol_thresh) & (df['rsi'] > 30) & (df['rsi'] < 70) & (df['macd_hist'] > 0)]
        stats = calculate_stats(test_df)
        rvol_results.append((f"RVOL > {rvol_thresh}", stats))
    
    # Test different RSI ranges
    rsi_ranges = [(20, 80), (25, 75), (28, 72), (30, 70), (32, 68), (35, 65)]
    rsi_results = []
    
    for rsi_min, rsi_max in rsi_ranges:
        test_df = df[df['alligator_aligned_bullish'] & (df['rvol'] > 1.5) & (df['rsi'] > rsi_min) & (df['rsi'] < rsi_max) & (df['macd_hist'] > 0)]
        stats = calculate_stats(test_df)
        rsi_results.append((f"RSI {rsi_min}-{rsi_max}", stats))
    
    # Display results
    console.print("\n[bold cyan]ADX Threshold Optimization:[/bold cyan]")
    adx_table = Table(show_header=True, header_style="bold magenta")
    adx_table.add_column("Threshold", style="cyan", width=20)
    adx_table.add_column("Signals", style="white")
    adx_table.add_column("Win Rate", style="green")
    adx_table.add_column("Avg Return", style="green")
    
    for name, stats in adx_results:
        if stats['count'] > 0:
            adx_table.add_row(name, str(stats['count']), f"{stats['win_rate']:.1%}", f"{stats['avg_return']:.2%}")
    
    console.print(adx_table)
    
    console.print("\n[bold cyan]RVOL Threshold Optimization:[/bold cyan]")
    rvol_table = Table(show_header=True, header_style="bold magenta")
    rvol_table.add_column("Threshold", style="cyan", width=15)
    rvol_table.add_column("Signals", style="white")
    rvol_table.add_column("Win Rate", style="green")
    rvol_table.add_column("Avg Return", style="green")
    
    for name, stats in rvol_results:
        if stats['count'] > 0:
            rvol_table.add_row(name, str(stats['count']), f"{stats['win_rate']:.1%}", f"{stats['avg_return']:.2%}")
    
    console.print(rvol_table)
    
    console.print("\n[bold cyan]RSI Range Optimization:[/bold cyan]")
    rsi_table = Table(show_header=True, header_style="bold magenta")
    rsi_table.add_column("Range", style="cyan", width=15)
    rsi_table.add_column("Signals", style="white")
    rsi_table.add_column("Win Rate", style="green")
    rsi_table.add_column("Avg Return", style="green")
    
    for name, stats in rsi_results:
        if stats['count'] > 0:
            rsi_table.add_row(name, str(stats['count']), f"{stats['win_rate']:.1%}", f"{stats['avg_return']:.2%}")
    
    console.print(rsi_table)
    
    # Find best combinations
    console.print("\n[bold yellow]=== BEST COMBINATIONS ===[/bold yellow]")
    
    best_adx = max(adx_results, key=lambda x: x[1]['win_rate'] if x[1]['count'] > 10 else 0)
    best_rvol = max(rvol_results, key=lambda x: x[1]['win_rate'] if x[1]['count'] > 50 else 0)
    best_rsi = max(rsi_results, key=lambda x: x[1]['win_rate'] if x[1]['count'] > 50 else 0)
    
    console.print(f"Best ADX Threshold: {best_adx[0]} ({best_adx[1]['win_rate']:.1%} win rate)")
    console.print(f"Best RVOL Threshold: {best_rvol[0]} ({best_rvol[1]['win_rate']:.1%} win rate)")
    console.print(f"Best RSI Range: {best_rsi[0]} ({best_rsi[1]['win_rate']:.1%} win rate)")

if __name__ == "__main__":
    main()
