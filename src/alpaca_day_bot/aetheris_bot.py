from __future__ import annotations

import argparse
import logging
import signal
import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import threading
import time
from datetime import datetime, timezone, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.live import Live

from alpaca_day_bot.config import load_settings
from alpaca_day_bot.data.stream import BarBuffer, MarketDataStreamer, OrderBookBuffer
from alpaca_day_bot.logging_utils import setup_json_logging
from alpaca_day_bot.risk.manager import RiskManager
from alpaca_day_bot.storage.ledger import Ledger
from alpaca_day_bot.strategy.v2_aetheris import V2AetherisSignalEngine
from alpaca_day_bot.trading.client import make_trading_client
from alpaca_day_bot.trading.executor import OrderExecutor
from alpaca_day_bot.trading.updates import TradingUpdatesStreamer
from alpaca_day_bot.data.stream import BarBuffer, MarketDataStreamer, OrderBookBuffer
from alpaca_day_bot.data.rest_bars import RestBarPoller
from alpaca_day_bot.ml.infer import load_model, predict_proba
from alpaca_day_bot.universe import build_liquid_universe
from alpaca_day_bot.trading.reviewer import TradeReviewer

# Premium UI/Console
console = Console()

def setup_premium_logging():
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )
    log = logging.getLogger("aetheris")
    return log

log = setup_premium_logging()

class AetherisBotPro:
    def __init__(self, settings, observe_only: bool = False):
        self.settings = settings
        self.observe_only = observe_only
        self.mock_positions = {}
        self.mock_realized_pnl = 0.0
        self.running = False
        
        # Infrastructure
        self.ledger = Ledger(str(Path(settings.state_dir) / "ledger.sqlite3"))
        self.trading_client = make_trading_client(settings)
        self.executor = OrderExecutor(self.trading_client, settings=settings, ledger=self.ledger)
        
        # Dynamic fallback: if starting_equity_usd <= 0.0, fetch actual broker account equity
        if getattr(self.settings, "starting_equity_usd", 0.0) <= 0.0:
            attempts = 3
            broker_equity = None
            for i in range(attempts):
                try:
                    broker_equity = self.executor.get_account_equity()
                    if broker_equity and broker_equity > 0:
                        self.settings.starting_equity_usd = float(broker_equity)
                        log.info(f"Dynamically set starting equity to broker account equity: [bold cyan]${self.settings.starting_equity_usd:,.2f}[/bold cyan]")
                        break
                    else:
                        log.warning(f"Attempt {i+1}/{attempts}: Broker returned invalid equity ({broker_equity}). Retrying...")
                except Exception as e:
                    log.warning(f"Attempt {i+1}/{attempts}: Error fetching broker equity ({e}). Retrying...")
                # exponential backoff
                time.sleep(2 ** i)
            else:
                log.error("Failed to fetch broker equity after multiple attempts. Aborting startup to avoid incorrect sizing.")
                raise SystemExit("Failed to fetch broker equity")
        
        
        # Risk Manager Initialization
        self.risk = RiskManager(
            max_gross_exposure_pct=settings.max_gross_exposure_pct,
            max_positions=settings.max_positions,
            max_trades_per_day=settings.max_trades_per_day,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            risk_per_trade_pct=settings.risk_per_trade_pct,
            max_notional_per_trade_usd=settings.max_notional_per_trade_usd,
            per_symbol_cooldown_s=settings.per_symbol_cooldown_s,
            daily_profit_target_usd=settings.daily_profit_target_usd
        )
        
        # Rehydrate risk manager same-day stats from SQLite ledger
        try:
            tz = ZoneInfo(settings.market_tz or "America/Chicago")
            now_ct = datetime.now(tz)
            self.risk.rehydrate_from_ledger(self.ledger, now_ct.date(), tz)
            log.info(f"[bold cyan]Risk Manager Rehydrated:[/bold cyan] {self.risk._trades_today} trades recorded today.")
        except Exception as e:
            log.error(f"Failed to rehydrate risk manager from ledger: {e}")

        
        self.buffer = BarBuffer(maxlen=settings.bar_buffer_maxlen)
        self.book_buffer = OrderBookBuffer(maxlen=100) # Deep Intelligence Cache
        
        # Strategy
        import os
        adx_trend = float(getattr(settings, "adx_trend_threshold", getattr(settings, "ADX_TREND_THRESHOLD", os.getenv("ADX_TREND_THRESHOLD", 25.0))))
        adx_chop = float(getattr(settings, "adx_chop_threshold", getattr(settings, "ADX_CHOP_THRESHOLD", os.getenv("ADX_CHOP_THRESHOLD", 20.0))))
        rsi_os = float(getattr(settings, "rsi_oversold", getattr(settings, "RSI_OVERSOLD", os.getenv("RSI_OVERSOLD", 30.0))))
        rsi_ob = float(getattr(settings, "rsi_overbought", getattr(settings, "RSI_OVERBOUGHT", os.getenv("RSI_OVERBOUGHT", 70.0))))


        self.strategy = V2AetherisSignalEngine(
            enable_shorts=settings.enable_shorts,
            min_confidence=0.40, # UNLEASHED: Entry on trend detection alone
            aggressive_mode=settings.aggressive_mode,
            adx_trend_threshold=adx_trend,
            adx_chop_threshold=adx_chop,
            rsi_oversold=rsi_os,
            rsi_overbought=rsi_ob,
        )

        
        # ML Models
        self.ml_long = load_model(settings.model_path_long)
        self.ml_short = load_model(settings.model_path_short)
        if self.ml_long:
            log.info("[bold green]ML Long Model Loaded.[/bold green]")
        if self.ml_short:
            log.info("[bold green]ML Short Model Loaded.[/bold green]")
        
        # Advanced Institutional Services
        from alpaca_day_bot.services.macro_engine import MacroEngine
        from alpaca_day_bot.services.sentiment_engine import SentimentEngine
        from alpaca_day_bot.services.fred_service import FredLiquidityService
        from alpaca_day_bot.services.moc_imbalance import MocImbalanceService
        self.macro = MacroEngine()
        self.sentiment = SentimentEngine()
        from alpaca_day_bot.services.sector_dispersion import SectorDispersionService
        from alpaca_day_bot.services.earnings_calendar import EarningsCalendarService
        self.fred_service = FredLiquidityService()
        self.moc_service = MocImbalanceService()
        self.sector_service = SectorDispersionService(settings.apca_api_key_id, settings.apca_api_secret_key)
        self.earnings_service = EarningsCalendarService(state_dir=settings.state_dir)
        from alpaca_day_bot.services.macro_calendar import MacroCalendarService
        self.macro_calendar = MacroCalendarService(state_dir=settings.state_dir)

        # Post-Mortem Diagnostic Brain
        self.reviewer = TradeReviewer(self.ledger, self.macro, self.sentiment)

        # Streamers
        self.data_streamer = MarketDataStreamer(settings, self.buffer, self.book_buffer)
        self.trade_streamer = TradingUpdatesStreamer(settings, on_update=self.reviewer.handle_update)

        # Dynamic Discovery Tracking
        self._last_universe_refresh = datetime.now(timezone.utc)
        self._universe_refresh_interval = timedelta(minutes=60)

    def stop(self, *args):
        log.info("[bold red]Stopping Aetheris Alpha Pro...[/bold red]")
        self.running = False
        self.data_streamer.stop()
        self.trade_streamer.stop()
        sys.exit(0)

    def _generate_status_table(self):
        table = Table(title="Aetheris Alpha Pro Status")
        table.add_column("Symbol", style="cyan")
        table.add_column("Signal", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("ML Proba", style="yellow")
        table.add_column("Action", style="bold white")
        
        # Example row
        # table.add_row("SPY", "BUY", "0.85", "0.72", "PENDING")
        return table

    def run(self, day_session: bool = False):
        log.info("[bold green]Aetheris Alpha Pro Initializing...[/bold green]")
        log.info(f"Target Capital: [bold cyan]${self.settings.starting_equity_usd:,.2f}[/bold cyan]")
        log.info(f"Accuracy Threshold: [bold yellow]High (0.75+)[/bold yellow]")
        
        # Reset session clock for reports
        try:
            with open("state/session_start.txt", "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
        except Exception: pass
        
        self.running = True
        
        # Start Prometheus metrics server
        from alpaca_day_bot.services.telemetry import start_metrics_server
        start_metrics_server(port=self.settings.prometheus_port)

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        if day_session:
            self._wait_for_session()

        # Refresh universe if enabled
        if self.settings.universe_enabled:
            log.info(f"[bold yellow]Building Master Universe ({self.settings.asset_class})...[/bold yellow]")
            master_path = str(Path(self.settings.state_dir) / "master_universe.json")
            from alpaca_day_bot.universe import build_master_universe_assets
            
            master_res = build_master_universe_assets(
                apca_api_key_id=self.settings.apca_api_key_id,
                apca_api_secret_key=self.settings.apca_api_secret_key,
                out_path=master_path,
                asset_class=self.settings.asset_class,
                max_symbols=5000
            )
            
            log.info(f"Master Universe built: {len(master_res['symbols'])} assets considered.")
            
            log.info("[bold yellow]Narrowing to Liquid Universe...[/bold yellow]")
            out_path = str(Path(self.settings.state_dir) / "universe_latest.json")
            res = build_liquid_universe(
                apca_api_key_id=self.settings.apca_api_key_id,
                apca_api_secret_key=self.settings.apca_api_secret_key,
                out_path=out_path,
                asset_class=self.settings.asset_class,
                candidate_symbols=master_res["symbols"],
                max_symbols=int(self.settings.universe_max_symbols),
                lookback_days=int(self.settings.universe_lookback_days),
                min_price=float(self.settings.universe_min_price),
                max_price=float(getattr(self.settings, "universe_max_price", 0.0) or 0.0),
                min_avg_dollar_vol=float(self.settings.universe_min_avg_dollar_vol)
            )
            if res.selected:
                self.settings.symbols.clear()
                self.settings.symbols.extend(res.selected)

        # Force SPY only for equity market context
        if self.settings.market_context_filter:
            if "SPY" not in self.settings.symbols:
                self.settings.symbols.append("SPY")
            
        log.info(f"Universe Updated: [bold cyan]{len(self.settings.symbols)} symbols found.[/bold cyan]")

        # Warm buffer with historical data so we can trade immediately
        log.info("[bold yellow]Warming data buffer with historical bars...[/bold yellow]")
        # 1. Critical: SPY Macro-Trend Fetch (EQUITY ONLY)
        spy_count = 0
        if self.settings.market_context_filter:
            log.info("[bold yellow]Performing Dedicated SPY Macro-Trend Fetch...[/bold yellow]")
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed
            from alpaca_day_bot.data.rest_bars import BarEvent, _to_float, _ts_utc

            client = StockHistoricalDataClient(self.settings.apca_api_key_id, self.settings.apca_api_secret_key)
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(minutes=300) # 5 hours of 1m bars
            
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=["SPY"],
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    feed=DataFeed.IEX
                )
                bars = client.get_stock_bars(req)
                if bars.df is not None and not bars.df.empty:
                    for idx, row in bars.df.iterrows():
                        evt = BarEvent(
                            symbol="SPY",
                            ts=_ts_utc(idx[1]),
                            open=_to_float(row["open"]),
                            high=_to_float(row["high"]),
                            low=_to_float(row["low"]),
                            close=_to_float(row["close"]),
                            volume=_to_float(row["volume"]),
                            vwap=_to_float(row.get("vwap", row["close"]))
                        )
                        self.buffer.append(evt)
                        spy_count += 1
                log.info(f"SPY Macro Buffer Warmed: [bold cyan]{spy_count} bars loaded.[/bold cyan]")
            except Exception as e:
                log.warning(f"SPY Macro Fetch Failed: {e}")

        # 2. General warmup for the rest of the universe
        log.info("[bold yellow]Warming data buffer with historical bars...[/bold yellow]")
        poller = RestBarPoller(self.settings, self.buffer)
        filled = poller.warm_buffer(rounds=2)
        
        log.info(f"Buffer Warmed: [bold cyan]{filled + spy_count} historical bars loaded.[/bold cyan]")

        # Start streamers/pollers in background threads
        threading.Thread(target=self.trade_streamer.run_forever, daemon=True).start()
        
        md_mode = (self.settings.market_data_mode or "rest").strip().lower()
        if md_mode == "websocket":
            log.info("[bold green]Starting websocket-based MarketDataStreamer...[/bold green]")
            threading.Thread(target=self.data_streamer.run_forever, daemon=True).start()
        else:
            log.info("[bold green]Starting REST-polling MarketDataPoller...[/bold green]")
            rp = RestBarPoller(self.settings, self.buffer)
            threading.Thread(target=rp.run_forever, name="market-data-rest", daemon=True).start()

        
        log.info("[bold blue]Core engine started. Entering main loop.[/bold blue]")
        
        while self.running:
            try:
                t0 = datetime.now(timezone.utc)
                
                # Check for session end
                if day_session and self._is_session_over(t0):
                    log.info("[bold red]Trading session over (3:00 PM CT). Shutting down.[/bold red]")
                    self.stop()
                    break

                self._tick(t0)
                time.sleep(self.settings.rest_bar_poll_interval_s)
            except Exception as e:
                log.exception(f"Error in main loop: {e}")
                time.sleep(10)

    def _wait_for_session(self):
        """Wait until settings.trade_start time."""
        tz = ZoneInfo(self.settings.market_tz or "America/Chicago")
        start_val = self.settings.trade_start or "08:30:00"
        
        if isinstance(start_val, str):
            start_time = datetime.strptime(start_val, "%H:%M:%S").time()
        else:
            start_time = start_val # Already a time object
        
        while self.running:
            now = datetime.now(tz)
            if now.time() >= start_time and now.weekday() < 5:
                log.info(f"[bold green]Session started at {now.strftime('%H:%M:%S')} {tz}.[/bold green]")
                break
            
            log.info(f"[dim]Waiting for session start ({start_val}). Current time: {now.strftime('%H:%M:%S')} {tz}[/dim]")
            time.sleep(10)

    def _is_session_over(self, t0: datetime) -> bool:
        """Check against settings.trade_end."""
        tz = ZoneInfo(self.settings.market_tz or "America/Chicago")
        now = t0.astimezone(tz)
        end_val = self.settings.trade_end or "15:00:00"
        
        if isinstance(end_val, str):
            end_time = datetime.strptime(end_val, "%H:%M:%S").time()
        else:
            end_time = end_val
        
        return now.time() >= end_time or now.weekday() >= 5

    def _is_extended_hours(self) -> bool:
        """Check if we are in extended hours (4:00 PM - 8:00 PM ET)."""
        tz = ZoneInfo("America/Chicago")
        now = datetime.now(tz=tz)
        # Regular session ends at 3:00 PM CT (4:00 PM ET)
        reg_end = dt_time(15, 0)
        return now.time() >= reg_end

    def _handle_regime_protection(self, market_regime: str):
        """
        Panic-exit or Harvest positions that contradict the broader market regime.
        This prevents holding Longs through a Bearish SPY flip.
        """
        if market_regime == "neutral":
            return
            
        positions = self.executor.get_all_positions()
        for p in positions:
            symbol = p.symbol
            try:
                qty = float(p.qty)
                unrealized_pnl = float(p.unrealized_pl)
            except (ValueError, TypeError):
                continue
            
            # Scenario A: Market is BEARISH, but we are LONG (qty > 0)
            if market_regime == "bearish" and qty > 0:
                # INSTANT PURGE: Close all longs if market is bearish
                log.info(f"[bold red]IRON CURTAIN:[/bold red] Market is BEARISH. Closing LONG {symbol} to prevent further losses.")
                self.executor.close_position(symbol)

            # Scenario B: Market is BULLISH, but we are SHORT (qty < 0)
            elif market_regime == "bullish" and qty < 0:
                if unrealized_pnl > 0:
                    log.info(f"[bold green]HARVESTING PROFIT:[/bold green] Market flipped BULLISH. Banking ${unrealized_pnl:.2f} on {symbol}")
                    self.executor.close_position(symbol)
                else:
                    log.info(f"[bold red]REGIME PANIC:[/bold red] Market flipped BULLISH. Cutting loss on SHORT {symbol}")
                    self.executor.close_position(symbol)

    def _handle_strategic_exits(self, t0: datetime) -> bool:
        """
        Phase 1-3 dynamic liquidation logic for the final configured minutes.
        Returns True if we are in the exit window (prevents new entries).
        """
        tz = ZoneInfo(self.settings.market_tz or "America/Chicago")
        now_dt = t0.astimezone(tz)
        now = now_dt.time()
        
        # Pre-holiday volume drop check for Memorial Day weekend (Friday, May 22, 2026)
        # Halts new entries after 2:00 PM EST (1:00 PM Chicago time)
        # if now_dt.year == 2026 and now_dt.month == 5 and now_dt.day == 22:
        #     if now >= dt_time(13, 0):
        #         log.info("[bold yellow]MEMORIAL DAY WEEKEND PRE-HOLIDAY: Halting new entries past 2:00 PM EST...[/bold yellow]")
        #         return True
        
        # Get dynamic trade end and flatten minutes
        trade_end_val = self.settings.trade_end
        fb_minutes = int(getattr(self.settings, "flatten_before_close_minutes", 20) or 20)
        
        if fb_minutes <= 0:
            return False

        # Normalize trade_end to a time object if it's a string
        if isinstance(trade_end_val, str):
            try:
                trade_end_time = datetime.strptime(trade_end_val, "%H:%M:%S").time()
            except Exception:
                trade_end_time = datetime.strptime(trade_end_val, "%H:%M").time()
        else:
            trade_end_time = trade_end_val

        trade_end_dt = datetime.combine(now_dt.date(), trade_end_time).replace(tzinfo=tz)
        exit_start_dt = trade_end_dt - timedelta(minutes=fb_minutes)
        final_sweep_dt = trade_end_dt - timedelta(minutes=5)
        trim_fat_dt = trade_end_dt - timedelta(minutes=min(15, fb_minutes))
        
        # Outside strategic exit window: business as usual
        if now_dt < exit_start_dt:
            return False
            
        # Phase 3: Final Sweep (final 5 minutes before trade_end)
        if now_dt >= final_sweep_dt:
            open_count = self.executor.open_positions_count()
            if open_count > 0:
                log.warning(f"[bold red]PHASE 3: VIOLENT LIQUIDATION ({open_count} positions remaining)...[/bold red]")
                self.executor.close_all_positions()
            return True
            
        # Phase 2: Trim the Fat (trim laggards and losers to let winners run)
        if now_dt >= trim_fat_dt:
            log.info("[bold yellow]PHASE 2: Trimming flat/losing positions...[/bold yellow]")
            positions = self.executor.get_all_positions()
            for p in positions:
                try:
                    pnl = float(p.unrealized_pl)
                    if pnl <= 0:
                        log.info(f"Closing laggard {p.symbol} at ${pnl:.2f}")
                        self.executor.close_position(p.symbol)
                except Exception:
                    continue
        
        return True # Halt entry logic inside strategic exit window


    def _tick(self, t0: datetime):
        """Single iteration with ML gating for maximum accuracy."""
        spy_df = None
        # Fetch macro data early to avoid UnboundLocalError during market condition scoring
        macro_data = self.macro.get_macro_context()
        # 0. Check for Dynamic Universe Refresh
        now_utc = datetime.now(timezone.utc)
        if (now_utc - self._last_universe_refresh) > self._universe_refresh_interval:
            self._refresh_universe()

        # --- STRATEGIC EXIT CHECK ---
        in_exit_window = self._handle_strategic_exits(t0)
        
        # Simulated Compounding Equity
        acct_equity = self.executor.get_account_equity() or self.settings.starting_equity_usd
        self.risk.reset_day_if_needed(t0.date(), acct_equity, self.settings.starting_equity_usd)
            
        true_daily_pnl = acct_equity - (self.risk._start_acct_equity or acct_equity)
        equity = self.settings.starting_equity_usd + true_daily_pnl
        
        # Adaptive threshold based on recent win rate
        recent_trades = self.risk.get_recent_trades(count=10)
        if recent_trades:
            win_rate = sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades)
            # Adjust threshold based on recent performance
            if win_rate < 0.3:  # Poor recent performance
                self.adaptive_threshold_multiplier = 1.2  # Require higher conviction
            elif win_rate > 0.5:  # Good recent performance
                self.adaptive_threshold_multiplier = 0.9  # Allow slightly lower conviction
            else:
                self.adaptive_threshold_multiplier = 1.0  # Normal threshold
        else:
            self.adaptive_threshold_multiplier = 1.0
        
        # Market condition scoring based on macro data
        self.market_condition_score = 1.0  # Default neutral
        if macro_data:
            vix = macro_data.get('vix', 20)
            spy_trend = macro_data.get('spy_trend_1h', 0)
            
            # Low VIX + bullish SPY = favorable conditions
            if vix < 15 and spy_trend > 0:
                self.market_condition_score = 1.1  # Favorable
            # High VIX + bearish SPY = unfavorable conditions
            elif vix > 25 or spy_trend < 0:
                self.market_condition_score = 0.9  # Unfavorable

        # Update Prometheus metrics
        from alpaca_day_bot.services.telemetry import ESTIMATED_EQUITY, DAILY_LOSS_PCT, PROCESS_MEMORY_MB
        ESTIMATED_EQUITY.set(float(equity))
        start_eq = self.risk._start_virtual_equity or equity
        if start_eq > 0:
            DAILY_LOSS_PCT.set(float(max(0.0, (start_eq - equity) / start_eq * 100)))

        # Hardware memory leak telemetry & Graceful safety restart trigger
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        PROCESS_MEMORY_MB.set(float(mem_mb))
        
        # Memory Leak Guard threshold — If memory utilization spikes above 750MB, trigger safety restart
        if mem_mb > 750.0:
            log.critical(f"[bold red]MEMORY LEAK GUARD TRIGGERED:[/bold red] Process RAM is {mem_mb:.2f} MB (> 750 MB). Initiating graceful shutdown for auto-restart.")
            if in_exit_window and self.executor.open_positions_count() > 0:
                log.critical("[bold red]EMERGENCY LIQUIDATION: Triggered memory shutdown while in exit window! Closing positions before death...[/bold red]")
                self.executor.close_all_positions()
            self.stop()
            return False

        gross = self.executor.gross_exposure_usd()
        open_pos = self.executor.open_positions_count()
        
        # Hard budget cap — starting equity × leverage multiplier
        budget = self.settings.starting_equity_usd * self.settings.max_gross_exposure_pct
        
        # --- MARKET REGIME & CONTEXT GATING ---
        market_regime = "neutral"
        
        if self.settings.market_context_filter:
            spy_df = self.buffer.snapshot_df("SPY")
            # We start calculating as soon as we have 10 bars, but use a 50-period EMA
            if spy_df is not None and len(spy_df) >= 10:
                import pandas as pd
                # EMA 50 on 1m is a proxy for the 5m-10m macro trend
                spy_df["ema_50"] = ta.ema(spy_df["close"], length=50)
                spy_last = spy_df.iloc[-1]
                close_val = spy_last.get("close")
                ema_val = spy_last.get("ema_50")
                
                # Safety: If close or EMA is missing or NaN, stay in a 'warming' state
                if close_val is None or ema_val is None or pd.isna(close_val) or pd.isna(ema_val):
                    log.warning(f"[bold yellow]MARKET GUARD WARMING:[/bold yellow] EMA50 or Close price is missing/NaN. Blocking new entries.")
                    market_regime = "warming"
                else:
                    # Check if the data is fresh (within last 5 mins)
                    now_utc = datetime.now(timezone.utc)
                    bar_age_m = (now_utc - spy_last.name).total_seconds() / 60
                    
                    if bar_age_m > 10:
                        log.warning(f"[bold yellow]MARKET GUARD LAGGING:[/bold yellow] SPY data is {bar_age_m:.1f}m old.")

                    # Hysteresis threshold
                    threshold = 0.15
                    if close_val > (ema_val + threshold):
                        market_regime = "bullish"
                    elif close_val < (ema_val - threshold):
                        market_regime = "bearish"
                    else:
                        market_regime = "neutral"
                    
                    diff = close_val - ema_val
                    log.info(f"[bold blue]MARKET GUARD (H50):[/bold blue] SPY is {market_regime.upper()} (Price: ${close_val:.2f}, EMA50: ${ema_val:.2f}, Diff: ${diff:.2f})")
            else:
                log.warning(f"[bold yellow]MARKET GUARD BLIND:[/bold yellow] Waiting for SPY data... (Current Buffer: {len(spy_df) if spy_df is not None else 0}/10 bars)")
                market_regime = "warming"
        
        # 0.1 Regime Protection: Exit non-aligned positions
        self._handle_regime_protection(market_regime)
        
        # 1. Manage Active Positions (Trailing Stops)
        self._manage_active_positions()
        if self.observe_only:
            self._manage_mock_positions()

        # Get currently held symbols and pending orders to prevent double-ups/spam
        held_symbols = set(self.executor.get_owned_symbols())

        # 1. Signal Detection (Block if warming)
        if market_regime == "warming":
            return

        # ── HARD MARKET HOURS GATE ──────────────────────────────────────────────
        # Refuse to open new positions outside TRADE_START–TRADE_END window.
        # This guards against launchd restarting the bot after session end and
        # firing entries on stale after-hours data (root cause of ALAB/SNOW/NOWL/ADM losses).
        tz_gate = ZoneInfo(self.settings.market_tz or "America/Chicago")
        now_gate = datetime.now(tz=tz_gate)
        try:
            trade_start_t = datetime.strptime(self.settings.trade_start or "08:00:00", "%H:%M:%S").time()
            trade_end_t   = datetime.strptime(self.settings.trade_end   or "15:00:00", "%H:%M:%S").time()
        except Exception:
            trade_start_t, trade_end_t = dt_time(8, 0), dt_time(15, 0)

        outside_hours = (
            now_gate.weekday() >= 5  # weekend
            or now_gate.time() < trade_start_t
            or now_gate.time() >= trade_end_t
        )
        if outside_hours:
            log.info(
                f"[dim]OUTSIDE TRADING WINDOW ({trade_start_t}–{trade_end_t} {self.settings.market_tz}). "
                f"No new entries. Current time: {now_gate.strftime('%H:%M:%S')}[/dim]"
            )
            return
        # ────────────────────────────────────────────────────────────────────────

        log.info(f"HUNT HEARTBEAT: Scanning {len(self.settings.symbols)} symbols for signals...")
        
        # --- PHASE A: SIGNAL COLLECTION (Parallel) ---
        # Score ALL symbols concurrently so the entire universe is evaluated in
        # one heartbeat — no symbol is starved by scan order or position limits.
        # Phase B then picks the highest ML-conviction signals to execute.
        import random
        from concurrent.futures import ThreadPoolExecutor, as_completed

        scan_order = [s for s in self.settings.symbols if s != "SPY" and s not in held_symbols]
        random.shuffle(scan_order)  # randomise to avoid any residual ordering bias

        # Pre-compute shared values (not per-symbol)
        spy_perf = None
        if spy_df is not None and not spy_df.empty:
            s_open = spy_df.iloc[0]["open"]
            s_curr = spy_df.iloc[-1]["close"]
            spy_perf = (s_curr - s_open) / s_open

        macro_data = self.macro.get_macro_context()
        fed_liq = self.fred_service.get_fed_liquidity_momentum()
        dispersion = self.sector_service.get_sector_dispersion()
        is_locked, lockout_reason = self.macro_calendar.is_lockout_active()

        def _score_symbol(sym):
            """Score a single symbol — runs in thread pool."""
            try:
                df_1m = self.buffer.snapshot_df(sym)
                df_15m = self.buffer.snapshot_resampled_df(sym, "15min")
                if df_1m is None or df_1m.empty:
                    return None

                ob_feats = self.book_buffer.latest_features(sym)

                sig = self.strategy.decide(
                    symbol=sym,
                    df_1m=df_1m,
                    df_15m=df_15m,
                    market_regime=market_regime,
                    spy_perf=spy_perf,
                    order_book=ob_feats
                )

                if sig:
                    sig.features["order_book"] = {f"val_{i}": v for i, v in enumerate(ob_feats)}

                if not sig or sig.action == "HOLD":
                    return None

                blocked = False
                block_reason = None

                if sig.action == "SHORT" and not self.settings.enable_shorts:
                    return None  # shorts disabled — skip silently

                if not self.settings.aggressive_mode:
                    if market_regime == "bearish" and sig.action == "BUY":
                        return None
                    elif market_regime == "bullish" and sig.action == "SHORT":
                        return None

                sig.features.update(macro_data)
                sig.features["fed_liquidity_momentum"] = fed_liq
                sig.features["sector_dispersion_factor"] = dispersion

                minutes_to_earnings = self.earnings_service.get_minutes_until_earnings(sym)
                sig.features["minutes_until_earnings_announcement"] = minutes_to_earnings

                if minutes_to_earnings < 240.0:
                    log.warning(f"[bold red]EARNINGS LOCKDOWN:[/bold red] {sym} earnings in {minutes_to_earnings:.1f}m. Blocking.")
                    return None

                if is_locked:
                    log.warning(f"[bold red]MACRO EVENT OVERRIDE:[/bold red] {lockout_reason} Blocking {sym}.")
                    return None

                moc_imbalance = self.moc_service.get_moc_imbalance_shares(sym, now_utc, self.book_buffer)
                sig.features["moc_imbalance_shares"] = moc_imbalance

                if sig.action == "BUY" and moc_imbalance <= -10000.0:
                    log.warning(f"[bold red]MOC SELLER LOCKDOWN:[/bold red] {sym} imbalance={moc_imbalance:.0f}. Blocking.")
                    return None

                spread_ratio = float(sig.features.get("spread_ratio", 0.0))
                if spread_ratio > 0.005:
                    log.warning(f"[bold red]SPREAD TOXICITY BLOCK:[/bold red] {sym} spread={spread_ratio:.2%}. Blocking.")
                    return None

                if sig.action == "BUY":
                    st_dir = float(sig.features.get("supertrend_dir", 1.0))
                    if st_dir <= 0:
                        log.warning(f"[bold red]SUPERTREND BLOCK:[/bold red] {sym} SuperTrend is BEARISH (dir={st_dir:.0f}). Blocking BUY to avoid contra-trend entry.")
                        return None

                if sig.action == "BUY":
                    htf_rsi_max = float(getattr(self.settings, "htf_rsi_max", 85.0))
                    htf_rsi_val = float(sig.features.get("htf_rsi") or sig.features.get("rsi_60") or 50.0)
                    if htf_rsi_val > htf_rsi_max:
                        log.warning(f"[bold red]HTF RSI BLOCK:[/bold red] {sym} 1H RSI={htf_rsi_val:.1f} > max={htf_rsi_max:.0f}. Blocking BUY — price is overbought.")
                        return None

                min_vol_ratio = float(getattr(self.settings, "min_volume_ratio_trade", 0.0))
                cur_vol_ratio = float(sig.features.get("volume_ratio") or sig.features.get("vol_ratio") or 1.0)
                if min_vol_ratio > 0 and cur_vol_ratio < min_vol_ratio:
                    log.warning(f"[bold red]VOLUME DROUGHT BLOCK:[/bold red] {sym} volume_ratio={cur_vol_ratio:.2f} < min={min_vol_ratio:.2f}. Blocking entry — no liquidity support.")
                    return None

                # ML probability scoring
                ml_proba = 0.0
                explainability = None
                threshold = 0.50

                model = self.ml_long if sig.action == "BUY" else self.ml_short
                if model:
                    ml_res = predict_proba(model_bundle=model, features=sig.features)
                    ml_proba = ml_res.proba or 0.0
                    explainability = ml_res.explainability
                    threshold = ml_res.threshold if ml_res.threshold is not None else float(model.get("threshold", 0.50))

                    import os
                    env_override = None
                    if sig.action == "BUY":
                        env_override = float(os.getenv("MODEL_MIN_PROBA_LONG")) if os.getenv("MODEL_MIN_PROBA_LONG") else \
                                       float(os.getenv("MODEL_MIN_PROBA")) if os.getenv("MODEL_MIN_PROBA") else None
                    else:
                        env_override = float(os.getenv("MODEL_MIN_PROBA_SHORT")) if os.getenv("MODEL_MIN_PROBA_SHORT") else \
                                       float(os.getenv("MODEL_MIN_PROBA")) if os.getenv("MODEL_MIN_PROBA") else None
                    if env_override is not None:
                        threshold = env_override

                    # Apply adaptive threshold multiplier based on recent performance
                    adaptive_threshold = threshold * getattr(self, 'adaptive_threshold_multiplier', 1.0)
                    
                    # Apply market condition score
                    market_score = getattr(self, 'market_condition_score', 1.0)
                    final_threshold = adaptive_threshold / market_score  # Lower threshold in favorable conditions
                    
                    # Filter loss-causing patterns (based on large dataset analysis)
                    LOSS_CAUSING_PATTERNS = {"quant_mr_oversold_z-2.1_osc"}
                    if sig.reason in LOSS_CAUSING_PATTERNS:
                        log.info(f"[bold red]PATTERN BLOCKED (Loss-causing):[/bold red] {sym} pattern '{sig.reason}' has 11% win rate - filtered")
                        return None
                    
                    # Volume confirmation filter (RVOL > 1.5 for better accuracy)
                    rvol = sig.features.get("rvol", 1.0)
                    if rvol < 1.5:
                        log.info(f"[bold yellow]VOLUME FILTER:[/bold yellow] {sym} RVOL {rvol:.2f} < 1.5 - filtered")
                        return None
                    
                    # Trend strength filter (ADX > 25 for better accuracy)
                    adx = sig.features.get("adx", 20)
                    if adx < 25:
                        log.info(f"[bold yellow]TRENGTH FILTER:[/bold yellow] {sym} ADX {adx:.1f} < 25 - filtered")
                        return None
                    
                    if not ml_res.ok or ml_proba < final_threshold:
                        log.info(f"[bold red]ML Blocked (Too Low Proba):[/bold red] {sym} ({sig.action}) - Proba: {ml_proba:.2%}, Threshold: {final_threshold:.2%} (base: {threshold:.2%}, mult: {getattr(self, 'adaptive_threshold_multiplier', 1.0):.2f}, market: {market_score:.2f})")
                        return None

                return {
                    "sym": sym,
                    "sig": sig,
                    "ml_proba": ml_proba,
                    "threshold": threshold,
                    "explainability": explainability,
                    "macro_data": macro_data,
                }
            except Exception as exc:
                log.debug(f"_score_symbol error for {sym}: {exc}")
                return None

        # Run all symbol scorers in parallel (I/O-bound: buffer reads + ML inference)
        # 16 workers balances CPU and avoids overwhelming shared data structures
        valid_signals = []
        with ThreadPoolExecutor(max_workers=32, thread_name_prefix="scan") as pool:
            futures = {pool.submit(_score_symbol, sym): sym for sym in scan_order}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    valid_signals.append(result)
        
        # --- PHASE B: RANK, SIZE, AND EXECUTE ---
        if not valid_signals:
            return
            
        valid_signals.sort(key=lambda x: x["ml_proba"], reverse=True)
        log.info(f"[bold cyan]BATCH EXECUTION: Found {len(valid_signals)} valid signals. Ranking by conviction...[/bold cyan]")
        
        for item in valid_signals:
            sym = item["sym"]
            sig = item["sig"]
            ml_proba = item["ml_proba"]
            explainability = item["explainability"]
            macro_data = item["macro_data"]
            
            if gross >= budget:
                log.warning(f"Budget cap reached: ${gross:.2f} >= ${budget:.2f}. Skipping remaining {sym}.")
                break
                
            if self.settings.max_positions > 0 and open_pos >= self.settings.max_positions:
                log.warning(f"Max positions reached: {open_pos} >= {self.settings.max_positions}. Skipping remaining {sym}.")
                break
                
            # Lazy Load News Sentiment (Only fetch for top candidates)
            from alpaca_day_bot.data.news import fetch_news_for_symbol
            news_res = fetch_news_for_symbol(
                symbol=sym,
                provider=self.settings.news_provider,
                alpaca_api_key_id=self.settings.apca_api_key_id,
                alpaca_secret_key=self.settings.apca_api_secret_key,
                alphavantage_api_key=getattr(self.settings, "alphavantage_api_key", None),
                lookback_hours=12.0,
                limit=10
            )
            news_sentiment = 0.0
            if news_res.get("ok"):
                texts = []
                for a in news_res.get("articles", []):
                    hl = a.get("headline") or ""
                    summary = a.get("summary") or ""
                    texts.append(f"{hl} {summary}")
                news_sentiment = self.sentiment.get_alpaca_news_sentiment(sym, texts)
            
            sig.features["news_sentiment_score"] = news_sentiment
            sentiment_data = self.sentiment.get_sentiment(sym)
            sig.features.update(sentiment_data)
            
            blocked = False
            block_reason = None
            if sig.action == "BUY" and news_sentiment <= -0.6:
                log.warning(f"[bold red]NEWS LOCKDOWN:[/bold red] {sym} news sentiment score is {news_sentiment:.2f} (<= -0.6). Blocking Mean Reversion long entry.")
                blocked = True
                block_reason = "news_lockdown"

            # Record Final Signal
            sig.features["ml_proba"] = ml_proba
            sig.features["blocked"] = blocked
            sig.features["blocked_reason"] = block_reason
            
            sig_id = self.ledger.record_signal(
                ts=now_utc,
                symbol=sym,
                action=sig.action,
                reason=sig.reason,
                features=sig.features,
                explainability=explainability,
                context=macro_data
            )
            
            if blocked:
                continue
                
            if in_exit_window:
                log.debug(f"Strategic Exit Window Active: Skipping entry for {sym}")
                continue

            # Cooldown validation
            if not self.risk.can_trade_symbol(sym, now_utc):
                log.debug(f"Risk Blocked: {sym} is in cooldown.")
                continue

            log.info(f"[bold yellow]ACCURATE SIGNAL EXECUTING:[/bold yellow] {sym} -> {sig.action} (ML Proba: {ml_proba:.2f})")

            
            # Dynamic Stop Distance
            from alpaca_day_bot.risk.garch_model import predict_garch_volatility
            from alpaca_day_bot.services.telemetry import GARCH_VOLATILITY, KELLY_RISK_PCT
            
            hist_df = self.buffer.snapshot_df(sym)
            garch_floor_pct = 0.0025
            if hist_df is not None and len(hist_df) >= 25:
                closes = hist_df["close"].to_numpy(dtype=np.float64)
                garch_floor_pct = predict_garch_volatility(closes)
            
            GARCH_VOLATILITY.labels(symbol=sym).set(float(garch_floor_pct))
            
            atr = sig.features.get("atr") or (sig.features.get("close") * 0.01)
            price = sig.features["close"]
            
            garch_floor = price * (garch_floor_pct * 3.0)
            absolute_floor = price * 0.0075
            stop_dist = max(atr * self.settings.stop_loss_atr_mult, garch_floor, absolute_floor)

            # Fix 5: Minimum Stop Distance Guard — prevent instant-stop scenarios
            min_stop_pct = float(getattr(self.settings, "min_stop_dist_pct", 0.005))
            min_stop_dist = price * min_stop_pct
            if stop_dist < min_stop_dist:
                log.debug(f"Stop distance {stop_dist:.4f} < min {min_stop_dist:.4f} for {sym}. Widening to {min_stop_pct:.1%} floor.")
                stop_dist = min_stop_dist
            
            remaining_budget = max(0.0, budget - gross)
            if remaining_budget < 50:
                log.debug(f"Remaining budget too small: ${remaining_budget:.2f}")
                break
                
            # TRUE KELLY SIZING
            b = float(self.settings.take_profit_r_mult) if hasattr(self.settings, "take_profit_r_mult") else 2.5
            p = ml_proba
            q = 1.0 - p
            raw_kelly = (b * p - q) / b
            kelly_val = min(0.05, max(0.0, raw_kelly * 0.20))  # Safe fractional Kelly capped at 5% risk
            KELLY_RISK_PCT.labels(symbol=sym).set(float(kelly_val))
            
            # Kelly sizes the RISK budget, not the NOTIONAL allocation
            risk_budget = equity * kelly_val
            
            # Prevent zero stop distance division
            if stop_dist > 0:
                target_qty = risk_budget / stop_dist
                kelly_notional = target_qty * price
            else:
                kelly_notional = 0.0
                
            target_notional = min(kelly_notional, remaining_budget)
            if self.settings.max_notional_per_trade_usd > 0:
                target_notional = min(target_notional, self.settings.max_notional_per_trade_usd)
            final_qty = int(target_notional / price)
            if final_qty <= 0:
                log.debug(f"Kelly Budget clamp: integer qty=0 for {sym} (notional: ${target_notional:.2f})")
                continue

            # Execute with SL/TP
            tp_price = 0.0
            sl_price = 0.0
            if sig.action == "BUY":
                tp_price = price + (stop_dist * self.settings.take_profit_r_mult)
                sl_price = price - stop_dist
            else:
                tp_price = price - (stop_dist * self.settings.take_profit_r_mult)
                sl_price = price + stop_dist

            from alpaca.trading.enums import OrderSide
            side = OrderSide.BUY if sig.action == "BUY" else OrderSide.SELL
            
            if self.observe_only:
                self.risk.register_trade(sym, now_utc)
                self.mock_positions[sym] = {
                    "action": sig.action,
                    "qty": final_qty,
                    "entry_price": price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "ts": datetime.now(timezone.utc)
                }
                log.info(f"[cyan]MOCK EXECUTED:[/cyan] {sym} {sig.action} {final_qty} shares @ ${price:.2f} (TP: ${tp_price:.2f}, SL: ${sl_price:.2f})")
                continue

            is_extended = self._is_extended_hours()
            if is_extended:
                res = self.executor.submit_simple_limit_chase(
                    symbol=sym,
                    side=side,
                    qty=final_qty,
                    limit_price=price,
                    chase_seconds=15
                )
            else:
                res = self.executor.submit_bracket_limit_chase(
                    symbol=sym,
                    side=side,
                    qty=final_qty,
                    limit_price=price,
                    stop_price=sl_price,
                    take_profit_price=tp_price,
                    chase_seconds=15
                )
                
            if res.submitted:
                self.risk.register_trade(sym, now_utc)
                order_notional = final_qty * price
                gross += order_notional
                open_pos += 1
                held_symbols.add(sym)
                log.info(f"[bold green]CHASE SUCCESS:[/bold green] {sym} {sig.action} filled {final_qty} shares at ${price:.2f} (Kelly: {kelly_val:.2%})")

            else:
                log.warning(f"[bold yellow]CHASE FAILED:[/bold yellow] {sym} - {res.reason}")
    def _refresh_universe(self):
        """
        Dynamically refreshes the trading universe mid-session to catch new momentum.
        """
        log.info("[bold cyan]DYNAMIC REFRESH: Re-building Institutional Universe...[/bold cyan]")
        try:
            from alpaca_day_bot.universe import build_liquid_universe
            res = build_liquid_universe(
                apca_api_key_id=self.settings.apca_api_key_id,
                apca_api_secret_key=self.settings.apca_api_secret_key,
                out_path=str(Path(self.settings.state_dir) / "universe.json"),
                asset_class=self.settings.asset_class,
                max_symbols=self.settings.universe_max_symbols,
                lookback_days=5,
                min_price=self.settings.universe_min_price,
                min_avg_dollar_vol=self.settings.universe_min_avg_dollar_vol
            )
            # Update the bot's live symbol list in-place
            self.settings.symbols.clear()
            self.settings.symbols.extend(res.selected)
            self._last_universe_refresh = datetime.now(timezone.utc)
            log.info(f"[bold green]DYNAMIC REFRESH COMPLETE: Hunting {len(self.settings.symbols)} momentum symbols.[/bold green]")
        except Exception as e:
            log.error(f"Dynamic Universe Refresh Failed: {e}")

    def _manage_active_positions(self):
        """
        Implements a dynamic Trailing Stop (Profit Guard) with extreme precision:
        - Dynamically fetches actual open stop order legs from Alpaca.
        - Calculates the exact risk distance (R) instead of relying on a static proxy.
        - Moves stop-loss to Break-Even at 0.7R (aggressive) / 1.5R.
        - Dynamically trails stop-loss at 1.2R (aggressive) / 3.0R to lock in profits.
        - Liquidates stagnant positions exceeding MAX_HOLD_MINUTES (Time-Stops).
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            be_trigger = getattr(self.settings, "be_trigger", 1.0)
            trail_trigger = getattr(self.settings, "trail_trigger", 2.0)
            max_hold = float(getattr(self.settings, "max_hold_minutes", 0.0) or 0.0)
            dynamic_hold = bool(getattr(self.settings, "dynamic_hold_enabled", True))
            
            # Fetch latest intents for the day to check entry timestamps
            tz = ZoneInfo(self.settings.market_tz or "America/Chicago")
            now_ct = datetime.now(tz)
            market_day = now_ct.date()
            intents = self.ledger.last_submitted_entry_intents_for_trading_date(market_day, tz)
            
            positions = self.executor._tc.get_all_positions()
            for p in positions:
                sym = p.symbol
                
                # Check for Time-Based Exits first
                intent = intents.get(sym)
                if intent and intent.get("ts"):
                    entry_ts = intent["ts"]
                    now_utc = datetime.now(timezone.utc)
                    age_m = (now_utc - entry_ts).total_seconds() / 60.0
                    
                    target = max_hold
                    extra = intent.get("extra") if isinstance(intent.get("extra"), dict) else {}
                    if dynamic_hold:
                        try:
                            target = float(extra.get("target_hold_minutes") or 0.0) or target
                        except Exception:
                            pass
                            
                    if target > 0 and age_m >= target:
                        log.info(f"[bold yellow]TIME EXIT:[/bold yellow] {sym} has been held for {age_m:.1f}m (>= {target}m limit). Liquidating position.")
                        res = self.executor.close_position(sym)
                        self.ledger.record_order_intent(
                            ts=now_utc,
                            symbol=sym,
                            side="close",
                            notional_usd=0.0,
                            stop_price=0.0,
                            take_profit_price=0.0,
                            client_order_id=None,
                            alpaca_order_id=res.alpaca_order_id,
                            submitted=res.submitted,
                            reason=f"time_exit:{res.reason}",
                            extra={"action": "EXIT_TIME", "age_minutes": age_m, "target_hold_minutes": target},
                        )
                        continue

                
                # Fetch actual open stop order for this symbol to get real stop price
                req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym], nested=True)
                orders = self.executor._tc.get_orders(req) or []
                
                stop_order = None
                for o in orders:
                    if str(o.type).split(".")[-1].lower() in ("stop", "stop_limit"):
                        stop_order = o
                        break
                
                if stop_order is None or stop_order.stop_price is None:
                    continue
                
                stop_price = float(stop_order.stop_price)
                entry = float(p.avg_entry_price)
                curr = float(p.current_price)
                qty = abs(float(p.qty))
                side = 1 if float(p.qty) > 0 else -1
                
                unrealized_pnl = float(p.unrealized_pl)
                
                # Exact initial stop distance
                r_dist = abs(entry - stop_price)
                if r_dist <= 0:
                    continue
                    
                # Precise R-multiple calculation
                r_multiple = (unrealized_pnl / (qty * r_dist)) if (qty * r_dist) > 0 else 0
                
                new_stop = None
                
                if side == 1:  # Long Position
                    target_stop = curr - (1.2 * r_dist)
                    # Profit Guard: Move to Break-Even
                    if r_multiple >= be_trigger and stop_price < entry:
                        new_stop = entry
                        log.info(f"[bold yellow]PROFIT GUARD (Long BE):[/bold yellow] {sym} is up {r_multiple:.2f}R. Moving SL to Break-Even (${entry:.2f})")
                    # Trailing Stop: Lock in profits
                    elif r_multiple >= trail_trigger and target_stop > stop_price and target_stop > entry:
                        new_stop = target_stop
                        log.info(f"[bold green]TRAILING PROFIT LOCK (Long):[/bold green] {sym} is up {r_multiple:.2f}R. Trailing SL to ${target_stop:.2f}")
                
                else:  # Short Position
                    target_stop = curr + (1.2 * r_dist)
                    # Profit Guard: Move to Break-Even
                    if r_multiple >= be_trigger and stop_price > entry:
                        new_stop = entry
                        log.info(f"[bold yellow]PROFIT GUARD (Short BE):[/bold yellow] {sym} is up {r_multiple:.2f}R. Moving SL to Break-Even (${entry:.2f})")
                    # Trailing Stop: Lock in profits
                    elif r_multiple >= trail_trigger and target_stop < stop_price and target_stop < entry:
                        new_stop = target_stop
                        log.info(f"[bold green]TRAILING PROFIT LOCK (Short):[/bold green] {sym} is up {r_multiple:.2f}R. Trailing SL to ${target_stop:.2f}")
                
                # Execute stop adjustment
                if new_stop is not None:
                    # Round stop price to 2 decimals for equity formatting
                    new_stop = round(new_stop, 2)
                    if abs(new_stop - stop_price) >= 0.01:
                        self.executor.update_stop_loss_for_symbol(sym, new_stop)
 
        except Exception as e:
            log.error(f"Trailing Stop Manager Error: {e}")

    def _manage_mock_positions(self):
        """Simulates bracket execution and P&L tracking for mock trading."""
        closed_syms = []
        for sym, pos in self.mock_positions.items():
            df = self.buffer.snapshot_df(sym)
            if df is None or len(df) == 0:
                continue
            last_bar = df.iloc[-1]
            high = last_bar['high']
            low = last_bar['low']
            
            outcome = None
            exit_price = 0.0
            
            # Check for Time-Based Exits in Mock Trading
            max_hold = float(getattr(self.settings, "max_hold_minutes", 0.0) or 0.0)
            if max_hold > 0 and 'ts' in pos:
                age_m = (datetime.now(timezone.utc) - pos['ts']).total_seconds() / 60.0
                if age_m >= max_hold:
                    outcome = "TIME_EXIT"
                    exit_price = last_bar['close']
                    
            if outcome is None:
                if pos['action'] == "BUY":
                    if high >= pos['tp_price']:
                        outcome = "WIN (TP)"
                        exit_price = pos['tp_price']
                    elif low <= pos['sl_price']:
                        outcome = "LOSS (SL)"
                        exit_price = pos['sl_price']
                else:
                    if low <= pos['tp_price']:
                        outcome = "WIN (TP)"
                        exit_price = pos['tp_price']
                    elif high >= pos['sl_price']:
                        outcome = "LOSS (SL)"
                        exit_price = pos['sl_price']

                    
            if outcome:
                # Calculate Realized P&L
                if pos['action'] == "BUY":
                    pnl = (exit_price - pos['entry_price']) * pos['qty']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['qty']
                
                self.mock_realized_pnl += pnl
                color = "green" if pnl > 0 else "red"
                log.info(f"[bold {color}]MOCK TRADE CLOSED:[/bold {color}] {sym} hit {outcome}. Realized P&L: ${pnl:.2f}. Total Mock Session P&L: ${self.mock_realized_pnl:.2f}")
                closed_syms.append(sym)
                
        for sym in closed_syms:
            del self.mock_positions[sym]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe-only", action="store_true", help="Do not place orders")
    parser.add_argument("--aggressive", action="store_true", help="Enable 3-of-4 and 2-of-4 confirmation (Max Frequency)")
    parser.add_argument("--day-session", action="store_true", help="Wait for 8:30 AM CT and exit at 3:00 PM CT")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    args = parser.parse_args()
    
    import os
    import dotenv
    dotenv.load_dotenv(args.env, override=True)
    os.environ["ENV_FILE"] = args.env
    settings = load_settings(args.env)
    
    if args.aggressive:
        settings.aggressive_mode = True
        log.info("[bold red]Aggressive Mode Enabled (Max Frequency).[/bold red]")
    
    bot = AetherisBotPro(settings, observe_only=args.observe_only)
    bot.run(day_session=args.day_session)

if __name__ == "__main__":
    main()
