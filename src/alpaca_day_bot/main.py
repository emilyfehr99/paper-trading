from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import pandas as pd

from alpaca_day_bot.config import load_settings
from alpaca_day_bot.data.news import fetch_news_for_symbol, news_bundle_should_block
from alpaca_day_bot.data.taapi import fetch_taapi_indicators_for_stock
from alpaca_day_bot.data.stream import BarBuffer, MarketDataStreamer
from alpaca_day_bot.logging_utils import setup_json_logging
from alpaca_day_bot.ml.infer import load_model as _load_ml_model, predict_proba as _ml_predict_proba
from alpaca_day_bot.reporting.report import write_daily_report, write_weekly_report
from alpaca_day_bot.risk.manager import RiskManager, now_utc
from alpaca_day_bot.storage.ledger import Ledger
from alpaca_day_bot.strategy.v1_rules import V1RulesSignalEngine
from alpaca_day_bot.trading.client import make_trading_client
from alpaca_day_bot.trading.executor import OrderExecutor
from alpaca_day_bot.trading.updates import TradeUpdateEvent, TradingUpdatesStreamer
from alpaca_day_bot.backtest import (
    run_backtest,
    run_cost_sensitivity_grid,
    run_param_sweep,
    run_walk_forward,
    time_of_day_breakdown,
    buy_and_hold_returns,
    compute_spy_regimes,
    label_trade_regime,
    symbol_daytrade_recommendations,
)
from alpaca_day_bot.universe import build_liquid_universe, load_universe_symbols


log = logging.getLogger("alpaca_day_bot")


def _acquire_single_instance_lock(state_dir: Path) -> object | None:
    """
    Prevent multiple live bots with the same API key (Alpaca returns
    'connection limit exceeded' on market-data websockets).
    """
    try:
        import fcntl
    except ImportError:
        return None
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "bot.lock"
    fp = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        raise SystemExit(
            f"Another alpaca_day_bot is already running (lock: {path}). "
            "Stop other instances (e.g. `pkill -f 'alpaca_day_bot'`), wait ~2 minutes, then retry."
        )
    fp.write(str(os.getpid()))
    fp.flush()
    return fp


def _try_acquire_single_instance_lock(state_dir: Path) -> object | None:
    """Like _acquire_single_instance_lock but returns None if another process holds the lock."""
    try:
        import fcntl
    except ImportError:
        return None
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "bot.lock"
    fp = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        return None
    fp.write(str(os.getpid()))
    fp.flush()
    return fp


def _base_symbols(settings) -> list[str]:
    """
    Universe selection.
    - Default: SYMBOLS from config.
    - If UNIVERSE_ENABLED=true: load `state/universe_latest.json` if present, else fall back to SYMBOLS.
    """
    base = list(settings.symbols)
    if not getattr(settings, "universe_enabled", False):
        return base
    try:
        p = Path(settings.state_dir) / "universe_latest.json"
        u = load_universe_symbols(str(p))
        if u:
            return u
    except Exception:
        pass
    return base


def _ordered_symbols(settings) -> list[str]:
    """Put `focus_symbols` from robustness JSON first (same universe as base universe)."""
    base = _base_symbols(settings)
    path_s = settings.recommendations_json
    p = Path(path_s) if path_s else Path(settings.reports_dir) / "day_trade_recommendations_latest.json"
    if not p.is_file():
        return base
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        focus = [x for x in (data.get("focus_symbols") or []) if x in base]
        rest = [s for s in base if s not in focus]
        if focus:
            log.info(
                "symbol_order from recommendations",
                extra={"extra_json": {"path": str(p), "focus_first": focus}},
            )
        return focus + rest
    except Exception:
        return base


def _label_buy_signals_forward_returns(
    *,
    ledger: Ledger,
    buffer: BarBuffer,
    settings,
    t0: datetime,
    market_day: date,
) -> None:
    if not settings.signal_accuracy_enabled:
        return
    pending = ledger.list_unlabeled_buy_signal_rows(
        market_day=market_day,
        tz=settings.tzinfo(),
        now_utc=t0,
        min_age_minutes=float(settings.signal_accuracy_min_age_minutes),
    )
    for signal_id, ts_s, sym, feat_json in pending:
        df = _run_sync(buffer.snapshot_df(sym))
        if df is None or getattr(df, "empty", True):
            continue
        try:
            feat = json.loads(feat_json)
            entry = feat.get("close")
            if entry is None:
                continue
            entry_f = float(entry)
            if entry_f <= 0:
                continue
        except Exception:
            continue
        try:
            ts_sig = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
        except Exception:
            continue
        if ts_sig.tzinfo is None:
            ts_sig = ts_sig.replace(tzinfo=timezone.utc)
        horizon_m = (t0 - ts_sig).total_seconds() / 60.0
        now_px = float(df["close"].iloc[-1])
        ret = (now_px - entry_f) / entry_f
        ledger.record_forward_return_label(
            signal_id=signal_id,
            evaluated_ts=t0,
            price_at_label=now_px,
            entry_close=entry_f,
            return_pct=ret,
            horizon_minutes=horizon_m,
        )


def _run_in_window_trading_cycle(
    *,
    settings,
    observe_only: bool,
    ledger: Ledger,
    executor: OrderExecutor,
    buffer: BarBuffer,
    strategy: V1RulesSignalEngine,
    risk: RiskManager,
    t0: datetime,
    market_day: date,
    last_signal_scan_ts: datetime,
    force_signal_scan: bool,
) -> datetime:
    """One pass: optional signal_scan + per-symbol decisions. Returns updated last_signal_scan_ts."""
    market_now = t0.astimezone(settings.tzinfo())

    equity = executor.get_account_equity()
    gross = executor.gross_exposure_usd()
    open_positions = executor.open_positions_count()

    scan_due = force_signal_scan or (t0 - last_signal_scan_ts) >= timedelta(
        seconds=float(settings.signal_scan_interval_s)
    )
    if scan_due:
        last_signal_scan_ts = t0
        candidates: list[dict] = []
        for sym in _ordered_symbols(settings):
            df_1m_s = _run_sync(buffer.snapshot_df(sym))
            df_15m_s = _run_sync(buffer.snapshot_resampled_df(sym, "15min"))
            row = strategy.evaluate_setup(symbol=sym, df_1m=df_1m_s, df_15m=df_15m_s)
            candidates.append(row)
        candidates.sort(key=lambda r: (-int(r.get("buy_score", 0)), str(r.get("symbol", ""))))
        top = candidates[:8]
        log.info(
            "signal_scan",
            extra={
                "extra_json": {
                    "market_day": str(market_day),
                    "candidates": top,
                    "note": "buy_score counts rsi_pullback, above_ema, macd_cross, above_vwap, volume_confirm",
                }
            },
        )

    market_open = market_now.replace(hour=9, minute=30, second=0, microsecond=0)
    if settings.open_delay_minutes > 0 and market_now < (
        market_open + timedelta(minutes=int(settings.open_delay_minutes))
    ):
        return last_signal_scan_ts

    if settings.market_context_filter:
        spy_5m = _run_sync(buffer.snapshot_resampled_df("SPY", "5min"))
        if spy_5m is not None and not getattr(spy_5m, "empty", True):
            try:
                import pandas_ta as ta

                tmp = spy_5m.copy()
                tmp["rsi"] = ta.rsi(tmp["close"], length=14)
                rsi = float(tmp["rsi"].iloc[-1])
                if rsi < float(settings.spy_5m_rsi_min):
                    return last_signal_scan_ts
            except Exception:
                pass

    def _try_submit_entry(
        *,
        sym: str,
        action: str,
        feat: dict,
        df_1m,
    ) -> None:
        """
        Single entry attempt with full checks + consistent ledger/risk logging.
        Used by both the direct rules path and the ML-ranked path.
        """
        nonlocal equity, gross, open_positions

        if action not in ("BUY", "SHORT"):
            return
        if df_1m is None or getattr(df_1m, "empty", True):
            return
        if executor.has_position(sym):
            return

        # Optional TAAPI confirmation layer (stocks RSI/MACD).
        if (
            (settings.indicator_provider or "").strip().lower() == "taapi"
            and bool(settings.taapi_confirm_on_trade)
            and (settings.taapi_secret or "").strip()
        ):
            ti = fetch_taapi_indicators_for_stock(secret=settings.taapi_secret or "", symbol=sym)
            feat["taapi"] = {
                "rsi_1m": ti.rsi_1m,
                "rsi_15m": ti.rsi_15m,
                "macd_1m": ti.macd_1m,
                "macd_signal_1m": ti.macd_signal_1m,
            }
            if action == "BUY":
                if ti.rsi_1m is None or ti.rsi_15m is None or ti.macd_1m is None or ti.macd_signal_1m is None:
                    if not bool(getattr(settings, "taapi_fail_open", True)):
                        return
                else:
                    ok = (
                        ti.rsi_1m <= float(settings.rsi_pullback_max)
                        and ti.rsi_15m >= float(settings.htf_rsi_min)
                        and ti.macd_1m >= ti.macd_signal_1m
                    )
                    if not ok:
                        return
            else:
                if ti.rsi_1m is None or ti.rsi_15m is None or ti.macd_1m is None or ti.macd_signal_1m is None:
                    if not bool(getattr(settings, "taapi_fail_open", True)):
                        return
                else:
                    ok = (
                        ti.rsi_15m <= float(settings.htf_rsi_max_short)
                        and ti.rsi_1m >= float(settings.rsi_rebound_min_short)
                        and ti.macd_1m <= ti.macd_signal_1m
                    )
                    if not ok:
                        return

        last_close = float(df_1m["close"].iloc[-1])
        last_range = float((df_1m["high"].iloc[-1] - df_1m["low"].iloc[-1]))
        stop_dist = max(0.01, last_range * settings.stop_loss_atr_mult)
        if action == "BUY":
            stop_price = last_close - stop_dist
            tp_price = last_close + stop_dist * settings.take_profit_r_mult
        else:
            stop_price = last_close + stop_dist
            tp_price = last_close - stop_dist * settings.take_profit_r_mult
        min_tick_buffer = max(0.25, abs(last_close) * 0.005)
        if action == "BUY":
            stop_price = min(stop_price, last_close - min_tick_buffer)
            tp_price = max(tp_price, last_close + min_tick_buffer)
        else:
            stop_price = max(stop_price, last_close + min_tick_buffer)
            tp_price = min(tp_price, last_close - min_tick_buffer)
        if tp_price <= 0:
            return

        rd = risk.decide_entry(
            symbol=sym,
            equity=equity,
            gross_exposure_usd=gross,
            open_positions=open_positions,
            now_utc=t0,
            trading_date=market_day,
            price=last_close,
            stop_distance=stop_dist,
        )
        if not rd.allow:
            return

        qty_int = int(float(rd.qty))
        if qty_int <= 0:
            return

        if observe_only:
            risk.register_trade(sym, t0)
            return

        if action == "BUY":
            res = executor.submit_bracket_buy(symbol=sym, qty=qty_int, stop_price=stop_price, take_profit_price=tp_price)
            side = "buy"
        else:
            res = executor.submit_bracket_short(symbol=sym, qty=qty_int, stop_price=stop_price, take_profit_price=tp_price)
            side = "sell"

        ledger.record_order_intent(
            ts=t0,
            symbol=sym,
            side=side,
            notional_usd=rd.notional_usd,
            stop_price=stop_price,
            take_profit_price=tp_price,
            client_order_id=res.client_order_id,
            alpaca_order_id=res.alpaca_order_id,
            submitted=res.submitted,
            reason=res.reason,
            extra={"qty": qty_int, "action": action},
        )
        if res.submitted:
            risk.register_trade(sym, t0)
            # Refresh risk state after opening a position
            equity = executor.get_account_equity()
            gross = executor.gross_exposure_usd()
            open_positions = executor.open_positions_count()

    ml_bundle = None
    if bool(getattr(settings, "model_enabled", False)):
        ml_bundle = _load_ml_model(getattr(settings, "model_path", "state/models/latest.joblib"))

    candidates_for_ml: list[dict] = []
    for sym in _ordered_symbols(settings):
        if executor.has_position(sym):
            continue

        df_1m = _run_sync(buffer.snapshot_df(sym))
        df_15m = _run_sync(buffer.snapshot_resampled_df(sym, "15min"))
        sig = strategy.decide(symbol=sym, df_1m=df_1m, df_15m=df_15m)
        if sig is None:
            continue

        feat: dict = dict(sig.features) if sig.features else {}
        action = sig.action
        reason = sig.reason
        # Add a couple fields so inference can use them as features.
        feat["reason"] = reason
        feat["signal_ts"] = t0.isoformat()

        if sig.action == "BUY" and settings.news_fetch_enabled:
            bundle = fetch_news_for_symbol(
                symbol=sym,
                provider=settings.news_provider,
                alpaca_api_key_id=settings.apca_api_key_id,
                alpaca_secret_key=settings.apca_api_secret_key,
                alphavantage_api_key=settings.alphavantage_api_key,
                lookback_hours=float(settings.news_lookback_hours),
                limit=int(settings.news_limit),
            )
            feat["news"] = bundle
            if news_bundle_should_block(
                bundle,
                settings.news_gate_mode,
                int(settings.news_busy_min_articles),
            ):
                action = "HOLD"
                reason = "news_gate"

        # ML scoring (filter + rank): score candidates but don't submit yet.
        if action in ("BUY", "SHORT") and ml_bundle is not None:
            # is_buy feature: for now, model is trained on BUY labels; we still score SHORTs but treat them separately later
            md = _ml_predict_proba(model_bundle=ml_bundle, features={**feat, "is_buy": 1.0 if action == "BUY" else 0.0})
            feat["model"] = {"ok": md.ok, "provider": md.provider, "proba": md.proba, "error": md.error}
            if md.ok and md.proba is not None:
                feat["model_proba"] = float(md.proba)
                candidates_for_ml.append(
                    {
                        "symbol": sym,
                        "action": action,
                        "reason": reason,
                        "features": feat,
                        "df_1m": df_1m,
                        "df_15m": df_15m,
                        "model_proba": float(md.proba),
                    }
                )
                # skip immediate trade; we'll rank later
                ledger.record_signal(ts=t0, symbol=sig.symbol, action=action, reason=reason, features=feat)
                continue

        ledger.record_signal(ts=t0, symbol=sig.symbol, action=action, reason=reason, features=feat)

        _try_submit_entry(sym=sym, action=action, feat=feat, df_1m=df_1m)

    if ml_bundle is not None and candidates_for_ml:
        _apply_ml_filter_rank_and_trade(
            settings=settings,
            observe_only=observe_only,
            ledger=ledger,
            executor=executor,
            buffer=buffer,
            strategy=strategy,
            risk=risk,
            t0=t0,
            market_day=market_day,
            equity=equity,
            gross=gross,
            open_positions=open_positions,
            candidates=candidates_for_ml,
            submit_entry=_try_submit_entry,
        )

    _label_buy_signals_forward_returns(
        ledger=ledger,
        buffer=buffer,
        settings=settings,
        t0=t0,
        market_day=market_day,
    )
    return last_signal_scan_ts


def _apply_ml_filter_rank_and_trade(
    *,
    settings,
    observe_only: bool,
    ledger: Ledger,
    executor: OrderExecutor,
    buffer: BarBuffer,
    strategy: V1RulesSignalEngine,
    risk: RiskManager,
    t0: datetime,
    market_day: date,
    equity: float,
    gross: float,
    open_positions: int,
    candidates: list[dict],
    submit_entry,
) -> None:
    """
    Take pre-scored candidates, filter by MODEL_MIN_PROBA and submit top-N.
    """
    if not candidates:
        return
    min_p = float(getattr(settings, "model_min_proba", 0.55))
    top_n = max(1, int(getattr(settings, "top_n_per_tick", 2)))
    keep = [c for c in candidates if float(c.get("model_proba", 0.0)) >= min_p]
    keep.sort(key=lambda c: (-float(c.get("model_proba", 0.0)), str(c.get("symbol", ""))))
    keep = keep[:top_n]

    for c in keep:
        sym = c["symbol"]
        action = c["action"]
        feat = c["features"]
        df_1m = c.get("df_1m")
        if df_1m is None or getattr(df_1m, "empty", True):
            continue

        submit_entry(sym=sym, action=action, feat=feat, df_1m=df_1m)



def cli() -> None:
    p = argparse.ArgumentParser(prog="alpaca-day-bot")
    p.add_argument("--observe-only", action="store_true", help="Compute signals/log only; do not place orders.")
    p.add_argument("--backtest", action="store_true", help="Run historical backtest and exit.")
    p.add_argument(
        "--robustness",
        action="store_true",
        help="Run backtest + cost grid + walk-forward + small parameter sweep and write a markdown report.",
    )
    p.add_argument("--start", type=str, default=None, help="Backtest start date YYYY-MM-DD (UTC).")
    p.add_argument("--end", type=str, default=None, help="Backtest end date YYYY-MM-DD (UTC).")
    p.add_argument(
        "--day-session",
        action="store_true",
        help="Wait until trade window opens, trade until it closes, then exit (good for cron/launchd).",
    )
    p.add_argument(
        "--scheduled-tick",
        action="store_true",
        help="Single trading pass then exit (for GitHub Actions / cron). Uses REST bar warmup; skips if bot.lock is held.",
    )
    p.add_argument(
        "--build-universe",
        action="store_true",
        help="Build a daily liquid universe (writes state/universe_latest.json) and exit.",
    )
    args = p.parse_args()
    run(
        observe_only_override=bool(args.observe_only),
        day_session=bool(args.day_session),
        scheduled_tick=bool(args.scheduled_tick),
        backtest=bool(args.backtest),
        robustness=bool(args.robustness),
        build_universe=bool(args.build_universe),
        start=args.start,
        end=args.end,
    )


def run(
    *,
    observe_only_override: bool,
    day_session: bool,
    scheduled_tick: bool = False,
    backtest: bool,
    robustness: bool,
    build_universe: bool = False,
    start: str | None,
    end: str | None,
) -> None:
    # Some environments set HTTPS_PROXY/HTTP_PROXY; Alpaca endpoints often fail behind
    # corporate proxies. Force bypass for Alpaca REST + websocket hosts.
    alpaca_hosts = [
        "paper-api.alpaca.markets",
        "api.alpaca.markets",
        "data.alpaca.markets",
        "stream.data.alpaca.markets",
        "stream.alpaca.markets",
    ]

    def _merge_no_proxy(key: str) -> None:
        cur = os.environ.get(key, "").strip()
        parts = [p.strip() for p in cur.split(",") if p.strip()] if cur else []
        for h in alpaca_hosts:
            if h not in parts:
                parts.append(h)
        os.environ[key] = ",".join(parts)

    _merge_no_proxy("NO_PROXY")
    _merge_no_proxy("no_proxy")

    # Clear proxy vars for this process to avoid 403 tunnel failures.
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        if k in os.environ:
            os.environ[k] = ""

    settings = load_settings()

    # Universe override (optional): use a persisted liquid universe list instead of static SYMBOLS.
    if getattr(settings, "universe_enabled", False):
        try:
            up = Path(settings.state_dir) / "universe_latest.json"
            u = load_universe_symbols(str(up))
            if u:
                settings.symbols = u  # type: ignore[assignment]
        except Exception:
            pass

    # CLI flag wins.
    observe_only = bool(settings.observe_only) or observe_only_override

    state_dir = Path(settings.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = state_dir / "logs"
    setup_json_logging(str(logs_dir))

    db_path = str(state_dir / "ledger.sqlite3")
    ledger = Ledger(db_path)

    def on_trade_update(evt: TradeUpdateEvent) -> None:
        ledger.record_trade_update(evt)
        o = (evt.payload or {}).get("order") or {}
        side = o.get("side", "")
        status = o.get("status", "")
        msg = (
            f"trade_update {evt.event} {evt.symbol or '?'} "
            f"side={side} status={status} "
            f"qty={evt.filled_qty if evt.filled_qty is not None else '-'} "
            f"avg_px={evt.filled_avg_price if evt.filled_avg_price is not None else '-'} "
            f"order_id={evt.order_id or '-'}"
        )
        log.info(
            msg,
            extra={
                "extra_json": {
                    "event": evt.event,
                    "symbol": evt.symbol,
                    "order_id": evt.order_id,
                    "client_order_id": evt.client_order_id,
                    "filled_qty": evt.filled_qty,
                    "filled_avg_price": evt.filled_avg_price,
                    "payload": evt.payload,
                }
            },
        )

    tc = make_trading_client(settings)
    executor = OrderExecutor(tc)

    buffer = BarBuffer(maxlen=int(settings.bar_buffer_maxlen))
    md = MarketDataStreamer(settings, buffer)
    tu = TradingUpdatesStreamer(settings, on_update=on_trade_update)

    risk = RiskManager(
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_positions=settings.max_positions,
        max_trades_per_day=settings.max_trades_per_day,
        max_daily_loss_pct=settings.max_daily_loss_pct,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        per_symbol_cooldown_s=settings.per_symbol_cooldown_s,
        daily_profit_target_usd=settings.daily_profit_target_usd,
    )
    strategy = V1RulesSignalEngine(
        rsi_pullback_max=settings.rsi_pullback_max,
        volume_confirm_mult=settings.volume_confirm_mult,
        htf_rsi_min=settings.htf_rsi_min,
        atr_regime_max_mult=settings.atr_regime_max_mult,
        enable_shorts=settings.enable_shorts,
        htf_rsi_max_short=settings.htf_rsi_max_short,
        rsi_rebound_min_short=settings.rsi_rebound_min_short,
        aggressive_mode=settings.aggressive_mode,
    )

    if build_universe:
        outp = str(Path(settings.state_dir) / "universe_latest.json")
        res = build_liquid_universe(
            apca_api_key_id=settings.apca_api_key_id,
            apca_api_secret_key=settings.apca_api_secret_key,
            out_path=outp,
            max_symbols=int(settings.universe_max_symbols),
            lookback_days=int(settings.universe_lookback_days),
            min_price=float(settings.universe_min_price),
            min_avg_dollar_vol=float(settings.universe_min_avg_dollar_vol),
        )
        log.info(
            "universe_built",
            extra={
                "extra_json": {
                    "out_path": outp,
                    "selected": len(res.selected),
                    "total_assets_seen": res.total_assets_seen,
                    "bars_symbols": res.bars_symbols,
                    "rejected_counts": res.rejected_counts,
                }
            },
        )
        ledger.close()
        return

    if backtest or robustness:
        _run_backtest(settings, start=start, end=end, robustness=robustness)
        return

    if scheduled_tick:
        lock_fp = _try_acquire_single_instance_lock(state_dir)
        if lock_fp is None:
            log.warning(
                "scheduled_tick skipped: another process holds bot.lock "
                "(or overlapping GitHub Actions run).",
            )
            ledger.close()
            return
    else:
        _acquire_single_instance_lock(state_dir)

    log.info(
        "startup",
        extra={
            "extra_json": {
                "observe_only": observe_only,
                "scheduled_tick": scheduled_tick,
                "symbols": settings.symbols,
                "bar_timeframe": settings.bar_timeframe,
                "max_positions": settings.max_positions,
                "max_gross_exposure_pct": settings.max_gross_exposure_pct,
                "max_daily_loss_pct": settings.max_daily_loss_pct,
                "daily_profit_target_usd": settings.daily_profit_target_usd,
                "market_data_mode": settings.market_data_mode,
                "news_gate_mode": settings.news_gate_mode,
                "news_provider": settings.news_provider,
                "signal_accuracy_enabled": settings.signal_accuracy_enabled,
            }
        },
    )

    # Market data: REST polling avoids opening a second Alpaca websocket (trading stream uses one).
    md_mode = (settings.market_data_mode or "rest").strip().lower()
    if scheduled_tick:
        from alpaca_day_bot.data.rest_bars import RestBarPoller

        rp = RestBarPoller(settings, buffer)
        warmed = rp.warm_buffer(rounds=2, pause_s=1.0)
        log.info("scheduled_tick rest bars warmed events=%s", warmed)
    elif md_mode == "websocket":
        md_thread = threading.Thread(target=md.run_forever, name="market-data-ws", daemon=True)
        md_thread.start()
    else:
        from alpaca_day_bot.data.rest_bars import RestBarPoller

        rp = RestBarPoller(settings, buffer)
        threading.Thread(target=rp.run_forever, name="market-data-rest", daemon=True).start()

    tu_thread = threading.Thread(target=tu.run_forever, name="trade-updates-ws", daemon=True)
    tu_thread.start()

    if scheduled_tick:
        time.sleep(5.0)

    last_equity_snap = datetime.min.replace(tzinfo=timezone.utc)
    last_report_day = None
    last_signal_scan_ts = datetime.min.replace(tzinfo=timezone.utc)

    if scheduled_tick:
        t0 = now_utc()
        market_now = t0.astimezone(settings.tzinfo())
        market_day = market_now.date()
        mt = market_now.time()
        in_window = settings.trade_start <= mt <= settings.trade_end
        risk.rehydrate_from_ledger(ledger, market_day, settings.tzinfo())
        equity = executor.get_account_equity()
        gross = executor.gross_exposure_usd()
        ledger.record_equity_snapshot(t0, equity, gross)
        last_report_day = market_day
        try:
            if not in_window:
                log.info(
                    "scheduled_tick outside trade window",
                    extra={"extra_json": {"market_time": str(mt), "market_day": str(market_day)}},
                )
            else:
                last_signal_scan_ts = _run_in_window_trading_cycle(
                    settings=settings,
                    observe_only=observe_only,
                    ledger=ledger,
                    executor=executor,
                    buffer=buffer,
                    strategy=strategy,
                    risk=risk,
                    t0=t0,
                    market_day=market_day,
                    last_signal_scan_ts=last_signal_scan_ts,
                    force_signal_scan=True,
                )
        finally:
            try:
                if last_report_day is not None:
                    write_daily_report(
                        db_path, settings.reports_dir, last_report_day, market_tz=settings.market_tz
                    )
                    write_weekly_report(db_path, settings.reports_dir, last_report_day, days=7)
            except Exception:
                pass
            ledger.close()
        return

    try:
        while True:
            t0 = now_utc()

            # Periodic equity snapshot.
            if (t0 - last_equity_snap) >= timedelta(seconds=60):
                equity = executor.get_account_equity()
                gross = executor.gross_exposure_usd()
                ledger.record_equity_snapshot(t0, equity, gross)
                last_equity_snap = t0

            # Daily report trigger (market tz date).
            market_now = t0.astimezone(settings.tzinfo())
            market_day = market_now.date()
            if last_report_day is None:
                last_report_day = market_day
            elif market_day != last_report_day:
                # Write report for previous day.
                write_daily_report(
                    db_path, settings.reports_dir, last_report_day, market_tz=settings.market_tz
                )
                write_weekly_report(db_path, settings.reports_dir, last_report_day, days=7)
                last_report_day = market_day

            # Trade window check.
            mt = market_now.time()
            in_window = settings.trade_start <= mt <= settings.trade_end

            if day_session and not in_window:
                # If we're running a single-day session, block until the window opens,
                # and exit after the window closes (handled below).
                if mt < settings.trade_start:
                    sleep_s = min(60.0, max(1.0, (datetime.combine(market_day, settings.trade_start, tzinfo=settings.tzinfo()) - market_now).total_seconds()))
                    time.sleep(sleep_s)
                    continue
                if mt > settings.trade_end:
                    break

            if in_window:
                last_signal_scan_ts = _run_in_window_trading_cycle(
                    settings=settings,
                    observe_only=observe_only,
                    ledger=ledger,
                    executor=executor,
                    buffer=buffer,
                    strategy=strategy,
                    risk=risk,
                    t0=t0,
                    market_day=market_day,
                    last_signal_scan_ts=last_signal_scan_ts,
                    force_signal_scan=False,
                )

            # Loop pacing.
            dt = (now_utc() - t0).total_seconds()
            time.sleep(max(1.0, 5.0 - dt))
    except KeyboardInterrupt:
        log.info("shutdown")
    finally:
        # Write a report for current market day on shutdown too.
        try:
            if last_report_day is not None:
                write_daily_report(
                    db_path, settings.reports_dir, last_report_day, market_tz=settings.market_tz
                )
                write_weekly_report(db_path, settings.reports_dir, last_report_day, days=7)
        except Exception:
            pass
        ledger.close()


def _run_sync(coro):
    # We use tiny async helpers in the buffer; keep main loop sync for simplicity.
    try:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already in an event loop (rare in CLI), fallback to thread-safe run.
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result(timeout=2.0)
        return asyncio.run(coro)
    except Exception:
        return []


def _run_backtest(settings, *, start: str | None, end: str | None, robustness: bool) -> None:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if not start or not end:
        raise SystemExit("Backtest requires --start YYYY-MM-DD and --end YYYY-MM-DD")

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    data_client = StockHistoricalDataClient(settings.apca_api_key_id, settings.apca_api_secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=settings.symbols,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df  # multi-index: (symbol, timestamp)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        raise SystemExit("No historical bars returned for backtest window.")

    # Normalize expected columns and index.
    for sym in settings.symbols:
        try:
            sdf = df.xs(sym, level=0).copy()
        except Exception:
            continue
        sdf.index = pd.to_datetime(sdf.index, utc=True)
        # alpaca-py uses columns: open, high, low, close, volume, trade_count, vwap
        need = ["open", "high", "low", "close", "volume"]
        if not all(c in sdf.columns for c in need):
            continue
        bars_by_symbol[sym] = sdf[need].sort_index()

    res = run_backtest(
        bars_by_symbol=bars_by_symbol,
        starting_equity=settings.starting_equity_usd,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        stop_loss_atr_mult=settings.stop_loss_atr_mult,
        take_profit_r_mult=settings.take_profit_r_mult,
        slippage_bps=settings.slippage_bps,
        commission_bps=settings.commission_bps,
        spread_proxy_k=settings.spread_proxy_k,
        spread_proxy_min=settings.spread_proxy_min,
        open_delay_minutes=settings.open_delay_minutes,
        market_context_filter=settings.market_context_filter,
        spy_5m_rsi_min=settings.spy_5m_rsi_min,
    )

    if not robustness:
        print("Backtest result")
        _print_backtest_result(res)
        return

    report_path = _write_robustness_report(
        settings=settings,
        start_dt=start_dt,
        end_dt=end_dt,
        bars_by_symbol=bars_by_symbol,
        base=res,
    )
    print(f"Wrote robustness report to: {report_path}")


def _print_backtest_result(res) -> None:
    print(f"Start equity: ${res.start_equity:,.2f}")
    print(f"End equity:   ${res.end_equity:,.2f}")
    print(f"Total return: {res.total_return*100:.2f}%")
    if res.sharpe_daily is not None:
        print(f"Sharpe (daily): {res.sharpe_daily:.2f}")
    if res.max_drawdown is not None:
        print(f"Max drawdown: {res.max_drawdown*100:.2f}%")
    if res.win_rate is not None:
        print(f"Win rate: {res.win_rate*100:.2f}%")
    if res.profit_factor is not None:
        print(f"Profit factor: {res.profit_factor:.2f}")
    if res.expectancy is not None:
        print(f"Expectancy ($/trade): {res.expectancy:.2f}")
    if getattr(res, "expectancy_r", None) is not None:
        print(f"Expectancy (R/trade): {res.expectancy_r:.3f}")
    if getattr(res, "turnover", None) is not None:
        print(f"Turnover (x start eq): {res.turnover:.2f}x")
    if getattr(res, "avg_hold_minutes", None) is not None:
        print(f"Avg hold (min): {res.avg_hold_minutes:.1f}")
    print(f"Trades: {len(res.trades)}")


def _write_robustness_report(*, settings, start_dt: datetime, end_dt: datetime, bars_by_symbol, base) -> str:
    from pathlib import Path

    def _phase(msg: str) -> None:
        print(f"[robustness] {msg}", flush=True)

    light = bool(getattr(settings, "robustness_light", False))

    Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)
    out = Path(settings.reports_dir) / f"backtest_robustness_{start_dt.date()}_{end_dt.date()}.md"

    # 1) Cost grid (full grid is slow on GitHub: ~60 backtests × huge 1m union index)
    slip_list = [0.5, 2.0, 6.0] if light else [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    comm_list = [0.0, 1.0] if light else [0.0, 0.5, 1.0]
    _phase(
        f"cost grid ({len(slip_list)}×{len(comm_list)} runs, ROBUSTNESS_LIGHT={light})…"
    )
    grid_rows = run_cost_sensitivity_grid(
        bars_by_symbol=bars_by_symbol,
        starting_equity=settings.starting_equity_usd,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        stop_loss_atr_mult=settings.stop_loss_atr_mult,
        take_profit_r_mult=settings.take_profit_r_mult,
        slippage_bps_list=slip_list,
        commission_bps_list=comm_list,
        strategy_params=None,
    )
    # Sort by sharpe then return
    grid_rows_sorted = sorted(
        grid_rows,
        key=lambda r: (-(r.sharpe_daily or -999), -r.total_return),
    )

    # 2) Walk-forward
    _phase("walk-forward folds…")
    folds = run_walk_forward(
        bars_by_symbol=bars_by_symbol,
        starting_equity=settings.starting_equity_usd,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        stop_loss_atr_mult=settings.stop_loss_atr_mult,
        take_profit_r_mult=settings.take_profit_r_mult,
        slippage_bps=settings.slippage_bps,
        commission_bps=settings.commission_bps,
        start_dt=start_dt,
        end_dt=end_dt,
        test_window_days=14 if light else 7,
        step_days=14 if light else 7,
        min_trades_per_fold=15 if light else 30,
        strategy_params=None,
    )

    # 3) Small param sweep (multiple-testing visibility)
    if light:
        sweep_grid = [
            {"rsi_pullback_max": r, "volume_confirm_mult": v, "atr_regime_max_mult": a}
            for r in (35.0, 40.0)
            for v in (1.0, 1.2)
            for a in (2.0, 2.5)
        ]
    else:
        sweep_grid = [
            {"rsi_pullback_max": r, "volume_confirm_mult": v, "atr_regime_max_mult": a}
            for r in (30.0, 35.0, 40.0)
            for v in (1.0, 1.2, 1.4)
            for a in (1.5, 2.0, 2.5)
        ]
    _phase(f"parameter sweep ({len(sweep_grid)} runs)…")
    sweep_rows = run_param_sweep(
        bars_by_symbol=bars_by_symbol,
        starting_equity=settings.starting_equity_usd,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        stop_loss_atr_mult=settings.stop_loss_atr_mult,
        take_profit_r_mult=settings.take_profit_r_mult,
        slippage_bps=settings.slippage_bps,
        commission_bps=settings.commission_bps,
        grid=sweep_grid,
    )
    sweep_sorted = sorted(sweep_rows, key=lambda r: (-(r.sharpe_daily or -999), -r.total_return))

    def fmt_pct(x):
        return "n/a" if x is None else f"{x*100:.2f}%"

    def fmt(x):
        return "n/a" if x is None else f"{x:.2f}"

    # Benchmarks
    bh = buy_and_hold_returns(bars_by_symbol)
    # Time of day
    tod = time_of_day_breakdown(base.trades)
    sym_recs, focus_syms = symbol_daytrade_recommendations(base.trades, min_trades=5)
    # Regimes
    adx_reg, vol_reg = compute_spy_regimes(bars_by_symbol)

    def regime_stats(kind: str):
        groups: dict[str, list] = {}
        for tr in base.trades:
            lbl = label_trade_regime(tr.entry_ts, adx_reg, vol_reg)
            key = getattr(lbl, kind)
            if key is None:
                continue
            groups.setdefault(str(key), []).append(tr)
        out = []
        for k, trs in sorted(groups.items()):
            b = None
            # reuse bucket stats helper via temporary function
            pnls = [t.pnl for t in trs]
            out.append(
                {
                    "label": k,
                    "trades": len(trs),
                    "pnl": float(sum(pnls)),
                    "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else None,
                    "exp": float(sum(pnls) / len(pnls)) if pnls else None,
                    "exp_r": (sum([t.pnl_r for t in trs if t.pnl_r is not None]) / max(1, len([t for t in trs if t.pnl_r is not None])))
                    if any(t.pnl_r is not None for t in trs)
                    else None,
                }
            )
        return out

    adx_stats = regime_stats("adx_regime")
    vol_stats = regime_stats("vol_regime")

    # Summary stats for sweep: best vs median
    sharpes = [r.sharpe_daily for r in sweep_rows if r.sharpe_daily is not None]
    returns = [r.total_return for r in sweep_rows]
    med_sharpe = float(pd.Series(sharpes).median()) if sharpes else None
    med_ret = float(pd.Series(returns).median()) if returns else None
    best = sweep_sorted[0] if sweep_sorted else None

    lines = []
    lines.append(f"## Backtest robustness report ({start_dt.date()} → {end_dt.date()})")
    if light:
        lines.append(
            "_**ROBUSTNESS_LIGHT**: smaller cost grid, wider walk-forward windows, "
            "and fewer sweep points (faster CI; less exhaustive than a full local run)._"
        )
    lines.append("")
    lines.append("### Base run (configured costs)")
    lines.append(f"- **Total return**: {fmt_pct(base.total_return)}")
    lines.append(f"- **Sharpe (daily)**: {fmt(base.sharpe_daily)}")
    lines.append(f"- **Max drawdown**: {fmt_pct(base.max_drawdown)}")
    lines.append(f"- **Win rate**: {fmt_pct(base.win_rate)}")
    lines.append(f"- **Profit factor**: {fmt(base.profit_factor)}")
    lines.append(f"- **Expectancy ($/trade)**: {fmt(base.expectancy)}")
    lines.append(f"- **Expectancy (R/trade)**: {fmt(getattr(base, 'expectancy_r', None))}")
    lines.append(f"- **Turnover (x start eq)**: {fmt(getattr(base, 'turnover', None))}")
    lines.append(f"- **Avg hold (min)**: {fmt(getattr(base, 'avg_hold_minutes', None))}")
    lines.append(f"- **Trades**: {len(base.trades)}")
    lines.append("")

    # Optional: model eval from ledger (if present in state/)
    try:
        from pathlib import Path as _Path

        from alpaca_day_bot.ml.eval import quick_walk_forward_eval

        dbp = _Path(settings.state_dir) / "ledger.sqlite3"
        if dbp.exists():
            _phase("ml walk-forward (ledger labels)…")
            folds_ml = quick_walk_forward_eval(db_path=str(dbp), min_horizon_minutes=15.0)
            lines.append("### ML quick walk-forward (from labeled signals in ledger)")
            for i, f in enumerate(folds_ml, 1):
                acc = "n/a" if f.test_acc is None else f"{f.test_acc*100:.1f}%"
                lines.append(f"- **Fold {i}**: train_n={f.train_n}, test_n={f.test_n}, acc={acc}")
            lines.append("")
    except Exception:
        pass

    lines.append("### Day-trade focus (ranked from backtest, not live advice)")
    lines.append(
        "- **How to read**: symbols sorted by historical **expectancy $/trade** in this window; "
        "`focus_symbols` has names with at least **5** simulated trades (less noise)."
    )
    if not sym_recs:
        lines.append("- No closed trades in base run — widen date range or relax strategy.")
    else:
        lines.append(f"- **focus_symbols** (≥5 trades): `{', '.join(focus_syms) if focus_syms else 'none'}`")
        for r in sym_recs[:12]:
            lines.append(
                f"- **{r.symbol}**: n={r.trades}, total_pnl ${r.total_pnl_usd:,.2f}, "
                f"win% {fmt_pct(r.win_rate)}, exp ${fmt(r.expectancy_usd)}, expR {fmt(r.expectancy_r)}"
            )
        if len(sym_recs) > 12:
            lines.append(f"- _…{len(sym_recs) - 12} more in JSON_")
    lines.append("")

    lines.append("### Time-of-day performance (by entry time, NY)")
    for b in tod:
        lines.append(
            f"- **{b.label}**: trades {b.trades}, pnl ${b.pnl:,.2f}, win_rate {fmt_pct(b.win_rate)}, exp ${fmt(b.expectancy)}, expR {fmt(b.expectancy_r)}"
        )
    lines.append("")

    lines.append("### Regime splits (SPY)")
    lines.append("- ADX regime on SPY 15m: trending if ADX(14)>25")
    if not adx_stats:
        lines.append("  - n/a")
    else:
        for r in adx_stats:
            lines.append(
                f"  - **{r['label']}**: trades {r['trades']}, pnl ${r['pnl']:,.2f}, win_rate {fmt_pct(r['win_rate'])}, exp ${fmt(r['exp'])}, expR {fmt(r['exp_r'])}"
            )
    lines.append("- Vol regime on SPY 1m: high/low by median rolling vol")
    if not vol_stats:
        lines.append("  - n/a")
    else:
        for r in vol_stats:
            lines.append(
                f"  - **{r['label']}**: trades {r['trades']}, pnl ${r['pnl']:,.2f}, win_rate {fmt_pct(r['win_rate'])}, exp ${fmt(r['exp'])}, expR {fmt(r['exp_r'])}"
            )
    lines.append("")

    lines.append("### Benchmarks (buy-and-hold over window)")
    if not bh:
        lines.append("- n/a")
    else:
        # show SPY first, then rest
        if "SPY" in bh:
            lines.append(f"- **SPY**: {fmt_pct(bh['SPY'])}")
        for sym in sorted(k for k in bh.keys() if k != "SPY"):
            lines.append(f"- **{sym}**: {fmt_pct(bh[sym])}")
    lines.append("")

    lines.append("### Cost sensitivity (top 10 by Sharpe)")
    lines.append("- If performance collapses with small bps increases, the edge is likely execution-cost fragile.")
    lines.append("")
    for r in grid_rows_sorted[:10]:
        lines.append(
            f"- **slip {r.slippage_bps:.1f} bps / comm {r.commission_bps:.1f} bps**: "
            f"ret {fmt_pct(r.total_return)}, sharpe {fmt(r.sharpe_daily)}, dd {fmt_pct(r.max_drawdown)}, trades {r.trades}"
        )
    lines.append("")

    lines.append("### Walk-forward (7d windows)")
    if not folds:
        lines.append("- n/a (insufficient history window)")
    else:
        for f in folds:
            inc = "included" if f.included else f"excluded({f.note})"
            lines.append(
                f"- **fold {f.fold}** {f.start.date()}→{f.end.date()} [{inc}]: ret {fmt_pct(f.total_return)}, sharpe {fmt(f.sharpe_daily)}, dd {fmt_pct(f.max_drawdown)}, trades {f.trades}"
            )
    lines.append("")

    lines.append("### Parameter sweep (multiple-testing warning)")
    lines.append(f"- Grid size: **{len(sweep_rows)}** combinations.")
    if best is not None:
        lines.append(
            f"- **Best**: ret {fmt_pct(best.total_return)}, sharpe {fmt(best.sharpe_daily)}, dd {fmt_pct(best.max_drawdown)}, params `{best.params}`"
        )
    lines.append(f"- **Median**: ret {fmt_pct(med_ret)}, sharpe {fmt(med_sharpe)}")
    lines.append("- Interpretation: if best ≫ median, you may be overfitting the backtest (false discovery risk).")
    lines.append("")

    _phase("writing markdown + recommendations JSON…")
    out.write_text("\n".join(lines), encoding="utf-8")

    best_tod_payload: dict | None = None
    nonempty_tod = [b for b in tod if b.trades >= 5]
    if nonempty_tod:
        bt = max(nonempty_tod, key=lambda b: (b.expectancy if b.expectancy is not None else -1e18))
        best_tod_payload = {
            "label": bt.label,
            "trades": bt.trades,
            "expectancy_usd": bt.expectancy,
            "expectancy_r": bt.expectancy_r,
        }

    rec_payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "backtest_window": {"start": str(start_dt.date()), "end": str(end_dt.date())},
        "disclaimer": "Historical simulation only; not investment advice. Paper trading.",
        "configured_slippage_bps": settings.slippage_bps,
        "configured_commission_bps": settings.commission_bps,
        "open_delay_minutes": settings.open_delay_minutes,
        "market_context_filter": settings.market_context_filter,
        "base_expectancy_usd": base.expectancy,
        "base_expectancy_r": getattr(base, "expectancy_r", None),
        "min_trades_for_focus": 5,
        "focus_symbols": focus_syms,
        "symbols_ranked": [asdict(r) for r in sym_recs],
        "best_time_of_day_ny": best_tod_payload,
        "robustness_report_path": str(out),
        "robustness_light": light,
    }
    reports_dir = Path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"day_trade_recommendations_{end_dt.date()}.json"
    latest_path = reports_dir / "day_trade_recommendations_latest.json"
    json_blob = json.dumps(rec_payload, indent=2, default=str)
    json_path.write_text(json_blob, encoding="utf-8")
    latest_path.write_text(json_blob, encoding="utf-8")
    print(f"Wrote day-trade recommendations JSON to: {json_path}")
    return str(out)

