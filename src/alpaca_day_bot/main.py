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
from alpaca_day_bot.data.tvta import fetch_tvta_indicators_for_stock
from alpaca_day_bot.data.stream import BarBuffer, MarketDataStreamer
from alpaca_day_bot.logging_utils import setup_json_logging
from alpaca_day_bot.ml.infer import load_model as _load_ml_model, predict_proba as _ml_predict_proba
from alpaca_day_bot.ml.regime_thresholds import learn_regime_min_proba_map, write_regime_thresholds_json
from alpaca_day_bot.options_sim import close_open_virtual_options
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
from alpaca_day_bot.universe import (
    build_master_universe_assets,
    build_liquid_universe,
    filter_universe_symbols_by_max_price,
    intraday_prefilter_symbols,
    load_universe_symbols,
)


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
            # Defensive: ensure universe remains compatible with whole-share sizing under $ cap.
            try:
                mp = float(getattr(settings, "universe_max_price", 0.0) or 0.0)
                if mp > 0:
                    u = filter_universe_symbols_by_max_price(
                        symbols=u,
                        max_price=mp,
                        apca_api_key_id=settings.apca_api_key_id,
                        apca_api_secret_key=settings.apca_api_secret_key,
                    )
            except Exception:
                pass
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


def _label_signals_forward_returns(
    *,
    ledger: Ledger,
    buffer: BarBuffer,
    settings,
    t0: datetime,
    market_day: date,
) -> None:
    if not settings.signal_accuracy_enabled:
        return
    if not bool(getattr(settings, "label_forward_returns_enabled", True)):
        return
    pending = ledger.list_unlabeled_signal_rows(
        market_day=market_day,
        tz=settings.tzinfo(),
        now_utc=t0,
        min_age_minutes=float(settings.signal_accuracy_min_age_minutes),
        actions=("BUY", "SHORT"),
    )
    for signal_id, ts_s, sym, action, feat_json in pending:
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
        act_u = (action or "").strip().upper()
        if act_u == "SHORT":
            # Positive label = price fell after signal (good for shorts)
            ret = (entry_f - now_px) / entry_f
        else:
            ret = (now_px - entry_f) / entry_f
        ledger.record_forward_return_label(
            signal_id=signal_id,
            evaluated_ts=t0,
            price_at_label=now_px,
            entry_close=entry_f,
            return_pct=ret,
            horizon_minutes=horizon_m,
        )


def _label_signals_triple_barrier(
    *,
    ledger: Ledger,
    buffer: BarBuffer,
    settings,
    t0: datetime,
    market_day: date,
) -> None:
    """
    Triple-barrier style labeling aligned to exits:
    - For BUY: TP if high >= tp_price before low <= sl_price; SL if low hits first; else TIMEOUT.
    - For SHORT: TP if low <= tp_price before high >= sl_price; SL if high hits first; else TIMEOUT.
    Uses bar path between signal_ts and now from the in-memory buffer snapshot.
    """
    if not settings.signal_accuracy_enabled:
        return
    pending = ledger.list_unlabeled_signal_rows_for_triple_barrier(
        market_day=market_day,
        tz=settings.tzinfo(),
        now_utc=t0,
        min_age_minutes=float(settings.signal_accuracy_min_age_minutes),
        actions=("BUY", "SHORT"),
    )
    for signal_id, ts_s, sym, action, feat_json in pending:
        try:
            feat = json.loads(feat_json) if feat_json else {}
        except Exception:
            feat = {}
        if not isinstance(feat, dict):
            continue

        entry = feat.get("close")
        tp = feat.get("tp_price")
        sl = feat.get("sl_price")
        if entry is None or tp is None or sl is None:
            continue
        try:
            entry_f = float(entry)
            tp_f = float(tp)
            sl_f = float(sl)
        except Exception:
            continue
        if entry_f <= 0 or tp_f <= 0 or sl_f <= 0:
            continue

        try:
            ts_sig = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
        except Exception:
            continue
        if ts_sig.tzinfo is None:
            ts_sig = ts_sig.replace(tzinfo=timezone.utc)
        horizon_m = (t0 - ts_sig).total_seconds() / 60.0

        df = _run_sync(buffer.snapshot_df(sym))
        if df is None or getattr(df, "empty", True):
            continue
        try:
            import pandas as pd

            dfx = df.copy()
            dfx.index = pd.to_datetime(dfx.index, utc=True, errors="coerce")
            dfx = dfx.sort_index()
            dfx_full = dfx
            # Some data sources/buffers can yield indices that don't compare cleanly to ts_sig
            # (or bars may not extend back far enough). If the filter removes everything,
            # fall back to the full snapshot so we still produce labels.
            try:
                dfx = dfx[dfx.index >= ts_sig]
                if dfx is None or getattr(dfx, "empty", True):
                    dfx = dfx_full
            except Exception:
                dfx = dfx_full
        except Exception:
            continue
        if dfx is None or getattr(dfx, "empty", True):
            continue
        if not all(c in dfx.columns for c in ("high", "low", "close")):
            continue

        act = (action or "").strip().upper()
        outcome = "timeout"
        px_at_eval = float(dfx["close"].iloc[-1])

        try:
            for _idx, row in dfx.iterrows():
                hi = float(row["high"])
                lo = float(row["low"])
                if act == "SHORT":
                    # TP is below, SL above
                    if lo <= tp_f:
                        outcome = "tp"
                        break
                    if hi >= sl_f:
                        outcome = "sl"
                        break
                else:
                    # BUY
                    if hi >= tp_f:
                        outcome = "tp"
                        break
                    if lo <= sl_f:
                        outcome = "sl"
                        break
        except Exception:
            outcome = "timeout"

        # realized return proxy at evaluation time (mark-to-market), but label is barrier-first.
        if act == "SHORT":
            realized_ret = (entry_f - px_at_eval) / entry_f
        else:
            realized_ret = (px_at_eval - entry_f) / entry_f

        ledger.record_triple_barrier_label(
            signal_id=int(signal_id),
            evaluated_ts=t0,
            entry_close=entry_f,
            tp_price=tp_f,
            sl_price=sl_f,
            outcome=outcome,
            realized_return_pct=float(realized_ret),
            horizon_minutes=float(horizon_m),
        )


def _run_in_window_trading_cycle(
    *,
    settings,
    observe_only: bool,
    scheduled_tick: bool,
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

    # Load ML bundles early so entries/exits can use them (separate BUY vs SHORT models).
    ml_bundle_buy = None
    ml_bundle_short = None
    if bool(getattr(settings, "model_enabled", False)):
        mp_buy = getattr(settings, "model_path_long", "") or ""
        mp_short = getattr(settings, "model_path_short", "") or ""
        ml_bundle_buy = _load_ml_model(mp_buy) if mp_buy else None
        ml_bundle_short = _load_ml_model(mp_short) if mp_short else None
        if ml_bundle_buy is None and ml_bundle_short is None:
            # Back-compat: try single-model path.
            mp = getattr(settings, "model_path", "state/models/latest.joblib")
            ml = _load_ml_model(mp)
            ml_bundle_buy = ml
            ml_bundle_short = ml
            if ml is None:
                log.warning("ml_model_not_loaded paths=%s,%s,%s", mp_buy, mp_short, mp)

    # Per-cycle caches (avoid repeated external calls in a single scheduled tick).
    taapi_cache: dict[str, dict] = {}
    taapi_disabled_for_tick = False
    tvta_cache: dict[str, dict] = {}
    news_cache: dict[str, dict] = {}

    def _maybe_add_taapi_features(*, sym: str, feat: dict) -> None:
        nonlocal taapi_disabled_for_tick
        try:
            if not (
                (settings.indicator_provider or "").strip().lower() == "taapi"
                and (settings.taapi_secret or "").strip()
            ):
                return
            # Avoid hammering TAAPI (429s are common on free tiers).
            ti_dict = taapi_cache.get(sym)
            if ti_dict is None and not taapi_disabled_for_tick:
                ti = fetch_taapi_indicators_for_stock(secret=settings.taapi_secret or "", symbol=sym)
                ti_dict = {
                    "rsi_1m": ti.rsi_1m,
                    "rsi_15m": ti.rsi_15m,
                    "macd_1m": ti.macd_1m,
                    "macd_signal_1m": ti.macd_signal_1m,
                }
                taapi_cache[sym] = ti_dict
                # If we got no usable TAAPI values, disable further TAAPI calls this tick.
                if all(v is None for v in ti_dict.values()):
                    taapi_disabled_for_tick = True
            feat["taapi"] = ti_dict or {"rsi_1m": None, "rsi_15m": None, "macd_1m": None, "macd_signal_1m": None}
            feat["taapi_ok"] = (
                0
                if all(feat["taapi"].get(k) is None for k in ("rsi_1m", "rsi_15m", "macd_1m", "macd_signal_1m"))
                else 1
            )
        except Exception:
            return

    def _maybe_add_tvta_features(*, sym: str, feat: dict) -> None:
        try:
            if (settings.indicator_provider or "").strip().lower() != "tvta":
                return
            base = (getattr(settings, "tvta_api_base_url", None) or "").strip()
            if not base:
                try:
                    log.warning("tvta_base_url_missing; set TVTA_API_BASE_URL to enable custom indicator API")
                except Exception:
                    pass
                return
            prefix = str(getattr(settings, "tvta_symbol_prefix", "NYSE") or "NYSE")
            ti_dict = tvta_cache.get(sym)
            if ti_dict is None:
                ti = fetch_tvta_indicators_for_stock(base_url=base, symbol=sym, symbol_prefix=prefix)
                ti_dict = {
                    "rsi_1m": ti.rsi_1m,
                    "rsi_15m": ti.rsi_15m,
                    "macd_1m": ti.macd_1m,
                    "macd_signal_1m": ti.macd_signal_1m,
                    "raw_1m": ti.raw_1m,
                    "raw_15m": ti.raw_15m,
                }
                tvta_cache[sym] = ti_dict
            feat["tvta"] = ti_dict or {}
            feat["tvta_ok"] = (
                0
                if all(
                    (feat.get("tvta") or {}).get(k) is None
                    for k in ("rsi_1m", "rsi_15m", "macd_1m", "macd_signal_1m")
                )
                else 1
            )
            # Provide a taapi-shaped view so downstream ML/features can stay stable.
            if "taapi" not in feat:
                feat["taapi"] = {
                    "rsi_1m": (feat.get("tvta") or {}).get("rsi_1m"),
                    "rsi_15m": (feat.get("tvta") or {}).get("rsi_15m"),
                    "macd_1m": (feat.get("tvta") or {}).get("macd_1m"),
                    "macd_signal_1m": (feat.get("tvta") or {}).get("macd_signal_1m"),
                }
                feat["taapi_ok"] = feat.get("tvta_ok", 0)
        except Exception:
            return

    def _maybe_add_news_features(*, sym: str, feat: dict) -> dict[str, float | int]:
        """
        Fetch once per symbol per cycle (small universe), store bundle + derived fields.
        Returns derived features dict (may be zeros).
        """
        try:
            if not bool(getattr(settings, "news_fetch_enabled", False)):
                return {"news_sent_wmean": 0.0, "news_event_risk": 0}
            bundle = news_cache.get(sym)
            if bundle is None:
                bundle = fetch_news_for_symbol(
                    symbol=sym,
                    provider=settings.news_provider,
                    alpaca_api_key_id=settings.apca_api_key_id,
                    alpaca_secret_key=settings.apca_api_secret_key,
                    alphavantage_api_key=settings.alphavantage_api_key,
                    lookback_hours=float(settings.news_lookback_hours),
                    limit=int(settings.news_limit),
                )
                news_cache[sym] = bundle if isinstance(bundle, dict) else {"ok": False, "count": 0, "articles": []}
            if isinstance(bundle, dict):
                feat["news"] = bundle
            nf = _derive_news_features(bundle if isinstance(bundle, dict) else None)
            feat.update(nf)
            return nf
        except Exception:
            return {"news_sent_wmean": 0.0, "news_event_risk": 0}

    def _derive_news_features(bundle: dict | None) -> dict[str, float | int]:
        """
        Compute lightweight, model-friendly features from the (possibly large) news bundle.
        Stored at top-level fields for easy querying/training.
        """
        try:
            if not isinstance(bundle, dict):
                return {"news_sent_wmean": 0.0, "news_event_risk": 0}
            arts = bundle.get("articles")
            if not isinstance(arts, list) or not arts:
                return {"news_sent_wmean": 0.0, "news_event_risk": 0}

            # Event risk keyword scan (cheap, robust).
            txt = " ".join(
                [
                    f"{(a.get('headline') or '')} {(a.get('summary') or '')}"
                    for a in arts
                    if isinstance(a, dict)
                ]
            ).lower()
            risk_words = (
                "earnings",
                "offering",
                "secondary",
                "sec ",
                "investigation",
                "lawsuit",
                "guidance",
                "halt",
                "bankruptcy",
                "merger",
                "acquisition",
            )
            event_risk = 1 if (txt and any(w in txt for w in risk_words)) else 0

            # Recency-weighted mean sentiment (half-life ~24h).
            from datetime import datetime, timezone

            now = datetime.now(tz=timezone.utc)
            half_life_h = 24.0
            w_sum = 0.0
            sw_sum = 0.0
            for a in arts:
                if not isinstance(a, dict):
                    continue
                sc = a.get("sentiment_score")
                if sc is None:
                    continue
                try:
                    s = float(sc)
                except Exception:
                    continue
                created_at = a.get("created_at")
                age_h = None
                if isinstance(created_at, str) and created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age_h = max(0.0, (now - dt).total_seconds() / 3600.0)
                    except Exception:
                        age_h = None
                # If timestamp missing/invalid, treat as stale.
                if age_h is None:
                    age_h = 7.0 * 24.0
                w = 0.5 ** (age_h / half_life_h) if half_life_h > 0 else 1.0
                w_sum += w
                sw_sum += w * s
            sent_wmean = (sw_sum / w_sum) if w_sum > 1e-12 else 0.0
            return {"news_sent_wmean": float(sent_wmean), "news_event_risk": int(event_risk)}
        except Exception:
            return {"news_sent_wmean": 0.0, "news_event_risk": 0}

    def _regime_label(*, df_1m, df_15m) -> str | None:
        """
        Lightweight regime label: {trend|chop}_{high_vol|low_vol} based on ADX(14) on 15m and ATR ratio on 1m.
        """
        try:
            import pandas_ta as ta

            if df_1m is None or getattr(df_1m, "empty", True) or df_15m is None or getattr(df_15m, "empty", True):
                return None

            atr = float(df_1m.get("atr").iloc[-1]) if "atr" in df_1m.columns else None
            atr_avg = float(df_1m.get("atr_avg").iloc[-1]) if "atr_avg" in df_1m.columns else None
            atr_ratio = (atr / atr_avg) if (atr is not None and atr_avg is not None and atr_avg > 1e-9) else None

            h = df_15m.copy()
            if "ADX_14" not in h.columns:
                adx = ta.adx(h["high"], h["low"], h["close"], length=14)
                if adx is not None:
                    h = h.join(adx)
            adx_v = None
            if "ADX_14" in h.columns:
                try:
                    adx_v = float(h["ADX_14"].iloc[-1])
                except Exception:
                    adx_v = None

            trend = "trend" if (adx_v is not None and adx_v >= 22.0) else "chop"
            vol = "high_vol" if (atr_ratio is not None and atr_ratio >= 1.15) else "low_vol"
            return f"{trend}_{vol}"
        except Exception:
            return None

    # Smarter exits: time-based and optional model-based exits (in addition to brackets).
    # We base age on the latest submitted intent timestamp for that symbol from the ledger.
    if float(getattr(settings, "max_hold_minutes", 0.0)) > 0 or bool(getattr(settings, "dynamic_hold_enabled", True)):
        try:
            tz = settings.tzinfo()
            intents = ledger.last_submitted_entry_intents_for_trading_date(market_day, tz)
            for sym, row in list(intents.items()):
                try:
                    if not executor.has_position(sym):
                        continue
                    entry_ts = row.get("ts")
                    if entry_ts is None:
                        continue
                    age_m = (t0 - entry_ts).total_seconds() / 60.0
                    target = float(getattr(settings, "max_hold_minutes", 0.0) or 0.0)
                    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
                    if bool(getattr(settings, "dynamic_hold_enabled", True)):
                        try:
                            target = float(extra.get("target_hold_minutes") or 0.0) or target
                        except Exception:
                            pass
                    if target <= 0:
                        continue
                    # Profit-protect exit: if move >= +1R, close and lock it in.
                    try:
                        entry_px = float(extra.get("entry_price") or 0.0) if isinstance(extra, dict) else 0.0
                        stop_dist = float(extra.get("stop_distance") or 0.0) if isinstance(extra, dict) else 0.0
                        act0 = str(extra.get("action") or "").upper() if isinstance(extra, dict) else ""
                        if entry_px > 0 and stop_dist > 0 and act0 in ("BUY", "SHORT"):
                            dfp = _run_sync(buffer.snapshot_df(sym))
                            if dfp is not None and not getattr(dfp, "empty", True):
                                last_px = float(dfp["close"].iloc[-1])
                                hit = (
                                    (act0 == "BUY" and (last_px - entry_px) >= stop_dist)
                                    or (act0 == "SHORT" and (entry_px - last_px) >= stop_dist)
                                )
                                if hit:
                                    res = executor.close_position_market(sym)
                                    ledger.record_order_intent(
                                        ts=t0,
                                        symbol=sym,
                                        side="close",
                                        notional_usd=0.0,
                                        stop_price=0.0,
                                        take_profit_price=0.0,
                                        client_order_id=None,
                                        alpaca_order_id=None,
                                        submitted=res.submitted,
                                        reason=f"profit_exit_1r:{res.reason}",
                                        extra={"action": "EXIT_PROFIT_1R"},
                                    )
                                    # Backfill late fills for closes (scheduled tick ends quickly).
                                    if (
                                        scheduled_tick
                                        and bool(getattr(settings, "fill_confirm_enabled", True))
                                        and res.alpaca_order_id
                                    ):
                                        try:
                                            evt = executor.poll_order_fill_event(
                                                order_id=str(res.alpaca_order_id),
                                                timeout_s=float(getattr(settings, "fill_confirm_timeout_s", 60.0)),
                                                poll_s=float(getattr(settings, "fill_confirm_poll_s", 3.0)),
                                            )
                                            if evt is not None:
                                                ledger.record_trade_update(evt)
                                        except Exception:
                                            pass
                                    continue
                    except Exception:
                        pass
                    if age_m < target:
                        continue
                    res = executor.close_position_market(sym)
                    ledger.record_order_intent(
                        ts=t0,
                        symbol=sym,
                        side="close",
                        notional_usd=0.0,
                        stop_price=0.0,
                        take_profit_price=0.0,
                        client_order_id=None,
                        alpaca_order_id=None,
                        submitted=res.submitted,
                        reason=f"time_exit:{res.reason}",
                        extra={"action": "EXIT_TIME", "age_minutes": age_m, "target_hold_minutes": target},
                    )
                    if (
                        scheduled_tick
                        and bool(getattr(settings, "fill_confirm_enabled", True))
                        and res.alpaca_order_id
                    ):
                        try:
                            evt = executor.poll_order_fill_event(
                                order_id=str(res.alpaca_order_id),
                                timeout_s=float(getattr(settings, "fill_confirm_timeout_s", 60.0)),
                                poll_s=float(getattr(settings, "fill_confirm_poll_s", 3.0)),
                            )
                            if evt is not None:
                                ledger.record_trade_update(evt)
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            pass

    # Model-based early exit: re-score open positions and close when confidence degrades.
    if bool(getattr(settings, "model_exit_enabled", False)) and ml_bundle is not None:
        try:
            tz = settings.tzinfo()
            intents = ledger.last_submitted_entry_intents_for_trading_date(market_day, tz)
            min_hold = float(getattr(settings, "model_exit_min_hold_minutes", 5.0) or 0.0)
            thr = float(getattr(settings, "model_exit_min_proba", 0.45) or 0.0)
            for sym, row in list(intents.items()):
                try:
                    if not executor.has_position(sym):
                        continue
                    entry_ts = row.get("ts")
                    if entry_ts is None:
                        continue
                    age_m = (t0 - entry_ts).total_seconds() / 60.0
                    if age_m < min_hold:
                        continue

                    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
                    entry_action = extra.get("action")
                    act = (
                        (str(entry_action).upper() if entry_action else None)
                        or ("BUY" if str(row.get("side")).lower() == "buy" else "SHORT")
                    )

                    df_1m = _run_sync(buffer.snapshot_df(sym))
                    df_15m = _run_sync(buffer.snapshot_resampled_df(sym, "15min"))
                    sig_now = strategy_cons.decide(symbol=sym, df_1m=df_1m, df_15m=df_15m)
                    if sig_now is None or not isinstance(sig_now.features, dict):
                        continue
                    feat_now = dict(sig_now.features)
                    feat_now["reason"] = sig_now.reason
                    feat_now["signal_ts"] = t0.isoformat()
                    feat_now["action"] = act
                    rlbl = _regime_label(df_1m=df_1m, df_15m=df_15m)
                    if rlbl:
                        feat_now["regime"] = rlbl

                    md = _ml_predict_proba(ml_bundle, feat_now)
                    if (not md.ok) or (md.proba is None):
                        continue
                    if float(md.proba) >= thr:
                        continue

                    res = executor.close_position_market(sym)
                    ledger.record_order_intent(
                        ts=t0,
                        symbol=sym,
                        side="close",
                        notional_usd=0.0,
                        stop_price=0.0,
                        take_profit_price=0.0,
                        client_order_id=None,
                        alpaca_order_id=None,
                        submitted=res.submitted,
                        reason=f"model_exit:{res.reason}",
                        extra={
                            "action": "EXIT_MODEL",
                            "age_minutes": age_m,
                            "model_provider": md.provider,
                            "model_proba": md.proba,
                            "model_exit_min_proba": thr,
                        },
                    )
                    if (
                        scheduled_tick
                        and bool(getattr(settings, "fill_confirm_enabled", True))
                        and res.alpaca_order_id
                    ):
                        try:
                            evt = executor.poll_order_fill_event(
                                order_id=str(res.alpaca_order_id),
                                timeout_s=float(getattr(settings, "fill_confirm_timeout_s", 60.0)),
                                poll_s=float(getattr(settings, "fill_confirm_poll_s", 3.0)),
                            )
                            if evt is not None:
                                ledger.record_trade_update(evt)
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            pass

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

        # Execution-quality filter: require sufficient relative volume.
        try:
            mvr = float(getattr(settings, "min_volume_ratio_trade", 0.0) or 0.0)
            vr = float(feat.get("volume_ratio") or 0.0)
            if mvr > 0 and vr > 0 and vr < mvr:
                ledger.record_order_intent(
                    ts=t0,
                    symbol=sym,
                    side=("buy" if action == "BUY" else "sell"),
                    notional_usd=0.0,
                    stop_price=0.0,
                    take_profit_price=0.0,
                    client_order_id=None,
                    alpaca_order_id=None,
                    submitted=False,
                    reason="min_volume_ratio",
                    extra={"action": action, "volume_ratio": vr, "min_volume_ratio_trade": mvr},
                )
                return
        except Exception:
            pass

        # Short guardrail: cap concurrent shorts.
        if action == "SHORT":
            try:
                mxs = int(getattr(settings, "max_short_positions", 0) or 0)
                if mxs > 0 and int(executor.short_positions_count()) >= mxs:
                    ledger.record_order_intent(
                        ts=t0,
                        symbol=sym,
                        side="sell",
                        notional_usd=0.0,
                        stop_price=0.0,
                        take_profit_price=0.0,
                        client_order_id=None,
                        alpaca_order_id=None,
                        submitted=False,
                        reason="max_short_positions",
                        extra={"action": action, "max_short_positions": mxs},
                    )
                    return
            except Exception:
                pass

        # Optional confirmation: require the SAME signal condition to persist for N bars.
        try:
            cb = int(getattr(settings, "confirm_bars", 1) or 1)
        except Exception:
            cb = 1
        if cb > 1:
            try:
                # Use the same timeframe as the strategy.
                use_15m = str(getattr(settings, "signal_timeframe", "15m") or "15m").strip().lower().startswith("15")
                df_base = _run_sync(buffer.snapshot_resampled_df(sym, "15min")) if use_15m else df_1m
                if df_base is None or getattr(df_base, "empty", True):
                    return
                if len(df_base) < (cb + 5):
                    return

                use_aggr = bool(int(feat.get("aggressive_used") or 0))
                strat = strategy_aggr if use_aggr else strategy_cons

                # Require last N bars all produce the same action (BUY/SHORT).
                n = len(df_base)
                for off in range(cb - 1, -1, -1):
                    dfx = df_base.iloc[: n - off]
                    if use_15m:
                        s = strat.decide(symbol=sym, df_1m=df_1m, df_15m=dfx)
                    else:
                        s = strat.decide(symbol=sym, df_1m=dfx, df_15m=_run_sync(buffer.snapshot_resampled_df(sym, "15min")))
                    if s is None or s.action != action:
                        return
            except Exception:
                # If confirmation calc fails, fail open (don't block trades).
                pass

        # Correlation guard: avoid stacking highly correlated names (accuracy + drawdown stability).
        max_corr = float(getattr(settings, "max_corr_with_open_positions", 0.0) or 0.0)
        if max_corr > 0:
            try:
                lookback = max(20, int(getattr(settings, "corr_lookback_bars_1m", 60) or 60))
                held = [h for h in executor.open_position_symbols() if h and h != sym]
                if held:
                    import numpy as np
                    import pandas as pd

                    s = df_1m["close"].astype(float).pct_change().dropna().tail(lookback)
                    for hs in held[:10]:
                        hdf = _run_sync(buffer.snapshot_df(hs))
                        if hdf is None or getattr(hdf, "empty", True) or "close" not in hdf.columns:
                            continue
                        hret = hdf["close"].astype(float).pct_change().dropna().tail(lookback)
                        # align by index
                        joined = pd.concat([s, hret], axis=1, join="inner").dropna()
                        if len(joined) < 15:
                            continue
                        c = float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])
                        if np.isfinite(c) and abs(c) >= max_corr:
                            return
            except Exception:
                pass

        # Optional TAAPI confirmation layer (stocks RSI/MACD).
        if (
            (settings.indicator_provider or "").strip().lower() == "taapi"
            and bool(settings.taapi_confirm_on_trade)
            and (settings.taapi_secret or "").strip()
        ):
            _maybe_add_taapi_features(sym=sym, feat=feat)
            trsi1 = feat["taapi"].get("rsi_1m")
            trsi15 = feat["taapi"].get("rsi_15m")
            tmacd = feat["taapi"].get("macd_1m")
            tmacds = feat["taapi"].get("macd_signal_1m")
            if action == "BUY":
                if trsi1 is None or trsi15 is None or tmacd is None or tmacds is None:
                    if not bool(getattr(settings, "taapi_fail_open", True)):
                        return
                else:
                    ok = (
                        float(trsi1) <= float(settings.rsi_pullback_max)
                        and float(trsi15) >= float(settings.htf_rsi_min)
                        and float(tmacd) >= float(tmacds)
                    )
                    if not ok:
                        return
            else:
                if trsi1 is None or trsi15 is None or tmacd is None or tmacds is None:
                    if not bool(getattr(settings, "taapi_fail_open", True)):
                        return
                else:
                    ok = (
                        float(trsi15) <= float(settings.htf_rsi_max_short)
                        and float(trsi1) >= float(settings.rsi_rebound_min_short)
                        and float(tmacd) <= float(tmacds)
                    )
                    if not ok:
                        return

        last_close = float(df_1m["close"].iloc[-1])
        # Triple-barrier/exit sizing: use ATR-based distance to reduce noise.
        atr = None
        try:
            atr = float(feat.get("atr")) if feat.get("atr") is not None else None
        except Exception:
            atr = None
        if atr is None or atr <= 0:
            try:
                import pandas_ta as ta

                s_atr = ta.atr(df_1m["high"], df_1m["low"], df_1m["close"], length=14)
                if s_atr is not None and not s_atr.dropna().empty:
                    atr = float(s_atr.dropna().iloc[-1])
            except Exception:
                atr = None
        stop_dist = max(0.01, float(atr or 0.0) * 2.0)
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

        asset_class = (getattr(settings, "asset_class", "equity") or "equity").strip().lower()
        if asset_class == "crypto":
            # Free weekend mode: crypto spot entries via market-by-notional.
            # Keep it simple (no bracket orders, no shorts). Exits are handled on later ticks.
            if action != "BUY":
                return
            max_notional = float(getattr(settings, "max_notional_per_trade_usd", 0.0) or 0.0)
            if max_notional <= 0:
                max_notional = 50.0
            notional = float(rd.notional_usd or max_notional)
            if max_notional > 0:
                notional = min(notional, max_notional)
            if notional <= 0:
                return

            if observe_only:
                risk.register_trade(sym, t0)
                return

            res = executor.submit_entry_buy_notional_market(symbol=sym, notional_usd=notional)
            ledger.record_order_intent(
                ts=t0,
                symbol=sym,
                side="buy",
                notional_usd=float(notional),
                stop_price=0.0,
                take_profit_price=0.0,
                client_order_id=res.client_order_id,
                alpaca_order_id=res.alpaca_order_id,
                submitted=res.submitted,
                reason=res.reason,
                extra={"action": action, "asset_class": "crypto", "entry_price": float(last_close)},
            )
            if res.submitted:
                risk.register_trade(sym, t0)
            return

        # Volatility gating (regime handling): if current ATR is unusually high vs "monthly" ATR,
        # halve the position size.
        qty_f = float(rd.qty)
        try:
            atr = float(feat.get("atr") or 0.0)
            atr_m = float(feat.get("atr_monthly") or feat.get("atr_avg") or 0.0)
            if atr_m > 1e-12 and atr > 1.5 * atr_m:
                if bool(getattr(settings, "vol_spike_skip_enabled", False)):
                    ledger.record_order_intent(
                        ts=t0,
                        symbol=sym,
                        side=("buy" if action == "BUY" else "sell"),
                        notional_usd=0.0,
                        stop_price=stop_price,
                        take_profit_price=tp_price,
                        client_order_id=None,
                        alpaca_order_id=None,
                        submitted=False,
                        reason="vol_spike_skip",
                        extra={"action": action, "atr": atr, "atr_monthly": atr_m},
                    )
                    return
                qty_f = qty_f * 0.5
                feat["vol_gated_size"] = 1
        except Exception:
            pass

        qty_int = int(qty_f)
        if qty_int <= 0:
            # Most commonly: MAX_NOTIONAL_PER_TRADE_USD cap + whole-share requirement.
            ledger.record_order_intent(
                ts=t0,
                symbol=sym,
                side=("buy" if action == "BUY" else "sell"),
                notional_usd=float((qty_int * last_close) if qty_int > 0 else (rd.notional_usd or 0.0)),
                stop_price=stop_price,
                take_profit_price=tp_price,
                client_order_id=None,
                alpaca_order_id=None,
                submitted=False,
                reason="cap_qty_zero",
                extra={
                    "qty": qty_int,
                    "action": action,
                    "price": float(last_close),
                    "max_notional_per_trade_usd": float(getattr(settings, "max_notional_per_trade_usd", 0.0) or 0.0),
                },
            )
            return

        if observe_only:
            risk.register_trade(sym, t0)
            return

        entry_type = (getattr(settings, "entry_order_type", "market") or "market").strip().lower()
        offset_bps = float(getattr(settings, "limit_entry_offset_bps", 0.0) or 0.0)
        limit_price = None
        if entry_type == "limit":
            k = (offset_bps / 10_000.0) if offset_bps > 0 else 0.0
            if action == "BUY":
                limit_price = last_close * (1.0 - k)
            else:
                limit_price = last_close * (1.0 + k)
            limit_price = max(0.01, float(limit_price))

        synthetic = bool(getattr(settings, "synthetic_exits_enabled", False))
        exit_side = None
        if synthetic:
            # Entry first, then OCO exits.
            if action == "BUY":
                if entry_type == "limit":
                    res = executor.submit_entry_buy_limit(
                        symbol=sym, qty=qty_int, limit_price=float(limit_price or last_close)
                    )
                else:
                    res = executor.submit_entry_buy_market(symbol=sym, qty=qty_int)
                side = "buy"
                exit_side = "sell"
            else:
                if entry_type == "limit":
                    res = executor.submit_entry_short_limit(
                        symbol=sym, qty=qty_int, limit_price=float(limit_price or last_close)
                    )
                else:
                    res = executor.submit_entry_short_market(symbol=sym, qty=qty_int)
                side = "sell"
                exit_side = "buy"
        else:
            # Native bracket exits.
            if action == "BUY":
                if entry_type == "limit":
                    res = executor.submit_bracket_buy_limit(
                        symbol=sym,
                        qty=qty_int,
                        limit_price=float(limit_price or last_close),
                        stop_price=stop_price,
                        take_profit_price=tp_price,
                    )
                else:
                    res = executor.submit_bracket_buy(
                        symbol=sym, qty=qty_int, stop_price=stop_price, take_profit_price=tp_price
                    )
                side = "buy"
            else:
                if entry_type == "limit":
                    res = executor.submit_bracket_short_limit(
                        symbol=sym,
                        qty=qty_int,
                        limit_price=float(limit_price or last_close),
                        stop_price=stop_price,
                        take_profit_price=tp_price,
                    )
                else:
                    res = executor.submit_bracket_short(
                        symbol=sym, qty=qty_int, stop_price=stop_price, take_profit_price=tp_price
                    )
                side = "sell"

        # Dynamic hold target (based on technical volatility + model confidence)
        target_hold = None
        try:
            if bool(getattr(settings, "dynamic_hold_enabled", True)):
                atr = float(feat.get("atr") or 0.0)
                atr_avg = float(feat.get("atr_avg") or 0.0)
                atr_ratio = (atr / atr_avg) if atr_avg > 1e-9 else 1.0
                model_p = None
                if "model_proba" in feat:
                    model_p = float(feat.get("model_proba"))
                else:
                    m = feat.get("model")
                    if isinstance(m, dict) and m.get("proba") is not None:
                        model_p = float(m.get("proba"))
                base = float(getattr(settings, "base_hold_minutes", 45.0))
                w_atr = float(getattr(settings, "hold_atr_ratio_weight", -15.0))
                w_mp = float(getattr(settings, "hold_model_proba_weight", 20.0))
                hold = base + w_atr * (atr_ratio - 1.0) + (w_mp * ((model_p or 0.5) - 0.5))
                hold = max(float(getattr(settings, "min_hold_minutes", 10.0)), hold)
                hold = min(float(getattr(settings, "max_hold_minutes_dynamic", 90.0)), hold)
                target_hold = float(hold)
        except Exception:
            target_hold = None

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
            extra={
                "qty": qty_int,
                "action": action,
                "target_hold_minutes": target_hold,
                "entry_order_type": entry_type,
                "limit_price": float(limit_price) if limit_price is not None else None,
                "entry_price": float(last_close),
                "stop_distance": float(stop_dist),
            },
        )
        if res.submitted:
            risk.register_trade(sym, t0)

            # Optional: simulate an options "call/put" trade (virtual; never sent to Alpaca).
            try:
                if bool(getattr(settings, "sim_options_enabled", False)):
                    side_opt = "call" if action == "BUY" else "put"
                    max_cap = float(getattr(settings, "max_notional_per_trade_usd", 0.0) or 0.0)
                    opt_cap = float(getattr(settings, "sim_options_notional_usd", 0.0) or 0.0)
                    notional = opt_cap if opt_cap > 0 else max_cap
                    if notional and notional > 0 and last_close > 0:
                        ledger.open_virtual_option_trade(
                            ts_open=t0,
                            symbol=sym,
                            side=side_opt,
                            notional_usd=float(notional),
                            leverage=float(getattr(settings, "sim_options_leverage", 6.0) or 6.0),
                            underlying_entry=float(last_close),
                            meta={
                                "source": "scheduled_tick",
                                "linked_action": action,
                                "strike": float(last_close),  # simple ATM strike proxy
                                "theta_decay_per_day": float(
                                    getattr(settings, "sim_options_theta_decay_per_day", 0.001) or 0.001
                                ),
                            },
                        )
            except Exception:
                pass

            # Refresh risk state after opening a position
            equity = executor.get_account_equity()
            gross = executor.gross_exposure_usd()
            open_positions = executor.open_positions_count()

            if synthetic and exit_side is not None:
                xres = executor.submit_exit_oco(
                    symbol=sym,
                    qty=qty_int,
                    side=exit_side,
                    take_profit_price=tp_price,
                    stop_price=stop_price,
                )
                ledger.record_order_intent(
                    ts=t0,
                    symbol=sym,
                    side="oco_exit",
                    notional_usd=0.0,
                    stop_price=stop_price,
                    take_profit_price=tp_price,
                    client_order_id=xres.client_order_id,
                    alpaca_order_id=xres.alpaca_order_id,
                    submitted=xres.submitted,
                    reason=f"synthetic_exit:{xres.reason}",
                    extra={"action": "EXIT_OCO", "qty": qty_int, "exit_side": exit_side},
                )

            # Scheduled ticks exit quickly; poll REST to confirm fills so we don't miss websocket events.
            if (
                scheduled_tick
                and bool(getattr(settings, "fill_confirm_enabled", True))
                and res.alpaca_order_id
            ):
                try:
                    evt = executor.poll_order_fill_event(
                        order_id=str(res.alpaca_order_id),
                        timeout_s=float(getattr(settings, "fill_confirm_timeout_s", 60.0)),
                        poll_s=float(getattr(settings, "fill_confirm_poll_s", 3.0)),
                    )
                    if evt is not None:
                        ledger.record_trade_update(evt)
                except Exception:
                    pass

    ml_bundle = None
    if bool(getattr(settings, "model_enabled", False)):
        ml_bundle = _load_ml_model(getattr(settings, "model_path", "state/models/latest.joblib"))

    candidates_for_ml: list[dict] = []
    for sym in _ordered_symbols(settings):
        if executor.has_position(sym):
            continue

        df_1m = _run_sync(buffer.snapshot_df(sym))
        df_15m = _run_sync(buffer.snapshot_resampled_df(sym, "15min"))
        # Use aggressive mode only in "good" regimes (trend + low vol).
        rlbl0 = _regime_label(df_1m=df_1m, df_15m=df_15m)
        use_aggr = (asset_class == "crypto") or (
            bool(getattr(settings, "aggressive_mode", False)) and (rlbl0 == "trend_low_vol")
        )
        sig = (strategy_aggr if use_aggr else strategy_cons).decide(symbol=sym, df_1m=df_1m, df_15m=df_15m)
        if sig is None:
            continue

        feat: dict = dict(sig.features) if sig.features else {}
        feat["aggressive_used"] = 1 if use_aggr else 0
        action = sig.action
        reason = sig.reason
        # Pre-filter: avoid short attempts on symbols Alpaca flags non-shortable.
        try:
            if action == "SHORT" and not executor.is_shortable(sym):
                ledger.record_order_intent(
                    ts=t0,
                    symbol=sym,
                    side="sell",
                    notional_usd=0.0,
                    stop_price=0.0,
                    take_profit_price=0.0,
                    client_order_id=None,
                    alpaca_order_id=None,
                    submitted=False,
                    reason="not_shortable_prefilter",
                    extra={"action": "SHORT"},
                )
                continue
        except Exception:
            pass
        # Add a couple fields so inference can use them as features.
        feat["reason"] = reason
        feat["signal_ts"] = t0.isoformat()
        feat["action"] = action
        # Precompute barriers at signal-time for triple-barrier labels.
        try:
            last_close = float(feat.get("close") or df_1m["close"].iloc[-1])
            last_range = float((df_1m["high"].iloc[-1] - df_1m["low"].iloc[-1]))
            # For ML labels: set barriers from ATR (2.0x) rather than bar-range.
            atr0 = None
            try:
                atr0 = float(feat.get("atr")) if feat.get("atr") is not None else None
            except Exception:
                atr0 = None
            stop_dist = max(0.01, float(atr0 or 0.0) * 2.0)
            if action == "BUY":
                sl_price = last_close - stop_dist
                tp_price = last_close + stop_dist
            else:
                sl_price = last_close + stop_dist
                tp_price = last_close - stop_dist
            feat["tp_price"] = float(tp_price)
            feat["sl_price"] = float(sl_price)
        except Exception:
            pass
        try:
            rlbl = _regime_label(df_1m=df_1m, df_15m=df_15m)
            if rlbl:
                feat["regime"] = rlbl
        except Exception:
            pass

        # Record news+indicator feature visibility even if this becomes HOLD.
        try:
            # Helps reporting: what indicator backend we used for enrichment.
            feat["indicator_provider"] = (settings.indicator_provider or "").strip().lower() or None
        except Exception:
            pass
        # Persist signal timestamp so ml.infer time-of-day features aren't NaN.
        try:
            feat["ts"] = t0.isoformat()
        except Exception:
            pass
        _maybe_add_taapi_features(sym=sym, feat=feat)
        _maybe_add_tvta_features(sym=sym, feat=feat)
        nf = _maybe_add_news_features(sym=sym, feat=feat)

        # Apply news gating only when we'd otherwise trade.
        if action in ("BUY", "SHORT"):
            try:
                if bool(getattr(settings, "news_block_on_event_risk", True)) and int(nf.get("news_event_risk", 0) or 0) == 1:
                    action = "HOLD"
                    reason = "news_event_risk"
            except Exception:
                pass
            try:
                bundle_now = feat.get("news") if isinstance(feat.get("news"), dict) else {"ok": False, "count": 0, "articles": []}
                if news_bundle_should_block(
                    bundle_now,
                    settings.news_gate_mode,
                    int(settings.news_busy_min_articles),
                ):
                    action = "HOLD"
                    reason = "news_gate"
            except Exception:
                pass

        # ML scoring (filter + rank): score candidates but don't submit yet.
        ml_for_action = (ml_bundle_buy if action == "BUY" else ml_bundle_short)
        if action in ("BUY", "SHORT") and ml_for_action is not None:
            md = _ml_predict_proba(
                model_bundle=ml_for_action,
                features={**feat, "action": action, "signal_action": action, "is_buy": 1.0 if action == "BUY" else 0.0},
            )
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

    if (ml_bundle_buy is not None or ml_bundle_short is not None) and candidates_for_ml:
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

    _label_signals_forward_returns(
        ledger=ledger,
        buffer=buffer,
        settings=settings,
        t0=t0,
        market_day=market_day,
    )
    _label_signals_triple_barrier(
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

    # Do not trust ML gating until we have enough executed outcomes.
    # We still record probabilities, but we avoid filtering/ranking by them too early.
    try:
        min_exec = int(getattr(settings, "min_executed_round_trips_for_model", 200) or 200)
    except Exception:
        min_exec = 200
    exec_n = None
    try:
        from alpaca_day_bot.reporting.executed_ml import executed_ml_summary

        es = executed_ml_summary(str(Path(getattr(settings, "state_dir", "state")) / "ledger.sqlite3"))
        exec_n = (es.n if es is not None else 0)
    except Exception:
        exec_n = 0
    if int(exec_n or 0) < int(min_exec):
        return
    min_p = float(getattr(settings, "model_min_proba", 0.55))
    min_p_long = float(getattr(settings, "model_min_proba_long", min_p) or min_p)
    min_p_short = float(getattr(settings, "model_min_proba_short", max(min_p, 0.65)) or max(min_p, 0.65))
    # Prefer dynamic regime thresholds file if present; otherwise allow env JSON override.
    reg_map = {}
    try:
        p = Path(getattr(settings, "state_dir", "state")) / "regime_thresholds_latest.json"
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            reg_map = data.get("model_min_proba_by_regime") or {}
        else:
            s = (getattr(settings, "model_min_proba_by_regime_json", "") or "").strip()
            if s:
                reg_map = json.loads(s)
    except Exception:
        reg_map = {}
    top_n = max(1, int(getattr(settings, "top_n_per_tick", 2)))
    keep = []
    for c in candidates:
        mp = float(c.get("model_proba", 0.0))
        act = str(c.get("action") or "").strip().upper()
        feat = c.get("features") or {}
        rlbl = None
        try:
            rlbl = (feat.get("regime") if isinstance(feat, dict) else None) or None
        except Exception:
            rlbl = None
        mp_req = (min_p_short if act == "SHORT" else min_p_long)
        try:
            if rlbl and isinstance(reg_map, dict) and rlbl in reg_map:
                mp_req = float(reg_map[rlbl])
        except Exception:
            mp_req = (min_p_short if act == "SHORT" else min_p_long)
        if mp >= mp_req:
            keep.append(c)
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
    p.add_argument(
        "--build-master-universe",
        action="store_true",
        help="Build a broad cached universe from Alpaca assets (writes state/universe_master.json) and exit.",
    )
    p.add_argument(
        "--write-report",
        action="store_true",
        help="Write daily+weekly report for the current market day and exit (no trading).",
    )
    p.add_argument(
        "--close-virtual-options",
        action="store_true",
        help="Close any open simulated call/put trades (virtual options) and exit.",
    )
    args = p.parse_args()
    run(
        observe_only_override=bool(args.observe_only),
        day_session=bool(args.day_session),
        scheduled_tick=bool(args.scheduled_tick),
        backtest=bool(args.backtest),
        robustness=bool(args.robustness),
        build_universe=bool(args.build_universe),
        build_master_universe=bool(args.build_master_universe),
        write_report=bool(args.write_report),
        close_virtual_options=bool(args.close_virtual_options),
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
    build_master_universe: bool = False,
    write_report: bool = False,
    close_virtual_options: bool = False,
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

    # Intraday prefilter: reduce the scan set BEFORE we warm bars / call TVTA/news/ML.
    # This keeps the master/liquid universe large but makes per-tick execution practical.
    try:
        asset_class = (getattr(settings, "asset_class", "equity") or "equity").strip().lower()
        if scheduled_tick and asset_class == "equity" and bool(getattr(settings, "prefilter_enabled", False)):
            pf_n = int(getattr(settings, "prefilter_max_symbols", 400) or 400)
            pf_m = str(getattr(settings, "prefilter_method", "movers_actives") or "movers_actives")
            pre = intraday_prefilter_symbols(
                apca_api_key_id=settings.apca_api_key_id,
                apca_api_secret_key=settings.apca_api_secret_key,
                method=pf_m,
                max_symbols=pf_n,
            )
            if pre:
                base_set = {str(s).strip().upper() for s in (settings.symbols or []) if str(s).strip()}
                pre_u = [s for s in pre if s in base_set] or pre
                settings.symbols = pre_u  # type: ignore[assignment]
                log.info(
                    "prefilter_applied",
                    extra={"extra_json": {"method": pf_m, "requested": pf_n, "selected": len(pre_u)}},
                )
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

    if close_virtual_options:
        try:
            res = close_open_virtual_options(
                ledger=ledger,
                apca_api_key_id=settings.apca_api_key_id,
                apca_api_secret_key=settings.apca_api_secret_key,
                ts_close=now_utc(),
            )
            log.info("virtual_options_closed", extra={"extra_json": res})
        except Exception as e:
            log.warning("virtual_options_close_failed err=%s", e, exc_info=True)
        finally:
            ledger.close()
        return

    if write_report:
        market_now = now_utc().astimezone(settings.tzinfo())
        market_day = market_now.date()
        try:
            try:
                mp_map, rows = learn_regime_min_proba_map(db_path=db_path)
                write_regime_thresholds_json(
                    out_path=str(Path(settings.state_dir) / "regime_thresholds_latest.json"),
                    mp_map=mp_map,
                    rows=rows,
                )
            except Exception:
                pass
            write_daily_report(db_path, settings.reports_dir, market_day, market_tz=settings.market_tz)
            write_weekly_report(db_path, settings.reports_dir, market_day, days=7)
        finally:
            ledger.close()
        return

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
        max_notional_per_trade_usd=settings.max_notional_per_trade_usd,
        per_symbol_cooldown_s=settings.per_symbol_cooldown_s,
        daily_profit_target_usd=settings.daily_profit_target_usd,
    )
    # Two strategy instances: conservative always; aggressive used only in good regimes.
    #
    # NOTE: Several helper functions reference these by name; bind them at module scope.
    global strategy_cons, strategy_aggr

    asset_class = (getattr(settings, "asset_class", "equity") or "equity").strip().lower()
    crypto_preset = asset_class == "crypto"
    # Crypto needed a looser preset; otherwise most ticks were HOLD (rsi_no_pullback) and no orders were placed.
    rsi_pb_max = float(settings.rsi_pullback_max)
    vol_mult = float(settings.volume_confirm_mult)
    htf_rsi_min = float(settings.htf_rsi_min)
    atr_regime_max = float(settings.atr_regime_max_mult)
    if crypto_preset:
        rsi_pb_max = max(rsi_pb_max, 55.0)
        vol_mult = min(vol_mult, 1.0)
        htf_rsi_min = min(htf_rsi_min, 40.0)
        atr_regime_max = max(atr_regime_max, 3.5)

    strategy_cons = V1RulesSignalEngine(
        rsi_pullback_max=rsi_pb_max,
        volume_confirm_mult=vol_mult,
        htf_rsi_min=htf_rsi_min,
        atr_regime_max_mult=atr_regime_max,
        enable_shorts=settings.enable_shorts,
        htf_rsi_max_short=settings.htf_rsi_max_short,
        rsi_rebound_min_short=settings.rsi_rebound_min_short,
        aggressive_mode=False,
        signal_timeframe=getattr(settings, "signal_timeframe", "15m"),
        macd_confirm_mode=getattr(settings, "macd_confirm_mode", "aligned_good_regime_else_cross"),
        crypto_momentum_setup=bool(crypto_preset),
    )
    strategy_aggr = V1RulesSignalEngine(
        rsi_pullback_max=rsi_pb_max,
        volume_confirm_mult=vol_mult,
        htf_rsi_min=htf_rsi_min,
        atr_regime_max_mult=atr_regime_max,
        enable_shorts=settings.enable_shorts,
        htf_rsi_max_short=settings.htf_rsi_max_short,
        rsi_rebound_min_short=settings.rsi_rebound_min_short,
        aggressive_mode=True,
        signal_timeframe=getattr(settings, "signal_timeframe", "15m"),
        macd_confirm_mode=getattr(settings, "macd_confirm_mode", "aligned_good_regime_else_cross"),
        crypto_momentum_setup=bool(crypto_preset),
    )

    if build_master_universe:
        outp = str(Path(settings.state_dir) / "universe_master.json")
        payload = build_master_universe_assets(
            apca_api_key_id=settings.apca_api_key_id,
            apca_api_secret_key=settings.apca_api_secret_key,
            out_path=outp,
            max_symbols=int(getattr(settings, "universe_master_max_symbols", 5000)),
            require_shortable=bool(getattr(settings, "universe_master_require_shortable", False)),
        )
        log.info(
            "master_universe_built",
            extra={"extra_json": {"out_path": outp, "symbols": len(payload.get("symbols") or [])}},
        )
        ledger.close()
        return

    if build_universe:
        outp = str(Path(settings.state_dir) / "universe_latest.json")
        master_path = Path(settings.state_dir) / "universe_master.json"
        candidates = load_universe_symbols(str(master_path)) if master_path.is_file() else []
        res = build_liquid_universe(
            apca_api_key_id=settings.apca_api_key_id,
            apca_api_secret_key=settings.apca_api_secret_key,
            out_path=outp,
            candidate_symbols=candidates if candidates else None,
            max_symbols=int(settings.universe_max_symbols),
            lookback_days=int(settings.universe_lookback_days),
            min_price=float(settings.universe_min_price),
            max_price=float(getattr(settings, "universe_max_price", 0.0) or 0.0),
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
        asset_class = (getattr(settings, "asset_class", "equity") or "equity").strip().lower()
        if asset_class == "crypto":
            from alpaca_day_bot.data.crypto_rest_bars import CryptoRestBarPoller

            rp = CryptoRestBarPoller(settings, buffer)
        else:
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
        asset_class = (getattr(settings, "asset_class", "equity") or "equity").strip().lower()
        # Crypto trades 24/7; don't apply US-equities time window gates.
        in_window = True if asset_class == "crypto" else (settings.trade_start <= mt <= settings.trade_end)
        risk.rehydrate_from_ledger(ledger, market_day, settings.tzinfo())
        equity = executor.get_account_equity()
        gross = executor.gross_exposure_usd()
        ledger.record_equity_snapshot(t0, equity, gross)
        last_report_day = market_day
        try:
            # Flatten before close (avoid overnight holds) — equities only.
            if asset_class != "crypto":
                try:
                    fb = int(getattr(settings, "flatten_before_close_minutes", 5) or 5)
                except Exception:
                    fb = 5
                if fb > 0:
                    try:
                        from datetime import datetime as _dt, timedelta as _td

                        close_cut = (
                            _dt.combine(market_day, settings.trade_end, tzinfo=settings.tzinfo()) - _td(minutes=fb)
                        ).time()
                        if mt >= close_cut:
                            for sym in executor.open_position_symbols():
                                try:
                                    res = executor.close_position_market(sym)
                                    ledger.record_order_intent(
                                        ts=t0,
                                        symbol=sym,
                                        side="close",
                                        notional_usd=0.0,
                                        stop_price=0.0,
                                        take_profit_price=0.0,
                                        client_order_id=None,
                                        alpaca_order_id=res.alpaca_order_id,
                                        submitted=res.submitted,
                                        reason=f"flatten_before_close:{res.reason}",
                                        extra={"action": "EXIT_FLATTEN"},
                                    )
                                    if (
                                        scheduled_tick
                                        and bool(getattr(settings, "fill_confirm_enabled", True))
                                        and res.alpaca_order_id
                                    ):
                                        try:
                                            evt = executor.poll_order_fill_event(
                                                order_id=str(res.alpaca_order_id),
                                                timeout_s=float(getattr(settings, "fill_confirm_timeout_s", 60.0)),
                                                poll_s=float(getattr(settings, "fill_confirm_poll_s", 3.0)),
                                            )
                                            if evt is not None:
                                                ledger.record_trade_update(evt)
                                        except Exception:
                                            pass
                                except Exception:
                                    continue
                            # Do not open new trades after flatten window.
                            in_window = False
                    except Exception:
                        pass

            if not in_window:
                log.info(
                    "scheduled_tick outside trade window",
                    extra={"extra_json": {"market_time": str(mt), "market_day": str(market_day)}},
                )
            else:
                last_signal_scan_ts = _run_in_window_trading_cycle(
                    settings=settings,
                    observe_only=observe_only,
                    scheduled_tick=True,
                    ledger=ledger,
                    executor=executor,
                    buffer=buffer,
                    strategy=strategy_cons,
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
                    scheduled_tick=False,
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

