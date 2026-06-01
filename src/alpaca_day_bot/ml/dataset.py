from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetResult:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # ts, symbol, action, horizon_minutes, return_pct


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _parse_iso_dt(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _news_features(bundle: dict[str, Any] | None, *, now_ts: datetime) -> dict[str, float]:
    if not isinstance(bundle, dict):
        return {
            "news_ok": 0.0,
            "news_count": 0.0,
            "news_recency_min": float("nan"),
            "news_sent_mean": float("nan"),
            "news_sent_wmean": float("nan"),
            "news_sent_present": 0.0,
            "news_event_risk": 0.0,
            "news_src_alpaca": 0.0,
            "news_src_alphavantage": 0.0,
            "news_src_google_rss": 0.0,
            "news_src_tickertick": 0.0,
        }

    ok = 1.0 if bool(bundle.get("ok")) else 0.0
    arts = bundle.get("articles")
    if not isinstance(arts, list):
        arts = []
    count = float(len(arts))

    # Recency: minutes since newest parsed article timestamp
    newest = None
    for a in arts:
        if not isinstance(a, dict):
            continue
        created = a.get("created_at")
        dt = _parse_iso_dt(created) if isinstance(created, str) else None
        if dt is None:
            continue
        newest = dt if newest is None else max(newest, dt)
    recency_min = float("nan")
    if newest is not None:
        recency_min = max(0.0, (now_ts - newest).total_seconds() / 60.0)

    # Provider counts and sentiment
    src_counts = {"alpaca": 0, "alphavantage": 0, "google_rss": 0, "tickertick": 0}
    sent = []
    sent_w = []
    event_risk = 0.0
    risk_words = (
        "earnings",
        "offering",
        "secondary",
        "sec ",
        "investigation",
        "lawsuit",
        "downgrade",
        "upgrade",
        "guidance",
        "merger",
        "acquisition",
        "halt",
        "bankruptcy",
    )
    for a in arts:
        if not isinstance(a, dict):
            continue
        prov = (a.get("provider") or "").strip().lower()
        if prov in src_counts:
            src_counts[prov] += 1
        txt = f"{a.get('headline') or ''} {a.get('summary') or ''}".strip().lower()
        if txt and any(w in txt for w in risk_words):
            event_risk = 1.0
        s = a.get("sentiment_score")
        if s is not None:
            try:
                sv = float(s)
                sent.append(sv)
                # Recency-weight (simple): fresher news gets higher weight.
                w = 1.0
                created = a.get("created_at")
                dt = _parse_iso_dt(created) if isinstance(created, str) else None
                if dt is not None:
                    age_min = max(0.0, (now_ts - dt).total_seconds() / 60.0)
                    w = 1.0 / (1.0 + (age_min / 60.0))
                sent_w.append((sv, w))
            except Exception:
                pass
    sent_mean = float(np.mean(sent)) if sent else float("nan")
    sent_wmean = float(sum(v * w for v, w in sent_w) / sum(w for _v, w in sent_w)) if sent_w else float("nan")
    sent_present = 1.0 if sent else 0.0

    return {
        "news_ok": ok,
        "news_count": count,
        "news_recency_min": recency_min,
        "news_sent_mean": sent_mean,
        "news_sent_wmean": sent_wmean,
        "news_sent_present": sent_present,
        "news_event_risk": float(event_risk),
        "news_src_alpaca": float(src_counts["alpaca"]),
        "news_src_alphavantage": float(src_counts["alphavantage"]),
        "news_src_google_rss": float(src_counts["google_rss"]),
        "news_src_tickertick": float(src_counts["tickertick"]),
    }


def _taapi_features(taapi: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(taapi, dict):
        return {
            "taapi_rsi_1m": float("nan"),
            "taapi_rsi_15m": float("nan"),
            "taapi_macd_1m": float("nan"),
            "taapi_macd_signal_1m": float("nan"),
            "taapi_present": 0.0,
        }
    rsi1 = _to_float(taapi.get("rsi_1m"))
    rsi15 = _to_float(taapi.get("rsi_15m"))
    macd = _to_float(taapi.get("macd_1m"))
    macds = _to_float(taapi.get("macd_signal_1m"))
    present = 1.0 if any(math.isfinite(v) for v in (rsi1, rsi15, macd, macds)) else 0.0
    return {
        "taapi_rsi_1m": rsi1,
        "taapi_rsi_15m": rsi15,
        "taapi_macd_1m": macd,
        "taapi_macd_signal_1m": macds,
        "taapi_present": present,
    }


def get_purged_embargoed_indices(event_times, test_times, embargo_pct=0.01):
    """
    Implements Purging and Embargoing to prevent data leakage.
    event_times: Series of (start, end) times for each sample.
    test_times: Tuple of (start, end) for the test set.
    embargo_pct: Percentage of the dataset to use as a buffer after the test set.
    """
    test_start, test_end = test_times
    # 1. Purging: Remove any training samples that overlap with the test set
    overlap = ((event_times['start'] <= test_end) & (event_times['end'] >= test_start))
    
    # 2. Embargoing: Remove training samples that occur immediately after the test set
    embargo_delta = (event_times['end'].max() - event_times['start'].min()) * embargo_pct
    embargo_end = test_end + embargo_delta
    after_test = ((event_times['start'] > test_end) & (event_times['start'] <= embargo_end))
    
    return ~(overlap | after_test)

def _order_book_features(feat: dict[str, Any]) -> dict[str, float]:
    """
    Extracts the 40 structural order book features for the CNN.
    """
    book = feat.get("order_book", {})
    if not isinstance(book, dict):
        return {f"ob_{i}": 0.0 for i in range(40)}
    
    # Map from the 40-feature list we built in stream.py
    # Level 1-10: Ask Price, Ask Size, Bid Price, Bid Size
    out = {}
    for i in range(40):
        val = book.get(f"val_{i}", 0.0)
        out[f"ob_{i}"] = float(val)
    return out

def label_all_unlabeled_signals(ledger: Ledger, settings: Settings, limit: int = 200) -> int:
    """
    Scans for signals that don't have a triple_barrier_label yet,
    fetches the necessary historical bars from Alpaca, and determines
    the outcome (TP, SL, or Time-Exit).
    """
    import sqlite3
    import json
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timedelta
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    
    conn = sqlite3.connect(ledger._path)
    # Find signals without a corresponding label
    sql = f"""
        SELECT s.id, s.ts, s.symbol, s.action, s.features_json, s.explainability_json
        FROM signals s
        LEFT JOIN triple_barrier_labels tb ON tb.signal_id = s.id
        WHERE tb.signal_id IS NULL AND s.action IN ('BUY', 'SHORT')
        LIMIT {limit}
    """
    unlabeled = conn.execute(sql).fetchall()
    conn.close()
    
    if not unlabeled:
        return 0

    client = StockHistoricalDataClient(settings.apca_api_key_id, settings.apca_api_secret_key)

    def process_signal(row):
        sid, ts_s, sym, action, feat_json, explain_json = row
        ts = _parse_iso_dt(ts_s)
        if not ts:
            return None
        
        feat = json.loads(feat_json or '{}')
        explain = json.loads(explain_json or '{}')
        price = _to_float(feat.get("close"))
        atr = _to_float(feat.get("atr")) or (price * 0.01)
        
        # Barriers (Matches the bot's entry logic exactly)
        stop_dist = max(atr * 2.0, price * 0.0075)
        tp_price = price + (stop_dist * 2.5) if action == "BUY" else price - (stop_dist * 2.5)
        sl_price = price - stop_dist if action == "BUY" else price + stop_dist
        
        # Look forward 120 minutes
        end_ts = ts + timedelta(minutes=120)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[sym],
                timeframe=TimeFrame.Minute,
                start=ts,
                end=end_ts,
                extended_hours=True,
                feed=DataFeed.IEX
            )
            bars = client.get_stock_bars(req)
            if bars.df is None or bars.df.empty:
                return None
                
            sdf = bars.df.xs(sym, level=0)
            outcome = "timeout"
            evaluated_ts = end_ts
            realized_return = 0.0
            
            for bar_ts, row_data in sdf.iterrows():
                b_ts = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
                h = float(row_data['high'])
                l = float(row_data['low'])
                
                # Check barriers
                if action == "BUY":
                    if h >= tp_price:
                        outcome = "tp"
                        evaluated_ts = b_ts
                        realized_return = (tp_price - price) / price
                        break
                    if l <= sl_price:
                        outcome = "sl"
                        evaluated_ts = b_ts
                        realized_return = (sl_price - price) / price
                        break
                else: # SHORT
                    if l <= tp_price:
                        outcome = "tp"
                        evaluated_ts = b_ts
                        realized_return = (price - tp_price) / price
                        break
                    if h >= sl_price:
                        outcome = "sl"
                        evaluated_ts = b_ts
                        realized_return = (price - sl_price) / price
                        break
            
            if outcome == "timeout":
                last_c = float(sdf.iloc[-1]['close'])
                realized_return = (last_c - price) / price if action == "BUY" else (price - last_c) / price

            # FAILURE ANALYSIS: If we lost, record the top reason from explainability
            failure_analysis = {}
            if outcome == "sl":
                failure_analysis = {
                    "primary_culprit": max(explain.items(), key=lambda x: abs(x[1]))[0] if explain else "unknown",
                    "explain_snapshot": explain
                }

            return (
                sid, outcome, realized_return * 100,
                (evaluated_ts - ts).total_seconds() / 60,
                evaluated_ts.isoformat(),
                price, tp_price, sl_price,
                json.dumps(failure_analysis)
            )
        except Exception as e:
            # Silence expected API errors but print stack trace for unexpected ones
            return None

    # Run signal evaluations in parallel
    print(f"Starting parallel evaluation of {len(unlabeled)} stock signals using 16 threads...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_signal, unlabeled))

    results = [r for r in results if r is not None]

    if results:
        conn = sqlite3.connect(ledger._path)
        try:
            conn.execute("BEGIN TRANSACTION;")
            conn.executemany("""
                INSERT INTO triple_barrier_labels (
                    signal_id, outcome, realized_return_pct, horizon_minutes, evaluated_ts, 
                    entry_close, tp_price, sl_price, failure_analysis_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, results)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Transaction failed: {e}")
            raise e
        finally:
            conn.close()

    return len(results)


def build_signal_label_dataset(
    *,
    db_path: str,
    min_horizon_minutes: float | None = None,
    actions: tuple[str, ...] = ("BUY", "SHORT"),
    limit: int | None = None,
    use_purging: bool = True
) -> DatasetResult:
    import sqlite3
    import json
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    
    # 1. Fetch only the IDs first (Extremely fast, minimal memory)
    id_sql = """
        SELECT s.id 
        FROM signals s
        JOIN triple_barrier_labels tb ON tb.signal_id = s.id
        WHERE s.action IN ({actions}) AND tb.outcome != 'timeout'
    """.format(actions=",".join(["?"] * len(actions)))
    
    all_ids = [row[0] for row in conn.execute(id_sql, list(actions)).fetchall()]
    
    # If no triple-barrier non-timeout labels exist, fall back to forward_return_labels
    if not all_ids:
        id_sql_fr = """
            SELECT s.id
            FROM signals s
            JOIN forward_return_labels f ON f.signal_id = s.id
            WHERE s.action IN ({actions})
        """.format(actions=",".join(["?"] * len(actions)))
        try:
            all_ids = [row[0] for row in conn.execute(id_sql_fr, list(actions)).fetchall()]
        except Exception:
            all_ids = []
    
    # 2. Randomly sample the IDs in Python (Instantaneous)
    if limit is not None and len(all_ids) > limit:
        import random
        sampled_ids = random.sample(all_ids, int(limit))
    else:
        sampled_ids = all_ids
        
    if not sampled_ids:
        conn.close()
        return DatasetResult(pd.DataFrame(), pd.Series(), [])
        
    # 3. Query only the 100,000 sampled rows using a temporary table JOIN
    # This bypasses SQLite's query variable limits completely and executes instantly
    conn.execute("CREATE TEMP TABLE sampled_ids_temp (id INTEGER PRIMARY KEY);")
    conn.executemany("INSERT INTO sampled_ids_temp VALUES (?);", [(i,) for i in sampled_ids])
    
    sql = """
        SELECT
          s.id, s.ts, s.symbol, s.action, s.reason, s.features_json,
          tb.realized_return_pct,
          tb.horizon_minutes,
          tb.outcome AS tb_outcome,
          tb.evaluated_ts AS end_ts,
          f.return_pct AS forward_return_pct
        FROM signals s
        LEFT JOIN triple_barrier_labels tb ON tb.signal_id = s.id
        LEFT JOIN forward_return_labels f ON f.signal_id = s.id
        JOIN sampled_ids_temp t ON t.id = s.id
    """
    
    rows = conn.execute(sql).fetchall()
    conn.execute("DROP TABLE sampled_ids_temp;")
    conn.close()

    feats_rows = []
    y_rows = []
    event_times = []

    for (sid, ts_s, sym, action, reason, feat_json, ret_pct, horizon_min, tb_outcome, end_ts_s, forward_return_pct) in rows:
        ts = _parse_iso_dt(ts_s)
        end_ts = _parse_iso_dt(end_ts_s) or ts
        if not ts: continue
        
        event_times.append({'ts': ts, 'end': end_ts})
        
        feat = json.loads(feat_json or '{}')
        # --- STATIONARY FEATURE ENGINEERING (100% Complete indicators) ---
        price = _to_float(feat.get("close"))
        vwap = _to_float(feat.get("vwap"))
        atr = _to_float(feat.get("atr", 0.0))
        avg50 = _to_float(feat.get("avg_50"))
        
        # Volatility normalization scale to prevent momentum compression bias
        norm_scale = atr if (atr and atr > 0) else (price * 0.01 if price > 0 else 1.0)
        
        # Extract order book level 0 details if available
        ob = feat.get("order_book", {})
        ask_price = _to_float(ob.get("val_0"))
        ask_size = _to_float(ob.get("val_1"))
        bid_price = _to_float(ob.get("val_2"))
        bid_size = _to_float(ob.get("val_3"))
        
        spread_ratio = (ask_price - bid_price) / price if (price > 0 and ask_price > 0 and bid_price > 0) else 0.0
        obi = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size and ask_size and (bid_size + ask_size) > 0) else 0.0
        
        # Multi-timeframe divergences
        htf_rsi = _to_float(feat.get("htf_rsi"))
        rsi_1m = _to_float(feat.get("rsi_14"))
        rsi_divergence = rsi_1m - htf_rsi if (math.isfinite(rsi_1m) and math.isfinite(htf_rsi)) else 0.0
        
        macd_val = _to_float(feat.get("macd"))
        macd_sig = _to_float(feat.get("macd_signal"))
        macd_convergence = macd_val / macd_sig if (macd_val and macd_sig and abs(macd_sig) > 0) else 0.0

        # Calculate vwap_ratio and avg50_ratio (volatility normalized)
        vwap_ratio = (price - vwap) / norm_scale if (price and vwap and norm_scale > 0) else 0.0
        
        avg50 = _to_float(feat.get("avg_50"))
        if math.isnan(avg50) or avg50 <= 0:
            ema21 = _to_float(feat.get("ema_21"))
            alligator = _to_float(feat.get("alligator_jaw"))
            if not math.isnan(ema21) and ema21 > 0:
                avg50 = ema21
            elif not math.isnan(alligator) and alligator > 0:
                avg50 = alligator
            else:
                avg50 = price
        
        avg50_ratio = (price - avg50) / norm_scale if (price and avg50 and norm_scale > 0) else 0.0

        vol_sma = _to_float(feat.get("vol_sma_20"))
        if not vol_sma or math.isnan(vol_sma) or vol_sma <= 0:
            vol_sma = _to_float(feat.get("volume"))
        if not vol_sma or math.isnan(vol_sma) or vol_sma <= 0:
            vol_sma = 1.0

        obv_val = _to_float(feat.get("obv", 0.0))
        obv_ema_val = _to_float(feat.get("obv_ema", 0.0))
        vol_imb_val = _to_float(feat.get("volume_imbalance", 0.0))
        supertrend_val = _to_float(feat.get("supertrend", 0.0))
        supertrend_norm = (supertrend_val - price) / price if (price > 0 and supertrend_val > 0) else 0.0

        def normalize_band(val_raw):
            val = _to_float(val_raw)
            if not val or math.isnan(val):
                return 0.0
            if abs(val) < 2.0:
                # Already normalized (close - band) / close in DB. Negate to get (band - close) / close.
                return -val
            return (val - price) / price if price > 0 else 0.0

        x = {
            "vwap_dist_ratio": vwap_ratio,
            "rsi_14": rsi_1m,
            "macd_line_ratio": _to_float(feat.get("macd_line", 0.0)) / norm_scale if norm_scale > 0 else 0.0,
            "macd_signal_ratio": _to_float(feat.get("macd_signal", 0.0)) / norm_scale if norm_scale > 0 else 0.0,
            "macd_hist_ratio": _to_float(feat.get("macd_hist", 0.0)) / norm_scale if norm_scale > 0 else 0.0,
            "alligator_jaw_ratio": (price - _to_float(feat.get("alligator_jaw", 0.0))) / norm_scale if norm_scale > 0 else 0.0,
            "alligator_teeth_ratio": (price - _to_float(feat.get("alligator_teeth", 0.0))) / norm_scale if norm_scale > 0 else 0.0,
            "alligator_lips_ratio": (price - _to_float(feat.get("alligator_lips", 0.0))) / norm_scale if norm_scale > 0 else 0.0,
            "alligator_convergence_index": _to_float(feat.get("alligator_convergence_index", 0.0)),
            "atr_norm": norm_scale / price if price > 0 else 0.0,
            "avg_50_ratio": avg50_ratio,
            "momentum_pct": _to_float(feat.get("momentum_pct")) if feat.get("momentum_pct") is not None and not math.isnan(_to_float(feat.get("momentum_pct"))) else ((price - avg50) / avg50 if (price and avg50 and avg50 > 0) else 0.0),
            
            # Deep Intelligence Features
            "news_sentiment_score": _to_float(feat.get("news_sentiment_score", 0.0)),
            "fed_liquidity_momentum": _to_float(feat.get("fed_liquidity_momentum", 0.0)),
            "sector_dispersion_factor": _to_float(feat.get("sector_dispersion_factor", 0.0)),
            "moc_imbalance_shares": _to_float(feat.get("moc_imbalance_shares", 0.0)),
            "minutes_until_earnings_announcement": _to_float(feat.get("minutes_until_earnings_announcement", 1440.0)),
            
            # Engineered Microstructure & Divergence Features
            "spread_ratio": spread_ratio,
            "order_book_imbalance": obi,
            "rsi_divergence": rsi_divergence,
            "macd_convergence_ratio": macd_convergence,

            # Primary live strategy rules indicators
            "ar1_rho": _to_float(feat.get("ar1_rho", 0.0)),
            "z_score_vwap": _to_float(feat.get("z_score_vwap", 0.0)),
            "willr": _to_float(feat.get("willr", -50.0)),
            "stoch_k": _to_float(feat.get("stoch_k", 50.0)),
            "stoch_d": _to_float(feat.get("stoch_d", 50.0)),
            "obv": obv_val / vol_sma,
            "obv_ema": obv_ema_val / vol_sma,
            "cmf": _to_float(feat.get("cmf", 0.0)),
            "supertrend": supertrend_norm,
            "supertrend_dir": _to_float(feat.get("supertrend_dir", 0.0)),
            "rvol": _to_float(feat.get("rvol", 1.0)),
            "ret_1m": _to_float(feat.get("ret_1m", 0.0)),
            "ret_5m": _to_float(feat.get("ret_5m", 0.0)),
            "ret_15m": _to_float(feat.get("ret_15m", 0.0)),
            # VWAP deviation bands
            "vwap_band_1": normalize_band(feat.get("vwap_band_1")),
            "vwap_band_2": normalize_band(feat.get("vwap_band_2")),
            "vwap_band_3": normalize_band(feat.get("vwap_band_3")),
            "vwap_band_neg1": normalize_band(feat.get("vwap_band_neg1")),
            "vwap_band_neg2": normalize_band(feat.get("vwap_band_neg2")),
            "vwap_band_neg3": normalize_band(feat.get("vwap_band_neg3")),
            # Volume imbalance
            "volume_imbalance": vol_imb_val / vol_sma,
            "volume_imbalance_norm": _to_float(feat.get("volume_imbalance_norm", 0.0)),
            # Regime metadata
            "regime": feat.get("regime", "neutral"),
        }
        
        # Get TAAPI features (nested JSON)
        taapi_feats = _taapi_features(feat.get("taapi"))
        x.update(taapi_feats)
        x["_signal_id_temp"] = sid
        
        # Store timestamp temporarily for broad-market asof join
        x["_ts_temp"] = ts
        
        # Target: Prefer triple-barrier outcome when available; otherwise use forward_return_pct sign
        if tb_outcome is not None:
            is_tp = (str(tb_outcome).strip().lower() == "tp")
            is_high_profit_timeout = (str(tb_outcome).strip().lower() == "timeout" and (ret_pct or 0.0) >= 0.75)
            y = 1 if (is_tp or is_high_profit_timeout) else 0
        else:
            # fallback: treat positive forward_return_pct as hit
            try:
                fr = float(forward_return_pct) if forward_return_pct is not None else 0.0
            except Exception:
                fr = 0.0
            y = 1 if fr > 0.0 else 0
        
        feats_rows.append(x)
        y_rows.append(y)

    if not feats_rows:
        return DatasetResult(X=pd.DataFrame(), y=pd.Series(), meta=pd.DataFrame())

    # Build primary features DataFrame
    X = pd.DataFrame(feats_rows)
    X["_target_y"] = y_rows
    X["_end_temp"] = [e["end"] for e in event_times]

    
    # Enrich with Pillar 2 (Broad Market Conditioning) and Pillar 3 (HTF Trend Alignment)
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca_day_bot.config import load_settings
        
        # 1. Initialize Alpaca historical client using settings
        settings = load_settings()
        client = StockHistoricalDataClient(
            settings.apca_api_key_id,
            settings.apca_api_secret_key,
        )
        
        # 2. Determine historical range (convert to timezone-aware UTC)
        timestamps = X["_ts_temp"]
        min_ts = timestamps.min() - pd.Timedelta(days=5) # 5 days buffer for EMAs
        max_ts = timestamps.max() + pd.Timedelta(days=1)
        
        # 3. Request 5-minute bars for SPY and QQQ
        from alpaca.data.enums import DataFeed
        req = StockBarsRequest(
            symbol_or_symbols=["SPY", "QQQ"],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=min_ts.to_pydatetime() if hasattr(min_ts, "to_pydatetime") else min_ts,
            end=max_ts.to_pydatetime() if hasattr(max_ts, "to_pydatetime") else max_ts,
            feed=DataFeed.IEX
        )
        
        bars = client.get_stock_bars(req)
        df_bars = bars.df
        
        if not df_bars.empty:
            # Pivot the dataframe to get closing prices for SPY and QQQ
            close_df = df_bars["close"].unstack(level="symbol").ffill()
            
            spy_series = close_df["SPY"]
            qqq_series = close_df["QQQ"]
            
            # Calculate rolling returns & true realized volatility (20-period log returns dispersion)
            spy_ret = spy_series.pct_change(1)
            qqq_ret = qqq_series.pct_change(1)
            
            spy_log_ret = np.log(spy_series / spy_series.shift(1))
            spy_vol = spy_log_ret.rolling(20).std()
            
            # Calculate EMAs for Pillar 3 Trend Alignment
            spy_ema_1h = spy_series.ewm(span=240, adjust=False).mean()
            spy_ema_4h = spy_series.ewm(span=960, adjust=False).mean()
            qqq_ema_1h = qqq_series.ewm(span=240, adjust=False).mean()
            qqq_ema_4h = qqq_series.ewm(span=960, adjust=False).mean()
            
            # Calculate realized volatilities (1h window = 12 5m bars)
            spy_log_ret = np.log(spy_series / spy_series.shift(1))
            qqq_log_ret = np.log(qqq_series / qqq_series.shift(1))
            spy_realized_vol_1h = spy_log_ret.rolling(12).std()
            qqq_realized_vol_1h = qqq_log_ret.rolling(12).std()

            # Construct macro features DataFrame
            macro_df = pd.DataFrame(index=close_df.index)
            macro_df["spy_ret_5m"] = spy_ret
            macro_df["qqq_ret_5m"] = qqq_ret
            macro_df["vix_roc_5m"] = spy_vol
            macro_df["spy_trend_1h"] = (spy_series > spy_ema_1h).astype(float)
            macro_df["spy_trend_4h"] = (spy_series > spy_ema_4h).astype(float)
            macro_df["qqq_trend_1h"] = (qqq_series > qqq_ema_1h).astype(float)
            macro_df["qqq_trend_4h"] = (qqq_series > qqq_ema_4h).astype(float)
            macro_df["spy_realized_vol_1h"] = spy_realized_vol_1h
            macro_df["qqq_realized_vol_1h"] = qqq_realized_vol_1h
            
            # Sort for pd.merge_asof (requires sorted keys)
            X = X.sort_values("_ts_temp").reset_index(drop=True)
            macro_df = macro_df.sort_index()
            
            # Ensure index timezone compatibility (convert all to timezone-aware UTC)
            if X["_ts_temp"].dt.tz is None:
                X["_ts_temp"] = X["_ts_temp"].dt.tz_localize("UTC")
            else:
                X["_ts_temp"] = X["_ts_temp"].dt.tz_convert("UTC")
                
            if macro_df.index.tz is None:
                macro_df.index = macro_df.index.tz_localize("UTC")
            else:
                macro_df.index = macro_df.index.tz_convert("UTC")
                
            # Perform high-speed Asof join (nearest matching tick from past)
            X_enriched = pd.merge_asof(
                X,
                macro_df,
                left_on="_ts_temp",
                right_index=True,
                direction="backward"
            )
            X = X_enriched
    except Exception as e:
        print(f"[ML Pipeline] Enrichment Warning: {e}")
        # Fallback to zero columns if Alpaca download fails
        for col in ["spy_ret_5m", "qqq_ret_5m", "vix_roc_5m", "spy_trend_1h", "spy_trend_4h", "qqq_trend_1h", "qqq_trend_4h", "spy_realized_vol_1h", "qqq_realized_vol_1h"]:
            if col not in X.columns:
                X[col] = 0.0

    # Final sort and reset index to guarantee order
    if "_ts_temp" in X.columns:
        X = X.sort_values("_ts_temp").reset_index(drop=True)

    y_ser = X["_target_y"].rename("y")
    X = X.drop(columns=["_target_y"])
    
    meta = pd.DataFrame({
        "ts": X["_ts_temp"],
        "end": X["_end_temp"],
        "signal_id": X["_signal_id_temp"]
    })
    X = X.drop(columns=["_end_temp", "_signal_id_temp"])

    # Drop temporary column used for joining
    if "_ts_temp" in X.columns:
        X = X.drop(columns=["_ts_temp"])
        
    X = X.fillna(0)
    
    return DatasetResult(X=X, y=y_ser, meta=meta)


