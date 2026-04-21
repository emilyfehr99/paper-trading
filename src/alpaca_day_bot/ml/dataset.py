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


def build_signal_label_dataset(
    *,
    db_path: str,
    min_horizon_minutes: float | None = None,
    actions: tuple[str, ...] = ("BUY",),
    limit: int | None = None,
) -> DatasetResult:
    """
    Build supervised dataset from:
      signals(ts,symbol,action,reason,features_json) JOIN forward_return_labels(signal_id,...,return_pct,horizon_minutes)
    """
    conn = sqlite3.connect(db_path)
    sql = """
        SELECT
          s.id,
          s.ts,
          s.symbol,
          s.action,
          s.reason,
          s.features_json,
          f.return_pct,
          f.horizon_minutes
        FROM signals s
        JOIN forward_return_labels f ON f.signal_id = s.id
        WHERE s.action IN ({actions})
    """.format(actions=",".join(["?"] * len(actions)))
    params: list[Any] = list(actions)
    if min_horizon_minutes is not None:
        sql += " AND f.horizon_minutes >= ?"
        params.append(float(min_horizon_minutes))
    sql += " ORDER BY s.ts ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    feats_rows: list[dict[str, Any]] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for (_sid, ts_s, sym, action, reason, feat_json, ret_pct, horizon_min) in rows:
        ts = _parse_iso_dt(ts_s) or datetime.now(tz=timezone.utc)
        now_ts = ts  # features are at signal-time; use ts as anchor for recency calcs

        # Time-of-day context (UTC; stable in CI). Model can learn when signals work.
        x_time = {
            "hour_utc": float(ts.hour),
            "minute_utc": float(ts.minute),
            "dow_utc": float(ts.weekday()),
        }

        feat = {}
        if feat_json and isinstance(feat_json, str):
            try:
                feat = json.loads(feat_json) or {}
            except Exception:
                feat = {}
        if not isinstance(feat, dict):
            feat = {}

        # Core technicals from rule-engine features
        x: dict[str, Any] = {
            "close": _to_float(feat.get("close")),
            "rsi_14": _to_float(feat.get("rsi_14")),
            "htf_rsi": _to_float(feat.get("htf_rsi")),
            "ema": _to_float(feat.get("ema")),
            "macd": _to_float(feat.get("macd")),
            "macd_signal": _to_float(feat.get("macd_signal")),
            "vwap": _to_float(feat.get("vwap")),
            "volume_ratio": _to_float(feat.get("volume_ratio")),
            "atr": _to_float(feat.get("atr")),
            "atr_avg": _to_float(feat.get("atr_avg")),
            "htf_ok_long": 1.0 if bool(feat.get("htf_ok_long")) else 0.0,
            "htf_ok_short": 1.0 if bool(feat.get("htf_ok_short")) else 0.0,
            "is_buy": 1.0 if str(action).upper() == "BUY" else 0.0,
        }
        x.update(x_time)

        # Setup-type context (reason string from StrategySignal)
        rs = (reason or "").strip().lower()
        x["setup_long_pullback"] = 1.0 if rs == "long_rsi_macd_vwap_volume" else 0.0
        x["setup_long_momo"] = 1.0 if rs == "long_momo" else 0.0
        x["setup_short_pullback"] = 1.0 if rs == "short_rsi_macd_vwap_volume" else 0.0
        x["setup_short_momo"] = 1.0 if rs == "short_momo" else 0.0

        # News bundle features (stored in features["news"] by main tick)
        news = feat.get("news") if isinstance(feat.get("news"), dict) else None
        x.update(_news_features(news, now_ts=now_ts))

        # TAAPI features (stored in features["taapi"])
        taapi = feat.get("taapi") if isinstance(feat.get("taapi"), dict) else None
        x.update(_taapi_features(taapi))

        # Target
        ret = float(ret_pct) if ret_pct is not None else 0.0
        y = 1 if ret > 0 else 0

        feats_rows.append(x)
        y_rows.append(y)
        meta_rows.append(
            {
                "ts": ts_s,
                "symbol": sym,
                "action": action,
                "reason": reason,
                "return_pct": ret,
                "horizon_minutes": float(horizon_min) if horizon_min is not None else float("nan"),
            }
        )

    X = pd.DataFrame(feats_rows)
    y_ser = pd.Series(y_rows, name="y", dtype=int)
    meta = pd.DataFrame(meta_rows)
    return DatasetResult(X=X, y=y_ser, meta=meta)

