from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from alpaca_day_bot.ml.dataset import DatasetResult, _news_features, _parse_iso_dt, _taapi_features, _to_float


def build_sim_trade_dataset(*, db_path: str, actions: tuple[str, ...] = ("BUY",), limit: int | None = None) -> DatasetResult:
    """
    Build a supervised dataset from simulated trades:
      sim_signals JOIN sim_trades ON sim_trades.sim_signal_id = sim_signals.id

    Label:
      y=1 iff trade pnl > 0 else 0
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
          t.pnl,
          t.entry_price,
          t.qty,
          t.hold_minutes
        FROM sim_signals s
        JOIN sim_trades t ON t.sim_signal_id = s.id
        WHERE s.action IN ({actions})
        ORDER BY s.ts ASC
    """.format(actions=",".join(["?"] * len(actions)))
    params: list[Any] = list(actions)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    feats_rows: list[dict[str, Any]] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for (_sid, ts_s, sym, action, reason, feat_json, pnl, entry_price, qty, hold_minutes) in rows:
        ts = _parse_iso_dt(ts_s) or datetime.now(tz=timezone.utc)
        now_ts = ts

        x_time = {
            "hour_utc": float(ts.hour),
            "minute_utc": float(ts.minute),
            "dow_utc": float(ts.weekday()),
        }

        feat: dict[str, Any] = {}
        if feat_json and isinstance(feat_json, str):
            try:
                feat = json.loads(feat_json) or {}
            except Exception:
                feat = {}
        if not isinstance(feat, dict):
            feat = {}

        x: dict[str, Any] = {
            "close": _to_float(feat.get("close")),
            "rsi_14": _to_float(feat.get("rsi_14")),
            "htf_rsi": _to_float(feat.get("htf_rsi")),
            "ema": _to_float(feat.get("ema")),
            "ema_9": _to_float(feat.get("ema_9")),
            "ema_21": _to_float(feat.get("ema_21")),
            "ema_9_21_bias": _to_float(feat.get("ema_9_21_bias")),
            "alligator_jaw": _to_float(feat.get("alligator_jaw")),
            "alligator_teeth": _to_float(feat.get("alligator_teeth")),
            "alligator_lips": _to_float(feat.get("alligator_lips")),
            "alligator_trend_up": _to_float(feat.get("alligator_trend_up")),
            "alligator_trend_down": _to_float(feat.get("alligator_trend_down")),
            "macd": _to_float(feat.get("macd")),
            "macd_signal": _to_float(feat.get("macd_signal")),
            "vwap": _to_float(feat.get("vwap")),
            "volume_ratio": _to_float(feat.get("volume_ratio")),
            "atr": _to_float(feat.get("atr")),
            "atr_avg": _to_float(feat.get("atr_avg")),
            "atr_monthly": _to_float(feat.get("atr_monthly")),
            "htf_ok_long": 1.0 if bool(feat.get("htf_ok_long")) else 0.0,
            "htf_ok_short": 1.0 if bool(feat.get("htf_ok_short")) else 0.0,
            "is_buy": 1.0 if str(action).upper() == "BUY" else 0.0,
        }
        x.update(x_time)

        rs = (reason or "").strip().lower()
        x["setup_long_pullback"] = 1.0 if rs == "long_rsi_macd_vwap_volume" else 0.0
        x["setup_long_momo"] = 1.0 if rs == "long_momo" else 0.0
        x["setup_short_pullback"] = 1.0 if rs == "short_rsi_macd_vwap_volume" else 0.0
        x["setup_short_momo"] = 1.0 if rs == "short_momo" else 0.0
        x["setup_short_rsi_overbought_fade"] = 1.0 if rs == "short_rsi_overbought_fade" else 0.0
        x["setup_short_bb_upper_fade"] = 1.0 if rs == "short_bb_upper_fade" else 0.0
        x["setup_short_break_retest"] = 1.0 if rs == "short_break_retest" else 0.0

        news = feat.get("news") if isinstance(feat.get("news"), dict) else None
        x.update(_news_features(news, now_ts=now_ts))

        taapi = feat.get("taapi") if isinstance(feat.get("taapi"), dict) else None
        x.update(_taapi_features(taapi))

        try:
            pnl_v = float(pnl) if pnl is not None else 0.0
            notional = float(entry_price) * float(qty) if entry_price is not None and qty is not None else 0.0
            ret_pct = (pnl_v / notional * 100.0) if notional > 1e-12 else 0.0
        except Exception:
            pnl_v = 0.0
            ret_pct = 0.0

        y = 1 if pnl_v > 0 else 0
        feats_rows.append(x)
        y_rows.append(y)
        meta_rows.append(
            {
                "ts": ts_s,
                "symbol": sym,
                "action": action,
                "reason": reason,
                "return_pct": float(ret_pct),
                "horizon_minutes": float(hold_minutes) if hold_minutes is not None else float("nan"),
                "tb_outcome": None,
            }
        )

    X = pd.DataFrame(feats_rows)
    y_ser = pd.Series(y_rows, name="y", dtype=int)
    meta = pd.DataFrame(meta_rows)
    return DatasetResult(X=X, y=y_ser, meta=meta)

