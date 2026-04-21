from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TradeWhyRow:
    ts: str
    symbol: str
    side: str
    qty: float | None
    action: str | None
    setup_reason: str | None
    model_proba: float | None
    news_count: int | None
    news_sent_mean: float | None
    taapi_present: bool | None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_features(features_json: str | None) -> dict[str, Any]:
    if not features_json:
        return {}
    try:
        v = json.loads(features_json)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def trade_whys_for_day(db_path: str, day_utc: str) -> list[TradeWhyRow]:
    """
    For each submitted order intent, find the most recent signal for that symbol within a short window
    and extract 'why' fields from signal features.
    """
    start = f"{day_utc}T00:00:00+00:00"
    end = f"{day_utc}T23:59:59+00:00"
    conn = sqlite3.connect(db_path)

    intents = conn.execute(
        """
        SELECT ts, symbol, side, raw_json
        FROM order_intents
        WHERE submitted=1 AND ts BETWEEN ? AND ?
        ORDER BY ts ASC
        """,
        (start, end),
    ).fetchall()

    out: list[TradeWhyRow] = []
    for ts, sym, side, raw in intents:
        extra = {}
        try:
            j = json.loads(raw) if raw else {}
            extra = (j.get("extra") or {}) if isinstance(j, dict) else {}
        except Exception:
            extra = {}
        qty = _safe_float(extra.get("qty"))
        action = extra.get("action")

        # Find most recent signal row for symbol at/just before intent time.
        sig = conn.execute(
            """
            SELECT ts, reason, features_json
            FROM signals
            WHERE symbol=? AND ts <= ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (sym, ts),
        ).fetchone()

        setup_reason = None
        model_proba = None
        news_count = None
        news_sent_mean = None
        taapi_present = None
        if sig:
            _sig_ts, sig_reason, feat_json = sig
            setup_reason = sig_reason
            feat = _parse_features(feat_json)
            # model
            if "model_proba" in feat:
                model_proba = _safe_float(feat.get("model_proba"))
            else:
                m = feat.get("model")
                if isinstance(m, dict):
                    model_proba = _safe_float(m.get("proba"))
            # news
            nb = feat.get("news")
            if isinstance(nb, dict):
                arts = nb.get("articles")
                if isinstance(arts, list):
                    news_count = len([a for a in arts if isinstance(a, dict)])
                    sents = []
                    for a in arts:
                        if not isinstance(a, dict):
                            continue
                        sc = a.get("sentiment_score")
                        if sc is None:
                            continue
                        try:
                            sents.append(float(sc))
                        except Exception:
                            pass
                    news_sent_mean = (sum(sents) / len(sents)) if sents else None
            # taapi
            t = feat.get("taapi")
            if isinstance(t, dict):
                taapi_present = any(t.get(k) is not None for k in ("rsi_1m", "rsi_15m", "macd_1m", "macd_signal_1m"))

        out.append(
            TradeWhyRow(
                ts=ts,
                symbol=str(sym),
                side=str(side),
                qty=qty,
                action=str(action) if action is not None else None,
                setup_reason=str(setup_reason) if setup_reason is not None else None,
                model_proba=model_proba,
                news_count=news_count,
                news_sent_mean=news_sent_mean,
                taapi_present=taapi_present,
            )
        )

    conn.close()
    return out

