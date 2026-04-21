from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelBucketRow:
    bucket: str
    n: int
    hit_rate: float | None


@dataclass(frozen=True)
class ModelDiagnostics:
    n_labeled: int
    n_with_proba: int
    buckets: list[ModelBucketRow]


def _parse_ts(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(tz=timezone.utc)


def model_diagnostics_for_day(db_path: str, day: date) -> ModelDiagnostics | None:
    conn = sqlite3.connect(db_path)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()
    rows = conn.execute(
        """
        SELECT s.ts, s.features_json, f.return_pct
        FROM signals s
        JOIN forward_return_labels f ON f.signal_id = s.id
        WHERE s.ts BETWEEN ? AND ? AND s.action='BUY'
        ORDER BY s.ts ASC
        """,
        (start, end),
    ).fetchall()
    conn.close()
    if not rows:
        return None

    probs = []
    y = []
    for _ts, feat_json, ret_pct in rows:
        feat = {}
        if feat_json:
            try:
                feat = json.loads(feat_json) or {}
            except Exception:
                feat = {}
        p = None
        if isinstance(feat, dict):
            if "model_proba" in feat:
                try:
                    p = float(feat["model_proba"])
                except Exception:
                    p = None
            else:
                m = feat.get("model")
                if isinstance(m, dict) and m.get("proba") is not None:
                    try:
                        p = float(m["proba"])
                    except Exception:
                        p = None
        probs.append(p)
        y.append(1 if float(ret_pct) > 0 else 0)

    n_labeled = len(y)
    idx = [i for i, p in enumerate(probs) if p is not None]
    n_with_proba = len(idx)

    if n_with_proba < 10:
        return ModelDiagnostics(n_labeled=n_labeled, n_with_proba=n_with_proba, buckets=[])

    df = pd.DataFrame({"p": [probs[i] for i in idx], "y": [y[i] for i in idx]})
    # Buckets: [0.0-0.55), [0.55-0.65), [0.65-0.75), [0.75+]
    bins = [0.0, 0.55, 0.65, 0.75, 1.01]
    labels = ["<0.55", "0.55-0.65", "0.65-0.75", ">=0.75"]
    df["b"] = pd.cut(df["p"], bins=bins, labels=labels, include_lowest=True, right=False)

    buckets: list[ModelBucketRow] = []
    for b in labels:
        sub = df[df["b"] == b]
        if sub.empty:
            continue
        hit = float(sub["y"].mean()) if len(sub) >= 3 else None
        buckets.append(ModelBucketRow(bucket=b, n=int(len(sub)), hit_rate=hit))

    return ModelDiagnostics(n_labeled=n_labeled, n_with_proba=n_with_proba, buckets=buckets)

