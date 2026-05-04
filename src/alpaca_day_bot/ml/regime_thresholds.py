from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score


@dataclass(frozen=True)
class RegimeThresholdRow:
    """Per (regime, action) bucket: best proba threshold from time-ordered train split."""

    regime: str
    action: str
    n: int
    best_min_proba: float | None
    metric_f1: float | None
    hit_rate: float | None


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _split_reg_act(key: str) -> tuple[str, str]:
    if "|" in key:
        reg, act = key.rsplit("|", 1)
        return reg.strip(), act.strip().upper()
    return key.strip(), ""


def learn_regime_min_proba_map(
    *,
    db_path: str,
    min_rows_per_bucket: int = 40,
    min_rows_per_regime: int | None = None,
    default_min_proba: float = 0.55,
) -> tuple[dict[str, float], list[RegimeThresholdRow]]:
    """
    Learn per-(regime, action) MODEL_MIN_PROBA thresholds from labeled signals that already
    record json `model_proba`.

    Map keys: "{regime}|BUY" and "{regime}|SHORT".

    Chooses threshold by F1 on the older 70% of each bucket; reports F1 and hit-rate on the newer 30%.
    """
    _ = default_min_proba  # reserved for callers that want a floor written beside the map
    if min_rows_per_regime is not None:
        min_rows_per_bucket = int(min_rows_per_regime)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT
          json_extract(s.features_json, '$.regime') AS regime,
          s.action AS sig_action,
          json_extract(s.features_json, '$.model_proba') AS model_proba,
          tb.outcome AS tb_outcome,
          f.return_pct AS return_pct
        FROM signals s
        LEFT JOIN triple_barrier_labels tb ON tb.signal_id = s.id
        LEFT JOIN forward_return_labels f ON f.signal_id = s.id
        WHERE (tb.signal_id IS NOT NULL OR f.signal_id IS NOT NULL)
          AND json_extract(s.features_json, '$.model_proba') IS NOT NULL
        ORDER BY s.ts ASC
        """
    ).fetchall()
    conn.close()

    def _y_from_row(tb_out: Any, ret: Any) -> int | None:
        if isinstance(tb_out, str) and tb_out.strip().lower() in ("tp", "sl", "timeout"):
            return 1 if tb_out.strip().lower() == "tp" else 0
        rr = _safe_float(ret)
        if rr is None:
            return None
        return 1 if rr > 0 else 0

    by_key: dict[str, list[tuple[float, int]]] = {}
    for regime, sig_action, mp, tb_out, ret in rows:
        r = (str(regime).strip() if regime is not None else "") or ""
        act = (str(sig_action).strip().upper() if sig_action is not None else "") or ""
        if not r or act not in ("BUY", "SHORT"):
            continue
        p = _safe_float(mp)
        if p is None:
            continue
        y = _y_from_row(tb_out, ret)
        if y is None:
            continue
        key = f"{r}|{act}"
        by_key.setdefault(key, []).append((float(p), int(y)))

    sweep = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    out_map: dict[str, float] = {}
    out_rows: list[RegimeThresholdRow] = []

    for key in sorted(by_key.keys()):
        pts = by_key[key]
        reg, act = _split_reg_act(key)
        if len(pts) < int(min_rows_per_bucket):
            out_rows.append(
                RegimeThresholdRow(regime=reg, action=act, n=len(pts), best_min_proba=None, metric_f1=None, hit_rate=None)
            )
            continue

        cut_i = max(1, int(len(pts) * 0.70))
        train_pts = pts[:cut_i]
        test_pts = pts[cut_i:]
        if len(test_pts) < max(8, int(0.15 * len(pts))):
            train_pts, test_pts = pts, pts

        p_tr = np.asarray([a for a, _ in train_pts], dtype=float)
        y_tr = np.asarray([b for _, b in train_pts], dtype=int)
        p_te = np.asarray([a for a, _ in test_pts], dtype=float)
        y_te = np.asarray([b for _, b in test_pts], dtype=int)
        if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
            out_rows.append(
                RegimeThresholdRow(regime=reg, action=act, n=len(pts), best_min_proba=None, metric_f1=None, hit_rate=None)
            )
            continue

        best_thr = None
        best_f1_tr = -1.0
        best_prec_tr = -1.0
        for thr in sweep:
            pred = (p_tr >= float(thr)).astype(int)
            if int(pred.sum()) < max(5, int(0.08 * len(train_pts))):
                continue
            f1v = float(f1_score(y_tr, pred, zero_division=0))
            pv = float(precision_score(y_tr, pred, zero_division=0))
            if f1v > best_f1_tr or (f1v == best_f1_tr and pv > best_prec_tr):
                best_f1_tr = f1v
                best_prec_tr = pv
                best_thr = float(thr)

        if best_thr is None:
            out_rows.append(
                RegimeThresholdRow(regime=reg, action=act, n=len(pts), best_min_proba=None, metric_f1=None, hit_rate=None)
            )
            continue

        pred_te = (p_te >= float(best_thr)).astype(int)
        f1_te = float(f1_score(y_te, pred_te, zero_division=0)) if len(y_te) else None
        taken = [(a, b) for a, b in test_pts if a >= best_thr]
        hr = (sum(b for _a, b in taken) / float(len(taken))) if taken else None

        out_map[key] = float(best_thr)
        out_rows.append(
            RegimeThresholdRow(
                regime=reg,
                action=act,
                n=len(pts),
                best_min_proba=float(best_thr),
                metric_f1=float(f1_te) if f1_te is not None else None,
                hit_rate=float(hr) if hr is not None else None,
            )
        )

    if not out_map:
        out_map = {}
    return out_map, out_rows


def write_regime_thresholds_json(*, out_path: str, mp_map: dict[str, float], rows: list[RegimeThresholdRow]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_min_proba_by_regime": mp_map,
        "rows": [
            {
                "regime": r.regime,
                "action": r.action,
                "n": r.n,
                "best_min_proba": r.best_min_proba,
                "metric_f1": r.metric_f1,
                "hit_rate": r.hit_rate,
            }
            for r in rows
        ],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
