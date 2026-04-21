from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegimeThresholdRow:
    regime: str
    n: int
    best_min_proba: float | None
    hit_rate: float | None


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def learn_regime_min_proba_map(
    *,
    db_path: str,
    min_rows_per_regime: int = 25,
    default_min_proba: float = 0.55,
) -> tuple[dict[str, float], list[RegimeThresholdRow]]:
    """
    Learn a per-regime MODEL_MIN_PROBA map from labeled signals.
    Uses triple_barrier_labels outcome if available; otherwise uses forward_return sign.

    We optimize for hit-rate among taken trades by sweeping thresholds.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT
          json_extract(s.features_json, '$.regime') AS regime,
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

    by_regime: dict[str, list[tuple[float, int]]] = {}
    for regime, mp, tb_out, ret in rows:
        r = (str(regime).strip() if regime is not None else "") or ""
        if not r:
            continue
        p = _safe_float(mp)
        if p is None:
            continue
        y = None
        if isinstance(tb_out, str) and tb_out.strip().lower() in ("tp", "sl", "timeout"):
            y = 1 if tb_out.strip().lower() == "tp" else 0
        else:
            rr = _safe_float(ret)
            if rr is None:
                continue
            y = 1 if rr > 0 else 0
        by_regime.setdefault(r, []).append((float(p), int(y)))

    sweep = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    out_map: dict[str, float] = {}
    out_rows: list[RegimeThresholdRow] = []
    for r, pts in sorted(by_regime.items()):
        if len(pts) < int(min_rows_per_regime):
            out_rows.append(RegimeThresholdRow(regime=r, n=len(pts), best_min_proba=None, hit_rate=None))
            continue
        best_thr = None
        best_hr = -1.0
        for thr in sweep:
            taken = [(p, y) for (p, y) in pts if p >= thr]
            if len(taken) < max(5, int(0.10 * len(pts))):
                continue
            hr = sum(y for _p, y in taken) / float(len(taken))
            if hr >= best_hr:
                best_hr = float(hr)
                best_thr = float(thr)
        if best_thr is None:
            out_rows.append(RegimeThresholdRow(regime=r, n=len(pts), best_min_proba=None, hit_rate=None))
            continue
        out_map[r] = float(best_thr)
        out_rows.append(RegimeThresholdRow(regime=r, n=len(pts), best_min_proba=float(best_thr), hit_rate=float(best_hr)))

    # Always keep a default in mind; caller can fall back.
    if not out_map:
        out_map = {}
    return out_map, out_rows


def write_regime_thresholds_json(*, out_path: str, mp_map: dict[str, float], rows: list[RegimeThresholdRow]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_min_proba_by_regime": mp_map,
        "rows": [r.__dict__ for r in rows],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

