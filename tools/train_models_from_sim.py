from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from alpaca_day_bot.ml.train import train_and_save


def _md_block(title: str, payload: dict) -> str:
    lines = [f"### {title}", ""]
    if payload.get("skipped"):
        lines.append(f"- **skipped**: true")
        lines.append(f"- **reason**: `{payload.get('skip_reason')}`")
        lines.append(f"- **n_labeled**: {payload.get('n_labeled')}")
    else:
        m = payload.get("metrics") or {}
        lines.append(f"- **task**: `{payload.get('task', 'classification')}`")
        lines.append(f"- **target_mode**: `{payload.get('target_mode', 'binary')}`")
        lines.append(f"- **provider**: `{payload.get('provider')}`")
        lines.append(f"- **dataset_kind**: `{payload.get('dataset_kind')}`")
        lines.append(f"- **rows_seen**: {payload.get('rows_seen')}")
        lines.append(f"- **test_n**: {m.get('n')}")
        if str(payload.get("task") or "").lower() == "regression":
            lines.append(f"- **test_rmse**: {m.get('rmse')}")
            lines.append(f"- **recommended_regression_min**: {payload.get('recommended_regression_min')}")
        else:
            lines.append(f"- **test_auc**: {m.get('auc')}")
            lines.append(f"- **test_acc**: {m.get('acc')}")
            lines.append(f"- **test_pos_rate**: {m.get('pos_rate')}")
            lines.append(f"- **recommended_min_proba**: {payload.get('recommended_min_proba')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="state/ledgers/sim_rollup.sqlite3")
    ap.add_argument("--out-dir", default="state/models")
    ap.add_argument("--report-path", default="reports/sim_training_latest.md")
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-class-count", type=int, default=50)
    ap.add_argument("--min-horizon-minutes", type=float, default=15.0)
    ap.add_argument(
        "--target-mode",
        default="binary",
        choices=["binary", "beat_fee_bps", "regression_r", "regression_return_pct"],
    )
    ap.add_argument("--min-edge-bps", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    buy_out = str(out_dir / "latest_buy_sim.joblib")
    short_out = str(out_dir / "latest_short_sim.joblib")

    buy_meta = train_and_save(
        db_path=str(args.db),
        out_path=buy_out,
        min_horizon_minutes=float(args.min_horizon_minutes),
        min_rows=int(args.min_rows),
        action="BUY",
        min_class_count=int(args.min_class_count),
        dataset_source="sim",
        target_mode=str(args.target_mode),
        min_edge_bps=float(args.min_edge_bps),
    )
    short_meta = train_and_save(
        db_path=str(args.db),
        out_path=short_out,
        min_horizon_minutes=float(args.min_horizon_minutes),
        min_rows=int(args.min_rows),
        action="SHORT",
        min_class_count=int(args.min_class_count),
        dataset_source="sim",
        target_mode=str(args.target_mode),
        min_edge_bps=float(args.min_edge_bps),
    )

    # Persist JSON sidecars for easy inspection.
    Path(buy_out).with_suffix(".json").write_text(json.dumps(buy_meta, indent=2), encoding="utf-8")
    Path(short_out).with_suffix(".json").write_text(json.dumps(short_meta, indent=2), encoding="utf-8")

    md = []
    md.append(f"# Sim model training\n")
    md.append(f"- **trained_at_utc**: {datetime.now(tz=timezone.utc).isoformat()}\n")
    md.append(f"- **db**: `{args.db}`\n")
    md.append("")
    md.append(_md_block("BUY (sim)", buy_meta))
    md.append(_md_block("SHORT (sim)", short_meta))
    report_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

