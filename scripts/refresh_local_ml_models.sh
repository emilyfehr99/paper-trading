#!/usr/bin/env bash
# Retrain ML artifacts under state/models/ from state/ledger.sqlite3 (same knobs as CI smoke).
# If training skips with n_labeled=0, backfill labels first (needs network):
#   alpaca-backfill-forward-labels --db state/ledger.sqlite3 --limit 500
#   alpaca-backfill-triple-barrier --db state/ledger.sqlite3 --limit 300
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="${ROOT}/state/ledger.sqlite3"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
if [[ ! -f "$DB" ]]; then
  echo "Missing ledger DB: $DB" >&2
  exit 1
fi
mkdir -p "${ROOT}/state/models"
cd "$ROOT"
for spec in "BUY:state/models/latest_buy.joblib" "SHORT:state/models/latest_short.joblib" "BUY:state/models/latest.joblib"; do
  IFS=: read -r act outpath <<<"$spec"
  echo "Training action=$act -> $outpath"
  "$PY" -m alpaca_day_bot.ml.train \
    --db "$DB" \
    --out "$outpath" \
    --action "$act" \
    --min-horizon-minutes 15 \
    --min-rows 10 \
    --min-class-count 5 \
    --dataset auto || true
done
echo "Done. Inspect state/models/*.json for skip_reason if rows are insufficient."
