#!/usr/bin/env bash
# Run the Alpaca paper-trading bot continuously (mock paper money).
# Usage: from repo root: ./scripts/run-paper-bot.sh
# Or:    bash scripts/run-paper-bot.sh --observe-only

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p state

if [[ -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

exec python3 -m alpaca_day_bot "$@"
