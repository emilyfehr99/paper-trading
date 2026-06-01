#!/bin/bash
# Aetheris Deep Intelligence: Stock Trading Grinder
# Single-mode focus on trading stocks with the $2500 trial portfolio (.env.stocks_trial)

PROJECT_DIR="/Users/emilyfehr8/CascadeProjects/alpaca-paper-day-bot"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
LOG_FILE="$PROJECT_DIR/state/aetheris_launch.log"

MODE="stocks"
ENV_FILE=".env.stocks_trial"

echo "$(date): [AETHERIS START] Mode: $MODE | Env: $ENV_FILE" >> "$LOG_FILE"

# --- PRE-SESSION SLEEP (Avoid spinning restart loop during weekend/night) ---
echo "$(date): [AETHERIS PRE-SESSION] Checking market status..." >> "$LOG_FILE"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/sleep_until_market.py" --env "$ENV_FILE" >> "$LOG_FILE" 2>&1

# --- AUTOMATIC CONTINUOUS RE-LEARNING (EVOLUTION) ---
# Automatically adapt and learn from recent live trade outcomes on every startup!
echo "$(date): [AETHERIS EVOLUTION] Running evolution/retraining..." >> "$LOG_FILE"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evolve_aetheris.py" --env "$ENV_FILE" >> "$LOG_FILE" 2>&1

# Execute with Persistence (launchd will handle restart if it fails)
"$PYTHON_BIN" -m alpaca_day_bot.aetheris_bot --env "$ENV_FILE" --aggressive --day-session >> "$LOG_FILE" 2>&1

echo "$(date): [AETHERIS] Service Cycle Complete." >> "$LOG_FILE"
