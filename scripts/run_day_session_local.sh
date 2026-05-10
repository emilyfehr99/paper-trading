#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/emilyfehr8/CascadeProjects/alpaca-paper-day-bot"

cd "$ROOT_DIR"

# Use the project's virtual environment.
exec "$ROOT_DIR/.venv/bin/python3" -m alpaca_day_bot --day-session

