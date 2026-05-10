#!/bin/bash

PROJECT_DIR="/Users/emilyfehr8/CascadeProjects/alpaca-paper-day-bot"
cd "$PROJECT_DIR"

echo "============================================================"
echo " 🚀 TRIPLE THREAT TRADING DASHBOARD"
echo "============================================================"

echo ""
echo "🔵 [1/3] CRYPTO (24/7 - \$2,500 Trial)"
export ENV_FILE=.env.crypto
export STATE_DIR=state/crypto
export REPORTS_DIR=reports/crypto
python3 scripts/day_update.py

echo ""
echo "🟢 [2/3] STOCKS (Big Cap - Standard)"
export ENV_FILE=.env.stocks
export STATE_DIR=state/stocks
export REPORTS_DIR=reports/stocks
python3 scripts/day_update.py

echo ""
echo "🟡 [3/3] STOCKS-TRIAL (Real Money Sim - \$2,500)"
export ENV_FILE=.env.stocks_trial
export STATE_DIR=state/stocks_trial
export REPORTS_DIR=reports/stocks_trial
python3 scripts/day_update.py

echo ""
echo "============================================================"
echo " ✅ All systems operational. Reports recorded in /reports/"
echo "============================================================"
