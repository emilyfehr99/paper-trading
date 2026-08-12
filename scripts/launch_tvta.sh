#!/usr/bin/env bash
# Start tv-ta-api (TradingView-style indicator server) for the trading bot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_DIR="$ROOT/tv-ta-api"
PORT="${TVTA_PORT:-8010}"
HOST="${TVTA_HOST:-127.0.0.1}"
LOG_FILE="${TVTA_LOG:-$ROOT/state/tvta_api.log}"
PID_FILE="${TVTA_PID_FILE:-$ROOT/state/tvta_api.pid}"

mkdir -p "$(dirname "$LOG_FILE")"

port_occupant() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true
}

PROBE="$ROOT/../macd-scanner-bot/scripts/tvta_curl_health.sh"

# Adopt a healthy listener even when pidfile is stale/mismatched — prevents
# ensure stampede ("port in use / not TVTA" → kill → relaunch death spiral).
adopt_if_healthy() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  # Under 4-stack load /health can take 10–15s; never recycle on a short probe timeout.
  if TVTA_API_CURL_MAX_SECS=25 bash "$PROBE" "$HOST" "$PORT" api 2>/dev/null; then
    echo "$pid" >"$PID_FILE"
    if TVTA_AUTH_CURL_MAX_SECS=60 bash "$PROBE" "$HOST" "$PORT" auth 2>/dev/null; then
      echo "tv-ta-api already running (pid $pid) http://${HOST}:${PORT} + TradingView auth OK"
    else
      echo "tv-ta-api already running (pid $pid) http://${HOST}:${PORT} (auth probe deferred)"
    fi
    return 0
  fi
  # Busy-but-alive: uvicorn listen → adopt even if /health timed out (load stall).
  local cmd
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if echo "$cmd" | grep -q 'uvicorn.*app.main:app'; then
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "$pid" >"$PID_FILE"
      echo "tv-ta-api already running (pid $pid) listen-adopt (health probe slow)"
      return 0
    fi
  fi
  return 1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    if adopt_if_healthy "$old_pid"; then
      exit 0
    fi
    # Only recycle if this pid actually owns OUR port. A stale pidfile must not
    # kill a healthy uvicorn on another port (classic 8000-default vs 8010 ops bug).
    owns_port=0
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | grep -qx "$old_pid"; then
      owns_port=1
    fi
    if [[ "$owns_port" == "1" ]]; then
      echo "tv-ta-api pid $old_pid unhealthy on :$PORT — recycling"
      kill "$old_pid" 2>/dev/null || true
      sleep 1
    else
      echo "tv-ta-api pidfile pid $old_pid not listening on :$PORT — clear pidfile, leave process"
    fi
    rm -f "$PID_FILE"
  fi
fi

occupy="$(port_occupant)"
if [[ -n "$occupy" ]]; then
  if adopt_if_healthy "$occupy"; then
    exit 0
  fi
  echo "Port $PORT already in use by pid $occupy (not healthy TVTA). Stop it or set TVTA_PORT." >&2
  ps -p "$occupy" -o pid=,command= 2>/dev/null || true
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -f "$API_DIR/.venv/bin/python3" ]]; then
    PYTHON_BIN="$API_DIR/.venv/bin/python3"
  elif [[ -f "$ROOT/.venv/bin/python3" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python3"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

# tvkit is required for TradingView OHLCV; install into tv-ta-api venv if missing.
if ! "$PYTHON_BIN" -c "import tvkit" 2>/dev/null; then
  echo "Installing tvkit into $(dirname "$PYTHON_BIN")..."
  "$PYTHON_BIN" -m pip install -q 'tvkit>=0.6.0'
fi

cd "$API_DIR"
if [[ ! -d .venv ]] && [[ ! -f requirements.txt ]]; then
  echo "Missing $API_DIR"
  exit 1
fi

# Legacy: activate venv for any subprocesses (uvicorn uses PYTHON_BIN below).
if [[ -f "$API_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$API_DIR/.venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

# Persisted session auth (TVKIT_SESSION_FILE) — optional browser for one-time bootstrap.
export TVTA_OHLCV_PROVIDER="${TVTA_OHLCV_PROVIDER:-tradingview}"
export TVKIT_SEGMENT_DELAY="${TVKIT_SEGMENT_DELAY:-1.0}"
export TVKIT_SESSION_FILE="${TVKIT_SESSION_FILE:-$ROOT/state/tradingview_session.json}"

nohup env TVTA_OHLCV_PROVIDER="$TVTA_OHLCV_PROVIDER" TVKIT_SEGMENT_DELAY="$TVKIT_SEGMENT_DELAY" \
  TVKIT_SESSION_FILE="$TVKIT_SESSION_FILE" \
  TVKIT_BROWSER="${TVKIT_BROWSER:-}" TVKIT_BROWSER_PROFILE="${TVKIT_BROWSER_PROFILE:-}" \
  TVKIT_AUTH_TOKEN="${TVKIT_AUTH_TOKEN:-}" \
  "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 2

if bash "$PROBE" "$HOST" "$PORT" api; then
  echo "tv-ta-api started pid=$(cat "$PID_FILE") http://${HOST}:${PORT}"
  echo "log: $LOG_FILE"
  if bash "$PROBE" "$HOST" "$PORT" auth; then
    echo "TradingView auth probe: OK"
  else
    echo "WARN: TradingView auth probe failed — run: python3 scripts/tvkit_refresh_session.py --bootstrap" >&2
    tail -15 "$LOG_FILE" 2>/dev/null || true
    exit 1
  fi
else
  echo "tv-ta-api failed to start — check $LOG_FILE"
  tail -20 "$LOG_FILE" 2>/dev/null || true
  exit 1
fi
