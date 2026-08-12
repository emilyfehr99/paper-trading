# tv-ta-api host (futures data plane)

Stock **Aetheris** bot removed 2026-06-11. This directory exists to run **TVTA** for `macd-scanner-bot`: Soft DNA / Fuse / Eval paper desks, DI×EMA alerts (RTY/ES/NQ), and archive backfill.

## Layout

| Path | Purpose |
|------|---------|
| `tv-ta-api/` | FastAPI indicator server (MACD, Alligator, STOCH V RSI, enrich, history_bulk) |
| `scripts/launch_tvta.sh` | Start / adopt on **`:8010`** (never `:8000`) |
| `state/tradingview_session.json` | TradingView session (do not delete) |
| `state/tvta_api.log` | Server log |
| `state/tvta_api.pid` | Running PID |

## Start (usually automatic)

Futures / paper supervisors and `com.macd.tvta` call `macd-scanner-bot/scripts/ensure_tvta.sh` → this launch script.

```bash
cd /Users/emilyfehr8/CascadeProjects/macd-scanner-bot
./scripts/ensure_tvta.sh
bash scripts/tvta_curl_health.sh 127.0.0.1 8010 poll
python3 scripts/audit_tvta_futures.py
```

Default port: **8010** only (`TVTA_PORT` / `TVTA_BASE_URL`). Canonical ops: `macd-scanner-bot/docs/TVTA.md`.

## Session refresh

```bash
cd /Users/emilyfehr8/CascadeProjects/macd-scanner-bot
python3 scripts/tvkit_refresh_session.py --bootstrap
```
