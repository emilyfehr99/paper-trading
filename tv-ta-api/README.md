# TradingView-style Technical Analysis API (FastAPI)

Custom **TradingView-style** indicator server (symbol prefixes, resolutions, batch API).
OHLCV comes from **TradingView via tvkit** (default for futures and date-range requests) or **yfinance**
(legacy 60-day intraday cap). Indicators are computed locally using the same conventions as TV charts.

**Ops port for macd-scanner-bot + GBT paper + DI alerts:** `http://127.0.0.1:8010` (`TVTA_PORT=8010`).

**Never use `:8000` for futures ops** — impostors (e.g. Prometheus) and wrong-port recycle stampedes historically killed a healthy `:8010` API. Canonical consumer runbook: `macd-scanner-bot/docs/TVTA.md`.

## Ops reliability (2026-08)

| Rule | Detail |
|------|--------|
| Default bind | `scripts/launch_tvta.sh` → `TVTA_PORT=8010` |
| Adopt | If `:8010` already serves `/health` `status:ok`, adopt — do not kill other ports’ pids |
| `/ws/ta` | OHLCV fetch off event loop (`asyncio.to_thread`); tip `count` capped |
| `/api/ta/history` + `/api/ta/tip` | **`count` ge=50** (422 below) — consumers clamp via `clamp_tvta_history_count` |
| Health | Consumers use `macd-scanner-bot/scripts/tvta_curl_health.sh` (15s `/health`, auth cache) |
| Live smoke | `macd-scanner-bot/scripts/smoke_live_imports.py` must PASS (enrich helpers + limits) |

## Futures symbols (important)

| Bot / yfinance | TradingView continuous | Notes |
|----------------|------------------------|-------|
| `MNQ=F` / `MNQ1` | `CME_MINI:MNQ1!` | Live primary |
| `MES=F` / `MES1` | `CME_MINI:MES1!` | Live primary |
| `ES=F` / `ES1` | **`CME_MINI:ES1!`** | Not `CME:ES1!` — free TV → **403 / 0 bars** |
| `NQ=F` / `NQ1` | **`CME_MINI:NQ1!`** | Same |
| `RTY=F` / `RTY1` | **`CME_MINI:RTY1!`** | Same |
| `M2K` / `MYM` / `YM` / `MCL` / `MGC` | `CME_MINI` / `CBOT_MINI` / `CBOT` / … | Research |

Deep history uses **named quarters** on the same exchange prefix, e.g. `CME_MINI:ESM2024`, `CME_MINI:MNQH2026`.

## Williams Alligator + MACD histogram

Batch indicator names:

- `alligator` — Williams Alligator (pandas_ta SMMA jaws/teeth/lips with TV displacements)
- `macd` or `macd:12-26-9` — MACD line, signal, **histogram** (`macd_hist`), plus:
  - `macd_bar_green` — histogram > 0 (green bar on TV)
  - `macd_cross_green` / `macd_cross_red` — histogram zero-cross
  - `macd_hist_fading` — still green but shrinking (“lighter green”)

Example (live snapshot):

```bash
curl -X POST "http://127.0.0.1:8010/api/ta/batch" \
  -H "Content-Type: application/json" \
  -d '{"resolution":"5","items":[{"symbol":"NASDAQ:AAPL","indicators":["rsi:14","macd","alligator"]}]}'
```

### Backtest / historical

| Endpoint | Use |
|----------|-----|
| `POST /api/ta/enrich_bars` | Upload OHLCV → per-bar MACD + Alligator (local; no TV chart pull) |
| `GET /api/ta/history` | OHLCV + indicators for a window (max ~5000 bars / request) |
| `POST /api/ta/history_bulk` | **Multi-year job** → parquet path (not a giant JSON body) |
| `GET /api/ta/history_bulk/{job_id}` | Job status / bar_count / parquet_path |
| `POST /api/ta/indicators_at` | Point-in-time snapshot (`as_of` unix); optional supplied bars |
| `POST /api/ta/batch` + `as_of` | Batch snapshot as of timestamp |

#### `history_bulk` (years of 5m bars)

Designed to pull ~400k bars without tripping TradingView hard rate-limits:

1. **Month-sized chunks** (default `chunk_days=28`) — far fewer round-trips than weekly
2. **Serial TV WebSocket** by default (`TVTA_BULK_SERIAL_TV=1`) — parallel sessions silently drop segments
3. **Adaptive pacing** (`app/rate_limit.py`) — slows before throttle, speeds up when healthy
4. **Disk segment cache** (`state/tvta_bulk_cache/`) — repeats never re-hit TV; stitches prior weekly cache into months
5. **Continuous-first fetch** (`fast_continuous=true`) — skips dual named-contract work when continuous is enough
6. Indicators via local `enrich_ohlcv_frame` (same as `enrich_bars`)

```bash
# Start job (returns immediately)
curl -s -X POST "http://127.0.0.1:8010/api/ta/history_bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ES=F",
    "resolution": "5",
    "start_ts": 1622523600,
    "end_ts": 1721520000,
    "chunk_days": 28,
    "fast_continuous": true,
    "use_cache": true,
    "write_parquet": true
  }'

# Poll
curl -s "http://127.0.0.1:8010/api/ta/history_bulk/<job_id>"
```

Bot-side archive helper (macd-scanner-bot):

```bash
PYTHONPATH=. python3 scripts/tvta_history_bulk_backfill.py \
  --tickers ES1,NQ1,RTY1 --start 2021-06-01 --end 2026-07-20
```

`enrich_bars` example:

```bash
curl -X POST "http://127.0.0.1:8010/api/ta/enrich_bars" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"NASDAQ:AAPL","resolution":"1","alligator_mode":"sma_shift","bars":[{"t":1717401600,"o":190,"h":191,"l":189.5,"c":190.5,"v":10000}]}'
```

## Features

- `GET /api/ta/indicators` compute latest values for multiple indicators
- `GET /api/ta/series` return a time series for one indicator
- `POST /api/ta/batch` compute many symbols/indicators in one call
- Optional caching (Redis if configured, otherwise in-memory TTL)
- Websocket stream for periodic indicator updates (demo)

## Quickstart

```bash
cd tv-ta-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macd-scanner-bot expects :8010 only (never :8000)
uvicorn app.main:app --host 127.0.0.1 --port 8010
# preferred: bash ../scripts/launch_tvta.sh
# from macd: bash scripts/ensure_tvta.sh && bash scripts/tvta_curl_health.sh 127.0.0.1 8010 poll
```

Open docs at `http://127.0.0.1:8010/docs`.

## Examples

Latest indicators:

```bash
curl "http://127.0.0.1:8010/api/ta/indicators?symbol=NASDAQ:NVDA&indicators=rsi,sma,macd,bbands&resolution=1D"
```

Daytrading set:

```bash
curl "http://127.0.0.1:8010/api/ta/indicators?symbol=NASDAQ:NVDA&resolution=15&indicators=rsi:14,bbands:20-2,donchian:20,willr:14,vwap"
```

Indicator series:

```bash
curl "http://127.0.0.1:8010/api/ta/series?symbol=NASDAQ:NVDA&indicator=rsi&period=14&resolution=1D&count=100"
```

Batch:

```bash
curl -X POST "http://127.0.0.1:8010/api/ta/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "resolution": "1D",
    "items": [
      {"symbol": "NASDAQ:NVDA", "indicators": ["rsi","sma","macd"]},
      {"symbol": "NASDAQ:MSFT", "indicators": ["rsi","bbands"]}
    ]
  }'
```

Websocket (demo):

```bash
python -c "import asyncio, websockets; \
async def main(): \
  async with websockets.connect('ws://127.0.0.1:8010/ws/ta?symbol=NASDAQ:NVDA&indicators=rsi,sma&resolution=1D') as ws: \
    print(await ws.recv()); \
asyncio.run(main())"
```

## Deep futures history

Set `TVTA_OHLCV_PROVIDER=tradingview` (default in `scripts/launch_tvta.sh`).

TVTA computes MACD + Alligator locally. OHLCV is fetched via **tvkit** using named
quarterly contracts (e.g. `CME_MINI:MNQH2026`) when continuous is thin.

| Env | Purpose |
|-----|---------|
| `TVTA_OHLCV_PROVIDER` | `auto` \| `tradingview` \| `yfinance` |
| `TVKIT_SEGMENT_DELAY` | Seconds between segmented requests (bulk sets `0`) |
| `TVTA_BULK_CHUNK_DAYS` | Bulk outer chunk size (default `28`) |
| `TVTA_BULK_SERIAL_TV` | `1` (default) = one TV WS at a time |
| `TVTA_BULK_START_INTERVAL` | Adaptive pacing start delay (default `0.75`) |
| `TVTA_BULK_CACHE_DIR` | Segment cache root (default `../state/tvta_bulk_cache`) |

Example — one month MNQ 5m:

```bash
curl "http://127.0.0.1:8010/api/ta/history?symbol=MNQ%3DF&resolution=5&count=5000\
&start_ts=1709251200&end_ts=1711929599&alligator_mode=williams"
```

Bot archive pull:

```bash
cd macd-scanner-bot
# Proven architecture (same as MES/MNQ 5yr): monitored weekly chunks + backfill_running.lock
nohup bash scripts/run_es_nq_rty_5m_backfill.sh >> logs/es_nq_rty_5m_backfill.nohup.log 2>&1 &

# Lower-level: month loop only (does NOT set backfill_running.lock — prefer the shell above)
PYTHONPATH=. python3 scripts/tvta_monitored_backfill.py \
  --tickers ES1 --start 2021-06-01 --end 2026-07-20 --bar-minutes 5
```

## Notes

- Default OHLCV for futures is TradingView (tvkit). yfinance remains fallback for equities.
- Symbols like `NASDAQ:NVDA` are normalized to `NVDA` for yfinance. Extend `app/symbols.py` for richer mapping.
- Resolutions accept TradingView-style: `1,5,15,30,60,120,240,1D,1W` (and back-compat: `1m,5m,15m,30m,1H,2H,4H`).
- Generic indicators are supported via `pandas_ta` naming, e.g. `stoch:length=14|smooth_k=3|smooth_d=3`.
- Full-size ES/NQ/RTY do **not** require Databento when mapped to `CME_MINI:*`.

## Automation (every 30 minutes)

This repo includes a GitHub Actions workflow at `tv-ta-api/.github/workflows/top-daytrading.yml` that runs on a `*/30` cron and uploads `tv-ta-api/out/top_daytrading.json` as an artifact.
