# TradingView-style Technical Analysis API (FastAPI)

Server-side indicator computation for feeding TradingView charts/widgets.

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

uvicorn app.main:app --reload --port 8000
```

Open docs at `http://127.0.0.1:8000/docs`.

## Examples

Latest indicators:

```bash
curl "http://127.0.0.1:8000/api/ta/indicators?symbol=NASDAQ:NVDA&indicators=rsi,sma,macd,bbands&resolution=1D"
```

Daytrading set (the ones you listed):

```bash
curl "http://127.0.0.1:8000/api/ta/indicators?symbol=NASDAQ:NVDA&resolution=15&indicators=rsi:14,bbands:20-2,donchian:20,willr:14,vwap"
```

Indicator series:

```bash
curl "http://127.0.0.1:8000/api/ta/series?symbol=NASDAQ:NVDA&indicator=rsi&period=14&resolution=1D&count=100"
```

Batch:

```bash
curl -X POST "http://127.0.0.1:8000/api/ta/batch" \
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
  async with websockets.connect('ws://127.0.0.1:8000/ws/ta?symbol=NASDAQ:NVDA&indicators=rsi,sma&resolution=1D') as ws: \
    print(await ws.recv()); \
asyncio.run(main())"
```

## Notes

- Data provider is `yfinance` for development. For production, swap in Polygon/Alpaca/Intrinio/etc.
- Symbols like `NASDAQ:NVDA` are normalized to `NVDA` for yfinance. Extend `app/symbols.py` for richer mapping.
- Resolutions accept TradingView-style: `1,5,15,30,60,120,240,1D,1W` (and back-compat: `1m,5m,15m,30m,1H,2H,4H`).
- Generic indicators are supported via `pandas_ta` naming, e.g. `stoch:length=14|smooth_k=3|smooth_d=3`.

## Automation (every 30 minutes)

This repo includes a GitHub Actions workflow at `tv-ta-api/.github/workflows/top-daytrading.yml` that runs on a `*/30` cron and uploads `tv-ta-api/out/top_daytrading.json` as an artifact.

