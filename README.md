# Alpaca paper-trading day bot (1-week)

Local-only, **paper-only** trading bot for US equities using Alpaca’s official Python SDK (`alpaca-py`).

## Safety
- **Paper trading only**: this project always initializes Alpaca with `paper=True`.
- Never put real credentials in git. Use `.env` (see `.env.example`).

## Setup

### 1) Create an Alpaca paper account + API keys
- Create API keys in the Alpaca paper dashboard.
- Put them in a local `.env` file (copy `.env.example`).

### 2) Create a virtualenv and install

```bash
cd alpaca-paper-day-bot
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

## Run

By default the bot loads **1-minute bars over REST** (`MARKET_DATA_MODE=rest`) so it does not open Alpaca’s **market-data websocket** alongside the **trading** websocket (many accounts hit `connection limit exceeded` with both). For live streaming bars, set `MARKET_DATA_MODE=websocket` in `.env` if your plan allows it.

### Observe-only (recommended first)
Connects to market data, computes features/signals, and logs — **no orders**.

```bash
python -m alpaca_day_bot --observe-only
```

### Paper live (orders enabled)

```bash
python -m alpaca_day_bot
```

### Run continuously (mock paper trading, local only)

No Colab—this runs on your Mac against Alpaca **paper** API (`paper=True`). Keep the process running during market hours (or 24/7; the bot only trades inside its configured window).

1. Copy `.env.example` → `.env` and add your **paper** API keys.
2. Install once (see above), then:

```bash
cd alpaca-paper-day-bot
source .venv/bin/activate   # if you use a venv
python3 -m alpaca_day_bot
```

Leave that terminal open, or run in the background (create `state/` first so the log file path exists):

```bash
mkdir -p state
nohup python3 -m alpaca_day_bot >> state/bot.log 2>&1 &
```

Or use **tmux** / **screen** so it survives closing the terminal. On macOS you can also use **launchd** or **cron** with `--day-session` to start each trading day.

### One trading-day session (auto start/stop)
Waits until the configured trading window opens, trades until it closes, then exits. This is handy if you want to schedule it daily with `cron`/`launchd`.

```bash
python -m alpaca_day_bot --day-session
```

### GitHub Actions (scheduled tick, every minute on weekdays)

The workflow `.github/workflows/paper-scheduled-tick.yml` runs **`python -m alpaca_day_bot --scheduled-tick`** every **minute, Monday–Friday (UTC cron)**—about as often as GitHub’s `schedule` allows (runs can still be delayed slightly when GitHub is busy). Each run performs one REST bar warmup, one signal scan, and at most one round of order logic, then exits. The `state/` folder (SQLite ledger + lock) is cached **per New York calendar day** so daily trade counts and cooldowns survive between runs.

1. Push this repo to GitHub.
2. **Settings → Secrets and variables → Actions**: add **`APCA_API_KEY_ID`** and **`APCA_API_SECRET_KEY`** (paper keys).
3. Enable Actions on the repo if prompted. Use **Actions → Paper scheduled tick → Run workflow** to test manually.

**Caveats:** GitHub cron is UTC and can drift a few minutes; the bot still only acts inside `TRADE_START` / `TRADE_END`. Do not run the same paper keys in Actions and on your laptop at the same time (Alpaca connection limits + duplicate orders). For a full trading-day loop with websockets, prefer `--day-session` on a VPS or your machine.

## Outputs
- `./state/ledger.sqlite3`: fills/orders/positions ledger
- `./reports/`: daily + weekly summaries
- `./state/logs/`: structured JSON logs

## Notes
- This is **not financial advice** and is for paper simulation and research only.
- V1 is deliberately conservative: **long-only**, strict limits, circuit breakers, and reconciliation on every loop.

