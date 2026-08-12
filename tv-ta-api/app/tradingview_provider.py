"""TradingView OHLCV via tvkit (WebSocket history + segmented date ranges)."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd

from .data_provider import Resolution, normalize_resolution

logger = logging.getLogger(__name__)

# Bot/yfinance tickers → TradingView continuous futures
# Note: full-size ES/NQ/RTY resolve on CME_MINI:* on free TV sessions; CME:ES1! etc. return 403/0.
_TV_SYMBOL_MAP: dict[str, str] = {
    "MNQ=F": "CME_MINI:MNQ1!",
    "MES=F": "CME_MINI:MES1!",
    "NQ=F": "CME_MINI:NQ1!",
    "ES=F": "CME_MINI:ES1!",
    "M2K=F": "CME_MINI:M2K1!",
    "MYM=F": "CBOT_MINI:MYM1!",
    "RTY=F": "CME_MINI:RTY1!",
    "YM=F": "CBOT:YM1!",
    "MCL=F": "NYMEX:MCL1!",
    "MGC=F": "COMEX:MGC1!",
    "MNQ1": "CME_MINI:MNQ1!",
    "MES1": "CME_MINI:MES1!",
    "NQ1": "CME_MINI:NQ1!",
    "ES1": "CME_MINI:ES1!",
    "M2K1": "CME_MINI:M2K1!",
    "MYM1": "CBOT_MINI:MYM1!",
    "RTY1": "CME_MINI:RTY1!",
    "YM1": "CBOT:YM1!",
    "MCL1": "NYMEX:MCL1!",
    "MGC1": "COMEX:MGC1!",
    "MNQ1!": "CME_MINI:MNQ1!",
    "MES1!": "CME_MINI:MES1!",
    "NQ1!": "CME_MINI:NQ1!",
    "ES1!": "CME_MINI:ES1!",
    "RTY1!": "CME_MINI:RTY1!",
}

_INTERVAL_SECONDS: dict[str, int] = {
    "1": 60,
    "5": 300,
    "15": 900,
    "30": 1800,
    "60": 3600,
    "120": 7200,
    "240": 14400,
    "1D": 86400,
    "1W": 604800,
}


def normalize_symbol_for_tradingview(symbol: str) -> str:
    """Map yfinance / bot symbols to EXCHANGE:SYMBOL for tvkit."""
    s = symbol.strip()
    if not s:
        return s
    upper = s.upper()
    if ":" in s:
        return s.replace("-", ":") if "-" in s and ":" not in s else s
    if upper in _TV_SYMBOL_MAP:
        return _TV_SYMBOL_MAP[upper]
    if upper.endswith("=F"):
        root = upper[:-2]
        return f"CME_MINI:{root}1!"
    if upper.endswith("1") and len(upper) >= 3:
        return f"CME_MINI:{upper}!"
    prefix = os.getenv("TVTA_SYMBOL_PREFIX", "NASDAQ").strip().upper()
    return f"{prefix}:{upper}" if prefix else upper


def is_futures_like_symbol(symbol: str) -> bool:
    s = symbol.strip().upper()
    if s in _TV_SYMBOL_MAP:
        return True
    if s.endswith("=F") or s.endswith("1!"):
        return True
    if ":" in s:
        exch = s.split(":", 1)[0].upper()
        return exch in {"CME", "CME_MINI", "CBOT", "CBOT_MINI", "NYMEX", "COMEX"}
    return s.endswith("1") and len(s) <= 5


def _bars_to_dataframe(bars: list[Any]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    idx: list[pd.Timestamp] = []
    rows: list[dict[str, float]] = []
    for bar in bars:
        ts = pd.to_datetime(int(bar.timestamp), unit="s", utc=True)
        idx.append(ts)
        rows.append(
            {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume or 0),
            }
        )
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    return df.sort_index()[~df.index.duplicated(keep="last")]


def _ohlcv_client_kwargs() -> dict[str, Any]:
    from .tvkit_session import get_live_auth_kwargs

    return get_live_auth_kwargs()


def _segment_delay() -> float:
    try:
        return float(os.getenv("TVKIT_SEGMENT_DELAY", "1.0"))
    except ValueError:
        return 1.0


def _estimate_bars_for_range(start: datetime, end: datetime, resolution: Resolution) -> int:
    """Conservative bar estimate for count-mode fallback."""
    resolution = normalize_resolution(resolution)
    interval = _INTERVAL_SECONDS.get(resolution, 900)
    span = max(int((end - start).total_seconds()), interval)
    # Futures trade ~23h/day; pad generously for weekends/holidays.
    est = int(span / interval * 0.75) + 400
    return min(max(est, 500), 20000)


def _clip_frame(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts = pd.Timestamp(start).tz_convert("UTC") if start.tzinfo else pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC") if end.tzinfo else pd.Timestamp(end, tz="UTC")
    return df[(df.index >= start_ts) & (df.index <= end_ts)]


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=0).sort_index()
    return df[~df.index.duplicated(keep="last")]


def _close_col(df: pd.DataFrame) -> str:
    if "close" in df.columns:
        return "close"
    if "Close" in df.columns:
        return "Close"
    raise KeyError("no close column")


def _ohlcv_jump_count(df: pd.DataFrame, *, threshold: float = 200.0) -> int:
    """Bars where |Δclose| exceeds threshold — flags blended contract/continuous data."""
    if df.empty:
        return 0
    col = _close_col(df)
    return int(df[col].diff().abs().gt(threshold).sum())


def _pick_single_ohlcv_frame(
    candidates: list[tuple[str, pd.DataFrame]],
    *,
    min_bars: int = 0,
) -> pd.DataFrame:
    """
    Choose one OHLCV series. Never concat continuous + named contracts — that
    interleaves absolute price levels (~300pt jumps on MNQ).
    """
    viable: list[tuple[str, pd.DataFrame, int, int]] = []
    for label, df in candidates:
        if df is None or df.empty:
            continue
        jumps = _ohlcv_jump_count(df)
        n = len(df)
        if n < min_bars:
            continue
        viable.append((label, df, n, jumps))

    if not viable:
        return pd.DataFrame()

    # Prefer fewer jumps; then more bars.
    viable.sort(key=lambda x: (x[3], -x[2]))
    best_label, best_df, best_n, best_jumps = viable[0]
    logger.info(
        "OHLCV source pick: %s bars=%d jumps_gt200=%d (candidates=%s)",
        best_label,
        best_n,
        best_jumps,
        [(l, len(d), _ohlcv_jump_count(d)) for l, d, _, _ in viable],
    )
    return best_df


def _week_date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


async def _gap_fill_same_symbol(
    df: pd.DataFrame,
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
    *,
    tv_symbol: str,
) -> pd.DataFrame:
    """Extend range/head/tail using one TradingView symbol only."""
    resolution = normalize_resolution(resolution)
    step = timedelta(seconds=_INTERVAL_SECONDS.get(resolution, 900))

    for _ in range(8):
        if df.empty:
            break
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        max_ts = df.index.max()
        if max_ts >= end_ts - step * 2:
            break
        tail_start = (max_ts + step).to_pydatetime()
        if tail_start >= end:
            break
        try:
            tail = await _fetch_range_once(symbol, resolution, tail_start, end, tv_symbol=tv_symbol)
        except Exception as exc:
            logger.debug("Tail gap-fill failed for %s: %s", tv_symbol, exc)
            break
        tail = _clip_frame(tail, tail_start, end)
        if tail.empty or tail.index.max() <= max_ts:
            break
        df = _clip_frame(_merge_frames([df, tail]), start, end)

    for _ in range(4):
        if df.empty:
            break
        start_ts = pd.Timestamp(start).tz_convert("UTC")
        min_ts = df.index.min()
        if min_ts <= start_ts + step * 2:
            break
        head_end = (min_ts - step).to_pydatetime()
        if head_end <= start:
            break
        try:
            head = await _fetch_range_once(symbol, resolution, start, head_end, tv_symbol=tv_symbol)
        except Exception as exc:
            logger.debug("Head gap-fill failed for %s: %s", tv_symbol, exc)
            break
        head = _clip_frame(head, start, head_end)
        if head.empty or head.index.min() >= min_ts:
            break
        df = _clip_frame(_merge_frames([head, df]), start, end)

    return df


async def _fetch_continuous_only(
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Continuous contract (MNQ1!) only — no quarterly overlay."""
    tv_symbol = normalize_symbol_for_tradingview(symbol)
    try:
        df = await _fetch_range_once(symbol, resolution, start, end, tv_symbol=tv_symbol)
    except Exception as exc:
        logger.warning("Continuous fetch failed %s: %s", tv_symbol, exc)
        df = pd.DataFrame()
    df = await _gap_fill_same_symbol(df, symbol, resolution, start, end, tv_symbol=tv_symbol)
    return _clip_frame(df, start, end)


async def _fetch_weekly_continuous(
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Weekly chunks on continuous symbol — deep 5m months that fail as one range."""
    start_d = start.astimezone(UTC).date() if start.tzinfo else start.date()
    end_d = end.astimezone(UTC).date() if end.tzinfo else end.date()
    tv_symbol = normalize_symbol_for_tradingview(symbol)
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _week_date_chunks(start_d, end_d):
        cs = datetime.combine(chunk_start, datetime.min.time(), tzinfo=UTC)
        ce = datetime.combine(chunk_end, datetime.max.time().replace(microsecond=0), tzinfo=UTC)
        cs = max(cs, start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC))
        ce = min(ce, end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC))
        if ce <= cs:
            continue
        try:
            part = await _fetch_range_once(symbol, resolution, cs, ce, tv_symbol=tv_symbol)
        except Exception as exc:
            logger.debug("Weekly continuous %s %s→%s: %s", tv_symbol, chunk_start, chunk_end, exc)
            continue
        part = _clip_frame(part, cs, ce)
        if not part.empty:
            frames.append(part)
    return _clip_frame(_merge_frames(frames), start, end)


async def _fetch_range_once(
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
    *,
    tv_symbol: str | None = None,
    max_attempts: int = 1,
) -> pd.DataFrame:
    from tvkit.api.chart.ohlcv import OHLCV

    tv_symbol = tv_symbol or normalize_symbol_for_tradingview(symbol)
    interval = normalize_resolution(resolution)
    delay = _segment_delay()
    kwargs = _ohlcv_client_kwargs()
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            async with OHLCV(**kwargs) as client:
                if kwargs:
                    try:
                        await client.wait_until_ready()
                    except Exception:
                        pass
                acct = client.account
                if acct and attempt == 1:
                    logger.info(
                        "TradingView session tier=%s max_bars=%s symbol=%s",
                        acct.tier,
                        acct.max_bars,
                        tv_symbol,
                    )
                bars = await client.get_historical_ohlcv(
                    tv_symbol,
                    interval,
                    start=start,
                    end=end,
                    segment_delay=delay,
                )
                df = _bars_to_dataframe(bars)
                if df is not None and not df.empty:
                    return df
                last_exc = RuntimeError(f"No historical bars received for symbol {tv_symbol}")
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "TV range fetch attempt %d/%d failed %s %s→%s: %s",
                attempt,
                max_attempts,
                tv_symbol,
                start,
                end,
                exc,
            )
        if attempt < max_attempts:
            await asyncio.sleep(0.75 * attempt)

    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


async def _fetch_contract_segments(
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch via named quarterly contracts (deep history without continuous symbol)."""
    from .futures_contracts import contract_segments_for_range

    if not is_futures_like_symbol(symbol):
        return pd.DataFrame()

    segments = contract_segments_for_range(symbol, start, end)
    if not segments:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    from .futures_contracts import _contract_tv_symbol, _parse_futures_root

    parsed = _parse_futures_root(symbol)
    exchange, root = parsed if parsed else ("CME_MINI", "MNQ")

    def _prior_quarter_symbol(tv_sym: str) -> str | None:
        if ":" not in tv_sym:
            return None
        exch, rest = tv_sym.split(":", 1)
        if len(rest) < 6 or not rest[-4:].isdigit():
            return None
        code, yr = rest[-5], int(rest[-4:])
        q = {"H": 3, "M": 6, "U": 9, "Z": 12}
        if code not in q:
            return None
        months = [3, 6, 9, 12]
        idx = months.index(q[code])
        if idx == 0:
            pm, py = 12, yr - 1
        else:
            pm, py = months[idx - 1], yr
        return _contract_tv_symbol(exch, root, py, pm)

    for seg in segments:
        part = pd.DataFrame()
        # Primary first; prior only if primary empty (roll overlap). Mid-quarter
        # hits primary and breaks — no doubled WS.
        for tv_sym in (seg.tv_symbol, _prior_quarter_symbol(seg.tv_symbol)):
            if not tv_sym:
                continue
            try:
                part = await _fetch_range_once(
                    symbol,
                    resolution,
                    seg.start,
                    seg.end,
                    tv_symbol=tv_sym,
                )
            except Exception as exc:
                logger.warning("Contract fetch failed %s: %s", tv_sym, exc)
                part = pd.DataFrame()
            if not part.empty:
                frames.append(part)
                logger.info(
                    "Contract OHLCV %s bars=%d (%s → %s)",
                    tv_sym,
                    len(part),
                    part.index.min(),
                    part.index.max(),
                )
                break
    return _clip_frame(_merge_frames(frames), start, end)


async def _fetch_contract_daily(
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Named-contract 1m: TV only serves sparse daily windows — fetch day-by-day."""
    from .futures_contracts import _contract_tv_symbol, _parse_futures_root, contract_segments_for_range

    if not is_futures_like_symbol(symbol):
        return pd.DataFrame()
    segments = contract_segments_for_range(symbol, start, end)
    if not segments:
        return pd.DataFrame()

    parsed = _parse_futures_root(symbol)
    exchange, root = parsed if parsed else ("CME_MINI", "MNQ")

    def _prior_quarter_symbol(tv_sym: str) -> str | None:
        if ":" not in tv_sym:
            return None
        exch, rest = tv_sym.split(":", 1)
        if len(rest) < 6 or not rest[-4:].isdigit():
            return None
        code, yr = rest[-5], int(rest[-4:])
        q = {"H": 3, "M": 6, "U": 9, "Z": 12}
        if code not in q:
            return None
        months = [3, 6, 9, 12]
        idx = months.index(q[code])
        pm, py = (12, yr - 1) if idx == 0 else (months[idx - 1], yr)
        return _contract_tv_symbol(exch, root, py, pm)

    frames: list[pd.DataFrame] = []
    start_d = start.astimezone(UTC).date()
    end_d = end.astimezone(UTC).date()
    for seg in segments:
        syms = [seg.tv_symbol]
        prior = _prior_quarter_symbol(seg.tv_symbol)
        if prior:
            syms.append(prior)
        cur = max(seg.start.date(), start_d)
        last = min(seg.end.date(), end_d)
        while cur <= last:
            cs = datetime.combine(cur, time.min, tzinfo=UTC)
            ce = datetime.combine(cur, time(23, 59, 59), tzinfo=UTC)
            for tv_sym in syms:
                try:
                    part = await _fetch_range_once(symbol, resolution, cs, ce, tv_symbol=tv_sym)
                except Exception:
                    part = pd.DataFrame()
                if not part.empty:
                    frames.append(part)
                    break
            cur += timedelta(days=1)
    return _clip_frame(_merge_frames(frames), start, end)


async def _fetch_count_once(
    symbol: str,
    resolution: Resolution,
    count: int,
) -> pd.DataFrame:
    from tvkit.api.chart.ohlcv import OHLCV

    tv_symbol = normalize_symbol_for_tradingview(symbol)
    interval = normalize_resolution(resolution)
    kwargs = _ohlcv_client_kwargs()

    async with OHLCV(**kwargs) as client:
        if kwargs:
            try:
                await client.wait_until_ready()
            except Exception:
                pass
        bars = await client.get_historical_ohlcv(
            tv_symbol,
            interval,
            bars_count=min(max(int(count), 50), 20000),
        )
        return _bars_to_dataframe(bars)


async def _fetch_range_async(
    symbol: str,
    resolution: Resolution,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV for [start, end] from a **single** coherent price series.

    Named quarterly contracts and continuous MNQ1! must not be merged — that
    produces alternating absolute price levels on 5m bars.
    """
    start = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
    end = end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC)
    if end <= start:
        return pd.DataFrame()

    est = _estimate_bars_for_range(start, end, resolution)
    min_viable = max(int(est * 0.15), 50)

    if normalize_resolution(resolution) == "1":
        df = await _fetch_contract_daily(symbol, resolution, start, end)
        if len(df) < min_viable:
            try:
                tail = _clip_frame(
                    await _fetch_count_once(symbol, resolution, max(est, 500)),
                    start,
                    end,
                )
                if not tail.empty:
                    df = _clip_frame(_merge_frames([df, tail]), start, end)
            except Exception as exc:
                logger.debug("1m continuous tail failed for %s: %s", symbol, exc)
        if not df.empty:
            logger.info(
                "TradingView 1m contract-daily %s bars=%d (%s → %s)",
                symbol,
                len(df),
                df.index.min(),
                df.index.max(),
            )
        return _clip_frame(df, start, end)

    contract_df = pd.DataFrame()
    try:
        contract_df = await _fetch_contract_segments(symbol, resolution, start, end)
    except Exception as exc:
        logger.warning("Contract segment fetch failed for %s: %s", symbol, exc)

    # Ground truth: continuous (ES1!) only holds ~5000 recent tip bars. For deep
    # history (>90d), chasing continuous after empty contracts wastes 10–20s/day
    # of empty retries (and never fills roll gaps — those need 60m on the next tip).
    continuous_df = pd.DataFrame()
    age_days = (datetime.now(UTC).date() - end.date()).days
    deep_history = age_days > 90
    if contract_df.empty and not deep_history:
        continuous_df = await _fetch_continuous_only(symbol, resolution, start, end)

    df = _pick_single_ohlcv_frame(
        [
            ("contracts", contract_df),
            ("continuous", continuous_df),
        ],
        min_bars=0,
    )

    # Thin fill via continuous only for recent windows.
    if contract_df.empty and not deep_history and len(df) < int(est * 0.45):
        weekly = await _fetch_weekly_continuous(symbol, resolution, start, end)
        if len(weekly) > len(df) and _ohlcv_jump_count(weekly) <= max(_ohlcv_jump_count(df), 20):
            df = weekly

    # Re-pick if jumps are bad (blended/wrong series).
    if _ohlcv_jump_count(df) > 20:
        df = _pick_single_ohlcv_frame(
            [
                ("contracts", contract_df),
                ("continuous", continuous_df),
            ],
            min_bars=min_viable,
        )

    # Last resort: wide count pull on continuous — recent only.
    if (not deep_history) and (df.empty or len(df) < min_viable):
        try:
            wide = _clip_frame(
                await _fetch_count_once(symbol, resolution, max(est, 500)),
                start,
                end,
            )
            if not wide.empty and _ohlcv_jump_count(wide) <= max(_ohlcv_jump_count(df), 25):
                df = wide
        except Exception as exc:
            logger.error("TradingView wide count fetch failed for %s: %s", symbol, exc)

    # Outer retry — recent only (deep empty → fail fast; caller uses 60m fallback).
    if df.empty and not deep_history:
        await asyncio.sleep(1.5)
        logger.warning(
            "TradingView empty range one-shot retry for %s %s→%s",
            symbol,
            start.date(),
            end.date(),
        )
        try:
            retry_df = await _fetch_weekly_continuous(symbol, resolution, start, end)
        except Exception as exc:
            logger.warning("Empty-range weekly retry failed: %s", exc)
            retry_df = pd.DataFrame()
        if retry_df.empty:
            try:
                retry_df = await _fetch_contract_segments(symbol, resolution, start, end)
            except Exception as exc:
                logger.warning("Empty-range contract retry failed: %s", exc)
                retry_df = pd.DataFrame()
        if not retry_df.empty:
            df = _clip_frame(retry_df, start, end)

    if not df.empty:
        logger.info(
            "TradingView range %s %s bars=%d jumps_gt200=%d (%s → %s)",
            symbol,
            resolution,
            len(df),
            _ohlcv_jump_count(df),
            df.index.min(),
            df.index.max(),
        )
    return _clip_frame(df, start, end)


async def _fetch_count_async(
    symbol: str,
    resolution: Resolution,
    count: int,
) -> pd.DataFrame:
    return await _fetch_count_once(symbol, resolution, count)


def _run_coro_sync(coro, *, timeout_s: float | None = None):
    """Run an async TV fetch from sync code — safe if a loop is already running.

    Always bounded: a hung TradingView websocket used to block forever via
    ``Future.result()`` with no timeout, exhaust ``_ohlcv_sem`` + the FastAPI
    threadpool, and freeze all 4 bots' exit loops on /history and /tip.
    """
    import os
    import concurrent.futures

    if timeout_s is None:
        try:
            timeout_s = float(os.getenv("TVTA_FETCH_TIMEOUT_S", "45") or 45)
        except (TypeError, ValueError):
            timeout_s = 45.0
    timeout_s = max(5.0, float(timeout_s))

    def _in_thread():
        return asyncio.run(coro)

    # Never ``with ThreadPoolExecutor`` here: on timeout, ``__exit__`` waits for
    # the hung worker and reintroduces the multi-minute freeze.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_in_thread).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def fetch_ohlcv_tradingview(
    symbol: str,
    resolution: Resolution,
    count: int,
    extra_bars: int = 200,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    """Sync entry: range mode if start/end set, else count mode."""
    import os

    try:
        tip_ish = start_ts is None and end_ts is None and int(count) + int(extra_bars) <= 300
        default_to = "30" if tip_ish else "45"
        timeout_s = float(os.getenv("TVTA_FETCH_TIMEOUT_S", default_to) or default_to)
    except (TypeError, ValueError):
        timeout_s = 45.0
    try:
        if start_ts is not None and end_ts is not None:
            start = datetime.fromtimestamp(int(start_ts), tz=UTC)
            end = datetime.fromtimestamp(int(end_ts), tz=UTC)
            return _run_coro_sync(
                _fetch_range_async(symbol, resolution, start, end),
                timeout_s=timeout_s,
            )
        n = min(max(int(count) + int(extra_bars), 50), 20000)
        return _run_coro_sync(
            _fetch_count_async(symbol, resolution, n), timeout_s=timeout_s
        )
    except Exception as exc:
        logger.error("TradingView fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()
