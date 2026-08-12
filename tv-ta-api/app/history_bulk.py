"""Parallel multi-year history fetch with adaptive pacing + disk segment cache.

Avoids TradingView hard rate-limits by:
  1. Adaptive spacing / concurrency (see rate_limit.AdaptiveTvLimiter)
  2. Disk cache — identical segments never re-hit TV
  3. Writing parquet / returning paths instead of giant JSON bodies

Indicators are computed locally (enrich_ohlcv_frame) — no TV indicator API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .data_provider import Resolution, normalize_resolution
from .rate_limit import AdaptiveTvLimiter
from .ta_enrich import enrich_ohlcv_frame, frame_to_points
from .tradingview_provider import (
    _fetch_continuous_only,
    _fetch_range_async,
    _merge_frames,
)
from .ohlcv_validate import OhlcvIntegrityError, validate_ohlcv_frame

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = asyncio.Lock()


def _cache_root() -> Path:
    raw = os.getenv("TVTA_BULK_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    # Default beside session file / TVTA state
    here = Path(__file__).resolve().parents[2]  # alpaca-paper-day-bot
    return here / "state" / "tvta_bulk_cache"


def _chunk_days() -> int:
    try:
        return max(7, min(62, int(os.getenv("TVTA_BULK_CHUNK_DAYS", "28"))))
    except ValueError:
        return 28


def _date_chunks(start: date, end: date, *, chunk_days: int | None = None) -> list[tuple[date, date]]:
    """Split range into ~month-sized chunks (default 28d). Fewer TV round-trips than weekly."""
    width = chunk_days if chunk_days is not None else _chunk_days()
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=width - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _stitch_from_finer_cache(
    cache_root: Path,
    symbol: str,
    resolution: str,
    start: date,
    end: date,
    alligator_mode: str,
) -> pd.DataFrame:
    """Reuse already-fetched weekly segment parquets inside a larger chunk."""
    safe = symbol.replace("=", "_").replace(":", "_").replace("/", "_")
    folder = cache_root / safe
    if not folder.is_dir():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob(f"{safe}_{resolution}_*_{alligator_mode}.parquet")):
        # Prefer regex on filename: SYM_res_YYYY-MM-DD_YYYY-MM-DD_mode.parquet
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_", path.name)
        if not m:
            continue
        try:
            a = date.fromisoformat(m.group(1))
            b = date.fromisoformat(m.group(2))
        except ValueError:
            continue
        if b < start or a > end:
            continue
        hit = _df_from_cache(path)
        if hit is not None and not hit.empty:
            frames.append(hit)
    if not frames:
        return pd.DataFrame()
    merged = _merge_frames(frames)
    if merged.empty:
        return merged
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time(), tzinfo=UTC))
    end_ts = pd.Timestamp(datetime.combine(end, datetime.max.time().replace(microsecond=0), tzinfo=UTC))
    out = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]
    # Require decent coverage before accepting stitch (avoid half-month holes).
    span_days = max((end - start).days + 1, 1)
    min_bars = max(200, int(span_days * 0.45 * (24 * 60 / 5) * 0.5))  # rough globex floor
    if len(out) < min_bars:
        return pd.DataFrame()
    return out


@dataclass
class BulkJobRequest:
    symbol: str
    resolution: Resolution
    start_ts: int
    end_ts: int
    alligator_mode: str = "williams"
    concurrency: int | None = None
    use_cache: bool = True
    write_parquet: bool = True
    out_dir: str | None = None
    chunk_days: int | None = None
    fast_continuous: bool = True


def _segment_cache_path(
    root: Path,
    symbol: str,
    resolution: str,
    start: date,
    end: date,
    alligator_mode: str,
) -> Path:
    safe = symbol.replace("=", "_").replace(":", "_").replace("/", "_")
    name = f"{safe}_{resolution}_{start.isoformat()}_{end.isoformat()}_{alligator_mode}.parquet"
    return root / safe / name


def _df_from_cache(path: Path) -> pd.DataFrame | None:
    """Return cached frame, or None on miss. Legacy .empty markers are deleted (miss)."""
    empty_marker = path.with_suffix(".empty")
    if empty_marker.exists():
        empty_marker.unlink(missing_ok=True)
        return None
    if not path.exists() or path.stat().st_size < 64:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Bulk cache read failed %s: %s", path, exc)
        return None
    if df is None or df.empty:
        return None
    return df


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    """Persist successful segments only — never cache empty as success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_marker = path.with_suffix(".empty")
    # Remove any legacy empty markers (they poisoned retries).
    if empty_marker.exists():
        empty_marker.unlink(missing_ok=True)
    if df is None or df.empty:
        return
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.replace(path)


def get_job(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    items = sorted(_JOBS.values(), key=lambda j: j.get("created_at", 0), reverse=True)
    return items[:limit]


async def start_bulk_job(req: BulkJobRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    job = {
        "job_id": job_id,
        "status": "queued",
        "symbol": req.symbol,
        "resolution": req.resolution,
        "start_ts": req.start_ts,
        "end_ts": req.end_ts,
        "created_at": now,
        "updated_at": now,
        "segments_total": 0,
        "segments_done": 0,
        "segments_cached": 0,
        "empty_segments": 0,
        "bar_count": 0,
        "elapsed_sec": 0.0,
        "parquet_path": None,
        "error": None,
        "warning": None,
        "serial_tv": None,
        "limiter": {},
        "preview_points": None,
    }
    async with _JOBS_LOCK:
        _JOBS[job_id] = job
    asyncio.create_task(_run_bulk_job(job_id, req))
    return dict(job)


async def _run_bulk_job(job_id: str, req: BulkJobRequest) -> None:
    t0 = time.perf_counter()
    job = _JOBS[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    # Prefer bulk limiter over per-segment TVKIT_SEGMENT_DELAY sleeps.
    os.environ.setdefault("TVKIT_SEGMENT_DELAY", "0")
    # Serial TV WebSocket by default — parallel sessions silently return empty weeks.
    serial_tv = os.getenv("TVTA_BULK_SERIAL_TV", "1").strip().lower() not in ("0", "false", "no")

    try:
        resolution = normalize_resolution(req.resolution)
        start = datetime.fromtimestamp(int(req.start_ts), tz=UTC)
        end = datetime.fromtimestamp(int(req.end_ts), tz=UTC)
        if end <= start:
            raise ValueError("end_ts must be after start_ts")

        chunks = _date_chunks(start.date(), end.date(), chunk_days=req.chunk_days)
        job["segments_total"] = len(chunks)
        job["chunk_days"] = req.chunk_days if req.chunk_days is not None else _chunk_days()
        limiter = AdaptiveTvLimiter()
        if req.concurrency is not None:
            limiter.concurrency = max(
                limiter.min_concurrency,
                min(int(req.concurrency), limiter.max_concurrency),
            )
        if serial_tv:
            limiter.concurrency = 1
            limiter.max_concurrency = 1

        cache_root = _cache_root()
        frames: list[pd.DataFrame] = []
        tv_lock = asyncio.Lock()
        empty_segments = 0

        async def _fetch_tv(cs: date, ce: date) -> pd.DataFrame:
            cs_dt = datetime.combine(cs, datetime.min.time(), tzinfo=UTC)
            ce_dt = datetime.combine(ce, datetime.max.time().replace(microsecond=0), tzinfo=UTC)
            cs_dt = max(cs_dt, start)
            ce_dt = min(ce_dt, end)
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                await limiter.acquire()
                try:
                    async with tv_lock:
                        raw = pd.DataFrame()
                        # Fast path: continuous-only (skips named-contract dual fetch).
                        if req.fast_continuous:
                            try:
                                raw = await _fetch_continuous_only(req.symbol, resolution, cs_dt, ce_dt)
                            except Exception as exc:
                                logger.debug("continuous-only failed %s: %s", req.symbol, exc)
                                raw = pd.DataFrame()
                        span_days = max((ce - cs).days + 1, 1)
                        min_ok = max(80, int(span_days * 40))  # ~half a light globex day * days
                        if raw is None or raw.empty or len(raw) < min_ok:
                            raw = await _fetch_range_async(req.symbol, resolution, cs_dt, ce_dt)
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if any(k in msg for k in ("rate", "throttl", "429", "too many", "limit")):
                        limiter.throttle(reason=str(exc)[:120])
                    else:
                        limiter.throttle(reason=f"fetch_error:{type(exc).__name__}")
                    await asyncio.sleep(min(limiter.interval, 5.0) * attempt)
                    continue
                if raw is not None and not raw.empty:
                    limiter.success()
                    return raw
                if attempt < 2:
                    await asyncio.sleep(0.5 * attempt)
                    continue
                limiter.success()
                return pd.DataFrame()
            if last_exc is not None:
                raise last_exc
            return pd.DataFrame()

        i = 0
        while i < len(chunks):
            batch_n = max(limiter.min_concurrency, min(limiter.concurrency, len(chunks) - i))
            batch = chunks[i : i + batch_n]

            async def _one(pair: tuple[date, date]) -> tuple[pd.DataFrame, bool]:
                cs, ce = pair
                path = _segment_cache_path(
                    cache_root, req.symbol, resolution, cs, ce, req.alligator_mode
                )
                start_i = int(pd.Timestamp(cs, tz="UTC").timestamp())
                end_i = int(pd.Timestamp(ce, tz="UTC").timestamp()) + 86399

                if req.use_cache:
                    hit = _df_from_cache(path)
                    if hit is not None and not hit.empty:
                        hit = validate_ohlcv_frame(
                            hit,
                            symbol=req.symbol,
                            start_ts=start_i,
                            end_ts=end_i,
                            resolution=resolution,
                        )
                        return hit, True
                    # Reuse prior weekly cache files inside this month chunk.
                    stitched = _stitch_from_finer_cache(
                        cache_root, req.symbol, resolution, cs, ce, req.alligator_mode
                    )
                    if not stitched.empty:
                        stitched = validate_ohlcv_frame(
                            stitched,
                            symbol=req.symbol,
                            start_ts=start_i,
                            end_ts=end_i,
                            resolution=resolution,
                        )
                        _write_cache(path, stitched)
                        return stitched, True

                raw = await _fetch_tv(cs, ce)
                if raw is None or raw.empty:
                    raise OhlcvIntegrityError(
                        f"empty_ohlcv segment {req.symbol} {cs}→{ce} resolution={resolution}"
                    )
                raw = validate_ohlcv_frame(
                    raw,
                    symbol=req.symbol,
                    start_ts=start_i,
                    end_ts=end_i,
                    resolution=resolution,
                )
                mode = (req.alligator_mode or "williams").strip().lower()
                enriched = enrich_ohlcv_frame(raw, alligator_mode=mode)
                if enriched.empty:
                    raise OhlcvIntegrityError(
                        f"enrich_empty segment {req.symbol} {cs}→{ce}"
                    )
                if req.use_cache:
                    _write_cache(path, enriched)
                return enriched, False

            results = await asyncio.gather(*[_one(p) for p in batch], return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.warning("Bulk segment failed: %s", res)
                    job["segments_done"] = int(job["segments_done"]) + 1
                    empty_segments += 1
                    raise res
                df, was_cached = res
                job["segments_done"] = int(job["segments_done"]) + 1
                if was_cached:
                    job["segments_cached"] = int(job["segments_cached"]) + 1
                if df is None or df.empty:
                    empty_segments += 1
                    raise OhlcvIntegrityError(
                        f"empty_segment_result {req.symbol} resolution={resolution}"
                    )
                frames.append(df)
                job["updated_at"] = time.time()
                job["elapsed_sec"] = round(time.perf_counter() - t0, 2)
                job["limiter"] = limiter.snapshot()
            i += batch_n

        merged = _merge_frames(frames)
        if merged.empty:
            raise OhlcvIntegrityError(
                f"empty_ohlcv bulk job {req.symbol} {start.date()}→{end.date()} "
                f"resolution={resolution} empty_segments={empty_segments}/{len(chunks)}"
            )
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        merged = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        start_i = int(pd.Timestamp(start).timestamp())
        end_i = int(pd.Timestamp(end).timestamp())
        merged = validate_ohlcv_frame(
            merged,
            symbol=req.symbol,
            start_ts=start_i,
            end_ts=end_i,
            resolution=resolution,
        )

        parquet_path = None
        if req.write_parquet:
            out_root = Path(req.out_dir).expanduser() if req.out_dir else (cache_root / "jobs")
            out_root.mkdir(parents=True, exist_ok=True)
            safe = req.symbol.replace("=", "_").replace(":", "_")
            parquet_path = str(
                out_root
                / f"{safe}_{resolution}_{start.date().isoformat()}_{end.date().isoformat()}_{job_id}.parquet"
            )
            merged.to_parquet(parquet_path)

        job["status"] = "done"
        job["bar_count"] = int(len(merged))
        job["empty_segments"] = empty_segments
        job["parquet_path"] = parquet_path
        job["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        job["limiter"] = limiter.snapshot()
        job["serial_tv"] = serial_tv
        job["updated_at"] = time.time()
        preview = enrich_ohlcv_frame(merged.head(3), alligator_mode=req.alligator_mode)
        job["preview_points"] = frame_to_points(preview)[:3]
        if empty_segments:
            job["warning"] = f"had_empty_retries empty_segments={empty_segments}/{len(chunks)}"
        logger.info(
            "Bulk job %s done bars=%d cached_segments=%s/%s empty=%d chunk_days=%s elapsed=%.1fs path=%s",
            job_id,
            job["bar_count"],
            job["segments_cached"],
            job["segments_total"],
            empty_segments,
            job.get("chunk_days"),
            job["elapsed_sec"],
            parquet_path,
        )
    except Exception as exc:
        logger.exception("Bulk job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(exc)
        job["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        job["updated_at"] = time.time()


async def wait_job(job_id: str, *, timeout_sec: float = 600.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        job = get_job(job_id)
        if job is None:
            return {"job_id": job_id, "status": "missing", "error": "unknown job_id"}
        if job.get("status") in ("done", "error"):
            return dict(job)
        await asyncio.sleep(0.5)
    job = get_job(job_id) or {"job_id": job_id}
    out = dict(job)
    out["status"] = "timeout"
    out["error"] = f"timed out after {timeout_sec}s"
    return out
