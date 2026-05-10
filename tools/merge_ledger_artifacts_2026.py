from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunPick:
    run_id: int
    created_at: str
    artifact_name: str


def _sh(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _try_sh(args: list[str]) -> str | None:
    try:
        return _sh(args)
    except Exception:
        return None


def _parse_artifacts_from_run_view(text: str) -> list[str]:
    if "ARTIFACTS" not in text:
        return []
    after = text.split("ARTIFACTS", 1)[1]
    names: list[str] = []
    for line in after.splitlines()[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("For more") or s.startswith("View this"):
            break
        if " " not in s and "\t" not in s:
            names.append(s)
    return names


def _pick_state_artifact(names: list[str]) -> str | None:
    # prefer "latest" snapshot; otherwise any state snapshot
    for n in names:
        if n.startswith("alpaca-state-latest-"):
            return n
    for n in names:
        if n.startswith("alpaca-state-"):
            return n
    return None


def list_2026_runs(*, workflow_name: str = "Paper scheduled tick", limit: int = 2000) -> list[dict]:
    raw = _sh(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow_name,
            "--limit",
            str(int(limit)),
            "--json",
            "databaseId,createdAt,conclusion,event",
        ]
    )
    runs = json.loads(raw)
    out = []
    for r in runs:
        ca = str(r.get("createdAt") or "")
        if not ca.startswith("2026-"):
            continue
        out.append(r)
    # newest first from gh; keep as-is
    return out


def pick_runs_with_state_artifacts(runs: Iterable[dict], *, max_days: int | None = None) -> list[RunPick]:
    """
    Return at most one run per day (newest with artifacts).
    """
    picked_by_day: dict[str, RunPick] = {}
    for r in runs:
        run_id = int(r["databaseId"])
        created_at = str(r["createdAt"])
        day = created_at[:10]
        if day in picked_by_day:
            continue
        txt = _try_sh(["gh", "run", "view", str(run_id)])
        if not txt:
            continue
        names = _parse_artifacts_from_run_view(txt)
        art = _pick_state_artifact(names)
        if not art:
            continue
        picked_by_day[day] = RunPick(run_id=run_id, created_at=created_at, artifact_name=art)
        if max_days is not None and len(picked_by_day) >= int(max_days):
            break
    # oldest to newest is nicer for merging/repro
    return [picked_by_day[d] for d in sorted(picked_by_day.keys())]


def download_ledger_for_run(*, pick: RunPick, out_dir: Path) -> Path | None:
    day = pick.created_at[:10]
    dest = out_dir / f"{day}_{pick.run_id}"
    dest.mkdir(parents=True, exist_ok=True)
    # If already downloaded, reuse.
    for cand in (dest / "ledger.sqlite3", dest / "state" / "ledger.sqlite3"):
        if cand.exists():
            return cand
    # download artifact into dest
    try:
        subprocess.check_call(["gh", "run", "download", str(pick.run_id), "-n", pick.artifact_name, "-D", str(dest)])
    except Exception as e:
        print("download_failed", pick.run_id, pick.artifact_name, "err=", str(e)[:200])
        return None
    # artifact layout is directly in dest (ledger.sqlite3)
    db = dest / "ledger.sqlite3"
    if db.exists():
        return db
    # some layouts may nest under state/
    db2 = dest / "state" / "ledger.sqlite3"
    return db2 if db2.exists() else None


def _table_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows]


def _max_id(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0


def _copy_with_id_offset(
    *,
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    id_offset: int,
    where: str | None = None,
) -> int:
    cols = _table_cols(src, table)
    # Some older ledgers may not include `id` for certain tables (schema drift).
    # In that case, fall back to copying common non-id columns and let SQLite assign ids.
    if "id" not in cols:
        dst_cols = _table_cols(dst, table)
        common = [c for c in cols if c in dst_cols and c != "id"]
        if not common:
            return 0
        sql = f"SELECT {', '.join(common)} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        rows = src.execute(sql).fetchall()
        if not rows:
            return 0
        ph = ", ".join(["?"] * len(common))
        ins = f"INSERT INTO {table} ({', '.join(common)}) VALUES ({ph})"
        dst.executemany(ins, rows)
        return len(rows)
    cols_wo_id = [c for c in cols if c != "id"]

    sql = f"SELECT {', '.join(cols)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    rows = src.execute(sql).fetchall()
    if not rows:
        return 0

    ins_cols = ["id"] + cols_wo_id
    ph = ", ".join(["?"] * len(ins_cols))
    ins = f"INSERT INTO {table} ({', '.join(ins_cols)}) VALUES ({ph})"

    out_rows = []
    for r in rows:
        d = dict(zip(cols, r))
        new_id = int(d["id"]) + int(id_offset)
        out_rows.append([new_id] + [d[c] for c in cols_wo_id])
    dst.executemany(ins, out_rows)
    return len(out_rows)


def _copy_labels_with_signal_offset(
    *,
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    signal_id_offset: int,
) -> int:
    cols = _table_cols(src, table)
    # Older ledgers may not have labeling tables at all (or may have a different schema).
    if "signal_id" not in cols:
        return 0

    rows = src.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
    if not rows:
        return 0
    ins_cols = cols
    ph = ", ".join(["?"] * len(ins_cols))
    ins = f"INSERT OR REPLACE INTO {table} ({', '.join(ins_cols)}) VALUES ({ph})"

    out_rows = []
    for r in rows:
        d = dict(zip(cols, r))
        d["signal_id"] = int(d["signal_id"]) + int(signal_id_offset)
        out_rows.append([d[c] for c in cols])
    dst.executemany(ins, out_rows)
    return len(out_rows)


def ensure_schema(dst_db: Path) -> None:
    # Reuse Ledger init to keep schema aligned with the bot
    from alpaca_day_bot.storage.ledger import Ledger

    ld = Ledger(str(dst_db))
    ld.close()


def merge_ledgers(*, src_dbs: list[Path], out_db: Path) -> dict:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()
    ensure_schema(out_db)

    dst = sqlite3.connect(str(out_db))
    dst.execute("PRAGMA journal_mode=WAL;")
    dst.execute("PRAGMA synchronous=NORMAL;")

    totals: dict[str, int] = {
        "src_dbs": 0,
        "signals": 0,
        "forward_return_labels": 0,
        "triple_barrier_labels": 0,
        "order_intents": 0,
        "trade_updates": 0,
        "equity_snapshots": 0,
        "virtual_option_trades": 0,
    }

    for dbp in src_dbs:
        src = sqlite3.connect(str(dbp))
        src.execute("PRAGMA query_only=ON;")

        # signal ids must be offset so labels keep pointing correctly
        sig_off = _max_id(dst, "signals")
        oi_off = _max_id(dst, "order_intents")
        tu_off = _max_id(dst, "trade_updates")
        es_off = _max_id(dst, "equity_snapshots")
        vo_off = _max_id(dst, "virtual_option_trades")

        dst.execute("BEGIN;")
        try:
            totals["signals"] += _copy_with_id_offset(src=src, dst=dst, table="signals", id_offset=sig_off)
            totals["order_intents"] += _copy_with_id_offset(
                src=src, dst=dst, table="order_intents", id_offset=oi_off
            )
            totals["trade_updates"] += _copy_with_id_offset(
                src=src, dst=dst, table="trade_updates", id_offset=tu_off
            )
            totals["equity_snapshots"] += _copy_with_id_offset(
                src=src, dst=dst, table="equity_snapshots", id_offset=es_off
            )
            totals["virtual_option_trades"] += _copy_with_id_offset(
                src=src, dst=dst, table="virtual_option_trades", id_offset=vo_off
            )
            totals["forward_return_labels"] += _copy_labels_with_signal_offset(
                src=src, dst=dst, table="forward_return_labels", signal_id_offset=sig_off
            )
            totals["triple_barrier_labels"] += _copy_labels_with_signal_offset(
                src=src, dst=dst, table="triple_barrier_labels", signal_id_offset=sig_off
            )
            dst.commit()
        except Exception:
            dst.rollback()
            raise
        finally:
            src.close()
        totals["src_dbs"] += 1

    dst.close()
    return totals


def summarize_short(*, db_path: Path) -> dict:
    from alpaca_day_bot.reporting.trades import _rows_to_fills, reconstruct_round_trips

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    by_reason = cur.execute(
        "SELECT reason, COUNT(1) FROM signals WHERE action='SHORT' GROUP BY reason ORDER BY COUNT(1) DESC"
    ).fetchall()

    labeled = cur.execute(
        """
        SELECT s.reason,
               COUNT(1) as n,
               SUM(CASE WHEN COALESCE(tb.outcome,'')='tp' THEN 1 ELSE 0 END) as tp,
               SUM(CASE WHEN COALESCE(tb.outcome,'')='sl' THEN 1 ELSE 0 END) as sl,
               SUM(CASE WHEN COALESCE(tb.outcome,'')='timeout' THEN 1 ELSE 0 END) as timeout,
               AVG(COALESCE(tb.realized_return_pct, f.return_pct)) as avg_ret
        FROM signals s
        LEFT JOIN triple_barrier_labels tb ON tb.signal_id=s.id
        LEFT JOIN forward_return_labels f ON f.signal_id=s.id
        WHERE s.action='SHORT' AND (tb.signal_id IS NOT NULL OR f.signal_id IS NOT NULL)
        GROUP BY s.reason
        ORDER BY n DESC
        """
    ).fetchall()

    tu = cur.execute(
        "SELECT ts, event, raw_json FROM trade_updates WHERE event IN ('fill','partial_fill','filled') ORDER BY ts"
    ).fetchall()
    fills = _rows_to_fills(tu)
    rts = reconstruct_round_trips(fills)

    conn.close()
    return {
        "short_signals_by_reason": [(str(r), int(n)) for r, n in by_reason],
        "short_labeled_by_reason": [
            {
                "reason": str(r),
                "n": int(n),
                "tp": int(tp or 0),
                "sl": int(sl or 0),
                "timeout": int(to or 0),
                "tp_rate": (float(tp or 0) / float(n) if n else None),
                "avg_ret": (None if avg_ret is None else float(avg_ret)),
            }
            for (r, n, tp, sl, to, avg_ret) in labeled
        ],
        "executed_round_trips_total": int(len(rts)),
        "executed_round_trips_short": int(sum(1 for rt in rts if rt.direction == "short")),
    }


def main() -> None:
    out_root = Path(os.environ.get("OUT_DIR", "tmp_2026_ledgers")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    merged_db = Path(os.environ.get("OUT_DB", "state/ledger_2026_merged.sqlite3")).resolve()

    print("listing 2026 runs…")
    runs = list_2026_runs()
    print("runs_in_2026:", len(runs))

    print("downloading one state ledger per day (best-effort)…")
    src_dbs: list[Path] = []
    got_days: set[str] = set()
    # Iterate newest->oldest; keep first successful ledger per day.
    for r in runs:
        created_at = str(r.get("createdAt") or "")
        day = created_at[:10]
        if not day or day in got_days:
            continue
        run_id = int(r["databaseId"])
        txt = _try_sh(["gh", "run", "view", str(run_id)])
        if not txt:
            continue
        names = _parse_artifacts_from_run_view(txt)
        art = _pick_state_artifact(names)
        if not art:
            continue
        db = download_ledger_for_run(pick=RunPick(run_id=run_id, created_at=created_at, artifact_name=art), out_dir=out_root)
        if db:
            src_dbs.append(db)
            got_days.add(day)
    print("days_with_ledgers:", len(got_days))

    print("downloaded_ledgers:", len(src_dbs))
    if not src_dbs:
        print("no_ledgers_downloaded")
        return

    print("merging ledgers into", merged_db)
    totals = merge_ledgers(src_dbs=src_dbs, out_db=merged_db)
    print("merge_totals:", json.dumps(totals, indent=2))

    print("short_summary:")
    summ = summarize_short(db_path=merged_db)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

