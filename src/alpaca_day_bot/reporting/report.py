from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from alpaca_day_bot.reporting.accuracy import forward_accuracy_for_calendar_day


@dataclass(frozen=True)
class ReportSummary:
    day: date
    start_equity: float | None
    end_equity: float | None
    pnl: float | None
    pnl_pct: float | None
    max_gross: float | None
    trades: int


def _parse_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(tz=timezone.utc)


def daily_summary(db_path: str, day: date) -> ReportSummary:
    conn = sqlite3.connect(db_path)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()

    rows = conn.execute(
        "SELECT ts, equity, gross_exposure FROM equity_snapshots WHERE ts BETWEEN ? AND ? ORDER BY ts ASC",
        (start, end),
    ).fetchall()
    trades = conn.execute(
        "SELECT COUNT(1) FROM trade_updates WHERE ts BETWEEN ? AND ? AND event IN ('fill','partial_fill','filled')",
        (start, end),
    ).fetchone()[0]

    conn.close()

    if not rows:
        return ReportSummary(day, None, None, None, None, None, trades=int(trades))

    start_equity = float(rows[0][1])
    end_equity = float(rows[-1][1])
    max_gross = max(float(r[2]) for r in rows)
    pnl = end_equity - start_equity
    pnl_pct = (pnl / start_equity) if start_equity else None
    return ReportSummary(day, start_equity, end_equity, pnl, pnl_pct, max_gross, trades=int(trades))


def write_daily_report(
    db_path: str, reports_dir: str, day: date, *, market_tz: str = "America/New_York"
) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    s = daily_summary(db_path, day)
    out = Path(reports_dir) / f"{day.isoformat()}.md"

    # Equity-curve derived stats
    conn = sqlite3.connect(db_path)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()
    eq = conn.execute(
        "SELECT ts, equity FROM equity_snapshots WHERE ts BETWEEN ? AND ? ORDER BY ts ASC",
        (start, end),
    ).fetchall()
    conn.close()

    sharpe_intraday = None
    max_dd = None
    if eq:
        idx = [datetime.fromisoformat(r[0]) for r in eq]
        vals = [float(r[1]) for r in eq]
        import pandas as pd

        ser = pd.Series(vals, index=pd.to_datetime(idx, utc=True))
        max_dd = _max_drawdown(ser)
        sharpe_intraday = _sharpe_from_returns(ser.pct_change().dropna(), annualization=252.0 * 6.5 * 60.0 / 5.0)

    acc = forward_accuracy_for_calendar_day(db_path, day, market_tz=market_tz)
    acc_lines: list[str] = []
    if acc is not None:
        hit_pct = (acc.directional_hits / acc.labeled_count) if acc.labeled_count else 0.0
        acc_lines = [
            "",
            "### Signal directional accuracy (labeled BUYs)",
            f"- **Labeled signals**: {acc.labeled_count}",
            f"- **Directional hit rate** (forward return > 0): {hit_pct*100:.1f}%",
            f"- **Mean forward return**: {fmt_pct(acc.mean_return_pct)}",
            f"- **Median forward return**: {fmt_pct(acc.median_return_pct)}",
            f"- _{acc.note}_",
        ]
    else:
        acc_lines = [
            "",
            "### Signal directional accuracy (labeled BUYs)",
            "- No labeled BUY signals for this calendar day yet (needs a later tick after the min age).",
        ]

    lines = [
        f"## Paper trading report: {day.isoformat()}",
        "",
        f"- **Start equity**: {fmt_money(s.start_equity)}",
        f"- **End equity**: {fmt_money(s.end_equity)}",
        f"- **PnL**: {fmt_money(s.pnl)} ({fmt_pct(s.pnl_pct)})",
        f"- **Max gross exposure**: {fmt_money(s.max_gross)}",
        f"- **Max drawdown (from snapshots)**: {fmt_pct(max_dd)}",
        f"- **Sharpe (snapshot returns, approx.)**: {('n/a' if sharpe_intraday is None else f'{sharpe_intraday:.2f}')}",
        f"- **Trade updates (fills)**: {s.trades}",
        *acc_lines,
        "",
        "Notes:",
        "- Equity snapshots are taken periodically during runtime.",
        "- Weekly summaries are written separately as `week_ending_YYYY-MM-DD.md`.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_weekly_report(db_path: str, reports_dir: str, week_ending: date, days: int = 7) -> str:
    """
    Simple rolling window aggregation over the last N calendar days ending at week_ending.
    (For a true trading-week calendar, we can refine later.)
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    daily: list[ReportSummary] = []
    for i in range(days):
        d = week_ending - timedelta(days=i)
        daily.append(daily_summary(db_path, d))
    daily = list(reversed(daily))

    start_equity = next((x.start_equity for x in daily if x.start_equity is not None), None)
    end_equity = next((x.end_equity for x in reversed(daily) if x.end_equity is not None), None)
    pnl = None if (start_equity is None or end_equity is None) else (end_equity - start_equity)
    pnl_pct = None if (pnl is None or start_equity in (None, 0.0)) else (pnl / start_equity)
    trades = sum(x.trades for x in daily)
    max_gross = max((x.max_gross or 0.0) for x in daily) if daily else None

    out = Path(reports_dir) / f"week_ending_{week_ending.isoformat()}.md"
    lines = [
        f"## Paper trading week summary (rolling {days}d): ending {week_ending.isoformat()}",
        "",
        f"- **Start equity**: {fmt_money(start_equity)}",
        f"- **End equity**: {fmt_money(end_equity)}",
        f"- **PnL**: {fmt_money(pnl)} ({fmt_pct(pnl_pct)})",
        f"- **Max gross exposure**: {fmt_money(max_gross)}",
        f"- **Trade updates (fills)**: {trades}",
        "",
        "### Daily breakdown",
        "",
    ]
    for d in daily:
        lines.append(
            f"- **{d.day.isoformat()}**: PnL {fmt_money(d.pnl)} ({fmt_pct(d.pnl_pct)}), trades {d.trades}, max_gross {fmt_money(d.max_gross)}"
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def fmt_money(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.2f}"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.2f}%"


def _max_drawdown(equity_curve):
    if equity_curve is None or equity_curve.empty:
        return None
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max - 1.0).min()
    return float(dd)


def _sharpe_from_returns(returns, *, annualization: float) -> float | None:
    # returns: periodic returns series
    if returns is None or len(returns) < 5:
        return None
    mu = float(returns.mean())
    sd = float(returns.std())
    if sd <= 1e-12:
        return None
    return (mu / sd) * (annualization**0.5)

