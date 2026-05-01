from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from alpaca_day_bot.reporting.accuracy import forward_accuracy_for_calendar_day
from alpaca_day_bot.reporting.model_diagnostics import model_diagnostics_for_day, model_diagnostics_for_day_by_action
from alpaca_day_bot.reporting.trades import realized_trade_stats_for_day
from alpaca_day_bot.reporting.trade_why import exit_intents_for_day, trade_whys_for_day
from alpaca_day_bot.reporting.virtual_options import virtual_options_stats_for_day
from alpaca_day_bot.reporting.executed_ml import executed_ml_summary
from alpaca_day_bot.ml.regime_thresholds import learn_regime_min_proba_map
from alpaca_day_bot.reporting.rollup_health import rollup_health


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

    def _acc_block(*, title: str, actions: tuple[str, ...]) -> list[str]:
        a = forward_accuracy_for_calendar_day(db_path, day, market_tz=market_tz, actions=actions)
        if a is None:
            return [
                "",
                f"### {title}",
                f"- No labeled {', '.join(actions)} signals for this calendar day yet (needs a later tick after the min age).",
            ]
        hit_pct = (a.directional_hits / a.labeled_count) if a.labeled_count else 0.0
        return [
            "",
            f"### {title}",
            f"- **Labeled signals**: {a.labeled_count}",
            f"- **Directional hit rate** (forward return > 0): {hit_pct*100:.1f}%",
            f"- **Mean forward return**: {fmt_pct(a.mean_return_pct)}",
            f"- **Median forward return**: {fmt_pct(a.median_return_pct)}",
            f"- _{a.note}_",
        ]

    acc_lines = [
        *_acc_block(title="Signal directional accuracy (labeled LONG / BUY)", actions=("BUY",)),
        *_acc_block(title="Signal directional accuracy (labeled SHORT)", actions=("SHORT",)),
    ]

    # Training metadata (written by ml.train next to the joblib artifact)
    model_train_lines: list[str] = []
    try:
        import json

        p = Path("state/models/latest.json")
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(j, dict) and not bool(j.get("skipped")):
                m = j.get("metrics") if isinstance(j.get("metrics"), dict) else {}
                try:
                    pos_rate_s = "n/a" if m.get("pos_rate") is None else f"{float(m.get('pos_rate'))*100:.1f}%"
                except Exception:
                    pos_rate_s = "n/a"
                try:
                    auc_s = "n/a" if m.get("auc") is None else f"{float(m.get('auc')):.3f}"
                except Exception:
                    auc_s = "n/a"
                try:
                    acc_s = "n/a" if m.get("acc") is None else f"{float(m.get('acc'))*100:.1f}%"
                except Exception:
                    acc_s = "n/a"
                model_train_lines = [
                    "",
                    "### Model training (latest)",
                    f"- **Trained at (UTC)**: {j.get('trained_at', 'n/a')}",
                    f"- **Dataset kind**: {j.get('dataset_kind', 'n/a')}",
                    f"- **Provider**: {j.get('provider', 'n/a')}",
                    f"- **Rows seen**: {j.get('rows_seen', 'n/a')}",
                    f"- **Test n**: {m.get('n', 'n/a')}",
                    f"- **Test pos rate**: {pos_rate_s}",
                    f"- **Test AUC**: {auc_s}",
                    f"- **Test accuracy**: {acc_s}",
                    f"- **Recommended min proba**: {j.get('recommended_min_proba', 'n/a')}",
                ]
            elif isinstance(j, dict) and bool(j.get("skipped")):
                model_train_lines = [
                    "",
                    "### Model training (latest)",
                    f"- Skipped: {j.get('skip_reason', 'n/a')} (n_labeled={j.get('n_labeled', 'n/a')}, min_required={j.get('min_required', 'n/a')})",
                ]
    except Exception:
        model_train_lines = []

    # Rollup health (long-lived training dataset)
    rollup_lines: list[str] = []
    try:
        rh = rollup_health(rollup_db_path=str(Path("state/ledgers/ledger_rollup.sqlite3")), state_dir="state")
        if rh is not None:
            rollup_lines = [
                "",
                "### Rollup health (long-lived training dataset)",
                f"- **Rollup DB**: `{rh.db_path}`",
                f"- **Signals (BUY / SHORT)**: {rh.signals_buy} / {rh.signals_short}",
                f"- **Triple-barrier labels (BUY / SHORT)**: {rh.tb_labels_buy} / {rh.tb_labels_short}",
                f"- **Executed round trips (total / BUY / SHORT)**: {rh.executed_round_trips_total} / {rh.executed_round_trips_buy} / {rh.executed_round_trips_short}",
                f"- **Latest BUY model**: {rh.model_buy_status or 'n/a'}",
                f"- **Latest SHORT model**: {rh.model_short_status or 'n/a'}",
            ]
    except Exception:
        rollup_lines = []

    md = model_diagnostics_for_day(db_path, day)
    model_lines: list[str] = []
    if md is not None:
        model_lines = [
            "",
            "### Model diagnostics (BUY labels)",
            f"- **Labeled signals**: {md.n_labeled}",
            f"- **With model probability recorded**: {md.n_with_proba}",
        ]
        if md.buckets:
            for b in md.buckets:
                hr = ("n/a" if b.hit_rate is None else f"{b.hit_rate*100:.1f}%")
                model_lines.append(f"- **{b.bucket}**: n={b.n}, hit={hr}")
        else:
            model_lines.append("- Not enough rows with probability to bucket yet.")

    # Optional: separate diagnostics by side (BUY vs SHORT)
    md_short = model_diagnostics_for_day_by_action(db_path, day, action="SHORT")
    if md_short is not None:
        model_lines.extend(
            [
                "",
                "### Model diagnostics (SHORT labels)",
                f"- **Labeled signals**: {md_short.n_labeled}",
                f"- **With model probability recorded**: {md_short.n_with_proba}",
            ]
        )
        if md_short.buckets:
            for b in md_short.buckets:
                hr = ("n/a" if b.hit_rate is None else f"{b.hit_rate*100:.1f}%")
                model_lines.append(f"- **{b.bucket}**: n={b.n}, hit={hr}")
        else:
            model_lines.append("- Not enough rows with probability to bucket yet.")

    # Realized trade stats (from fills)
    tstats = realized_trade_stats_for_day(db_path, start, end)
    if tstats.trades > 0:
        trade_lines = [
            "",
            "### Realized trade stats (from fills, FIFO lot matching)",
            f"- **Trades (round trips)**: {tstats.trades}",
            f"- **Win rate**: {('n/a' if tstats.win_rate is None else f'{tstats.win_rate*100:.1f}%')}",
            f"- **Profit factor**: {('n/a' if tstats.profit_factor is None else f'{tstats.profit_factor:.2f}')}",
            f"- **Expectancy ($/trade)**: {('n/a' if tstats.expectancy is None else f'{tstats.expectancy:.2f}')}",
            f"- **Avg win**: {fmt_money(tstats.avg_win)}",
            f"- **Avg loss**: {fmt_money(tstats.avg_loss)}",
            f"- **Realized PnL (sum)**: {fmt_money(tstats.total_pnl)}",
        ]
    else:
        trade_lines = [
            "",
            "### Realized trade stats (from fills)",
            "- No completed round trips yet in this calendar day.",
        ]

    vopt = virtual_options_stats_for_day(db_path, day)
    vopt_lines = [
        "",
        "### Virtual options (simulated calls/puts)",
        f"- **Closed virtual option trades**: {vopt.n_closed}",
        f"- **Realized PnL (sum)**: {fmt_money(vopt.total_pnl_usd)}",
    ]

    exec_sum = executed_ml_summary(db_path)
    exec_lines = [
        "",
        "### Executed-trade learning (fills → FIFO round trips)",
    ]
    if exec_sum is None:
        exec_lines.append("- Not enough executed round trips yet to train/evaluate.")
    else:
        exec_lines.append(f"- **Round trips available**: {exec_sum.n}")
        exec_lines.append(
            f"- **Win rate**: {('n/a' if exec_sum.win_rate is None else f'{exec_sum.win_rate*100:.1f}%')}"
        )
        exec_lines.append(f"- **Total realized PnL**: {fmt_money(exec_sum.total_pnl)}")

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
        *model_train_lines,
        *rollup_lines,
        *model_lines,
        *trade_lines,
        *exec_lines,
        *vopt_lines,
        "",
        "### Regime threshold suggestions (learned from ledger)",
        *(_regime_threshold_lines(db_path)),
        "",
        "### Trades (why this fired)",
        *(_trade_why_lines(db_path, day)),
        "",
        "### Exits (submitted closes)",
        *(_exit_intent_lines(db_path, day)),
        "",
        "Notes:",
        "- Equity snapshots are taken periodically during runtime.",
        "- Weekly summaries are written separately as `week_ending_YYYY-MM-DD.md`.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def _regime_threshold_lines(db_path: str) -> list[str]:
    try:
        mp_map, rows = learn_regime_min_proba_map(db_path=db_path)
    except Exception:
        return ["- n/a"]
    if not rows:
        return ["- Not enough labeled rows yet."]
    out = [f"- **Default**: 0.55", f"- **Regimes learned**: {len(mp_map)}"]
    for r in rows[:12]:
        thr = "n/a" if r.best_min_proba is None else f"{r.best_min_proba:.2f}"
        hr = "n/a" if r.hit_rate is None else f"{r.hit_rate*100:.1f}%"
        out.append(f"- **{r.regime}**: n={r.n}, thr={thr}, hit={hr}")
    return out

def _trade_why_lines(db_path: str, day: date) -> list[str]:
    rows = trade_whys_for_day(db_path, day.isoformat())
    if not rows:
        return ["- No submitted entry orders recorded for this calendar day."]
    out: list[str] = []
    for r in rows:
        qty = "n/a" if r.qty is None else str(int(r.qty))
        mp = "n/a" if r.model_proba is None else f"{r.model_proba:.3f}"
        nn = "n/a" if r.news_count is None else str(r.news_count)
        tp = "n/a" if r.taapi_present is None else str(r.taapi_present).lower()
        prov = "n/a" if r.model_provider is None else str(r.model_provider)
        ip = "n/a" if r.indicator_provider is None else str(r.indicator_provider)
        inds = "" if not r.indicators_used else f" inds={','.join(r.indicators_used[:10])}"
        rv = ""
        try:
            # prefer the votes that match the action when available
            act = (r.action or "").strip().upper()
            if isinstance(r.rule_votes, dict):
                if act == "SHORT" and isinstance(r.rule_votes.get("short"), dict):
                    ok = [k for k, v in (r.rule_votes.get("short") or {}).items() if v]
                    rv = f" votes={','.join(ok[:8])}"
                elif act == "BUY" and isinstance(r.rule_votes.get("long"), dict):
                    ok = [k for k, v in (r.rule_votes.get("long") or {}).items() if v]
                    rv = f" votes={','.join(ok[:8])}"
        except Exception:
            rv = ""
        out.append(
            f"- **{r.symbol}** {str(r.side).upper()} qty={qty} "
            f"setup={r.setup_reason or 'n/a'} model={prov} model_p={mp} "
            f"news_n={nn} taapi={tp} ind_provider={ip}{inds}{rv}"
        )
    return out


def _exit_intent_lines(db_path: str, day: date) -> list[str]:
    rows = exit_intents_for_day(db_path, day.isoformat())
    if not rows:
        return ["- No submitted exit orders recorded for this calendar day."]
    # Keep this concise; exits can be batchy (flatten/time exits).
    syms = [s for _ts, s, _reason, _raw in rows if s]
    uniq = []
    for s in syms:
        if s not in uniq:
            uniq.append(s)
    top = ", ".join(uniq[:15])
    more = "" if len(uniq) <= 15 else f" (+{len(uniq)-15} more)"
    return [f"- **Exit intents submitted**: {len(rows)} (symbols: {top}{more})"]


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

    # Meta-label take-rate / hit-rate summary (from labeled signals).
    try:
        conn = sqlite3.connect(db_path)
        rr = conn.execute(
            """
            SELECT
              COUNT(1) AS n_labeled,
              SUM(CASE WHEN tb.outcome = 'tp' THEN 1 ELSE 0 END) AS n_tp
            FROM triple_barrier_labels tb
            """,
        ).fetchone()
        conn.close()
        if rr and rr[0]:
            n_l = int(rr[0] or 0)
            n_tp = int(rr[1] or 0)
            hr = (n_tp / n_l) if n_l else 0.0
            lines.extend(
                [
                    "### Meta-label summary (triple-barrier outcomes)",
                    f"- **Labeled signals**: {n_l}",
                    f"- **TP outcomes**: {n_tp}",
                    f"- **Hit rate (TP)**: {hr*100:.1f}%",
                    "",
                ]
            )
    except Exception:
        pass

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

