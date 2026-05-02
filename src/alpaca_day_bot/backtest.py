from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import sqrt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpaca_day_bot.data.stream import resample_ohlcv
from alpaca_day_bot.strategy.v1_rules import V1RulesSignalEngine


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    qty: float
    entry_notional: float
    exit_notional: float
    pnl: float
    pnl_r: float | None
    risk_r: float | None
    hold_minutes: float
    entry_cost_usd: float | None
    exit_cost_usd: float | None


@dataclass(frozen=True)
class BacktestResult:
    start_equity: float
    end_equity: float
    total_return: float
    sharpe_daily: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    expectancy_r: float | None
    turnover: float | None
    avg_hold_minutes: float | None
    trades: list[Trade]
    equity_curve: pd.Series


def run_backtest(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    starting_equity: float,
    risk_per_trade_pct: float,
    max_gross_exposure_pct: float,
    stop_loss_atr_mult: float,
    take_profit_r_mult: float,
    slippage_bps: float = 1.0,
    commission_bps: float = 0.5,
    spread_proxy_k: float = 0.10,
    spread_proxy_min: float = 0.01,
    open_delay_minutes: int = 0,
    market_context_filter: bool = False,
    spy_5m_rsi_min: float = 40.0,
    strategy_params: dict | None = None,
    on_signal: callable | None = None,
    on_trade: callable | None = None,
) -> BacktestResult:
    """
    Lightweight event-driven backtest:
    - Uses V1RulesSignalEngine on 1m + 15m resampled data.
    - Assumes marketable entries at next bar open.
    - Uses stop/take-profit simulated on bar highs/lows.
    - Applies modeled slippage/commission in bps (round-trip).
    """
    engine = V1RulesSignalEngine(**(strategy_params or {}))

    # Create a merged timestamp index across all symbols (1m).
    all_ts = sorted({ts for df in bars_by_symbol.values() for ts in df.index})
    if not all_ts:
        ec = pd.Series([], dtype=float)
        return BacktestResult(
            start_equity=starting_equity,
            end_equity=starting_equity,
            total_return=0.0,
            sharpe_daily=None,
            max_drawdown=None,
            win_rate=None,
            profit_factor=None,
            expectancy=None,
            expectancy_r=None,
            turnover=None,
            avg_hold_minutes=None,
            trades=[],
            equity_curve=ec,
        )

    equity = float(starting_equity)
    cash = float(starting_equity)
    positions: dict[str, dict] = {}  # symbol -> {qty, entry_price, stop, tp, entry_ts}
    trades: list[Trade] = []
    equity_curve = []
    equity_index = []

    def cost_multiplier() -> float:
        # round-trip cost approx; applied on entry+exit in price terms
        return (slippage_bps + commission_bps) / 10000.0

    def half_spread_proxy(df_sym: pd.DataFrame, ts_: pd.Timestamp) -> float:
        # Conservative proxy: a fraction of bar range with a minimum.
        try:
            bar = df_sym.loc[ts_]
            rng = float(bar["high"]) - float(bar["low"])
        except Exception:
            rng = 0.0
        return max(float(spread_proxy_min), float(spread_proxy_k) * max(0.0, rng)) / 2.0

    ny_tz = ZoneInfo("America/New_York")

    def is_in_open_delay(ts_: pd.Timestamp) -> bool:
        if open_delay_minutes <= 0:
            return False
        dt = ts_.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        m = dt.astimezone(ny_tz)
        open_dt = m.replace(hour=9, minute=30, second=0, microsecond=0)
        return m < (open_dt + timedelta(minutes=int(open_delay_minutes)))

    # Precompute 15m frames
    df15_by_symbol: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        df15_by_symbol[sym] = resample_ohlcv(df, rule="15min")

    # Precompute SPY 5m RSI context if enabled
    spy_5m_rsi = None
    if market_context_filter and "SPY" in bars_by_symbol:
        import pandas_ta as ta

        spy5 = resample_ohlcv(bars_by_symbol["SPY"], rule="5min")
        if spy5 is not None and not spy5.empty:
            spy5["rsi"] = ta.rsi(spy5["close"], length=14)
            spy_5m_rsi = spy5["rsi"].dropna()

    def market_context_bad(ts_: pd.Timestamp) -> bool:
        if not market_context_filter or spy_5m_rsi is None or spy_5m_rsi.empty:
            return False
        s = spy_5m_rsi[spy_5m_rsi.index <= ts_]
        if s.empty:
            return False
        return float(s.iloc[-1]) < float(spy_5m_rsi_min)

    for i, ts in enumerate(all_ts):
        # Mark-to-market equity
        mtm = cash
        for sym, pos in positions.items():
            df = bars_by_symbol[sym]
            if ts in df.index:
                px = float(df.loc[ts, "close"])
                mtm += float(pos["qty"]) * px
            else:
                # fallback to last known close
                prev = df[df.index <= ts]
                if not prev.empty:
                    px = float(prev["close"].iloc[-1])
                    mtm += float(pos["qty"]) * px
        equity = mtm
        equity_curve.append(equity)
        equity_index.append(ts)

        # Manage exits first (stop/tp)
        for sym in list(positions.keys()):
            df = bars_by_symbol[sym]
            if ts not in df.index:
                continue
            bar = df.loc[ts]
            hi = float(bar["high"])
            lo = float(bar["low"])
            pos = positions[sym]
            stop = float(pos["stop"])
            tp = float(pos["tp"])

            exit_px = None
            # Conservative ordering: stop first if both touched.
            if lo <= stop:
                exit_px = stop
            elif hi >= tp:
                exit_px = tp

            if exit_px is not None:
                qty = float(pos["qty"])
                entry_px = float(pos["entry_price"])
                entry_px_eff = float(pos.get("entry_px_eff", entry_px))
                entry_ts = pos["entry_ts"]
                sim_signal_id = pos.get("sim_signal_id")
                stop_dist = float(pos.get("stop_dist", 0.0))
                risk_r = (qty * stop_dist) if stop_dist > 0 else None

                # Apply modeled costs on exit as worse price for long
                c = cost_multiplier()
                hs = float(pos.get("half_spread", 0.0))
                exit_px_eff = (exit_px - hs) * (1.0 - c)
                cash += qty * exit_px_eff
                pnl = qty * (exit_px_eff - entry_px_eff)
                pnl_r = (pnl / risk_r) if (risk_r is not None and risk_r > 1e-12) else None
                entry_cost_usd = qty * (entry_px_eff - entry_px) if qty > 0 else None
                exit_cost_usd = qty * (exit_px - exit_px_eff) if qty > 0 else None

                hold_minutes = (ts - entry_ts).total_seconds() / 60.0
                trades.append(
                    Trade(
                        symbol=sym,
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        entry_price=entry_px,
                        exit_price=exit_px_eff,
                        qty=qty,
                        entry_notional=qty * entry_px,
                        exit_notional=qty * exit_px_eff,
                        pnl=pnl,
                        pnl_r=pnl_r,
                        risk_r=risk_r,
                        hold_minutes=hold_minutes,
                        entry_cost_usd=entry_cost_usd,
                        exit_cost_usd=exit_cost_usd,
                    )
                )
                if callable(on_trade):
                    try:
                        on_trade(trades[-1], sim_signal_id)
                    except TypeError:
                        try:
                            on_trade(trades[-1])
                        except Exception:
                            pass
                    except Exception:
                        pass
                del positions[sym]

        # Entry decisions
        gross_exposure = 0.0
        for sym, pos in positions.items():
            df = bars_by_symbol[sym]
            prev = df[df.index <= ts]
            if prev.empty:
                continue
            px = float(prev["close"].iloc[-1])
            gross_exposure += abs(float(pos["qty"]) * px)

        max_gross = equity * float(max_gross_exposure_pct)
        remaining_gross = max(0.0, max_gross - gross_exposure)

        for sym, df in bars_by_symbol.items():
            if sym in positions:
                continue
            if ts not in df.index:
                continue
            if is_in_open_delay(ts):
                continue
            if market_context_bad(ts):
                continue

            # Need at least 50 bars to stabilize indicators
            df_hist = df.loc[:ts].copy()
            if len(df_hist) < 80:
                continue
            df15 = df15_by_symbol.get(sym)
            if df15 is None or df15.empty:
                continue
            df15_hist = df15.loc[:ts].copy()

            sig = engine.decide(symbol=sym, df_1m=df_hist, df_15m=df15_hist)
            sim_signal_id = None
            if sig is not None and callable(on_signal):
                try:
                    sim_signal_id = on_signal(ts.to_pydatetime(), sym, sig)
                except Exception:
                    sim_signal_id = None
            if sig is None or sig.action != "BUY":
                continue

            # Entry at next bar open (if exists)
            if i + 1 >= len(all_ts):
                continue
            next_ts = all_ts[i + 1]
            if next_ts not in df.index:
                continue
            entry_px = float(df.loc[next_ts, "open"])

            # Stop distance proxy: ATR-like via recent true range average
            tr = (df_hist["high"] - df_hist["low"]).rolling(14).mean().iloc[-1]
            if not np.isfinite(tr) or tr <= 0:
                continue
            stop_dist = float(tr) * float(stop_loss_atr_mult)
            stop = entry_px - stop_dist
            tp = entry_px + stop_dist * float(take_profit_r_mult)
            if stop <= 0:
                continue

            # Size by risk: risk_per_trade% of equity / stop_dist gives qty
            risk_budget = equity * float(risk_per_trade_pct)
            qty = risk_budget / stop_dist
            notional = qty * entry_px

            # Cap by remaining gross exposure and cash
            cap_notional = min(remaining_gross, cash)
            if cap_notional <= 0:
                continue
            if notional > cap_notional:
                qty = cap_notional / entry_px
                notional = qty * entry_px
            if notional < 5.0:
                continue

            # Apply modeled costs on entry as worse price for long
            c = cost_multiplier()
            hs = half_spread_proxy(df, next_ts)
            entry_px_eff = (entry_px + hs) * (1.0 + c)
            cash -= qty * entry_px_eff
            remaining_gross -= notional

            positions[sym] = {
                "qty": qty,
                "entry_price": entry_px,
                "entry_px_eff": entry_px_eff,
                "entry_ts": next_ts,
                "stop": stop,
                "tp": tp,
                "stop_dist": stop_dist,
                "half_spread": hs,
                "sim_signal_id": sim_signal_id,
            }

    ec = pd.Series(equity_curve, index=pd.to_datetime(equity_index, utc=True))
    end_equity = float(ec.iloc[-1]) if not ec.empty else float(starting_equity)
    total_return = (end_equity / starting_equity - 1.0) if starting_equity else 0.0

    dd = max_drawdown(ec)
    sharpe = sharpe_from_equity_curve_daily(ec)
    wr, pf, exp = trade_stats(trades)
    exp_r = expectancy_r(trades)
    turnover = turnover_from_trades(trades, start_equity=starting_equity)
    avg_hold = float(np.mean([t.hold_minutes for t in trades])) if trades else None

    return BacktestResult(
        start_equity=float(starting_equity),
        end_equity=end_equity,
        total_return=float(total_return),
        sharpe_daily=sharpe,
        max_drawdown=dd,
        win_rate=wr,
        profit_factor=pf,
        expectancy=exp,
        expectancy_r=exp_r,
        turnover=turnover,
        avg_hold_minutes=avg_hold,
        trades=trades,
        equity_curve=ec,
    )


def max_drawdown(equity_curve: pd.Series) -> float | None:
    if equity_curve is None or equity_curve.empty:
        return None
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max - 1.0).min()
    return float(dd)


def sharpe_from_equity_curve_daily(equity_curve: pd.Series) -> float | None:
    if equity_curve is None or equity_curve.empty:
        return None
    # Convert to daily closes (UTC date). This is coarse but stable.
    daily = equity_curve.resample("1D").last().dropna()
    if len(daily) < 3:
        return None
    rets = daily.pct_change().dropna()
    if rets.std() <= 1e-12:
        return None
    return float((rets.mean() / rets.std()) * sqrt(252.0))


def trade_stats(trades: list[Trade]) -> tuple[float | None, float | None, float | None]:
    if not trades:
        return None, None, None
    pnls = np.asarray([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float((pnls > 0).mean())
    profit_factor = None
    if losses.size:
        profit_factor = float(wins.sum() / abs(losses.sum())) if wins.size else 0.0
    expectancy = float(pnls.mean())
    return win_rate, profit_factor, expectancy


def expectancy_r(trades: list[Trade]) -> float | None:
    rs = [t.pnl_r for t in trades if t.pnl_r is not None]
    if not rs:
        return None
    return float(np.mean(rs))


@dataclass(frozen=True)
class SymbolDaytradeRec:
    """Per-symbol stats from a backtest, used to rank names for day-trade focus."""

    symbol: str
    trades: int
    total_pnl_usd: float
    win_rate: float | None
    expectancy_usd: float | None
    expectancy_r: float | None


def symbol_daytrade_recommendations(
    trades: list[Trade], *, min_trades: int = 5
) -> tuple[list[SymbolDaytradeRec], list[str]]:
    """
    Aggregate closed trades by symbol and rank by expectancy ($/trade), then total PnL.

    Returns:
        (all ranked rows with at least 1 trade, focus_symbols with trades >= min_trades)
    """
    by_sym: dict[str, list[Trade]] = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t)

    rows: list[SymbolDaytradeRec] = []
    for sym, ts in sorted(by_sym.items()):
        pnls = [x.pnl for x in ts]
        rs = [x.pnl_r for x in ts if x.pnl_r is not None]
        n = len(ts)
        wr = float(sum(1 for p in pnls if p > 0) / n) if n else None
        exp = float(np.mean(pnls)) if pnls else None
        exp_r = float(np.mean(rs)) if rs else None
        rows.append(
            SymbolDaytradeRec(
                symbol=sym,
                trades=n,
                total_pnl_usd=float(sum(pnls)),
                win_rate=wr,
                expectancy_usd=exp,
                expectancy_r=exp_r,
            )
        )

    rows.sort(
        key=lambda r: (
            -(r.expectancy_usd if r.expectancy_usd is not None else -1e18),
            -r.total_pnl_usd,
            -r.trades,
        )
    )
    focus = [r.symbol for r in rows if r.trades >= min_trades]
    return rows, focus


def turnover_from_trades(trades: list[Trade], *, start_equity: float) -> float | None:
    """
    Turnover proxy: total traded notional / starting equity.
    Uses entry+exit notionals for each closed trade.
    """
    if not trades or start_equity <= 0:
        return None
    total = float(sum(t.entry_notional + t.exit_notional for t in trades))
    return total / float(start_equity)


@dataclass(frozen=True)
class RobustnessGridRow:
    slippage_bps: float
    commission_bps: float
    total_return: float
    sharpe_daily: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    turnover: float | None
    trades: int


def run_cost_sensitivity_grid(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    starting_equity: float,
    risk_per_trade_pct: float,
    max_gross_exposure_pct: float,
    stop_loss_atr_mult: float,
    take_profit_r_mult: float,
    slippage_bps_list: list[float],
    commission_bps_list: list[float],
    strategy_params: dict | None = None,
) -> list[RobustnessGridRow]:
    rows: list[RobustnessGridRow] = []
    for s in slippage_bps_list:
        for c in commission_bps_list:
            res = run_backtest(
                bars_by_symbol=bars_by_symbol,
                starting_equity=starting_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                max_gross_exposure_pct=max_gross_exposure_pct,
                stop_loss_atr_mult=stop_loss_atr_mult,
                take_profit_r_mult=take_profit_r_mult,
                slippage_bps=float(s),
                commission_bps=float(c),
                strategy_params=strategy_params,
            )
            rows.append(
                RobustnessGridRow(
                    slippage_bps=float(s),
                    commission_bps=float(c),
                    total_return=float(res.total_return),
                    sharpe_daily=res.sharpe_daily,
                    max_drawdown=res.max_drawdown,
                    win_rate=res.win_rate,
                    profit_factor=res.profit_factor,
                    expectancy=res.expectancy,
                    turnover=res.turnover,
                    trades=len(res.trades),
                )
            )
    return rows


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    start: datetime
    end: datetime
    total_return: float
    sharpe_daily: float | None
    max_drawdown: float | None
    trades: int
    included: bool
    note: str | None = None


def run_walk_forward(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    starting_equity: float,
    risk_per_trade_pct: float,
    max_gross_exposure_pct: float,
    stop_loss_atr_mult: float,
    take_profit_r_mult: float,
    slippage_bps: float,
    commission_bps: float,
    start_dt: datetime,
    end_dt: datetime,
    test_window_days: int = 7,
    step_days: int = 7,
    min_trades_per_fold: int = 30,
    strategy_params: dict | None = None,
) -> list[WalkForwardFold]:
    """
    Walk-forward on calendar windows. Each fold runs an independent backtest on that window.
    (For rule-based strategies this is still valuable to detect regime sensitivity.)
    """
    folds: list[WalkForwardFold] = []
    fold = 0
    cur = start_dt
    while cur < end_dt:
        fold_start = cur
        fold_end = min(end_dt, fold_start + timedelta(days=int(test_window_days)))
        if (fold_end - fold_start).total_seconds() < 24 * 3600:
            break

        window_bars: dict[str, pd.DataFrame] = {}
        for sym, df in bars_by_symbol.items():
            w = df[(df.index >= fold_start) & (df.index <= fold_end)].copy()
            if not w.empty:
                window_bars[sym] = w

        res = run_backtest(
            bars_by_symbol=window_bars,
            starting_equity=starting_equity,
            risk_per_trade_pct=risk_per_trade_pct,
            max_gross_exposure_pct=max_gross_exposure_pct,
            stop_loss_atr_mult=stop_loss_atr_mult,
            take_profit_r_mult=take_profit_r_mult,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            strategy_params=strategy_params,
        )
        included = len(res.trades) >= int(min_trades_per_fold)
        note = None if included else f"low_trades<{min_trades_per_fold}"
        folds.append(
            WalkForwardFold(
                fold=fold,
                start=fold_start,
                end=fold_end,
                total_return=float(res.total_return),
                sharpe_daily=res.sharpe_daily,
                max_drawdown=res.max_drawdown,
                trades=len(res.trades),
                included=included,
                note=note,
            )
        )

        fold += 1
        cur = cur + timedelta(days=int(step_days))

    return folds


@dataclass(frozen=True)
class BucketStats:
    label: str
    trades: int
    pnl: float
    win_rate: float | None
    expectancy: float | None
    expectancy_r: float | None


def time_of_day_breakdown(trades: list[Trade]) -> list[BucketStats]:
    """
    Buckets by entry time in America/New_York:
    - open:   09:30–11:00
    - midday: 11:00–15:00
    - late:   15:00–16:00
    """
    ny = ZoneInfo("America/New_York")

    def bucket(dt_utc: datetime) -> str:
        dt = dt_utc.astimezone(ny)
        t = dt.time()
        if (t.hour < 11) and not (t.hour < 9 or (t.hour == 9 and t.minute < 30)):
            return "open"
        if t.hour < 15:
            return "midday"
        return "late"

    groups: dict[str, list[Trade]] = {"open": [], "midday": [], "late": []}
    for tr in trades:
        groups[bucket(tr.entry_ts)].append(tr)

    return [_bucket_stats(k, groups[k]) for k in ("open", "midday", "late")]


def _bucket_stats(label: str, trades: list[Trade]) -> BucketStats:
    if not trades:
        return BucketStats(label, 0, 0.0, None, None, None)
    pnls = np.asarray([t.pnl for t in trades], dtype=float)
    pnl = float(pnls.sum())
    win_rate = float((pnls > 0).mean())
    expectancy = float(pnls.mean())
    rs = [t.pnl_r for t in trades if t.pnl_r is not None]
    exp_r = float(np.mean(rs)) if rs else None
    return BucketStats(label, len(trades), pnl, win_rate, expectancy, exp_r)


def buy_and_hold_returns(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym, df in bars_by_symbol.items():
        if df is None or df.empty:
            continue
        try:
            start_px = float(df["close"].iloc[0])
            end_px = float(df["close"].iloc[-1])
            if start_px > 0:
                out[sym] = end_px / start_px - 1.0
        except Exception:
            continue
    return out


@dataclass(frozen=True)
class RegimeLabel:
    adx_regime: str | None  # "trending" | "ranging" | None
    vol_regime: str | None  # "high_vol" | "low_vol" | None


def compute_spy_regimes(bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (adx_regime_15m, vol_regime_1m) series indexed by UTC timestamps."""
    if "SPY" not in bars_by_symbol:
        return None, None

    spy = bars_by_symbol["SPY"]
    if spy is None or spy.empty:
        return None, None

    import pandas_ta as ta

    spy15 = resample_ohlcv(spy, rule="15min")
    adx_reg = None
    if spy15 is not None and not spy15.empty and len(spy15) > 30:
        adx = ta.adx(spy15["high"], spy15["low"], spy15["close"], length=14)
        if adx is not None and "ADX_14" in adx.columns:
            s = adx["ADX_14"].dropna()
            adx_reg = (s > 25).map(lambda x: "trending" if x else "ranging")

    closes = spy["close"].astype(float)
    logret = np.log(closes).diff()
    vol = logret.rolling(60).std().dropna()
    vol_reg = None
    if not vol.empty:
        med = float(vol.median())
        vol_reg = (vol > med).map(lambda x: "high_vol" if x else "low_vol")

    return adx_reg, vol_reg


def label_trade_regime(entry_ts: datetime, adx_reg_15m: pd.Series | None, vol_reg_1m: pd.Series | None) -> RegimeLabel:
    ts = pd.Timestamp(entry_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    adx_label = None
    if adx_reg_15m is not None and not adx_reg_15m.empty:
        s = adx_reg_15m[adx_reg_15m.index <= ts]
        if not s.empty:
            adx_label = str(s.iloc[-1])

    vol_label = None
    if vol_reg_1m is not None and not vol_reg_1m.empty:
        s = vol_reg_1m[vol_reg_1m.index <= ts]
        if not s.empty:
            vol_label = str(s.iloc[-1])

    return RegimeLabel(adx_label, vol_label)


@dataclass(frozen=True)
class ParamSweepRow:
    params: dict
    total_return: float
    sharpe_daily: float | None
    max_drawdown: float | None
    trades: int


def run_param_sweep(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    starting_equity: float,
    risk_per_trade_pct: float,
    max_gross_exposure_pct: float,
    stop_loss_atr_mult: float,
    take_profit_r_mult: float,
    slippage_bps: float,
    commission_bps: float,
    grid: list[dict],
) -> list[ParamSweepRow]:
    rows: list[ParamSweepRow] = []
    for params in grid:
        res = run_backtest(
            bars_by_symbol=bars_by_symbol,
            starting_equity=starting_equity,
            risk_per_trade_pct=risk_per_trade_pct,
            max_gross_exposure_pct=max_gross_exposure_pct,
            stop_loss_atr_mult=stop_loss_atr_mult,
            take_profit_r_mult=take_profit_r_mult,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            strategy_params=params,
        )
        rows.append(
            ParamSweepRow(
                params=params,
                total_return=float(res.total_return),
                sharpe_daily=res.sharpe_daily,
                max_drawdown=res.max_drawdown,
                trades=len(res.trades),
            )
        )
    return rows

