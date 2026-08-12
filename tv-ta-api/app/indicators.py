from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import pandas_ta as ta

from app.bill_williams import macd_histogram_latest, williams_alligator_latest
from app.stoch_v_rsi import stoch_v_rsi_latest


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    params: dict[str, Any]


_SPEC_RE = re.compile(r"^(?P<name>[a-zA-Z_]+)(?::(?P<arg>[^,]+))?$")


def parse_indicator_list(indicators: str) -> list[IndicatorSpec]:
    """
    Accepts:
      - "rsi,sma,macd,bbands"
      - "rsi:14,sma:50,bbands:20-2,macd:12-26-9"
    """
    out: list[IndicatorSpec] = []
    for raw in [p.strip() for p in indicators.split(",") if p.strip()]:
        m = _SPEC_RE.match(raw)
        if not m:
            continue
        name = m.group("name").lower()
        arg = m.group("arg")
        if name == "rsi":
            length = int(arg) if arg else 14
            out.append(IndicatorSpec(name="rsi", params={"length": length}))
        elif name == "sma":
            length = int(arg) if arg else 20
            out.append(IndicatorSpec(name="sma", params={"length": length}))
        elif name == "ema":
            length = int(arg) if arg else 20
            out.append(IndicatorSpec(name="ema", params={"length": length}))
        elif name == "macd":
            if arg and "-" in arg:
                fast, slow, signal = [int(x) for x in arg.split("-", 2)]
            else:
                fast, slow, signal = 12, 26, 9
            out.append(IndicatorSpec(name="macd", params={"fast": fast, "slow": slow, "signal": signal}))
        elif name in {"bbands", "bb"}:
            if arg and "-" in arg:
                length_s, std_s = arg.split("-", 1)
                length = int(length_s)
                std = float(std_s)
            else:
                length, std = 20, 2.0
            out.append(IndicatorSpec(name="bbands", params={"length": length, "std": std}))
        elif name in {"donchian", "donchian_channels", "dc"}:
            length = int(arg) if arg else 20
            out.append(IndicatorSpec(name="donchian", params={"length": length}))
        elif name in {"williamsr", "williams", "willr", "wr"}:
            length = int(arg) if arg else 14
            out.append(IndicatorSpec(name="willr", params={"length": length}))
        elif name == "vwap":
            out.append(IndicatorSpec(name="vwap", params={}))
        elif name == "atr":
            length = int(arg) if arg else 14
            out.append(IndicatorSpec(name="atr", params={"length": length}))
        elif name in {"alligator", "williams_alligator", "ag"}:
            out.append(IndicatorSpec(name="alligator", params={}))
        elif name in {"stochrsi", "stoch_v_rsi", "stoch_rsi"}:
            if arg and "|" in arg:
                params = {}
                for part in arg.split("|"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k.strip()] = int(v.strip())
            elif arg and "-" in arg:
                parts = [int(x) for x in arg.split("-")]
                params = {
                    "length_rsi": parts[0],
                    "length_stoch": parts[1] if len(parts) > 1 else 14,
                    "smooth_k": parts[2] if len(parts) > 2 else 3,
                    "smooth_d": parts[3] if len(parts) > 3 else 3,
                }
            else:
                params = {}
            out.append(IndicatorSpec(name="stochrsi", params=params))
        else:
            # Generic form: indicator[:k=v|k=v|...] or indicator (no params)
            params: dict[str, Any] = {}
            if arg:
                # "length=20|std=2" → {"length": 20, "std": 2}
                for part in [x.strip() for x in arg.split("|") if x.strip()]:
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if not k:
                        continue
                    # best-effort typing
                    try:
                        if "." in v:
                            params[k] = float(v)
                        else:
                            params[k] = int(v)
                    except Exception:
                        params[k] = v
            out.append(IndicatorSpec(name=name, params=params))
    return out


def _require_ohlc(df: pd.DataFrame) -> None:
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")


def compute_latest(df: pd.DataFrame, specs: Iterable[IndicatorSpec]) -> dict[str, float]:
    _require_ohlc(df)
    if df.empty:
        return {}

    out: dict[str, float] = {}

    for spec in specs:
        if spec.name == "rsi":
            length = int(spec.params["length"])
            s = ta.rsi(df["close"], length=length)
            key = f"rsi_{length}"
            val = float(s.dropna().iloc[-1]) if s is not None and not s.dropna().empty else float("nan")
            out[key] = val

        elif spec.name == "sma":
            length = int(spec.params["length"])
            s = ta.sma(df["close"], length=length)
            key = f"sma_{length}"
            val = float(s.dropna().iloc[-1]) if s is not None and not s.dropna().empty else float("nan")
            out[key] = val

        elif spec.name == "ema":
            length = int(spec.params["length"])
            s = ta.ema(df["close"], length=length)
            key = f"ema_{length}"
            val = float(s.dropna().iloc[-1]) if s is not None and not s.dropna().empty else float("nan")
            out[key] = val

        elif spec.name == "macd":
            fast = int(spec.params["fast"])
            slow = int(spec.params["slow"])
            signal = int(spec.params["signal"])
            out.update(macd_histogram_latest(df, fast=fast, slow=slow, signal=signal))

        elif spec.name == "alligator":
            out.update(williams_alligator_latest(df))

        elif spec.name == "stochrsi":
            out.update(
                stoch_v_rsi_latest(
                    df,
                    length_rsi=int(spec.params.get("length_rsi", 14)),
                    length_stoch=int(spec.params.get("length_stoch", 14)),
                    smooth_k=int(spec.params.get("smooth_k", 3)),
                    smooth_d=int(spec.params.get("smooth_d", 3)),
                )
            )

        elif spec.name == "bbands":
            length = int(spec.params["length"])
            std = float(spec.params["std"])
            bb = ta.bbands(df["close"], length=length, std=std)
            if bb is None or bb.empty:
                out["bb_upper"] = float("nan")
                out["bb_middle"] = float("nan")
                out["bb_lower"] = float("nan")
            else:
                # columns: BBU_{len}_{std}, BBM_..., BBL_...
                upper = next((c for c in bb.columns if c.startswith("BBU_")), None)
                mid = next((c for c in bb.columns if c.startswith("BBM_")), None)
                lower = next((c for c in bb.columns if c.startswith("BBL_")), None)
                if upper:
                    s = bb[upper].dropna()
                    out["bb_upper"] = float(s.iloc[-1]) if not s.empty else float("nan")
                if mid:
                    s = bb[mid].dropna()
                    out["bb_middle"] = float(s.iloc[-1]) if not s.empty else float("nan")
                if lower:
                    s = bb[lower].dropna()
                    out["bb_lower"] = float(s.iloc[-1]) if not s.empty else float("nan")

        elif spec.name == "donchian":
            length = int(spec.params.get("length", 20))
            dc = ta.donchian(df["high"], df["low"], lower_length=length, upper_length=length)
            if dc is None or dc.empty:
                out["donchian_upper"] = float("nan")
                out["donchian_middle"] = float("nan")
                out["donchian_lower"] = float("nan")
            else:
                upper = next((c for c in dc.columns if c.startswith("DCU_")), None)
                mid = next((c for c in dc.columns if c.startswith("DCM_")), None)
                lower = next((c for c in dc.columns if c.startswith("DCL_")), None)
                if upper:
                    s = dc[upper].dropna()
                    out["donchian_upper"] = float(s.iloc[-1]) if not s.empty else float("nan")
                if mid:
                    s = dc[mid].dropna()
                    out["donchian_middle"] = float(s.iloc[-1]) if not s.empty else float("nan")
                if lower:
                    s = dc[lower].dropna()
                    out["donchian_lower"] = float(s.iloc[-1]) if not s.empty else float("nan")

        elif spec.name == "willr":
            length = int(spec.params.get("length", 14))
            s = ta.willr(df["high"], df["low"], df["close"], length=length)
            key = f"willr_{length}"
            val = float(s.dropna().iloc[-1]) if s is not None and not s.dropna().empty else float("nan")
            out[key] = val

        elif spec.name == "vwap":
            if "volume" not in df.columns:
                out["vwap"] = float("nan")
            else:
                s = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
                val = float(s.dropna().iloc[-1]) if s is not None and not s.dropna().empty else float("nan")
                out["vwap"] = val

        elif spec.name == "atr":
            length = int(spec.params.get("length", 14))
            s = ta.atr(df["high"], df["low"], df["close"], length=length)
            key = f"atr_{length}"
            val = float(s.dropna().iloc[-1]) if s is not None and not s.dropna().empty else float("nan")
            out[key] = val

        else:
            # Generic pandas_ta: df.ta.<indicator>(**params) returning Series or DataFrame
            try:
                fn = getattr(df.ta, spec.name)
            except Exception:
                continue
            if not callable(fn):
                continue
            try:
                res = fn(**spec.params)
            except Exception:
                continue

            if isinstance(res, pd.Series):
                s = res.dropna()
                if not s.empty:
                    key = (res.name or spec.name).lower()
                    out[key] = float(s.iloc[-1])
            elif isinstance(res, pd.DataFrame):
                for col in res.columns:
                    s = res[col].dropna()
                    if s.empty:
                        continue
                    out[str(col).lower()] = float(s.iloc[-1])

    return out


def compute_series(
    df: pd.DataFrame,
    indicator: str,
    period: int,
    count: int,
) -> list[dict[str, Any]]:
    _require_ohlc(df)
    if df.empty:
        return []

    ind = indicator.lower()
    if ind == "rsi":
        s = ta.rsi(df["close"], length=period)
        key_series = s
    elif ind == "sma":
        key_series = ta.sma(df["close"], length=period)
    elif ind == "ema":
        key_series = ta.ema(df["close"], length=period)
    elif ind in {"willr", "williamsr", "wr"}:
        key_series = ta.willr(df["high"], df["low"], df["close"], length=period)
    elif ind == "vwap":
        if "volume" not in df.columns:
            raise ValueError("VWAP requires volume")
        # VWAP isn't really "period"-based, but we keep the query shape stable.
        key_series = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    elif ind == "macd":
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            return []
        macd_col = next((c for c in macd_df.columns if str(c).startswith("MACD_")), None)
        sig_col = next((c for c in macd_df.columns if str(c).startswith("MACDs_")), None)
        hist_col = next((c for c in macd_df.columns if str(c).startswith("MACDh_")), None)
        if not (macd_col and sig_col and hist_col):
            return []
        macd_df = macd_df.dropna().tail(count)
        pts: list[dict[str, Any]] = []
        for ts, row in macd_df.iterrows():
            pts.append({
                "t": int(pd.Timestamp(ts).timestamp()),
                "macd": float(row[macd_col]),
                "signal": float(row[sig_col]),
                "hist": float(row[hist_col]),
            })
        return pts
    elif ind in {"alligator", "williams_alligator", "ag"}:
        ag_df = ta.alligator(df["close"])
        if ag_df is None or ag_df.empty:
            return []
        jaw_c = next((c for c in ag_df.columns if str(c).startswith("AGj")), None)
        teeth_c = next((c for c in ag_df.columns if str(c).startswith("AGt")), None)
        lips_c = next((c for c in ag_df.columns if str(c).startswith("AGl")), None)
        if not (jaw_c and teeth_c and lips_c):
            return []
        ag_df = ag_df.dropna().tail(count)
        pts: list[dict[str, Any]] = []
        for ts, row in ag_df.iterrows():
            pts.append({
                "t": int(pd.Timestamp(ts).timestamp()),
                "jaw": float(row[jaw_c]),
                "teeth": float(row[teeth_c]),
                "lips": float(row[lips_c]),
            })
        return pts
    else:
        raise ValueError(f"Unsupported indicator for series: {indicator}")

    if key_series is None or key_series.empty:
        return []

    key_series = key_series.dropna().tail(count)
    pts_simple: list[dict[str, Any]] = []
    for ts, v in key_series.items():
        pts_simple.append({"t": int(pd.Timestamp(ts).timestamp()), "v": float(v)})
    return pts_simple

