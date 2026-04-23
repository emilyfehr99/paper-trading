from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import pandas_ta as ta


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
            macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
            if macd_df is None or macd_df.empty:
                out["macd"] = float("nan")
                out["macd_signal"] = float("nan")
            else:
                # pandas_ta column names: MACD_{fast}_{slow}_{signal}, MACDs_..., MACDh_...
                macd_col = next((c for c in macd_df.columns if c.startswith("MACD_")), None)
                sig_col = next((c for c in macd_df.columns if c.startswith("MACDs_")), None)
                if macd_col:
                    s = macd_df[macd_col].dropna()
                    out["macd"] = float(s.iloc[-1]) if not s.empty else float("nan")
                if sig_col:
                    s = macd_df[sig_col].dropna()
                    out["macd_signal"] = float(s.iloc[-1]) if not s.empty else float("nan")

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
) -> list[dict[str, float | int]]:
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
    else:
        raise ValueError(f"Unsupported indicator for series: {indicator}")

    if key_series is None or key_series.empty:
        return []

    key_series = key_series.dropna().tail(count)
    pts: list[dict[str, float | int]] = []
    for ts, v in key_series.items():
        pts.append({"t": int(pd.Timestamp(ts).timestamp()), "v": float(v)})
    return pts

