"""
Candlestick Pattern Detection Engine
=====================================
Implements all major single, double, and triple candle patterns following
the 4 golden rules:
  1. Context  — only fire at the end of an established trend
  2. Location — pattern must be at a key level (VWAP, EMA, support/resistance)
  3. Closed   — only reads completed, closed candles (caller's responsibility)
  4. Volume   — high-volume patterns weighted more strongly

Patterns implemented:
  Single  : Doji, Hammer, Hanging Man, Shooting Star, Inverted Hammer,
            Marubozu (Bullish/Bearish)
  Double  : Bullish Engulfing, Bearish Engulfing, Tweezer Bottom/Top,
            Bullish/Bearish Harami
  Triple  : Morning Star, Evening Star, Rising Three Methods,
            Falling Three Methods, Three White Soldiers, Three Black Crows
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class CandlePattern:
    name: str                        # e.g. "hammer", "bullish_engulfing"
    direction: str                   # "bullish" or "bearish"
    strength: float                  # 0.0–1.0  (quality score)
    volume_confirmed: bool = False   # True if vol > avg vol
    at_key_level: bool = False       # True if price near VWAP / EMA / S&R
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helper geometry
# ─────────────────────────────────────────────────────────────────────────────

def _body(o, c):       return abs(c - o)
def _range(h, l):      return h - l if h > l else 1e-9
def _upper_wick(o, c, h): return h - max(o, c)
def _lower_wick(o, c, l): return min(o, c) - l
def _is_green(o, c):   return c > o
def _is_red(o, c):     return c < o
def _body_ratio(o, c, h, l): return _body(o, c) / _range(h, l)


# ─────────────────────────────────────────────────────────────────────────────
# Single-candle patterns
# ─────────────────────────────────────────────────────────────────────────────

def detect_doji(o, c, h, l, *, doji_ratio: float = 0.10) -> CandlePattern | None:
    """Body < 10 % of total range → indecision / potential reversal warning."""
    if _body_ratio(o, c, h, l) < doji_ratio:
        return CandlePattern("doji", "neutral", strength=0.5,
                             description="Open ≈ Close — market indecision")
    return None


def detect_hammer(o, c, h, l) -> CandlePattern | None:
    """
    Hammer (bullish reversal at downtrend bottom):
      - Small body in the upper third of the candle range
      - Lower wick ≥ 2× body
      - Upper wick ≤ body
    """
    body = _body(o, c)
    rng  = _range(h, l)
    lower = _lower_wick(o, c, l)
    upper = _upper_wick(o, c, h)
    if body < 1e-9 or rng < 1e-9:
        return None
    body_pct = body / rng
    if body_pct < 0.05 or body_pct > 0.40:   # body must be small but real
        return None
    if lower < 2.0 * body:
        return None
    if upper > body:
        return None
    strength = min(1.0, lower / body / 3.0)   # longer wick = stronger signal
    return CandlePattern("hammer", "bullish", strength=strength,
                         description=f"Hammer — sellers rejected, lower wick={lower:.4f}")


def detect_shooting_star(o, c, h, l) -> CandlePattern | None:
    """
    Shooting Star (bearish reversal at uptrend top):
      - Small body in the lower third
      - Upper wick ≥ 2× body
      - Lower wick ≤ body
    """
    body  = _body(o, c)
    rng   = _range(h, l)
    upper = _upper_wick(o, c, h)
    lower = _lower_wick(o, c, l)
    if body < 1e-9 or rng < 1e-9:
        return None
    body_pct = body / rng
    if body_pct < 0.05 or body_pct > 0.40:
        return None
    if upper < 2.0 * body:
        return None
    if lower > body:
        return None
    strength = min(1.0, upper / body / 3.0)
    return CandlePattern("shooting_star", "bearish", strength=strength,
                         description=f"Shooting Star — buyers rejected, upper wick={upper:.4f}")


def detect_inverted_hammer(o, c, h, l) -> CandlePattern | None:
    """Inverted hammer — same geometry as shooting star but occurs at a bottom (bullish)."""
    body  = _body(o, c)
    rng   = _range(h, l)
    upper = _upper_wick(o, c, h)
    lower = _lower_wick(o, c, l)
    if body < 1e-9 or rng < 1e-9:
        return None
    body_pct = body / rng
    if body_pct < 0.05 or body_pct > 0.40:
        return None
    if upper < 2.0 * body:
        return None
    if lower > body:
        return None
    strength = min(1.0, upper / body / 3.0) * 0.8   # slightly weaker than hammer
    return CandlePattern("inverted_hammer", "bullish", strength=strength,
                         description="Inverted Hammer — potential bullish reversal")


def detect_marubozu(o, c, h, l, *, wick_ratio: float = 0.05) -> CandlePattern | None:
    """
    Marubozu — body fills ≥ 95 % of range (almost no wicks).
    Strong momentum candle in its own color direction.
    """
    rng = _range(h, l)
    if _body_ratio(o, c, h, l) >= (1.0 - wick_ratio):
        direction = "bullish" if _is_green(o, c) else "bearish"
        return CandlePattern("marubozu", direction, strength=0.9,
                             description=f"Marubozu — pure {direction} momentum")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Two-candle patterns  (pass prev then curr)
# ─────────────────────────────────────────────────────────────────────────────

def detect_bullish_engulfing(po, pc, ph, pl, o, c, h, l) -> CandlePattern | None:
    """
    Bullish Engulfing:
      - Prev candle is red
      - Current candle is green AND its body completely engulfs the prev body
    """
    if not _is_red(po, pc):
        return None
    if not _is_green(o, c):
        return None
    if c > po and o < pc:   # curr body engulfs prev body
        body_ratio = _body(o, c) / max(_body(po, pc), 1e-9)
        strength = min(1.0, body_ratio / 2.0)
        return CandlePattern("bullish_engulfing", "bullish", strength=strength,
                             description=f"Bullish Engulfing — momentum surge (body×{body_ratio:.1f})")
    return None


def detect_bearish_engulfing(po, pc, ph, pl, o, c, h, l) -> CandlePattern | None:
    """
    Bearish Engulfing:
      - Prev candle is green
      - Current candle is red AND its body completely engulfs the prev body
    """
    if not _is_green(po, pc):
        return None
    if not _is_red(o, c):
        return None
    if c < po and o > pc:
        body_ratio = _body(o, c) / max(_body(po, pc), 1e-9)
        strength = min(1.0, body_ratio / 2.0)
        return CandlePattern("bearish_engulfing", "bearish", strength=strength,
                             description=f"Bearish Engulfing — sellers overwhelm buyers (body×{body_ratio:.1f})")
    return None


def detect_bullish_harami(po, pc, ph, pl, o, c, h, l) -> CandlePattern | None:
    """
    Bullish Harami (inside bar):
      - Prev candle is large red
      - Current small green candle body is completely inside prev body
    """
    if not _is_red(po, pc) or not _is_green(o, c):
        return None
    prev_body = _body(po, pc)
    if prev_body < 1e-9:
        return None
    if o > pc and c < po:      # inside prev body
        ratio = _body(o, c) / prev_body
        if ratio < 0.5:        # small inside bar
            return CandlePattern("bullish_harami", "bullish", strength=0.55,
                                 description="Bullish Harami — inside bar, momentum stalling")
    return None


def detect_bearish_harami(po, pc, ph, pl, o, c, h, l) -> CandlePattern | None:
    """
    Bearish Harami — inverse of bullish harami.
    """
    if not _is_green(po, pc) or not _is_red(o, c):
        return None
    prev_body = _body(po, pc)
    if prev_body < 1e-9:
        return None
    if o < pc and c > po:
        ratio = _body(o, c) / prev_body
        if ratio < 0.5:
            return CandlePattern("bearish_harami", "bearish", strength=0.55,
                                 description="Bearish Harami — inside bar, upside stalling")
    return None


def detect_tweezer_bottom(po, pc, ph, pl, o, c, h, l, *, tol: float = 0.001) -> CandlePattern | None:
    """Tweezer Bottom: two consecutive candles share the same low (±tol %)."""
    if abs(pl - l) / max(abs(l), 1e-9) <= tol and _is_green(o, c):
        return CandlePattern("tweezer_bottom", "bullish", strength=0.65,
                             description=f"Tweezer Bottom — double-tested support at {l:.4f}")
    return None


def detect_tweezer_top(po, pc, ph, pl, o, c, h, l, *, tol: float = 0.001) -> CandlePattern | None:
    """Tweezer Top: two consecutive candles share the same high (±tol %)."""
    if abs(ph - h) / max(abs(h), 1e-9) <= tol and _is_red(o, c):
        return CandlePattern("tweezer_top", "bearish", strength=0.65,
                             description=f"Tweezer Top — double-tested resistance at {h:.4f}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Three-candle patterns  (pass c1→c2→c3, oldest first)
# ─────────────────────────────────────────────────────────────────────────────

def detect_morning_star(c1, c2, c3) -> CandlePattern | None:
    """
    Morning Star (bullish reversal):
      c1: large red candle
      c2: small body (doji-like) that gaps down from c1
      c3: large green candle that closes well into c1's body
    """
    o1,c1_c,h1,l1 = c1
    o2,c2_c,h2,l2 = c2
    o3,c3_c,h3,l3 = c3

    if not _is_red(o1, c1_c):     return None
    if not _is_green(o3, c3_c):   return None

    c1_body = _body(o1, c1_c)
    c2_body = _body(o2, c2_c)
    c3_body = _body(o3, c3_c)

    if c1_body < 1e-9 or c3_body < 1e-9:
        return None
    # c2 must be small relative to c1
    if c2_body > 0.35 * c1_body:
        return None
    # c3 must close above midpoint of c1's body
    c1_mid = (o1 + c1_c) / 2.0
    if c3_c <= c1_mid:
        return None
    strength = min(1.0, c3_body / c1_body)
    return CandlePattern("morning_star", "bullish", strength=strength,
                         description="Morning Star — three-candle bullish reversal")


def detect_evening_star(c1, c2, c3) -> CandlePattern | None:
    """
    Evening Star (bearish reversal):
      c1: large green candle
      c2: small body that gaps up from c1
      c3: large red candle closing well into c1's body
    """
    o1,c1_c,h1,l1 = c1
    o2,c2_c,h2,l2 = c2
    o3,c3_c,h3,l3 = c3

    if not _is_green(o1, c1_c):   return None
    if not _is_red(o3, c3_c):     return None

    c1_body = _body(o1, c1_c)
    c2_body = _body(o2, c2_c)
    c3_body = _body(o3, c3_c)

    if c1_body < 1e-9 or c3_body < 1e-9:
        return None
    if c2_body > 0.35 * c1_body:
        return None
    c1_mid = (o1 + c1_c) / 2.0
    if c3_c >= c1_mid:
        return None
    strength = min(1.0, c3_body / c1_body)
    return CandlePattern("evening_star", "bearish", strength=strength,
                         description="Evening Star — three-candle bearish reversal")


def detect_three_white_soldiers(c1, c2, c3) -> CandlePattern | None:
    """Three consecutive green candles each closing higher — strong trend confirmation."""
    o1,c1_c,h1,l1 = c1
    o2,c2_c,h2,l2 = c2
    o3,c3_c,h3,l3 = c3
    if _is_green(o1,c1_c) and _is_green(o2,c2_c) and _is_green(o3,c3_c):
        if c2_c > c1_c and c3_c > c2_c:
            if o2 > o1 and o3 > o2:   # each opens within prior body
                return CandlePattern("three_white_soldiers", "bullish", strength=0.85,
                                     description="Three White Soldiers — strong bullish trend")
    return None


def detect_three_black_crows(c1, c2, c3) -> CandlePattern | None:
    """Three consecutive red candles each closing lower — strong bearish trend."""
    o1,c1_c,h1,l1 = c1
    o2,c2_c,h2,l2 = c2
    o3,c3_c,h3,l3 = c3
    if _is_red(o1,c1_c) and _is_red(o2,c2_c) and _is_red(o3,c3_c):
        if c2_c < c1_c and c3_c < c2_c:
            if o2 < o1 and o3 < o2:
                return CandlePattern("three_black_crows", "bearish", strength=0.85,
                                     description="Three Black Crows — strong bearish trend")
    return None


def detect_rising_three_methods(candles) -> CandlePattern | None:
    """
    Rising Three Methods (5-candle bullish continuation):
      c1: large green candle
      c2-c4: 3 small red candles contained within c1's range
      c5: large green candle closing above c1's high
    """
    if len(candles) < 5:
        return None
    o1,c1_c,h1,l1 = candles[0]
    o5,c5_c,h5,l5 = candles[4]
    if not _is_green(o1,c1_c) or not _is_green(o5,c5_c):
        return None
    if c5_c <= h1:   # must break above c1's high
        return None
    # middle 3 must be red and contained
    for i in range(1,4):
        oi,ci_c,hi,li = candles[i]
        if not _is_red(oi,ci_c): return None
        if hi > h1 or li < l1:  return None
    return CandlePattern("rising_three_methods", "bullish", strength=0.80,
                         description="Rising Three Methods — institutional buyers resuming")


def detect_falling_three_methods(candles) -> CandlePattern | None:
    """
    Falling Three Methods (5-candle bearish continuation):
      c1: large red candle
      c2-c4: 3 small green candles contained within c1's range
      c5: large red candle closing below c1's low
    """
    if len(candles) < 5:
        return None
    o1,c1_c,h1,l1 = candles[0]
    o5,c5_c,h5,l5 = candles[4]
    if not _is_red(o1,c1_c) or not _is_red(o5,c5_c):
        return None
    if c5_c >= l1:
        return None
    for i in range(1,4):
        oi,ci_c,hi,li = candles[i]
        if not _is_green(oi,ci_c): return None
        if hi > h1 or li < l1:    return None
    return CandlePattern("falling_three_methods", "bearish", strength=0.80,
                         description="Falling Three Methods — institutional sellers resuming")


# ─────────────────────────────────────────────────────────────────────────────
# Main scanner — runs all patterns on a DataFrame, returns best match
# ─────────────────────────────────────────────────────────────────────────────

def scan_patterns(
    df: pd.DataFrame,
    *,
    vwap: float | None = None,
    ema_fast: float | None = None,
    ema_slow: float | None = None,
    avg_volume: float | None = None,
    key_level_tol: float = 0.005,   # 0.5% proximity counts as "at a level"
) -> list[CandlePattern]:
    """
    Run all pattern detectors on the last 5 closed candles of `df`.
    Returns a list of detected patterns sorted by strength descending.

    Golden-rule enrichment:
      - volume_confirmed = True  if current candle volume > avg_volume
      - at_key_level     = True  if close is within key_level_tol of VWAP or EMA
    """
    if df is None or len(df) < 5:
        return []

    # Work on a view of the last 5 bars — all already closed (caller ensures this)
    tail = df.tail(5)
    rows = [(r["open"], r["close"], r["high"], r["low"]) for _, r in tail.iterrows()]

    # Unpack last 3 for convenience
    *_, c3_row, c2_row, c1_row = rows[-3], rows[-2], rows[-1]  # c1=most recent
    o1, c1, h1, l1 = rows[-1]
    o2, c2, h2, l2 = rows[-2]
    o3, c3, h3, l3 = rows[-3]

    vol_now = float(tail.iloc[-1].get("volume", 0))
    vol_confirmed = (avg_volume is not None and avg_volume > 0 and vol_now > avg_volume)

    close = c1
    at_key = False
    if vwap is not None and abs(close - vwap) / max(abs(vwap), 1e-9) <= key_level_tol:
        at_key = True
    if ema_fast is not None and abs(close - ema_fast) / max(abs(ema_fast), 1e-9) <= key_level_tol:
        at_key = True
    if ema_slow is not None and abs(close - ema_slow) / max(abs(ema_slow), 1e-9) <= key_level_tol:
        at_key = True

    detected: list[CandlePattern] = []

    def _add(p):
        if p is not None:
            p.volume_confirmed = vol_confirmed
            p.at_key_level = at_key
            detected.append(p)

    # Single-candle
    _add(detect_doji(o1, c1, h1, l1))
    _add(detect_hammer(o1, c1, h1, l1))
    _add(detect_shooting_star(o1, c1, h1, l1))
    _add(detect_inverted_hammer(o1, c1, h1, l1))
    _add(detect_marubozu(o1, c1, h1, l1))

    # Two-candle
    _add(detect_bullish_engulfing(o2, c2, h2, l2, o1, c1, h1, l1))
    _add(detect_bearish_engulfing(o2, c2, h2, l2, o1, c1, h1, l1))
    _add(detect_bullish_harami(o2, c2, h2, l2, o1, c1, h1, l1))
    _add(detect_bearish_harami(o2, c2, h2, l2, o1, c1, h1, l1))
    _add(detect_tweezer_bottom(o2, c2, h2, l2, o1, c1, h1, l1))
    _add(detect_tweezer_top(o2, c2, h2, l2, o1, c1, h1, l1))

    # Three-candle
    _add(detect_morning_star(rows[-3], rows[-2], rows[-1]))
    _add(detect_evening_star(rows[-3], rows[-2], rows[-1]))
    _add(detect_three_white_soldiers(rows[-3], rows[-2], rows[-1]))
    _add(detect_three_black_crows(rows[-3], rows[-2], rows[-1]))

    # Five-candle
    if len(rows) >= 5:
        _add(detect_rising_three_methods(rows[-5:]))
        _add(detect_falling_three_methods(rows[-5:]))

    # Boost strength for volume-confirmed & at-key-level patterns
    for p in detected:
        if p.volume_confirmed: p.strength = min(1.0, p.strength * 1.25)
        if p.at_key_level:     p.strength = min(1.0, p.strength * 1.20)

    # Sort strongest first, exclude neutral doji unless nothing else found
    bullish  = sorted([p for p in detected if p.direction == "bullish"],
                       key=lambda x: x.strength, reverse=True)
    bearish  = sorted([p for p in detected if p.direction == "bearish"],
                       key=lambda x: x.strength, reverse=True)
    neutral  = [p for p in detected if p.direction == "neutral"]

    return bullish + bearish + neutral
