"""Map futures date ranges to specific TradingView contract symbols (no auth required)."""

from __future__ import annotations

from calendar import monthcalendar
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

# CME quarterly index futures: Mar(H), Jun(M), Sep(U), Dec(Z)
_QUARTERLY: tuple[tuple[int, str], ...] = ((3, "H"), (6, "M"), (9, "U"), (12, "Z"))
_MONTH_CODES = "FGHJKMNQUVXZ"

# Bot / yfinance root → (exchange prefix, root ticker)
# Full-size ES/NQ/RTY use CME_MINI on TradingView free sessions (CME:ES1! → 403/0 bars).
_FUTURES_ROOTS: dict[str, tuple[str, str]] = {
    "MNQ=F": ("CME_MINI", "MNQ"),
    "MES=F": ("CME_MINI", "MES"),
    "M2K=F": ("CME_MINI", "M2K"),
    "MCL=F": ("NYMEX", "MCL"),
    "MGC=F": ("COMEX", "MGC"),
    "MYM=F": ("CBOT_MINI", "MYM"),
    "NQ=F": ("CME_MINI", "NQ"),
    "ES=F": ("CME_MINI", "ES"),
    "RTY=F": ("CME_MINI", "RTY"),
    "YM=F": ("CBOT", "YM"),
    "MNQ1": ("CME_MINI", "MNQ"),
    "MES1": ("CME_MINI", "MES"),
    "M2K1": ("CME_MINI", "M2K"),
    "MCL1": ("NYMEX", "MCL"),
    "MGC1": ("COMEX", "MGC"),
    "MYM1": ("CBOT_MINI", "MYM"),
    "NQ1": ("CME_MINI", "NQ"),
    "ES1": ("CME_MINI", "ES"),
    "RTY1": ("CME_MINI", "RTY"),
    "YM1": ("CBOT", "YM"),
    "MNQ1!": ("CME_MINI", "MNQ"),
    "MES1!": ("CME_MINI", "MES"),
    "M2K1!": ("CME_MINI", "M2K"),
    "NQ1!": ("CME_MINI", "NQ"),
    "ES1!": ("CME_MINI", "ES"),
    "RTY1!": ("CME_MINI", "RTY"),
    "YM1!": ("CBOT", "YM"),
    "MYM1!": ("CBOT_MINI", "MYM"),
}

# Full-size / non-mini roots — used when =F suffix is ambiguous.
_FULL_SIZE_ROOTS: frozenset[str] = frozenset({"NQ", "ES", "RTY", "YM", "CL", "GC"})
_MINI_ROOTS: frozenset[str] = frozenset({"MNQ", "MES", "M2K", "MCL", "MGC", "MYM"})
# TV free-tier exchange for full-size index roots (not CME:).
_TV_FULL_SIZE_EXCHANGE: dict[str, str] = {
    "NQ": "CME_MINI",
    "ES": "CME_MINI",
    "RTY": "CME_MINI",
    "YM": "CBOT",
}


@dataclass(frozen=True)
class ContractSegment:
    tv_symbol: str
    start: datetime
    end: datetime


def _third_friday(year: int, month: int) -> date:
    fridays = [week[4] for week in monthcalendar(year, month) if week[4] != 0]
    return date(year, month, fridays[2])


def _contract_expiry(year: int, month: int) -> date:
    """Approximate last trading day (3rd Friday of contract month)."""
    return _third_friday(year, month)


def _contract_tv_symbol(exchange: str, root: str, year: int, month: int) -> str:
    code = _MONTH_CODES[month - 1]
    return f"{exchange}:{root}{code}{year}"


def _parse_futures_root(symbol: str) -> tuple[str, str] | None:
    s = symbol.strip().upper()
    if s in _FUTURES_ROOTS:
        return _FUTURES_ROOTS[s]
    if s.endswith("=F"):
        root = s[:-2]
        if root in _FULL_SIZE_ROOTS:
            if root in {"CL", "GC"}:
                return ("NYMEX" if root == "CL" else "COMEX", root)
            return (_TV_FULL_SIZE_EXCHANGE.get(root, "CME_MINI"), root)
        if root in _MINI_ROOTS:
            if root == "MYM":
                return ("CBOT_MINI", root)
            if root == "MCL":
                return ("NYMEX", root)
            if root == "MGC":
                return ("COMEX", root)
            return ("CME_MINI", root)
        return ("CME_MINI", root)
    if s.endswith("1") and len(s) <= 5:
        root = s[:-1]
        if root in _FULL_SIZE_ROOTS:
            if root in {"CL", "GC"}:
                return ("NYMEX" if root == "CL" else "COMEX", root)
            return (_TV_FULL_SIZE_EXCHANGE.get(root, "CME_MINI"), root)
        if root == "MYM":
            return ("CBOT_MINI", root)
        if root == "MCL":
            return ("NYMEX", root)
        if root == "MGC":
            return ("COMEX", root)
        return ("CME_MINI", root)
    if ":" in s:
        exch, rest = s.split(":", 1)
        if rest.endswith("1!"):
            root = rest[:-2]
            if root in _FULL_SIZE_ROOTS:
                return (exch, root)
            return (exch, root)
        if len(rest) >= 5 and rest[-5] in "FGHJKMNQUVXZ" and rest[-4:].isdigit():
            return (exch, rest[:-5])
    return None


def _quarterly_contracts_between(start: date, end: date) -> list[tuple[int, int]]:
    """(year, month) pairs for H/M/U/Z contracts that may cover [start, end]."""
    out: list[tuple[int, int]] = []
    for year in range(start.year - 1, end.year + 2):
        for month, _ in _QUARTERLY:
            expiry = _contract_expiry(year, month)
            # Contract can trade ~1 year before expiry; include if overlap with range.
            window_start = expiry - timedelta(days=365)
            if window_start <= end and expiry >= start:
                out.append((year, month))
    return out


def contract_segments_for_range(
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[ContractSegment]:
    """
    Build specific contract fetches for a futures date range.

    Anonymous TradingView access returns deep history on named contracts
    (e.g. CME_MINI:MNQH2026) but not on continuous MNQ1!.
    """
    parsed = _parse_futures_root(symbol)
    if parsed is None:
        return []

    exchange, root = parsed
    start_d = start.astimezone(UTC).date() if start.tzinfo else start.date()
    end_d = end.astimezone(UTC).date() if end.tzinfo else end.date()
    if end_d < start_d:
        return []

    segments: list[ContractSegment] = []
    for year, month in _quarterly_contracts_between(start_d, end_d):
        expiry = _contract_expiry(year, month)
        # Active window: day after prior quarterly expiry → this contract expiry (inclusive).
        prev_month, prev_year = month, year
        idx = [m for m, _ in _QUARTERLY].index(month)
        if idx == 0:
            prev_month, prev_year = 12, year - 1
        else:
            prev_month = _QUARTERLY[idx - 1][0]
        active_from = _contract_expiry(prev_year, prev_month) + timedelta(days=1)
        active_to = expiry

        seg_start = max(start_d, active_from)
        seg_end = min(end_d, active_to)
        if seg_start > seg_end:
            continue

        tv_sym = _contract_tv_symbol(exchange, root, year, month)
        seg_start_dt = datetime.combine(seg_start, time.min, tzinfo=UTC)
        seg_end_dt = datetime.combine(seg_end, time(23, 59, 59), tzinfo=UTC)
        segments.append(ContractSegment(tv_sym, seg_start_dt, seg_end_dt))

    return segments
