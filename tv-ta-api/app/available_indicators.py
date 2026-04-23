from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import pandas as pd


def _is_indicator_method(name: str, obj: Any) -> bool:
    if name.startswith("_"):
        return False
    if not callable(obj):
        return False
    # Exclude common non-indicator helpers on df.ta
    if name in {
        "help",
        "constants",
        "categories",
        "strategies",
        "strategy",
        "reverse",
        "adjusted",
        "datetime_ordered",
        "to_utc",
        "study",
        "indicator",
        "indicators",
        "cores",
        "version",
        "data",
        "config",
        "utils",
        "signals",
    }:
        return False
    return True


@lru_cache(maxsize=1)
def available_indicators_catalog() -> list[dict[str, Any]]:
    """
    Build a catalog from pandas_ta as installed in this environment.
    This is intentionally cached since introspection is relatively expensive.
    """
    df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    ta = df.ta

    items: list[dict[str, Any]] = []
    for name in dir(ta):
        try:
            obj = getattr(ta, name)
        except Exception:
            continue
        if not _is_indicator_method(name, obj):
            continue

        sig = None
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = None

        doc = None
        try:
            d = inspect.getdoc(obj) or ""
            doc = d.splitlines()[0].strip() if d else None
        except Exception:
            doc = None

        items.append({"name": name, "signature": sig, "doc": doc})

    items.sort(key=lambda x: x["name"])
    return items

