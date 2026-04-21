from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger("alpaca_day_bot.news")

_ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
_TICKERTICK_FEED = "https://api.tickertick.com/feed"


def _normalize_gate_mode(mode: str) -> str:
    m = (mode or "log_only").strip().lower()
    allowed = {"off", "log_only", "skip_if_any", "skip_if_busy"}
    if m not in allowed:
        return "log_only"
    return m


def news_bundle_should_block(bundle: dict[str, Any], gate_mode: str, busy_min_articles: int) -> bool:
    mode = _normalize_gate_mode(gate_mode)
    if mode in ("off", "log_only"):
        return False
    if not bundle.get("ok", False):
        return False
    n = int(bundle.get("count", 0) or 0)
    if mode == "skip_if_any":
        return n > 0
    if mode == "skip_if_busy":
        return n >= max(1, int(busy_min_articles))
    return False


def fetch_symbol_news(
    *,
    api_key: str,
    secret_key: str,
    symbol: str,
    lookback_hours: float,
    limit: int,
) -> dict[str, Any]:
    """
    Alpaca Market Data news (same keys as bars; plan may limit access).
    """
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest

    sym = (symbol or "").strip().upper()
    if not sym:
        return {"ok": False, "error": "empty_symbol", "count": 0, "articles": []}

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=max(0.25, float(lookback_hours)))
    lim = max(1, min(int(limit), 50))

    try:
        client = NewsClient(api_key, secret_key)
        req = NewsRequest(start=start, end=end, symbols=sym, limit=lim)
        ns = client.get_news(req)
        raw_list = ns.data.get("news", []) if getattr(ns, "data", None) else []
    except Exception as e:
        log.warning("news_fetch_failed symbol=%s err=%s", sym, e)
        return {"ok": False, "error": str(e), "count": 0, "articles": [], "symbol": sym}

    articles: list[dict[str, Any]] = []
    for a in raw_list:
        try:
            created = getattr(a, "created_at", None)
            articles.append(
                {
                    "id": getattr(a, "id", None),
                    "headline": (getattr(a, "headline", None) or "")[:240],
                    "source": getattr(a, "source", None),
                    "created_at": created.isoformat() if created is not None else None,
                }
            )
        except Exception:
            continue

    return {
        "ok": True,
        "symbol": sym,
        "count": len(articles),
        "articles": articles,
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
    }


def _alphavantage_time_published_to_iso(tp: str | None) -> str | None:
    if not tp or not isinstance(tp, str) or len(tp) < 15:
        return None
    try:
        y, m, d = tp[0:4], tp[4:6], tp[6:8]
        hh, mm, ss = tp[9:11], tp[11:13], tp[13:15]
        return f"{y}-{m}-{d}T{hh}:{mm}:{ss}"
    except Exception:
        return None


def fetch_alphavantage_news(*, api_key: str, symbol: str, limit: int) -> dict[str, Any]:
    """
    Alpha Vantage NEWS_SENTIMENT (see https://www.alphavantage.co/documentation/).
    Free tier is rate-limited (~5 calls/min); prefer NEWS_PROVIDER=alphavantage only when needed.
    """
    sym = (symbol or "").strip().upper()
    key = (api_key or "").strip()
    if not sym:
        return {"ok": False, "error": "empty_symbol", "provider": "alphavantage", "count": 0, "articles": []}
    if not key:
        return {"ok": False, "error": "missing_api_key", "provider": "alphavantage", "symbol": sym, "count": 0, "articles": []}

    lim = max(1, min(int(limit), 50))
    params = urllib.parse.urlencode(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": sym,
            "limit": str(lim),
            "apikey": key,
        }
    )
    url = f"{_ALPHAVANTAGE_URL}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "alpaca-paper-day-bot/0.1"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        log.warning("alphavantage_news_http symbol=%s err=%s", sym, e)
        return {"ok": False, "error": str(e), "provider": "alphavantage", "symbol": sym, "count": 0, "articles": []}
    except Exception as e:
        log.warning("alphavantage_news_failed symbol=%s err=%s", sym, e)
        return {"ok": False, "error": str(e), "provider": "alphavantage", "symbol": sym, "count": 0, "articles": []}

    if not isinstance(data, dict):
        return {"ok": False, "error": "invalid_response", "provider": "alphavantage", "symbol": sym, "count": 0, "articles": []}

    note = data.get("Note") or data.get("Information")
    if note:
        log.warning("alphavantage_news_note symbol=%s note=%s", sym, str(note)[:160])
        return {
            "ok": False,
            "error": str(note)[:300],
            "provider": "alphavantage",
            "symbol": sym,
            "count": 0,
            "articles": [],
        }

    feed = data.get("feed")
    if not isinstance(feed, list):
        return {"ok": False, "error": "no_feed", "provider": "alphavantage", "symbol": sym, "count": 0, "articles": []}

    articles: list[dict[str, Any]] = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "")[:240]
        tp = item.get("time_published")
        articles.append(
            {
                "headline": title,
                "source": item.get("source"),
                "created_at": _alphavantage_time_published_to_iso(tp if isinstance(tp, str) else None),
                "sentiment_score": item.get("overall_sentiment_score"),
                "url": ((item.get("url") or "")[:500] or None),
            }
        )

    return {
        "ok": True,
        "provider": "alphavantage",
        "symbol": sym,
        "count": len(articles),
        "articles": articles,
    }


def _normalize_news_provider(raw: str) -> str:
    p = (raw or "alpaca").strip().lower()
    # "both" kept for backwards compat; "combo" merges all sources we can reach.
    if p not in ("alpaca", "alphavantage", "google_rss", "tickertick", "both", "combo"):
        return "alpaca"
    return p


def fetch_tickertick_news(*, symbol: str, limit: int) -> dict[str, Any]:
    """
    No-key stock news feed: TickerTick (https://api.tickertick.com/feed).
    """
    sym = (symbol or "").strip().lower()
    lim = max(1, min(int(limit), 50))
    if not sym:
        return {"ok": False, "error": "empty_symbol", "provider": "tickertick", "count": 0, "articles": []}

    # Query language: tt:<ticker>
    params = urllib.parse.urlencode({"q": f"tt:{sym}", "n": str(lim)})
    url = f"{_TICKERTICK_FEED}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "alpaca-paper-day-bot/0.1"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"ok": False, "error": str(e), "provider": "tickertick", "symbol": sym.upper(), "count": 0, "articles": []}

    stories = data.get("stories") if isinstance(data, dict) else None
    if not isinstance(stories, list):
        return {"ok": False, "error": "no_stories", "provider": "tickertick", "symbol": sym.upper(), "count": 0, "articles": []}

    arts: list[dict[str, Any]] = []
    for st in stories[:lim]:
        if not isinstance(st, dict):
            continue
        title = (st.get("title") or "").strip()
        if not title:
            continue
        # time is ms since epoch
        created_at = None
        try:
            tms = st.get("time")
            if isinstance(tms, (int, float)):
                created_at = datetime.fromtimestamp(float(tms) / 1000.0, tz=timezone.utc).isoformat()
        except Exception:
            created_at = None

        arts.append(
            {
                "headline": title[:240],
                "source": st.get("site"),
                "created_at": created_at,
                "url": ((st.get("url") or "")[:500] or None),
                "id": st.get("id"),
            }
        )

    return {"ok": True, "provider": "tickertick", "symbol": sym.upper(), "count": len(arts), "articles": arts}


def fetch_google_news_rss(*, symbol: str, limit: int) -> dict[str, Any]:
    """
    No-key fallback: Google News RSS search for the ticker.
    Returns headlines only (no sentiment).
    """
    sym = (symbol or "").strip().upper()
    lim = max(1, min(int(limit), 20))
    if not sym:
        return {"ok": False, "error": "empty_symbol", "provider": "google_rss", "count": 0, "articles": []}

    q = f"{sym} stock"
    params = urllib.parse.urlencode({"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    url = f"{_GOOGLE_NEWS_RSS}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "alpaca-paper-day-bot/0.1"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            xml = resp.read().decode(errors="ignore")
        root = ET.fromstring(xml)
    except Exception as e:
        return {"ok": False, "error": str(e), "provider": "google_rss", "symbol": sym, "count": 0, "articles": []}

    # RSS: channel/item/title + pubDate
    items = root.findall(".//item")
    arts: list[dict[str, Any]] = []
    for it in items[:lim]:
        title = (it.findtext("title") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title:
            continue
        arts.append({"headline": title[:240], "source": "GoogleNewsRSS", "created_at": pub or None, "url": link or None})

    return {"ok": True, "provider": "google_rss", "symbol": sym, "count": len(arts), "articles": arts}


def fetch_news_for_symbol(
    *,
    symbol: str,
    provider: str,
    alpaca_api_key_id: str,
    alpaca_secret_key: str,
    alphavantage_api_key: str | None,
    lookback_hours: float,
    limit: int,
) -> dict[str, Any]:
    """
    Dispatch: Alpaca news, Alpha Vantage NEWS_SENTIMENT, or merge both (deduped by headline).
    """
    prov = _normalize_news_provider(provider)
    sym = (symbol or "").strip().upper()
    lim_cap = max(1, min(int(limit), 50))

    if prov == "alpaca":
        b = fetch_symbol_news(
            api_key=alpaca_api_key_id,
            secret_key=alpaca_secret_key,
            symbol=sym,
            lookback_hours=lookback_hours,
            limit=lim_cap,
        )
        return {**b, "provider": "alpaca"}

    if prov == "alphavantage":
        return fetch_alphavantage_news(api_key=alphavantage_api_key or "", symbol=sym, limit=lim_cap)

    if prov == "google_rss":
        return fetch_google_news_rss(symbol=sym, limit=lim_cap)

    if prov == "tickertick":
        return fetch_tickertick_news(symbol=sym, limit=lim_cap)

    merged: list[dict[str, Any]] = []
    any_ok = False

    b_alp = fetch_symbol_news(
        api_key=alpaca_api_key_id,
        secret_key=alpaca_secret_key,
        symbol=sym,
        lookback_hours=lookback_hours,
        limit=lim_cap,
    )
    if b_alp.get("ok") and prov in ("both", "combo"):
        any_ok = True
        for a in b_alp.get("articles", []):
            merged.append({**a, "provider": "alpaca"})

    b_av = fetch_alphavantage_news(api_key=alphavantage_api_key or "", symbol=sym, limit=lim_cap)
    if b_av.get("ok") and prov in ("both", "combo"):
        any_ok = True
        for a in b_av.get("articles", []):
            merged.append({**a, "provider": "alphavantage"})

    b_rss = fetch_google_news_rss(symbol=sym, limit=min(10, lim_cap))
    if b_rss.get("ok"):
        any_ok = True
        for a in b_rss.get("articles", []):
            merged.append({**a, "provider": "google_rss"})

    b_tt = fetch_tickertick_news(symbol=sym, limit=min(20, lim_cap))
    if b_tt.get("ok") and prov == "combo":
        any_ok = True
        for a in b_tt.get("articles", []):
            merged.append({**a, "provider": "tickertick"})

    if not any_ok:
        err_a = b_alp.get("error")
        err_v = b_av.get("error")
        err_r = b_rss.get("error")
        err_t = b_tt.get("error") if prov == "combo" else None
        return {
            "ok": False,
            "error": f"alpaca={err_a}; alphavantage={err_v}; google_rss={err_r}; tickertick={err_t}",
            "provider": prov,
            "symbol": sym,
            "count": 0,
            "articles": [],
        }

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for a in merged:
        h = str(a.get("headline", "")).strip()
        key_d = h.lower() if h else str(id(a))
        if key_d in seen:
            continue
        seen.add(key_d)
        deduped.append(a)

    return {
        "ok": True,
        "provider": prov,
        "symbol": sym,
        "count": len(deduped),
        "articles": deduped[: min(len(deduped), lim_cap * 2)],
    }
