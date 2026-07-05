"""Multi-source news aggregator.

Hits Alpaca (Benzinga), Yahoo Finance (yfinance), and optional API-key
sources Finnhub / MarketAux / NewsAPI in parallel, dedupes by headline,
and returns a unified list sorted most-recent-first.

Each item is normalized to:
    {ts, headline, summary, source, url}

A source is silently skipped if its API key isn't set or the call fails —
the agent keeps working with whatever feeds are live.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional


def _norm_headline(h: str) -> str:
    """Lowercased, alphanumeric-only headline for dedupe matching."""
    return re.sub(r"[^a-z0-9 ]", "", (h or "").lower()).strip()


def _safe_iso(dt) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    if isinstance(dt, (int, float)):
        try:
            return datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
        except Exception:
            return ""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def fetch_alpaca(broker, symbol: str, lookback_hours: int, limit: int) -> list[dict]:
    """Alpaca's Benzinga news feed (already wrapped in broker.news)."""
    try:
        return broker.news(symbol, lookback_hours=lookback_hours, limit=limit)
    except Exception:
        return []


def fetch_yahoo(symbol: str, lookback_hours: int, limit: int) -> list[dict]:
    """Yahoo Finance via yfinance — no API key, broad source diversity
    (Reuters, Bloomberg, MarketWatch, Motley Fool, etc.)."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        items = t.news or []
    except Exception:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).timestamp()
    out = []
    for it in items:
        # yfinance's recent format wraps under "content"
        c = it.get("content") if isinstance(it, dict) else None
        if c:
            ts_str = c.get("pubDate") or c.get("displayTime") or ""
            try:
                ts_obj = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_obj.timestamp() < cutoff:
                    continue
            except Exception:
                pass
            provider = (c.get("provider") or {}).get("displayName", "yahoo")
            url = ((c.get("canonicalUrl") or {}).get("url")
                   or (c.get("clickThroughUrl") or {}).get("url") or "")
            out.append({
                "ts": ts_str,
                "headline": c.get("title") or "",
                "summary": (c.get("summary") or c.get("description") or "")[:280],
                "source": provider.lower(),
                "url": url,
            })
        else:
            # Older yfinance schema (flat dict)
            ts_unix = it.get("providerPublishTime", 0)
            if ts_unix and ts_unix < cutoff:
                continue
            out.append({
                "ts": _safe_iso(ts_unix),
                "headline": it.get("title") or "",
                "summary": "",
                "source": (it.get("publisher") or "yahoo").lower(),
                "url": it.get("link") or "",
            })
        if len(out) >= limit:
            break
    return out


def fetch_finnhub(symbol: str, api_key: str, lookback_hours: int, limit: int) -> list[dict]:
    """Finnhub /company-news — sources include Reuters, Bloomberg, CNBC,
    MarketWatch, SeekingAlpha. Free tier: 60 calls/min."""
    if not api_key:
        return []
    try:
        import requests
    except Exception:
        return []
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": symbol.upper(),
                "from": start.date().isoformat(),
                "to": end.date().isoformat(),
                "token": api_key,
            },
            timeout=8,
        )
        if r.status_code != 200:
            return []
        rows = r.json() or []
    except Exception:
        return []
    out = []
    for it in rows[:limit]:
        ts_unix = it.get("datetime", 0)
        if ts_unix and datetime.fromtimestamp(ts_unix, tz=timezone.utc) < start:
            continue
        out.append({
            "ts": _safe_iso(ts_unix),
            "headline": it.get("headline") or "",
            "summary": (it.get("summary") or "")[:280],
            "source": (it.get("source") or "finnhub").lower(),
            "url": it.get("url") or "",
        })
    return out


def fetch_marketaux(symbol: str, api_key: str, lookback_hours: int, limit: int) -> list[dict]:
    """MarketAux /news/all — equity-focused with real source attribution.
    Free tier: 100 calls/day."""
    if not api_key:
        return []
    try:
        import requests
    except Exception:
        return []
    published_after = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    try:
        r = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={
                "symbols": symbol.upper(),
                "filter_entities": "true",
                "language": "en",
                "limit": min(limit, 50),
                "published_after": published_after,
                "api_token": api_key,
            },
            timeout=8,
        )
        if r.status_code != 200:
            return []
        rows = (r.json() or {}).get("data", [])
    except Exception:
        return []
    out = []
    for it in rows:
        out.append({
            "ts": it.get("published_at") or "",
            "headline": it.get("title") or "",
            "summary": (it.get("description") or it.get("snippet") or "")[:280],
            "source": (it.get("source") or "marketaux").lower(),
            "url": it.get("url") or "",
        })
    return out


def fetch_google_news(symbol: str, lookback_hours: int, limit: int) -> list[dict]:
    """Google News RSS — free, no API key, broad publisher coverage.
    Returns headlines mentioning the ticker symbol."""
    try:
        import requests
        from xml.etree import ElementTree as ET
    except Exception:
        return []
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": f"{symbol.upper()} stock", "hl": "en-US",
                    "gl": "US", "ceid": "US:en"},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (FinBot/1.0)"},
        )
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    out = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        src_el = item.find("source")
        if title_el is None:
            continue
        # Google News titles are often "Headline - Publisher"
        raw_title = (title_el.text or "").strip()
        source = (src_el.text or "").strip() if src_el is not None else ""
        if not source and " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            if len(parts) == 2 and len(parts[1]) < 60:
                source = parts[1].strip()
                raw_title = parts[0].strip()
        try:
            ts_obj = datetime.strptime(date_el.text, "%a, %d %b %Y %H:%M:%S %Z")
            ts_obj = ts_obj.replace(tzinfo=timezone.utc)
            if ts_obj < cutoff:
                continue
            ts_iso = ts_obj.isoformat()
        except Exception:
            ts_iso = ""
        out.append({
            "ts": ts_iso,
            "headline": raw_title[:280],
            "summary": "",
            "source": (source or "google news").lower(),
            "url": (link_el.text or "").strip() if link_el is not None else "",
        })
        if len(out) >= limit:
            break
    return out


def fetch_newsapi(symbol: str, api_key: str, lookback_hours: int, limit: int) -> list[dict]:
    """NewsAPI.org /everything — broadest source list. Free tier: 100/day,
    24h delay (paid tier removes the delay)."""
    if not api_key:
        return []
    try:
        import requests
    except Exception:
        return []
    from_dt = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": symbol,
                "from": from_dt,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": api_key,
            },
            timeout=8,
        )
        if r.status_code != 200:
            return []
        rows = (r.json() or {}).get("articles", [])
    except Exception:
        return []
    out = []
    for it in rows:
        src = (it.get("source") or {}).get("name", "newsapi")
        out.append({
            "ts": it.get("publishedAt") or "",
            "headline": it.get("title") or "",
            "summary": (it.get("description") or "")[:280],
            "source": src.lower(),
            "url": it.get("url") or "",
        })
    return out


def aggregate(broker, symbol: str, cfg, lookback_hours: int = 72,
              per_source_limit: int = 20, total_limit: int = 30) -> list[dict]:
    """Fetch all enabled sources in parallel, dedupe by normalized headline,
    sort by recency (newest first), and return up to total_limit items."""
    sources = [
        ("alpaca",    lambda: fetch_alpaca(broker, symbol, lookback_hours, per_source_limit)),
        ("yahoo",     lambda: fetch_yahoo(symbol, lookback_hours, per_source_limit)),
        ("google",    lambda: fetch_google_news(symbol, lookback_hours, per_source_limit)),
        ("finnhub",   lambda: fetch_finnhub(symbol, cfg.finnhub_api_key, lookback_hours, per_source_limit)),
        ("marketaux", lambda: fetch_marketaux(symbol, cfg.marketaux_api_key, lookback_hours, per_source_limit)),
        ("newsapi",   lambda: fetch_newsapi(symbol, cfg.newsapi_api_key, lookback_hours, per_source_limit)),
    ]
    all_items: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = {pool.submit(fn): name for name, fn in sources}
        for fut in as_completed(futures):
            try:
                all_items.extend(fut.result() or [])
            except Exception:
                pass

    # Dedupe by normalized headline; keep the first occurrence (most-likely
    # most-detailed since sources are added in roughly that order)
    seen: set[str] = set()
    deduped: list[dict] = []
    for it in all_items:
        key = _norm_headline(it.get("headline", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    # Sort newest first; missing ts goes to the end
    def _sort_key(it: dict):
        ts = it.get("ts") or ""
        try:
            return -datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0
    deduped.sort(key=_sort_key)
    return deduped[:total_limit]
