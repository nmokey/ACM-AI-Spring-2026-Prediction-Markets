"""
nlp/news_client.py
────────────────────
GNews API client — fetches headlines and stores them in SQLite.

Team 2 — Modeling & Intelligence (NLP half) — implement all methods marked with TODO.

GNews docs:    https://gnews.io/docs/v4
Academic access: email support@gnews.io from your UCLA address for a free key.
GDELT fallback (free, no key): https://api.gdeltproject.org/api/v2/doc/doc
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH = Path("data/store/headlines.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

GNEWS_BASE = "https://gnews.io/api/v4"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")


def init_db() -> sqlite3.Connection:
    """
    Create the headlines SQLite table if it doesn't exist and return the connection.

    Schema:
        id        TEXT PRIMARY KEY
        text      TEXT NOT NULL      -- headline title + description concatenated
        source    TEXT               -- publisher name
        url       TEXT
        timestamp TEXT               -- ISO 8601 UTC string
        query     TEXT               -- the search query that found this headline

    TODO (Week 2): write the CREATE TABLE IF NOT EXISTS statement and return conn.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS headlines (
            id        TEXT PRIMARY KEY,
            text      TEXT NOT NULL,
            source    TEXT,
            url       TEXT,
            timestamp TEXT,
            query     TEXT
        )
        """
    )
    conn.commit()
    return conn


class NewsClient:

    def __init__(self) -> None:
        self.api_key = GNEWS_API_KEY
        self.session = requests.Session()
        self.conn = init_db()

    def fetch_headlines(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """
        Query GNews for recent headlines matching `query`.

        Args:
            query:       search string (e.g. "bitcoin price" or "New York rain today")
            max_results: max articles (GNews free tier: up to 10 per call)

        Returns:
            List of dicts with keys: id, text, source, url, timestamp, query
        """
        if not self.api_key:
            return self._fetch_gdelt(query, max_results)

        params = {
            "q":       query,
            "lang":    "en",
            "max":     max_results,
            "apikey":  self.api_key,
            "sortby":  "publishedAt",
        }

        resp = self.session.get(f"{GNEWS_BASE}/search", params=params, timeout=10)
        if resp.status_code in (403, 429):
            logger.warning("GNews %d — falling back to GDELT for query: %r", resp.status_code, query)
            return self._fetch_gdelt(query, max_results)
        resp.raise_for_status()

        headlines = []
        for a in resp.json().get("articles", []):
            title = a.get("title", "")
            desc  = a.get("description", "") or ""
            headlines.append({
                "id":        str(uuid.uuid4()),
                "text":      f"{title}. {desc}".strip(". "),
                "source":    a.get("source", {}).get("name", ""),
                "url":       a.get("url", ""),
                "timestamp": a.get("publishedAt", ""),
                "query":     query,
            })

        return headlines

    def _fetch_gdelt(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """
        GDELT fallback — free, no auth required, but noisier data.

        Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
        Params: query=..., mode=artlist, maxrecords=..., format=json, timespan=1d

        TODO (Week 2): implement as a backup when GNews is unavailable.
        Parse resp.json()["articles"] — each has: title, domain, url, seendate.
        """
        url = 'https://api.gdeltproject.org/api/v2/doc/doc'

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_results,
            "format": "json",
            "timespan": "1d",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            print(f"[GDELT] Request failed: {e}")
            return []

        results = []
        for i, article in enumerate(articles):
            title = article.get("title", "").strip()
            if not title:
                continue
            results.append({
                "id":        f"gdelt-{i}-{hash(article.get('url', ''))}",
                "text":      title,
                "source":    article.get("domain", "gdelt"),
                "url":       article.get("url", ""),
                "timestamp": article.get("seendate", ""),
                "query":     query,
            })

        return results

    
    def store_headlines(self, headlines: list[dict[str, Any]]) -> int:
        """
        Insert headlines into the SQLite store.
        Returns the number of new rows inserted (use INSERT OR IGNORE to skip duplicates).

        TODO (Week 2): iterate over headlines and insert each one.
        Don't forget to call self.conn.commit() after all inserts.
        """
        if not headlines:
            return 0
        rows = [
            (h["id"], h["text"], h.get("source"), h.get("url"),
             h.get("timestamp"), h.get("query"))
            for h in headlines
        ]
        before = self.conn.total_changes
        self.conn.executemany(
            "INSERT OR IGNORE INTO headlines (id, text, source, url, timestamp, query) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def get_recent_headlines(self, since_iso: str | None = None) -> list[dict[str, Any]]:
        """
        Retrieve stored headlines from SQLite.
        If since_iso is provided, only return headlines after that timestamp.
        Used by nlp/relevance.py to get headlines for a given time window.

        TODO (Week 2): write the SELECT query and return a list of dicts.
        Hint: cursor.description gives you the column names.
        """
        cursor = self.conn.cursor()
        if since_iso:
            cursor.execute("SELECT * FROM headlines WHERE timestamp > ?", (since_iso,))
        else:
            cursor.execute("SELECT * FROM headlines")

        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def poll_for_contracts(
        self,
        contract_titles: list[str],
        sleep_between_calls: float = 1.0,
    ) -> int:
        """
        Fetch and store headlines for a list of Kalshi contract titles.
        Extracts a short keyword query from each title before calling GNews.

        TODO (Week 3):
            - For each title, call _extract_query(title) to get a search string
            - Call self.fetch_headlines(query) and self.store_headlines(headlines)
            - Sleep between calls to respect rate limits
            - Return total new headlines stored
        """
        total_stored = 0

        for title in contract_titles:
            query = _extract_query(title)
            logger.info(f"Polling contract: {title!r} → query: {query!r}")

            headlines = self.fetch_headlines(query)
            stored = self.store_headlines(headlines)
            total_stored += stored
            time.sleep(sleep_between_calls)

        return total_stored


def _extract_query(contract_title: str) -> str:
    """
    Extract a short, newsworthy search query from a Kalshi contract title.
    Returns a topic-level query (no numbers, dates, or thresholds) so that
    multiple contracts on the same underlying map to the same API call.
    """
    low = contract_title.lower()

    # Crypto
    if re.search(r"\bbitcoin\b|\bbtc\b", low):
        return "bitcoin price"
    if re.search(r"\bethereum\b|\beth\b", low):
        return "ethereum price"
    if re.search(r"\bsolana\b|\bsol\b", low):
        return "solana price"
    if re.search(r"\bdogecoin\b|\bdoge\b", low):
        return "dogecoin price"
    if re.search(r"\bbnb\b", low):
        return "BNB crypto price"
    if re.search(r"\bxrp\b|\bripple\b", low):
        return "XRP crypto price"

    # Macro / economic
    if re.search(r"\bfederal funds\b|\bfed\b.*\brate\b|\bfomc\b", low):
        return "Federal Reserve interest rate decision"
    if re.search(r"\bcpi\b|\bconsumer price index\b|\binflation\b", low):
        return "US inflation CPI"
    if re.search(r"\bgdp\b|\bgross domestic product\b", low):
        return "US GDP growth"
    if re.search(r"\badp\b|\bemployment change\b|\bjobs\b", low):
        return "US employment jobs report"
    if re.search(r"\bwti\b|\bcrude oil\b|\boil price\b", low):
        return "WTI crude oil price"
    if re.search(r"\beur.*usd\b|\beurusd\b", low):
        return "EUR USD exchange rate"
    if re.search(r"\busd.*jpy\b|\busdjpy\b", low):
        return "USD JPY exchange rate"

    # Weather — map city abbreviations to full names
    city_map = {
        r"\bla\b|\blos angeles\b|\blax\b": "Los Angeles",
        r"\bnyc\b|\bnew york\b|\bny\b": "New York",
        r"\bchi\b|\bchicago\b": "Chicago",
        r"\bmiami\b|\bmia\b": "Miami",
        r"\bdenver\b|\bden\b": "Denver",
        r"\baustin\b|\baus\b": "Austin",
        r"\bseattle\b|\bsea\b": "Seattle",
        r"\bsan francisco\b|\bsfo\b": "San Francisco",
        r"\bphoenix\b|\bphx\b": "Phoenix",
        r"\bhouston\b|\bhou\b": "Houston",
        r"\bwashington\b|\bdc\b|\btdc\b": "Washington DC",
    }
    if re.search(r"\bhigh temp\b|\blow temp\b|\bmaximum temp\b|\bminimum temp\b"
                 r"|\btemperature\b|\brain\b|\bsnow\b|\bprecip\b|\bweather\b", low):
        for pat, city in city_map.items():
            if re.search(pat, low):
                return f"{city} weather forecast"
        return "weather forecast"

    # Sports
    if re.search(r"\bmlb\b|\bbaseball\b", low):
        return "MLB baseball"
    if re.search(r"\bnba\b|\bbasketball\b", low):
        return "NBA basketball"
    if re.search(r"\bnhl\b|\bhockey\b|\bstanley cup\b", low):
        return "NHL hockey Stanley Cup"
    if re.search(r"\bf1\b|\bformula 1\b|\bformula one\b", low):
        return "Formula 1 F1 racing"
    # MLB player stat lines (e.g. "Mike Trout: 5+ hits + runs + RBIs?")
    player_match = re.match(r"([A-Z][a-z]+ [A-Z][a-z]+):", contract_title)
    if player_match:
        return f"{player_match.group(1)} MLB stats"

    # Fallback: strip numbers, dates, punctuation, and short stop words
    stop = {
        "will", "the", "a", "an", "be", "is", "are", "was", "were",
        "on", "in", "at", "by", "to", "of", "for", "it", "today",
        "tonight", "tomorrow", "this", "that", "above", "below",
        "more", "than", "up", "next", "mins", "price", "range",
    }
    words = re.findall(r"[A-Za-z]{3,}", contract_title)
    kept = [w for w in words if w.lower() not in stop]
    return " ".join(kept[:5]) if kept else contract_title[:60]


# ── Week 2 smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO (Week 1): without using the class above, make a raw requests.get()
    # call to GNews searching for "bitcoin" and print 3 headline titles.
    # Push your notebook to notebooks/week1_team3_nlp.ipynb.
    raw = requests.get(
        f"{GNEWS_BASE}/search",
        params={"q": "bitcoin", "lang": "en", "max": 3, "apikey": GNEWS_API_KEY},
        timeout=10,
    )
    articles = raw.json().get("articles", [])
    for article in articles:
        print(article["title"])