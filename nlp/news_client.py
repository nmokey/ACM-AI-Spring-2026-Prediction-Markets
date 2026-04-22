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
import sqlite3
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

        TODO (Week 2):
            - If self.api_key is empty, call self._fetch_gdelt(query, max_results) instead
            - Otherwise, GET https://gnews.io/api/v4/search with params:
                q, lang="en", max=max_results, apikey, sortby="publishedAt"
            - Parse each article from resp.json()["articles"] into the dict format above
            - Give each headline a unique id (hint: import uuid; str(uuid.uuid4()))
            - Concatenate title + ". " + description for the "text" field
        """
        raise NotImplementedError

    def _fetch_gdelt(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """
        GDELT fallback — free, no auth required, but noisier data.

        Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
        Params: query=..., mode=artlist, maxrecords=..., format=json, timespan=1d

        TODO (Week 2): implement as a backup when GNews is unavailable.
        Parse resp.json()["articles"] — each has: title, domain, url, seendate.
        """
        raise NotImplementedError

    def store_headlines(self, headlines: list[dict[str, Any]]) -> int:
        """
        Insert headlines into the SQLite store.
        Returns the number of new rows inserted (use INSERT OR IGNORE to skip duplicates).

        TODO (Week 2): iterate over headlines and insert each one.
        Don't forget to call self.conn.commit() after all inserts.
        """
        raise NotImplementedError

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
        raise NotImplementedError


def _extract_query(contract_title: str) -> str:
    """
    Extract a short, clean search query from a Kalshi contract title.

    E.g. "Will BTC close above $100k on April 20?" → "bitcoin price"
         "Will it rain in New York today?"          → "weather forecast New York"

    TODO (Week 2): write simple keyword heuristics for crypto, weather, and sports.
    For Week 3, consider using NER or embeddings for smarter extraction.
    """
    raise NotImplementedError


# ── Week 1 hello world ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO (Week 1): without using the class above, make a raw requests.get()
    # call to GNews searching for "bitcoin" and print 3 headline titles.
    # Push your notebook to notebooks/week1_team3_nlp.ipynb.
    print("Hello from GNews! Implement me.")
