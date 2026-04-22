"""
data/ingestion/kalshi_client.py
────────────────────────────────
Kalshi REST API client.

Team 1 — implement all methods marked with TODO.

Docs:    https://trading-api.kalshi.com/docs
Auth:    KALSHI_API_KEY in your .env file (Bearer token)
Base URL: https://api.elections.kalshi.com/trade-api/v2
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")


class KalshiClient:

    def __init__(self) -> None:
        self.api_key = os.getenv("KALSHI_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

    # ── Market discovery ─────────────────────────────────────────────────────

    def get_markets(
        self,
        status: str = "open",
        category: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Fetch a list of markets from Kalshi.

        Args:
            status:   filter by market status — "open", "closed", or "settled"
            category: optional category tag (e.g. "weather", "crypto", "sports")
            limit:    max results per page

        Returns:
            List of raw market dicts from the Kalshi API.

        TODO (Week 2):
            - Build the params dict and call self._get("/markets", params=...)
            - Return the "markets" key from the response
            - Check the Kalshi docs for the exact query parameter names
        """
        params: dict[str, Any] = {"status": status, "limit": limit}
        if category is not None:
            params["category"] = category
        return self._get("/markets", params=params)["markets"]

    def get_market(self, ticker: str) -> dict[str, Any]:
        """
        Fetch a single market by its ticker.

        TODO (Week 2): call self._get with the correct endpoint path.
        """
        return self._get(f"/markets/{ticker}")["market"]

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """
        Fetch the current order book for a market.

        Returns orderbook_fp dict with yes_dollars and no_dollars price levels:
            {"yes_dollars": [["0.55", "100.00"], ...], "no_dollars": [...]}
        """
        return self._get(f"/markets/{ticker}/orderbook")["orderbook_fp"]

    # ── Historical data (for Team 2 training) ────────────────────────────────

    def get_resolved_markets(
        self,
        category: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> tuple[list[dict], str | None]:
        """
        Fetch settled (resolved) markets for use as training data.

        Returns:
            (markets, next_cursor) — pass next_cursor into the next call to paginate.

        TODO (Week 3):
            - Call get_markets with status="settled"
            - Return both the market list AND the pagination cursor from the response
        """
        raise NotImplementedError

    def backfill_all_resolved(
        self, category: str | None = None, max_pages: int = 20
    ) -> list[dict[str, Any]]:
        """
        Paginate through all resolved markets and return a flat list.

        TODO (Week 3):
            - Call get_resolved_markets in a loop, passing the cursor each time
            - Stop when cursor is None or max_pages is reached
            - Add a small time.sleep() between calls to avoid rate limiting
        """
        raise NotImplementedError

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        limit_price: int,
        order_type: str = "limit",
    ) -> dict[str, Any]:
        """
        Place a limit order on Kalshi.

        NOTE: Never call this directly — always go through execution/order_manager.py
        so that dry_run mode is respected.

        Args:
            ticker:      Kalshi market ticker
            side:        "yes" or "no"
            count:       number of contracts
            limit_price: price in Kalshi cents (0–100)

        TODO (Week 5):
            - Build the payload dict and call self._post("/portfolio/orders", json=payload)
            - Read the Kalshi docs for the exact required fields
        """
        raise NotImplementedError

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        """
        Make a GET request to BASE_URL + path.

        TODO (Week 2):
            - Use self.session.get()
            - Call raise_for_status() to catch HTTP errors
            - Return resp.json()
        """
        resp = self.session.get(BASE_URL + path, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> dict[str, Any]:
        """Make a POST request to BASE_URL + path. TODO (Week 5): same pattern as _get."""
        raise NotImplementedError


if __name__ == "__main__":
    client = KalshiClient()
    markets = client.get_markets(limit=5)
    print(f"Fetched {len(markets)} markets:")
    for m in markets:
        print(f"  {m.get('ticker')} — {m.get('title')}")
