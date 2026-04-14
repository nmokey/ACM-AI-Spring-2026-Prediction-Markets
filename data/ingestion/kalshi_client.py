"""
data/ingestion/kalshi_client.py
────────────────────────────────
Kalshi REST API client.

Team 1 — implement all methods marked with TODO.

Docs:    https://trading-api.kalshi.com/docs
Auth:    KALSHI_API_KEY + KALSHI_API_SECRET in your .env file
Base URL: https://trading-api.kalshi.com/trade-api/v2
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("KALSHI_BASE_URL", "https://trading-api.kalshi.com/trade-api/v2")


class KalshiClient:

    def __init__(self) -> None:
        self.api_key = os.getenv("KALSHI_API_KEY", "")
        self.api_secret = os.getenv("KALSHI_API_SECRET", "")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        # TODO (Week 1): add your API key to the session headers.
        # Read the Kalshi auth docs to understand the required header format.

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
        raise NotImplementedError

    def get_market(self, ticker: str) -> dict[str, Any]:
        """
        Fetch a single market by its ticker.

        TODO (Week 2): call self._get with the correct endpoint path.
        """
        raise NotImplementedError

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """
        Fetch the current order book for a market.

        TODO (Week 2): call self._get with the correct endpoint path.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def _post(self, path: str, json: dict | None = None) -> dict[str, Any]:
        """Make a POST request to BASE_URL + path. TODO (Week 5): same pattern as _get."""
        raise NotImplementedError


# ── Week 1 hello world ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO (Week 1): without using the class above, write a raw requests.get()
    # call to the Kalshi /markets endpoint and print the titles of 5 open contracts.
    # Goal: just prove you can hit the API and read the response.
    # Then copy your working code into notebooks/week1_team1.ipynb and push it.
    print("Hello from Kalshi! Implement me.")
