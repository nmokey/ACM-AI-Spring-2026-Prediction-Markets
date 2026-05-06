"""
data/ingestion/kalshi_client.py
────────────────────────────────
Kalshi REST API client.

Team 1 — implement all methods marked with TODO.

Docs:    https://trading-api.kalshi.com/docs
Auth:    RSA-PSS signature — KALSHI_API_KEY (UUID) + KALSHI_API_SECRET (PEM private key)
Base URL: https://trading-api.kalshi.com/trade-api/v2
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("KALSHI_BASE_URL", "https://trading-api.kalshi.com/trade-api/v2")


class KalshiClient:

    def __init__(self) -> None:
        self.api_key = os.getenv("KALSHI_API_ID", "") or os.getenv("KALSHI_API_KEY", "")
        raw_secret = os.getenv("KALSHI_API_SECRET", "")
        # Reconstruct PEM if the .env collapsed newlines.
        # MIIEpA prefix = PKCS#1 RSA key; MIIEvA/MIIEv = PKCS#8.
        if raw_secret and "-----" not in raw_secret:
            if raw_secret.startswith("MIIEpA") or raw_secret.startswith("MIIEo"):
                header, footer = "-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"
            else:
                header, footer = "-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"
            raw_secret = (
                header + "\n"
                + "\n".join(raw_secret[i:i+64] for i in range(0, len(raw_secret), 64))
                + "\n" + footer
            )
        self._private_key = serialization.load_pem_private_key(
            raw_secret.encode(), password=None
        ) if raw_secret else None
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate RSA-PSS auth headers for a single request."""
        ts = str(int(time.time() * 1000))
        # Strip query string — sign only the path component
        parsed_path = urlparse(path).path
        msg = (ts + method.upper() + parsed_path).encode()
        sig = self._private_key.sign(msg, padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ), hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    # ── Market discovery ─────────────────────────────────────────────────────

    def get_markets(
        self,
        status: str = "open",
        category: str | None = None,
        series_ticker: str | None = None,
        limit_per_page: int = 100,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Fetch markets from Kalshi with cursor pagination.

        Args:
            status:         filter by market status — "open", "closed", or "settled"
            category:       optional category tag (e.g. "weather", "crypto", "sports")
            series_ticker:  optional Kalshi series ticker to fetch a specific series
            limit_per_page: max results per page (Kalshi caps at 100)
            max_pages:      safety cap on total pages fetched

        Returns:
            Flat list of all market dicts across all pages.
        """
        all_markets: list[dict[str, Any]] = []
        cursor: str | None = None

        for _ in range(max_pages):
            params: dict[str, Any] = {"status": status, "limit": limit_per_page}
            if category is not None:
                params["category"] = category
            if series_ticker is not None:
                params["series_ticker"] = series_ticker
            if cursor is not None:
                params["cursor"] = cursor

            resp = self._get("/markets", params=params)
            page = resp.get("markets", [])
            all_markets.extend(page)

            cursor = resp.get("cursor") or None
            if not cursor or not page:
                break
            time.sleep(0.25)

        return all_markets

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
        params: dict[str, Any] = {"status": "settled", "limit": limit}
        if category is not None:
            params["category"] = category
        if cursor is not None:
            params["cursor"] = cursor
        resp = self._get("/markets", params=params)
        return resp["markets"], resp.get("cursor") or None

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
        import time
        all_markets: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            page, cursor = self.get_resolved_markets(category=category, cursor=cursor)
            all_markets.extend(page)
            if cursor is None:
                break
            time.sleep(0.5)
        return all_markets

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

        NOTE: limit_price is in Kalshi cents (0–100). The API expects yes_price
        or no_price depending on side, both in cents.
        """
        price_key = "yes_price" if side.lower() == "yes" else "no_price"
        payload = {
            "ticker": ticker,
            "side": side.lower(),
            "action": "buy",
            "type": order_type,
            "count": count,
            price_key: limit_price,
        }
        return self._post("/portfolio/orders", json=payload)["order"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        """Make a GET request to BASE_URL + path."""
        full_path = "/trade-api/v2" + path
        resp = self.session.get(BASE_URL + path, params=params, headers=self._auth_headers("GET", full_path))
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> dict[str, Any]:
        """Make a POST request to BASE_URL + path."""
        full_path = "/trade-api/v2" + path
        resp = self.session.post(BASE_URL + path, json=json, headers=self._auth_headers("POST", full_path))
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    client = KalshiClient()
    markets = client.get_markets(limit_per_page=5, max_pages=1)
    print(f"Fetched {len(markets)} markets:")
    for m in markets:
        print(f"  {m.get('ticker')} — {m.get('title')}")
