"""
data/ingestion/crypto_client.py
────────────────────────────────
Binance public REST API client for spot prices.

Team 1 — implement all methods marked with TODO.

Docs:    https://binance-docs.github.io/apidocs/spot/en/
Auth:    None required for public market data endpoints.
Base URL: https://api.binance.com/api/v3

Useful endpoints:
    GET /ticker/price           → current spot price for a symbol
    GET /ticker/24hr            → 24-hour rolling stats (price change %, volume, etc.)
    GET /klines                 → candlestick (OHLCV) data for any interval
"""

from __future__ import annotations

import requests
from typing import Any

BASE_URL = "https://api.binance.com/api/v3"
DEFAULT_PAIRS = ["BTCUSDT", "ETHUSDT"]


class CryptoClient:

    def __init__(self) -> None:
        self.session = requests.Session()

    def get_price(self, symbol: str) -> float:
        """
        Return the current spot price for a symbol (e.g. "BTCUSDT").

        TODO (Week 2):
            - GET /ticker/price with params={"symbol": symbol}
            - Return float(resp.json()["price"])
        """
        raise NotImplementedError

    def get_24h_stats(self, symbol: str) -> dict[str, Any]:
        """
        Return 24-hour rolling stats for a symbol.

        Response includes: priceChange, priceChangePercent, lastPrice, volume, etc.

        TODO (Week 2): GET /ticker/24hr with the symbol param.
        """
        raise NotImplementedError

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent candlestick (OHLCV) data.

        Args:
            symbol:   e.g. "BTCUSDT"
            interval: "1m" | "5m" | "1h" | "4h" | "1d"
            limit:    number of candles (max 1000)

        Returns:
            List of dicts with keys: open_time, open, high, low, close, volume.
            Tip: Binance returns raw lists — you'll need to parse each candle.
            Index 0 = open_time (ms timestamp), 1 = open, 2 = high, 3 = low,
            4 = close, 5 = volume.

        TODO (Week 2): GET /klines, parse the raw list-of-lists response into
        a list of readable dicts. Convert open_time from milliseconds to a datetime.
        """
        raise NotImplementedError

    def compute_price_changes(self, symbol: str) -> dict[str, float]:
        """
        Compute 1h and 6h price changes for a symbol.
        This is the main output used by Team 1's feature engineer.

        Returns:
            {
                "current_price": float,
                "price_change_1h": float,   # e.g. 0.012 means +1.2%
                "price_change_6h": float,
            }

        TODO (Week 3):
            - Call self.get_klines with interval="1h", limit=7
            - current   = candles[-1]["close"]
            - 1h ago    = candles[-2]["close"]
            - 6h ago    = candles[-7]["close"]
            - change    = (current - past) / past
        """
        raise NotImplementedError


# ── Week 1 hello world ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO (Week 1): make a raw requests.get() call to Binance to fetch the
    # current BTC price and print it. No class needed yet — just prove the API works.
    # Push your notebook to notebooks/week1_team1.ipynb.
    print("Hello from Binance! Implement me.")
