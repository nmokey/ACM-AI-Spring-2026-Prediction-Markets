"""
data/ingestion/crypto_client.py
────────────────────────────────
Coinbase API for spot prices
Base URL: https://api.coinbase.com/api/v3/brokerage
Docs:   https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-overview

Useful endpoints:
    GET /market/products/{id}/ticker  → current spot price for a symbol
    GET /market/products/{id}         → 24-hour rolling stats (price change %, volume, etc.)
    GET /market/products/{id}/candles → candlestick (OHLCV) data for any interval
"""

from __future__ import annotations

import requests
from datetime import datetime, timezone
from typing import Any

BASE_URL = "https://api.coinbase.com/api/v3/brokerage/market"
DEFAULT_PAIRS = ["BTC-USD", "ETH-USD"]


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
        url = f"{BASE_URL}/products/{symbol}/ticker"
        resp = self.session.get(url)
        resp.raise_for_status()
        return float(resp.json()["price"])

    def get_24h_stats(self, symbol: str) -> dict[str, Any]:
        """
        Return 24-hour rolling stats for a symbol.

        Response includes: priceChange, priceChangePercent, lastPrice, volume, etc.

        TODO (Week 2): GET /ticker/24hr with the symbol param.
        """
        url = f"{BASE_URL}/products/{symbol}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

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
        url = f"{BASE_URL}/products/{symbol}/candles"
        # Coinbase uses start/end timestamps or a granularity string
        params = {"granularity": interval}
        
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        
        raw_candles = resp.json().get("candles", [])
        
        parsed_candles = []
        for c in raw_candles:
            # Coinbase V3 returns: start (sec), low, high, open, close, volume
            parsed_candles.append({
                "open_time": datetime.fromtimestamp(int(c["start"]), tz=timezone.utc),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"]),
            })

        parsed_candles.sort(key=lambda x: x["open_time"])
        return parsed_candles[-limit:]

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
        candles = self.get_klines(symbol=symbol, interval="ONE_HOUR", limit=7)
        
        if len(candles) < 7:
            raise ValueError("Not enough candle data returned.")

        current_price = candles[-1]["close"]
        price_1h_ago = candles[-2]["close"]
        price_6h_ago = candles[-7]["close"]

        return {
            "current_price": current_price,
            "price_change_1h": (current_price - price_1h_ago) / price_1h_ago,
            "price_change_6h": (current_price - price_6h_ago) / price_6h_ago,
        }


# ── Week 1 hello world ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO (Week 1): make a raw requests.get() call to Binance to fetch the
    # current BTC price and print it. No class needed yet — just prove the API works.
    # Push your notebook to notebooks/week1_team1.ipynb.
    try:
        test_url = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/ticker"
        response = requests.get(test_url)
        btc_price = response.json()["price"]
        print(f"Hello from Coinbase! Current BTC Price: ${btc_price}")
    except Exception as e:
        print(f"Connection failed: {e}")
