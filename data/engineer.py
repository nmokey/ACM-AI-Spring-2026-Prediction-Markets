"""
data/engineer.py
─────────────────
Feature engineering pipeline — Team 1 deliverable.

Pulls live data from CryptoClient and KalshiClient, combines it into one
row per open Kalshi contract, and writes data/features/live_features.parquet
for Team 2 (Modeling & Intelligence) to consume in models/predict.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from data.ingestion.crypto_client import CryptoClient
from data.ingestion.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
with open(ROOT / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

FEATURES_PATH = ROOT / CONFIG["data"]["features_path"]
FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

CRYPTO_PAIRS = CONFIG["markets"]["crypto_pairs"]


def build_features() -> pd.DataFrame:
    """
    Fetch live crypto and Kalshi data and return a feature DataFrame.
    One row per open Kalshi contract.
    """
    crypto = CryptoClient()
    kalshi = KalshiClient()

    # Fetch price changes for each configured crypto pair
    crypto_features: dict[str, dict] = {}
    for pair in CRYPTO_PAIRS:
        try:
            crypto_features[pair] = crypto.compute_price_changes(pair)
        except Exception:
            logger.warning("Failed to fetch crypto data for %s", pair)
            crypto_features[pair] = {"current_price": None, "price_change_1h": None, "price_change_6h": None}

    # Flatten crypto features into individual columns
    btc = crypto_features.get("BTC-USD", {})
    eth = crypto_features.get("ETH-USD", {})

    # Fetch open Kalshi markets
    markets = kalshi.get_markets(status="open")

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for m in markets:
        yes_ask = m.get("yes_ask_dollars")
        yes_bid = m.get("yes_bid_dollars")
        if yes_ask is not None and yes_bid is not None:
            ask, bid = float(yes_ask), float(yes_bid)
            mid = (ask + bid) / 2
            market_price = mid if mid > 0 else (ask if ask > 0 else None)
        else:
            last = m.get("last_price_dollars")
            market_price = float(last) if last is not None else None

        rows.append({
            "contract_id":    m.get("ticker"),
            "title":          m.get("title"),
            "category":       m.get("category"),
            "market_price":   market_price,
            "btc_price":      btc.get("current_price"),
            "btc_change_1h":  btc.get("price_change_1h"),
            "btc_change_6h":  btc.get("price_change_6h"),
            "eth_price":      eth.get("current_price"),
            "eth_change_1h":  eth.get("price_change_1h"),
            "eth_change_6h":  eth.get("price_change_6h"),
            "fetched_at":     fetched_at,
        })

    return pd.DataFrame(rows)


def save_features(df: pd.DataFrame) -> None:
    """Write the feature DataFrame to live_features.parquet."""
    df.to_parquet(FEATURES_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(df), FEATURES_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_features()
    save_features(df)
    print(df.head())
    print(f"\nSaved {len(df)} contracts to {FEATURES_PATH}")
