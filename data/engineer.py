"""
data/engineer.py
─────────────────
Feature engineering pipeline — Team 1 deliverable.

Pulls live data from CryptoClient, KalshiClient, and WeatherClient, combines
it into one row per open Kalshi contract, and writes
data/features/live_features.parquet for Team 2 (Modeling & Intelligence)
to consume in models/predict.py.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from data.ingestion.crypto_client import CryptoClient
from data.ingestion.kalshi_client import KalshiClient
from data.ingestion.weather_client import WeatherClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
with open(ROOT / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

FEATURES_PATH = ROOT / CONFIG["data"]["features_path"]
FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_PATH = ROOT / CONFIG["data"]["snapshots_path"]

CRYPTO_PAIRS = CONFIG["markets"]["crypto_pairs"]
TARGET_CITIES = CONFIG["markets"]["target_cities"]

WEATHER_SERIES = [
    "KXHIGHLAX", "KXHIGHCHI", "KXHIGHNY", "KXHIGHDEN", "KXHIGHMIA",
    "KXHIGHDAL", "KXHIGHBOS", "KXHIGHAUS", "KXHIGHOU",
    "KXHIGHTSFO", "KXHIGHTSEA", "KXHIGHTPHX", "KXHIGHTDC",
    "KXLOWTLAX", "KXLOWTCHI", "KXLOWTNYC", "KXLOWTDEN",
    "KXLOWTHOU", "KXLOWTDFW", "KXLOWTSFO", "KXLOWTSEA",
]

CRYPTO_SERIES = [
    "KXBTCD", "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M",
    "KXDOGE15M", "KXDOGE", "KXADA15M", "KXAVAX15M", "KXLINK15M",
    "KXBNB15M", "KXEURUSD", "KXUSDJPY", "KXGBPUSD",
]

SPORTS_SERIES = [
    "KXNBA", "KXNHL", "KXMLB", "KXF1",
    "KXNBACHMP", "KXNHLCHMP", "KXMLBHRR", "KXNCAAMB",
]

ECONOMICS_SERIES = [
    "KXCPI", "KXNFP", "KXADP", "KXUNRATE", "KXPPI", "KXPCE",
    "KXJOBLESS", "KXICSA", "KXISM", "KXISMMFG", "KXISMSVC", "KXGDP",
    "KXFED", "KXFFR", "KXDGS10", "KXDGS2",
]

EQUITY_SERIES = [
    "KXSPX", "KXSPXD", "KXNDX", "KXINXD", "KXDOW",
]

ENERGY_SERIES = [
    "KXWTI", "KXOIL", "KXNATGAS", "KXNG",
]


def _fetch_crypto() -> dict[str, dict]:
    """Return price-change dicts keyed by pair. Fills None on failure."""
    crypto = CryptoClient()
    result = {}
    for pair in CRYPTO_PAIRS:
        try:
            result[pair] = crypto.compute_price_changes(pair)
        except Exception:
            logger.warning("Failed to fetch crypto data for %s", pair)
            result[pair] = {"current_price": None, "price_change_1h": None, "price_change_6h": None}
    return result


def _fetch_weather() -> dict[str, float | None]:
    """Return today's max precip probability (0–100) keyed by city. Fills None on failure."""
    weather = WeatherClient()
    result = {}
    for city in TARGET_CITIES:
        try:
            result[city] = weather.get_todays_precip_prob(city)
        except Exception:
            logger.warning("Failed to fetch weather data for %s", city)
            result[city] = None
    return result


def _market_price(m: dict) -> float | None:
    """Compute mid price in [0, 1] from yes_ask/bid_dollars. Falls back to last_price."""
    ask = m.get("yes_ask_dollars")
    bid = m.get("yes_bid_dollars")
    if ask is not None and bid is not None:
        ask_f, bid_f = float(ask), float(bid)
        mid = (ask_f + bid_f) / 2
        if mid > 0:
            return mid
        if ask_f > 0:
            return ask_f
    last = m.get("last_price_dollars")
    return float(last) if last is not None else None


def _days_to_resolution(m: dict, now: datetime) -> float | None:
    """Days from now until the market's expected expiration."""
    close_str = m.get("expected_expiration_time") or m.get("close_time")
    if not close_str:
        return None
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        return max((close_dt - now).total_seconds() / 86400, 0.0)
    except ValueError:
        return None


def _fetch_markets_by_series(kalshi: KalshiClient) -> list[dict]:
    """Fetch markets per series group, tag market_category, deduplicate by ticker."""
    seen: set[str] = set()
    all_markets: list[dict] = []

    groups = [
        ("weather",    WEATHER_SERIES),
        ("crypto",     CRYPTO_SERIES),
        ("sports",     SPORTS_SERIES),
        ("economics",  ECONOMICS_SERIES),
        ("equity",     EQUITY_SERIES),
        ("energy",     ENERGY_SERIES),
    ]

    for category_tag, series_list in groups:
        for series in series_list:
            try:
                page = kalshi.get_markets(status="open", series_ticker=series)
                for m in page:
                    ticker = m.get("ticker", "")
                    if ticker and ticker not in seen:
                        seen.add(ticker)
                        m["_category_tag"] = category_tag
                        all_markets.append(m)
            except Exception as e:
                logger.warning("Failed to fetch series %s: %s", series, e)
            time.sleep(0.35)  # pace between series requests — Kalshi rate-limits bursts

    logger.info("Fetched %d unique markets across all series", len(all_markets))
    return all_markets


def build_features() -> pd.DataFrame:
    """
    Fetch live data from all sources and return a feature DataFrame.
    One row per open Kalshi contract.

    Columns (aligned with data/features/schema.py MarketFeatures):
        contract_id, title, market_category,
        market_price, volume_24h, open_interest,
        days_to_resolution,
        btc_price, btc_change_1h, btc_change_6h,
        eth_price, eth_change_1h, eth_change_6h,
        precip_prob_new_york, precip_prob_los_angeles, precip_prob_chicago,
        fetched_at
    """
    crypto_data = _fetch_crypto()
    weather_data = _fetch_weather()

    kalshi = KalshiClient()
    markets = _fetch_markets_by_series(kalshi)

    btc = crypto_data.get("BTC-USD", {})
    eth = crypto_data.get("ETH-USD", {})

    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat()

    rows = []
    for m in markets:
        rows.append({
            # Identity
            "contract_id":              m.get("ticker"),
            "title":                    m.get("title"),
            "market_category":          m.get("_category_tag") or m.get("category") or None,
            # Kalshi market features
            "market_price":             _market_price(m),
            "volume_24h":               float(m.get("volume_24h_fp") or 0),
            "open_interest":            float(m.get("open_interest_fp") or 0),
            "days_to_resolution":       _days_to_resolution(m, now),
            # Crypto features
            "btc_price":                btc.get("current_price"),
            "btc_change_1h":            btc.get("price_change_1h"),
            "btc_change_6h":            btc.get("price_change_6h"),
            "eth_price":                eth.get("current_price"),
            "eth_change_1h":            eth.get("price_change_1h"),
            "eth_change_6h":            eth.get("price_change_6h"),
            # Weather features
            "precip_prob_new_york":     float(weather_data["New York"]) if weather_data.get("New York") is not None else None,
            "precip_prob_los_angeles":  float(weather_data["Los Angeles"]) if weather_data.get("Los Angeles") is not None else None,
            "precip_prob_chicago":      float(weather_data["Chicago"]) if weather_data.get("Chicago") is not None else None,
            # Metadata
            "fetched_at":               fetched_at,
        })

    return pd.DataFrame(rows)


def save_features(df: pd.DataFrame) -> None:
    """Write the feature DataFrame to live_features.parquet."""
    df.to_parquet(FEATURES_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(df), FEATURES_PATH)


def append_snapshot(df: pd.DataFrame) -> None:
    """
    Append the current live feature rows to the rolling snapshot file.

    Each row gets a resolved_yes=None placeholder — the label_resolved.py
    script fills these in later once Kalshi settles the contracts.

    Deduplicates on (contract_id, fetched_at) so re-running engineer.py
    within the same minute doesn't double-write.
    """
    snapshot_cols = [
        "contract_id", "title", "market_category",
        "market_price", "volume_24h", "open_interest", "days_to_resolution",
        "btc_price", "btc_change_1h", "btc_change_6h",
        "eth_price", "eth_change_1h", "eth_change_6h",
        "precip_prob_new_york", "precip_prob_los_angeles", "precip_prob_chicago",
        "sentiment_score", "sentiment_confidence",
        "fetched_at", "resolved_yes",
    ]

    # Join current sentiment cache onto the snapshot rows
    sentiment_path = ROOT / CONFIG["data"]["sentiment_path"]
    if sentiment_path.exists():
        import json
        with open(sentiment_path) as f:
            raw = json.load(f)
        sent_df = pd.DataFrame.from_dict(raw, orient="index")[
            ["sentiment_score", "sentiment_confidence"]
        ]
        sent_df.index.name = "contract_id"
        new_rows = df.set_index("contract_id").join(sent_df, how="left").reset_index()
    else:
        new_rows = df.copy()
        new_rows["sentiment_score"] = float("nan")
        new_rows["sentiment_confidence"] = float("nan")

    new_rows["resolved_yes"] = pd.NA

    for col in snapshot_cols:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    new_rows = new_rows[snapshot_cols]

    if SNAPSHOTS_PATH.exists():
        existing = pd.read_parquet(SNAPSHOTS_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["contract_id", "fetched_at"], keep="first")
    else:
        combined = new_rows

    combined.to_parquet(SNAPSHOTS_PATH, index=False)
    logger.info("Snapshot: %d total rows in %s (+%d new)", len(combined), SNAPSHOTS_PATH, len(new_rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_features()
    save_features(df)
    append_snapshot(df)
    print(df.head())
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nSaved {len(df)} contracts to {FEATURES_PATH}")
