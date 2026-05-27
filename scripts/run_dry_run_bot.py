"""
execution/kalshi_dry_run.py
────────────────────────────
Dry-run trader for weather arbitrage on live Kalshi contracts.

Fetches open weather contracts from Kalshi, computes rolling climatology signals
against their implied probabilities, identifies edge, and logs trades without
executing them.

Usage:
    python kalshi_dry_run.py --city LAX --threshold 85 --lookback 30 --min-edge 0.05

Team 3 — Execution owns this file.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

# Import strategy module
from extensions.weather_arb.strategy import (
    compute_signals,
    fetch_noaa_tmax_data,
)

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.yaml"
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {
        "data": {
            "dry_run_log_path": "logs/dry_run_trades.csv",
            "live_log_path": "logs/live_trades.csv",
        },
        "kalshi": {
            "api_base": "https://api.kalshi.com/trade-api/v2",
        },
    }

DRY_RUN_LOG = Path(CONFIG["data"]["dry_run_log_path"])
DRY_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

KALSHI_API_BASE = CONFIG["kalshi"]["api_base"]


# ────────────────────────────────────────────────────────────────────────────
# Data Models
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Record of a single trade (real or dry-run)."""
    contract_id: str
    timestamp: datetime
    side: str  # "buy" or "sell"
    size: int
    limit_price: float
    p_model: float
    market_price: float
    edge: float
    mode: str  # "dry_run" or "live"
    contract_name: str = ""
    resolution_date: Optional[str] = None


@dataclass
class KalshiContract:
    """Live contract from Kalshi."""
    contract_id: str
    ticker: str
    title: str
    underlying_type: str
    strike_price: Optional[float]
    expiration_date: str
    resolution_date: Optional[str]
    status: str
    latest_price: float  # mid-market price
    bid: float
    ask: float
    last_price: float


# ────────────────────────────────────────────────────────────────────────────
# Kalshi API
# ────────────────────────────────────────────────────────────────────────────

def get_kalshi_auth_token() -> str:
    """Get Kalshi API authentication token."""
    token = os.getenv("KALSHI_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "KALSHI_API_TOKEN is not set. Add it to your .env file or environment variables."
        )
    return token


def fetch_open_weather_contracts() -> list[KalshiContract]:
    """
    Fetch all open weather contracts from Kalshi.
    
    Filters for contracts that are currently tradeable (status='active')
    and have weather-related underlying types.
    """
    headers = {"Authorization": f"Bearer {get_kalshi_auth_token()}"}
    url = f"{KALSHI_API_BASE}/contracts"
    
    params = {
        "status": "active",
        "limit": 100,
    }
    
    contracts = []
    offset = 0
    
    while True:
        params["offset"] = offset
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch contracts from Kalshi: {e}")
            break
        
        data = resp.json()
        results = data.get("contracts", [])
        
        if not results:
            break
        
        for item in results:
            # Filter for weather contracts (adjust keywords as needed)
            if any(kw in item.get("title", "").lower() for kw in ["weather", "temp", "high", "low"]):
                contract = KalshiContract(
                    contract_id=item["id"],
                    ticker=item["ticker"],
                    title=item["title"],
                    underlying_type=item.get("underlying_type", ""),
                    strike_price=item.get("strike_price"),
                    expiration_date=item.get("expiration_date", ""),
                    resolution_date=item.get("resolution_date"),
                    status=item["status"],
                    latest_price=item.get("last_price", 0.5),
                    bid=item.get("bid", item.get("last_price", 0.5)),
                    ask=item.get("ask", item.get("last_price", 0.5)),
                    last_price=item.get("last_price", 0.5),
                )
                contracts.append(contract)
        
        if len(results) < params["limit"]:
            break
        
        offset += params["limit"]
    
    return contracts


def match_contract_to_signal(
    contract: KalshiContract,
    signals: pd.DataFrame,
    threshold_f: float,
) -> Optional[tuple[float, float, float]]:
    """
    Match a Kalshi contract to a computed signal.
    
    Returns:
        (p_model, market_price, edge) if a match is found, else None.
    
    Strategy:
        - Extract the threshold from the contract title (e.g., "High > 85°F")
        - Match it against signals computed with that same threshold
        - Use the most recent signal (today's forecast)
    """
    # Simple parsing: look for "> XX" or "above XX" patterns in title
    import re
    
    match = re.search(r'(?:>|above)\s*(\d+)', contract.title)
    if not match:
        return None
    
    contract_threshold = float(match.group(1))
    
    # Only match if thresholds align
    if abs(contract_threshold - threshold_f) > 0.5:
        return None
    
    # Use the most recent signal
    if signals.empty:
        return None
    
    latest_signal = signals.iloc[-1]
    
    p_model = latest_signal["p_model"]
    market_price = latest_signal["market_price"]
    edge = abs(p_model - market_price)
    
    return p_model, market_price, edge


def compute_trade_size(
    edge: float,
    max_notional: float = 100.0,
    edge_multiplier: float = 2.0,
) -> int:
    """
    Compute trade size (number of contracts) based on edge.
    
    A larger edge warrants a larger position.
    """
    if edge <= 0:
        return 0
    size = int(np.clip(edge * edge_multiplier * max_notional, 1, max_notional))
    return size


def compute_limit_price(
    market_price: float,
    p_model: float,
    side: str,
    slip: float = 0.02,
) -> float:
    """
    Compute a conservative limit price with slippage buffer.
    
    If p_model > market_price (underpriced YES), we want to BUY at market_price - slip.
    If p_model < market_price (overpriced YES), we want to SELL at market_price + slip.
    """
    if side == "buy":
        return float(np.clip(market_price - slip, 0.01, 0.99))
    else:  # sell
        return float(np.clip(market_price + slip, 0.01, 0.99))


def log_dry_run_trade(record: TradeRecord) -> None:
    """Log a dry-run trade to CSV."""
    log_path = DRY_RUN_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = log_path.exists()
    fieldnames = [
        "timestamp",
        "contract_id",
        "contract_name",
        "side",
        "size",
        "limit_price",
        "p_model",
        "market_price",
        "edge",
        "mode",
        "resolution_date",
    ]
    
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": record.timestamp.isoformat(),
            "contract_id": record.contract_id,
            "contract_name": record.contract_name,
            "side": record.side,
            "size": record.size,
            "limit_price": record.limit_price,
            "p_model": record.p_model,
            "market_price": record.market_price,
            "edge": record.edge,
            "mode": record.mode,
            "resolution_date": record.resolution_date or "",
        })


# ────────────────────────────────────────────────────────────────────────────
# Main Dry-Run Loop
# ────────────────────────────────────────────────────────────────────────────

def run_dry_run(
    city: str = "LAX",
    threshold_f: float = 85.0,
    lookback_days: int = 30,
    min_edge: float = 0.05,
    max_trades: Optional[int] = None,
) -> None:
    """
    Execute the dry-run: fetch data, compute signals, match contracts, log trades.
    
    Args:
        city:           City code (e.g., "LAX", "NYC").
        threshold_f:    Temperature threshold for the arbitrage signal.
        lookback_days:  Rolling window size for climatology.
        min_edge:       Minimum edge to consider a trade.
        max_trades:     Max number of trades to generate (for testing).
    """
    logger.info("=" * 80)
    logger.info("Weather Arbitrage Dry-Run")
    logger.info("=" * 80)
    
    # Map city code to NOAA station ID
    station_map = {
        "LAX": "GHCND:USW00023174",
        "NYC": "GHCND:USW00014732",
        "ORD": "GHCND:USW00094846",
        "DFW": "GHCND:USW00013960",
    }
    
    stationid = station_map.get(city.upper())
    if not stationid:
        logger.error(f"Unknown city code: {city}. Available: {list(station_map.keys())}")
        return
    
    logger.info(f"Fetching NOAA temperature data for {city} (station {stationid})...")
    try:
        historical_temps = fetch_noaa_tmax_data(
            stationid=stationid,
            startdate="2024-01-01",
            enddate="2025-01-31",
        )
    except Exception as e:
        logger.error(f"Failed to fetch NOAA data: {e}")
        return
    
    if historical_temps.empty:
        logger.error("No NOAA data retrieved.")
        return
    
    logger.info(f"Retrieved {len(historical_temps)} days of temperature data.")
    
    logger.info(f"Computing signals (threshold={threshold_f}°F, lookback={lookback_days} days)...")
    signals = compute_signals(
        historical_temps,
        threshold_f=threshold_f,
        lookback_days=lookback_days,
        market_noise=0.05,
        min_edge=min_edge,
    )
    
    if signals.empty:
        logger.warning("No signals generated (edge threshold not met).")
        return
    
    logger.info(f"Generated {len(signals)} signals with sufficient edge.")
    
    logger.info("Fetching live Kalshi weather contracts...")
    try:
        contracts = fetch_open_weather_contracts()
    except Exception as e:
        logger.error(f"Failed to fetch Kalshi contracts: {e}")
        return
    
    logger.info(f"Found {len(contracts)} open weather contracts.")
    
    # Match contracts to signals and generate trades
    trade_count = 0
    matched_count = 0
    
    for contract in contracts:
        if max_trades and trade_count >= max_trades:
            break
        
        result = match_contract_to_signal(contract, signals, threshold_f)
        if not result:
            continue
        
        matched_count += 1
        p_model, market_price, edge = result
        
        # Skip if edge is too small
        if edge < min_edge:
            continue
        
        # Decide side: if p_model > market_price, the YES is underpriced → BUY
        if p_model > market_price:
            side = "buy"
        else:
            side = "sell"
        
        size = compute_trade_size(edge)
        limit_price = compute_limit_price(market_price, p_model, side)
        
        record = TradeRecord(
            contract_id=contract.contract_id,
            timestamp=datetime.now(),
            side=side,
            size=size,
            limit_price=limit_price,
            p_model=p_model,
            market_price=market_price,
            edge=edge,
            mode="dry_run",
            contract_name=contract.title,
            resolution_date=contract.resolution_date,
        )
        
        log_dry_run_trade(record)
        trade_count += 1
        
        logger.info(
            f"[DRY RUN] {contract.ticker}: {side.upper()} {size} @ {limit_price:.3f} "
            f"(model={p_model:.3f}, market={market_price:.3f}, edge={edge:.3f})"
        )
    
    logger.info("=" * 80)
    logger.info(f"Dry-run complete. Matched {matched_count} contracts, logged {trade_count} trades.")
    logger.info(f"Trades logged to: {DRY_RUN_LOG}")
    logger.info("=" * 80)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dry-run weather arbitrage trades on live Kalshi contracts."
    )
    parser.add_argument(
        "--city",
        type=str,
        default="LAX",
        help="City code (LAX, NYC, ORD, DFW). Default: LAX",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Temperature threshold (°F). Default: 85",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Lookback window (days). Default: 30",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge to trade. Default: 0.05",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Max trades to generate (for testing). Default: None (all)",
    )
    
    args = parser.parse_args()
    
    run_dry_run(
        city=args.city,
        threshold_f=args.threshold,
        lookback_days=args.lookback,
        min_edge=args.min_edge,
        max_trades=args.max_trades,
    )


if __name__ == "__main__":
    main()