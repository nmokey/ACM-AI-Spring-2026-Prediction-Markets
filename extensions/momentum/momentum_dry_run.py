"""
extensions/momentum/momentum_dry_run.py
───────────────────────────────────────
Minimal dry-run script for the momentum strategy.
Bypasses geo-restrictions by fetching live data from US-available exchanges,
calculates the momentum signal, and saves it using execution/dry_run.py.
"""

import sys
from pathlib import Path

# 1. Path Fix: Ensure Python can find the main execution and data packages
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from datetime import datetime, timezone
import ccxt
import pandas as pd

# Import strategy and infrastructure modules
from strategy import compute_signals
from execution.dry_run import log_dry_run_trade
from data.schema import TradeRecord


def fetch_live_data() -> pd.DataFrame | None:
    """Fetches public BTC/USDT data, falling back to US-accessible endpoints."""
    # Kraken is highly stable for US public data fetching without API keys
    exchanges_to_try = [
        ("Kraken", ccxt.kraken()),
        ("Binance.US", ccxt.binanceus())
    ]
    
    for name, exchange in exchanges_to_try:
        try:
            print(f"Attempting to fetch live data from {name}...")
            # Request 150 bars to comfortably cover the strategy's 48h rolling window
            raw = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=150)
            
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            print(f"  └─ {name} failed: {e}")
            continue
            
    return None


def fetch_and_log_trade():
    df = fetch_live_data()
    if df is None:
        print("CRITICAL: All exchange endpoints failed or were restricted. Exiting.")
        return

    print("Computing momentum signals...")
    # Matches lookback requirements from your strategy configuration
    signals = compute_signals(df, lookback_hours=4, threshold=0.0)
    
    if signals.empty:
        print("No valid signals generated.")
        return

    # Extract the most recent hourly signal
    latest = signals.iloc[-1]
    p_model = float(latest['p_model'])
    market_price = float(latest['market_price'])
    edge = abs(p_model - market_price)
    
    # Determine direction relative to baseline market price
    side = "YES" if p_model > market_price else "NO"
    
    # Require a small minimum edge to log a mock transaction
    if edge < 0.01:
        print(f"Edge too small ({edge:.3f}). Skipping trade allocation.")
        return

    # Populate trade schema mapping
    record = TradeRecord(
        contract_id="KXBTC-MOMENTUM-DRY",
        timestamp=datetime.now(timezone.utc),
        side=side,
        size=10,  # Default mock target volume
        limit_price=int(market_price * 100),
        p_model=p_model,
        market_price=market_price,
        edge=edge,
        mode="dry_run"
    )

    print(f"\n🚀 Signal Found! P_Model: {p_model:.3f} | Market Price: {market_price:.3f} | Edge: {edge:.3f}")
    print(f"Writing {side} trade into your dry-run logger...")
    
    # Passes object directly to your Team 3 execution code
    log_dry_run_trade(record)
    print("✨ Performance log updated successfully.")


if __name__ == "__main__":
    fetch_and_log_trade()