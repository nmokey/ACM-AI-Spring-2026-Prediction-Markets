"""
extensions/mean_reversion/mean_reversion_dry_run.py
─────────────────────────────────────────────────────
Continuous dry-run daemon for the Mean-Reversion strategy.
Polls for new hourly candles matching the execution style of trader.py,
ensuring trades are only evaluated and logged once per bar.
"""

import sys
from pathlib import Path

# Path Fix: Ensure Python can find the main execution and data packages
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from datetime import datetime, timezone
import ccxt
import pandas as pd
import yaml

# Import strategy and infrastructure modules
from strategy import compute_signals
from execution.dry_run import log_dry_run_trade
from data.schema import TradeRecord
from data.ingestion.kalshi_client import KalshiClient


def fetch_live_data() -> pd.DataFrame | None:
    """Fetches public BTC/USDT data, falling back to US-accessible endpoints."""
    exchanges_to_try = [
        ("Kraken", ccxt.kraken()),
        ("Binance.US", ccxt.binanceus())
    ]
    
    for name, exchange in exchanges_to_try:
        try:
            raw = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=150)
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            return df
        except Exception:
            continue
            
    return None


def evaluate_and_log_trade(df: pd.DataFrame, kalshi: KalshiClient):
    """Computes signals on the provided dataset and logs qualified trades."""
    signals = compute_signals(df, lookback_hours=4, threshold=0.0)
    if signals.empty:
        return

    # Extract the most recent hourly completed signal
    latest = signals.iloc[-1]
    
    # Mean Reversion Inversion
    p_model = float(latest['p_model'])

    # ─── REAL KALSHI PRICE INTEGRATION ───────────────────────────────────────
    # We define the specific contract ticker we want to trade. 
    # (e.g., today's daily BTC contract)
    target_contract = "KXBTC-TODAY" # Update this to your dynamic ticker logic
    
    try:
        # Fetch the actual live YES price from the Kalshi order book
        # NOTE: Update '.get_market_price()' to whatever method your team 
        # actually wrote inside kalshi_client.py!
        live_kalshi_data = kalshi.get_market_price(target_contract)
        market_price = float(live_kalshi_data) 
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Kalshi API fetch failed: {e}. Falling back to 0.50 baseline.")
        market_price = 0.50
    # ─────────────────────────────────────────────────────────────────────────

    edge = abs(p_model - market_price)
    
    side = "YES" if p_model > market_price else "NO"
    
    if edge < 0.01:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Edge too small ({edge:.3f}). No trade action taken.")
        return

    # Populate trade schema mapping
    record = TradeRecord(
        contract_id="KXBTC-REVERSION-DRY",
        timestamp=datetime.now(timezone.utc),
        side=side,
        size=10,  
        limit_price=int(market_price * 100),
        p_model=p_model,
        market_price=market_price,
        edge=edge,
        mode="dry_run"
    )

    # Print the exact performance log entry to the terminal
    print(f"\n🚀 [MOMENTUM TRIGGERED]")
    print("-" * 50)
    print(f"  Timestamp    : {record.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Contract ID  : {record.contract_id}")
    print(f"  Action       : BUY {record.side}")
    print(f"  Size         : {record.size} contracts")
    print(f"  Limit Price  : {record.limit_price}¢")
    print(f"  Model Prob   : {record.p_model:.1%}")
    print(f"  Market Impl  : {record.market_price:.1%}")
    print(f"  Edge         : {record.edge:.1%}")
    print("-" * 50)
    
    print("Writing to logs/dry_run_trades.csv...")
    log_dry_run_trade(record)
    print("✨ Performance log updated successfully.\n")


def main():
    # Load configuration settings dynamically like trader.py
    try:
        config_path = REPO_ROOT / "config" / "settings.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        poll_interval = config.get("pipeline", {}).get("poll_interval_sec", 60)
    except Exception:
        poll_interval = 60  # Fallback to 60 seconds if configuration reading fails

    print("=" * 60)
    print(f"Starting Mean Reversion Dry Run Daemon")
    print(f"Polling execution loop every {poll_interval}s...")
    print("=" * 60)

    last_processed_ts = None

    while True:
        try:
            df = fetch_live_data()
            if df is not None and not df.empty:
                latest_candle_ts = df.index[-1]
                
                # Deduplication check: Has a new minute bar actually arrived?
                if last_processed_ts == latest_candle_ts:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No new minute candle. (Current bar: {latest_candle_ts.strftime('%Y-%m-%d %H:%M')}). Sleeping...")
                else:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New minute candle detected ({latest_candle_ts.strftime('%Y-%m-%d %H:%M')}). Evaluating...")
                    evaluate_and_log_trade(df)
                    last_processed_ts = latest_candle_ts
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Failed to fetch data from exchanges.")
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in execution loop: {e}")
        
        # Poll sleep interval matching pipeline guidelines
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()