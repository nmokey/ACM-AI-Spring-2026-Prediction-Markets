"""
extensions/momentum/strategy.py
─────────────────────────────────
Crypto momentum signal for Kalshi "Will BTC be up today?" contracts.

See extensions/SPEC.md for full instructions and the backtest loop to run.

Quick start:
    import ccxt, pandas as pd
    exchange = ccxt.binance()
    raw = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=2160)
    df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    signals = compute_signals(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid: maps R → (0, 1)


def compute_signals(
    df: pd.DataFrame,
    lookback_hours: int = 4,
    threshold: float = 0.0,
    p_scale: float = 3.0,
) -> pd.DataFrame:
    """
    Generate a p_model signal from BTC price momentum.

    Args:
        df:             OHLCV DataFrame with DatetimeIndex and 'close' column.
        lookback_hours: number of hourly bars to measure momentum over.
        threshold:      minimum absolute z-score to include a row (filter low-edge rows).
        p_scale:        scaling factor before sigmoid — higher = more extreme p_model values.

    Returns:
        DataFrame with columns: p_model, market_price, resolved_yes.
        One row per hourly bar where |z_score| >= threshold.

    Hints:
        1. Compute rolling % change: df['close'].pct_change(lookback_hours)
        2. Z-score over a 48h rolling window: (x - mean) / std
        3. p_model = expit(z_score * p_scale)  ← momentum direction
           OR:      expit(-z_score * p_scale)  ← mean-reversion (fade the move)
        4. resolved_yes = 1 if next bar's close > this bar's close, else 0
        5. market_price = 0.50 (naive baseline) or add ±Gaussian noise
    """

    # 1. Compute rolling % change
    momentum = df['close'].pct_change(periods=lookback_hours)
    
    # 2. Z-score over a 48h rolling window
    rolling_mean = momentum.rolling(window=48).mean()
    rolling_std = momentum.rolling(window=48).std()
    
    # Adding a small epsilon to avoid division by zero in case of constant prices
    z_score = (momentum - rolling_mean) / (rolling_std + 1e-9)
    
    # Initialize the output DataFrame
    out = pd.DataFrame(index=df.index)
    
    # 3. p_model = expit(z_score * p_scale)  ← momentum direction
    out['p_model'] = expit(z_score * p_scale)
    
    # 4. resolved_yes = 1 if next bar's close > this bar's close, else 0
    # Using shift(-1) to look ahead to the next bar
    out['resolved_yes'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 5. market_price = 0.50 (naive baseline)
    out['market_price'] = 0.50
    
    # Save z_score temporarily for filtering purposes
    out['z_score'] = z_score
    
    # Drop rows where z_score is NaN (due to rolling windows/pct_change)
    out = out.dropna(subset=['z_score'])
    
    # Filter rows based on the absolute z-score threshold
    out = out[out['z_score'].abs() >= threshold]
    
    # Return strictly the requested columns
    return out[['p_model', 'market_price', 'resolved_yes']]
