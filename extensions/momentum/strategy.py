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
    raise NotImplementedError("Implement me! See extensions/SPEC.md for full hints.")
