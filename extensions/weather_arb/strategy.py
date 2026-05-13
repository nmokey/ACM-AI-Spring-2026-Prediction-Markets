"""
extensions/weather_arb/strategy.py
────────────────────────────────────
Weather arbitrage signal: rolling climatology vs. Kalshi implied probability.

See extensions/SPEC.md for full instructions, data download steps, and the
backtest loop to run.

Quick start (after downloading a NOAA CDO CSV for your city):
    import pandas as pd
    df = pd.read_csv("noaa_lax_temps.csv", parse_dates=["DATE"])
    df = df[["DATE", "TMAX"]].dropna()
    signals = compute_signals(df, threshold_f=85.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_signals(
    historical_temps: pd.DataFrame,
    threshold_f: float = 85.0,
    lookback_days: int = 30,
    market_noise: float = 0.05,
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate p(daily high > threshold_f) from a rolling historical window.

    Args:
        historical_temps: DataFrame with columns 'DATE' (datetime) and 'TMAX' (°F).
        threshold_f:      temperature threshold for the simulated Kalshi contract.
        lookback_days:    rolling window size for the climatology estimate.
        market_noise:     std of Gaussian noise added to market_price (simulates
                          the market not perfectly tracking the forecast).
        min_edge:         drop rows where |p_model - market_price| < min_edge.

    Returns:
        DataFrame with columns: date, p_model, market_price, resolved_yes.

    Hints:
        1. Sort by DATE, reset index.
        2. For each day i, look at the previous lookback_days rows:
               window = historical_temps['TMAX'].iloc[max(0, i-lookback_days):i]
               p_model = (window > threshold_f).mean()
        3. resolved_yes = 1 if that day's TMAX > threshold_f, else 0.
        4. market_price = 0.50 + np.random.normal(0, market_noise), clipped to [0.05, 0.95].
        5. Filter out rows where |p_model - market_price| < min_edge.
        6. Drop the first lookback_days rows (insufficient history).
    """
    raise NotImplementedError("Implement me! See extensions/SPEC.md for full hints.")
