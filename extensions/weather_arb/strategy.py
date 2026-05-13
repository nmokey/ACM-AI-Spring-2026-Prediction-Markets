import numpy as np
import pandas as pd

def compute_signals(
    historical_temps: pd.DataFrame,
    threshold_f: float = 85.0,
    lookback_days: int = 30,
    market_noise: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate p(daily high > threshold_f) using a rolling historical window.
    Args:
        historical_temps: DataFrame with columns 'DATE' (datetime) and 'TMAX' (°F)
        threshold_f:      temperature threshold (e.g. 85°F for LA in summer)
        lookback_days:    rolling window to estimate climatology probability
        market_noise:     std of Gaussian noise added to market_price (simulates
                          market not tracking forecast perfectly)
    Returns:
        DataFrame with columns: date, p_model, market_price, resolved_yes
    Implementation hints:
        1. Sort by DATE. For each day i, look at the last lookback_days days.
        2. p_model = (number of days in window where TMAX > threshold_f) / lookback_days
           This is a rolling empirical probability — your "model."
        3. resolved_yes = 1 if that day's TMAX > threshold_f, else 0.
        4. market_price = 0.50 + random noise (mean 0, std=market_noise, clipped to [0.05, 0.95])
           This simulates a naive market. Better: use actual Kalshi prices from
           data/features/snapshots.parquet if you want a more realistic test.
        5. Return one row per day where abs(p_model - market_price) > 0.05 (min edge filter).
    """
    # 1. Make a copy and sort by DATE
    df = historical_temps.copy()
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Ensure TMAX is numeric
    df["TMAX"] = pd.to_numeric(df["TMAX"], errors="coerce")
    
    # 2. Compute rolling p_model (empirical probability)
    # For each day, count how many days in the last lookback_days had TMAX > threshold_f
    df["exceeds_threshold"] = (df["TMAX"] > threshold_f).astype(int)
    df["p_model"] = df["exceeds_threshold"].rolling(
        window=lookback_days, min_periods=1
    ).mean()
    
    # 3. Compute resolved_yes (actual outcome for that day)
    df["resolved_yes"] = df["exceeds_threshold"]
    
    # 4. Generate market_price with noise
    # Start with base 0.50, add Gaussian noise, clip to [0.05, 0.95]
    np.random.seed(42)  # For reproducibility; remove if you want different runs
    noise = np.random.normal(loc=0, scale=market_noise, size=len(df))
    df["market_price"] = 0.50 + noise
    df["market_price"] = df["market_price"].clip(0.05, 0.95)
    
    # 5. Filter for meaningful edge (abs difference > 0.05) and return
    df["edge"] = np.abs(df["p_model"] - df["market_price"])
    signals = df[df["edge"] > 0.05][["p_model", "market_price", "resolved_yes"]].copy()
    
    return signals.reset_index(drop=True)


# Example usage / testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    temps = np.random.uniform(50, 90, size=100)
    sample_df = pd.DataFrame({"DATE": dates, "TMAX": temps})
    
    # Run the function
    result = compute_signals(sample_df, threshold_f=75.0, lookback_days=30)
    print(result.head(10))
    print(f"\nTotal signals: {len(result)}")
    print(f"p_model range: [{result['p_model'].min():.3f}, {result['p_model'].max():.3f}]")
    print(f"market_price range: [{result['market_price'].min():.3f}, {result['market_price'].max():.3f}]")