import pandas as pd
import numpy as np

def compute_live_signal_for_today(
    historical_temps: pd.DataFrame,
    threshold_f: float = 85.0,
    lookback_days: int = 30,
) -> float:
    """
    Computes today's model probability based on the last N days of real historical observations.
    """
    df = historical_temps.copy()
    df = df.sort_values("DATE").reset_index(drop=True)
    df["TMAX"] = pd.to_numeric(df["TMAX"], errors="coerce")
    
    # Take the most recent lookback window
    recent_window = df.tail(lookback_days)
    if len(recent_window) == 0:
        return 0.50  # Fallback if no data
        
    exceeds = (recent_window["TMAX"] > threshold_f).sum()
    p_model = exceeds / len(recent_window)
    return float(p_model)