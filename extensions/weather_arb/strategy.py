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

import os

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

CDO_API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"


def get_noaa_token() -> str:
    token = os.getenv("NOAA_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "NOAA_TOKEN is not set. Add it to your .env file or environment variables."
        )
    return token


def fetch_noaa_tmax_data(
    stationid: str = "GHCND:USW00023174",
    startdate: str = "2022-01-01",
    enddate: str = "2025-01-01",
    units: str = "standard",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch NOAA daily TMAX data for a station and return a clean DataFrame."""
    headers = {"token": get_noaa_token()}
    params = {
        "datasetid": "GHCND",
        "stationid": stationid,
        "datatypeid": "TMAX",
        "startdate": startdate,
        "enddate": enddate,
        "limit": limit,
        "units": units,
        "offset": 1,
    }

    rows: list[dict[str, object]] = []
    while True:
        resp = requests.get(CDO_API_BASE, params=params, headers=headers)
        resp.raise_for_status()
        page = resp.json().get("results", [])
        if not page:
            break

        rows.extend(page)
        if len(page) < limit:
            break

        params["offset"] += limit

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["DATE", "TMAX"])

    df["DATE"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"value": "TMAX"})
    return df[["DATE", "TMAX"]].dropna().reset_index(drop=True)


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
    historical_temps = historical_temps.copy()
    historical_temps = historical_temps.sort_values("DATE").reset_index(drop=True)

    dates = historical_temps["DATE"]
    temps = historical_temps["TMAX"]

    p_model_list = []
    market_price_list = []
    resolved_yes_list = []

    n = len(historical_temps)
    for i in range(n):
        if i < lookback_days:
            p_model_list.append(np.nan)
            market_price_list.append(np.nan)
            resolved_yes_list.append(np.nan)
            continue

        window = temps.iloc[i - lookback_days : i]
        p_model = float((window > threshold_f).mean())
        resolved_yes = int(temps.iloc[i] > threshold_f)
        market_price = float(np.clip(0.50 + np.random.normal(0.0, market_noise), 0.05, 0.95))

        p_model_list.append(p_model)
        market_price_list.append(market_price)
        resolved_yes_list.append(resolved_yes)

    signals = pd.DataFrame(
        {
            "date": dates,
            "p_model": p_model_list,
            "market_price": market_price_list,
            "resolved_yes": resolved_yes_list,
        }
    )

    signals = signals.iloc[lookback_days:].reset_index(drop=True)
    signals = signals.loc[
        signals["p_model"].sub(signals["market_price"]).abs() >= min_edge
    ]
    signals = signals.reset_index(drop=True)
    signals["resolved_yes"] = signals["resolved_yes"].astype(int)

    return signals
