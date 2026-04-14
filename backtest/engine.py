"""
backtest/engine.py
────────────────────
Backtesting engine — simulates the trading pipeline on historical
resolved Kalshi contracts to evaluate strategy performance before going live.

Shared by all teams. Use notebooks/backtest_results.ipynb for visualization.

Usage: python -m backtest.engine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from execution.kelly import kelly_fraction
from backtest.metrics import compute_metrics, print_metrics

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

TRADING_CFG = CONFIG["trading"]


def run_backtest(
    features_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
    model=None,
    starting_balance: float = 1000.0,
) -> pd.DataFrame:
    """
    Simulate trading on historical resolved contracts.

    Args:
        features_path:    path to historical features parquet
        sentiment_path:   path to sentiment.json (optional)
        model:            fitted model with predict_proba(). If None, loads from disk.
        starting_balance: simulated starting bankroll in dollars

    Returns:
        DataFrame of simulated trades with columns:
        contract_id, p_model, market_price, side, n_contracts, cost,
        edge, resolved_yes, won, pnl, balance, cumulative_pnl

    TODO (Week 6):
        1. Load the model (joblib.load from models/trained/xgb_v1.joblib)
        2. Load the features parquet — filter to rows where resolved_yes is not NaN
        3. Join sentiment signals if available (default 0.0)
        4. Run model.predict_proba(X)[:, 1] to get p_model per contract
        5. For each contract, simulate the trade:
               a. Compute edge = |p_model - market_price|
               b. Skip if edge < TRADING_CFG["min_edge"]
               c. Call kelly_fraction() to get bet_dollars and side
               d. Compute n_contracts and cost
               e. Determine if we won:
                      won = (side=="YES" and resolved_yes==1) or (side=="NO" and resolved_yes==0)
               f. Compute pnl: if won → n_contracts * (1 - price), else → -cost
               g. Update running balance
        6. Build trade_log list of dicts and return as DataFrame
        7. Call print_metrics(trades_df, starting_balance) at the end
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = run_backtest()
    if not trades.empty:
        trades.to_csv("logs/backtest_trades.csv", index=False)
        logger.info("Backtest complete. Results in logs/backtest_trades.csv")
