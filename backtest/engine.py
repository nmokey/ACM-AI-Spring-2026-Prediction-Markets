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
import random
from pathlib import Path

import pandas as pd
import yaml

from execution.kelly import kelly_fraction, dollars_to_contracts
from backtest.metrics import compute_metrics, print_metrics

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

TRADING_CFG = CONFIG["trading"]
DATA_CFG = CONFIG["data"]

_PREDICTIONS_PATH = Path(__file__).parents[1] / DATA_CFG["predictions_path"]


def _load_dummy_trades(
    predictions_path: Path,
    starting_balance: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a simulated trade log directly from predictions.json without a
    trained model. Outcomes are randomly sampled from a Bernoulli(p_model)
    distribution — i.e. we assume the model is perfectly calibrated.

    Market prices are synthetically offset from p_model by ±0.05–0.15 to
    create realistic edges. This lets the full metrics pipeline run before
    real historical data is available.
    """
    rng = random.Random(seed)

    with open(predictions_path) as f:
        raw = json.load(f)

    balance = starting_balance
    cumulative_pnl = 0.0
    trade_log = []
    open_positions: dict[str, float] = {}

    for contract_id, entry in raw.items():
        p_model: float = entry["p_model"]
        confidence: float = entry["confidence"]

        if confidence < TRADING_CFG["min_confidence"]:
            logger.debug("Skipping %s — confidence too low", contract_id)
            continue

        # Synthetic market price: offset p_model by a random amount so we have edge
        offset = rng.uniform(0.05, 0.15) * rng.choice([-1, 1])
        market_price = max(0.05, min(0.95, p_model + offset))

        edge = abs(p_model - market_price)
        if edge < TRADING_CFG["min_edge"]:
            logger.debug("Skipping %s — edge %.3f below min", contract_id, edge)
            continue

        total_exposure = sum(open_positions.values())
        if total_exposure >= TRADING_CFG["max_total_exposure_pct"] * balance:
            logger.debug("Skipping %s — exposure cap reached", contract_id)
            continue

        bet_dollars, side = kelly_fraction(
            p_model=p_model,
            market_price=market_price,
            bankroll=balance,
            kelly_multiplier=TRADING_CFG["kelly_fraction"],
            max_position_pct=TRADING_CFG["max_position_pct"],
        )
        if bet_dollars <= 0:
            continue

        price = market_price if side == "YES" else (1 - market_price)
        n_contracts = dollars_to_contracts(bet_dollars, price)
        if n_contracts == 0:
            continue

        cost = n_contracts * price
        open_positions[contract_id] = cost

        # Simulate resolution: Bernoulli(p_model) — perfectly calibrated model
        resolved_yes = int(rng.random() < p_model)

        won = (side == "YES" and resolved_yes == 1) or (side == "NO" and resolved_yes == 0)
        payout_per_contract = (1 - price) if won else 0.0
        pnl = n_contracts * payout_per_contract - (0 if won else cost)
        balance += pnl
        cumulative_pnl += pnl

        del open_positions[contract_id]

        trade_log.append(
            {
                "contract_id": contract_id,
                "timestamp": entry["timestamp"],
                "p_model": p_model,
                "market_price": market_price,
                "side": side,
                "n_contracts": n_contracts,
                "cost": round(cost, 4),
                "edge": round(edge, 4),
                "resolved_yes": resolved_yes,
                "won": won,
                "pnl": round(pnl, 4),
                "balance": round(balance, 2),
                "cumulative_pnl": round(cumulative_pnl, 4),
            }
        )

    return pd.DataFrame(trade_log)


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

    When features_path is None and model is None, falls back to dummy mode:
    runs directly from signals/predictions.json with synthetically sampled
    outcomes so the full metrics pipeline can be exercised without real data.

    TODO (Week 6):
        1. Load the model (joblib.load from models/trained/xgb_v1.joblib)
        2. Load the features parquet — filter to rows where resolved_yes is not NaN
        3. Join sentiment signals if available (default 0.0)
        4. Run model.predict_proba(X)[:, 1] to get p_model per contract
        5. Replace _load_dummy_trades() with the real simulation loop above
    """
    if features_path is None and model is None:
        logger.info("No features/model provided — running in dummy mode from %s", _PREDICTIONS_PATH)
        trades = _load_dummy_trades(_PREDICTIONS_PATH, starting_balance)
    else:
        raise NotImplementedError("Real backtest (features + model) not yet implemented — Week 6 task")

    if not trades.empty:
        print_metrics(trades, starting_balance)

    return trades


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = run_backtest()
    if not trades.empty:
        Path("logs").mkdir(exist_ok=True)
        trades.to_csv("logs/backtest_trades.csv", index=False)
        logger.info("Backtest complete. Results in logs/backtest_trades.csv")
