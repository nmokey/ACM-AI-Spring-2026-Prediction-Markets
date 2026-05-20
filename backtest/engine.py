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

import joblib
import pandas as pd
import yaml

from execution.kelly import kelly_fraction, dollars_to_contracts
from execution.risk import check_trade
from backtest.metrics import compute_metrics, print_metrics
from models.train import FEATURE_COLS

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

TRADING_CFG = CONFIG["trading"]
DATA_CFG = CONFIG["data"]

_PREDICTIONS_PATH = Path(__file__).parents[1] / DATA_CFG["predictions_path"]
_MODEL_PATH = Path(__file__).parents[1] / "models" / "trained" / "xgb_v1.joblib"
_SENTIMENT_PATH = Path(__file__).parents[1] / DATA_CFG["sentiment_path"]


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

        # Synthetic market price: offset p_model by a random amount so we have edge
        offset = rng.uniform(0.05, 0.15) * rng.choice([-1, 1])
        market_price = max(0.05, min(0.95, p_model + offset))

        risk = check_trade(
            p_model=p_model,
            market_price=market_price,
            confidence=confidence,
            open_positions=open_positions,
            account_balance=balance,
        )
        if not risk.passed:
            logger.debug("Skipping %s — %s", contract_id, risk.reason)
            continue

        edge = abs(p_model - market_price)

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


def _run_real_backtest(
    features_path: Path,
    sentiment_path: Path | None,
    model,
    starting_balance: float,
) -> pd.DataFrame:
    """Simulate trading on real historical resolved contracts."""
    # 1. Load model from disk if not supplied
    if model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {_MODEL_PATH} — run models/train.py first")
        model = joblib.load(_MODEL_PATH)
        logger.info("Loaded model from %s", _MODEL_PATH)

    # 2. Load features, keep only rows with a real label
    df = pd.read_parquet(features_path)
    before = len(df)
    df = df.dropna(subset=["resolved_yes"])
    logger.info("Loaded %d labeled rows (dropped %d unresolved) from %s", len(df), before - len(df), features_path)

    if df.empty:
        logger.warning("No resolved contracts in %s — nothing to backtest", features_path)
        return pd.DataFrame()

    # 3. One row per contract: last snapshot before resolution gives the most
    #    informed features without look-ahead bias
    df = df.sort_values("fetched_at").groupby("contract_id").last().reset_index()
    logger.info("%d unique resolved contracts after dedup", len(df))

    # 4. Join sentiment (default 0.0 when missing)
    resolved_sentiment = sentiment_path or _SENTIMENT_PATH
    if resolved_sentiment and Path(resolved_sentiment).exists():
        with open(resolved_sentiment) as f:
            raw = json.load(f)
        sent_df = pd.DataFrame.from_dict(raw, orient="index")[
            ["sentiment_score", "sentiment_confidence"]
        ]
        sent_df.index.name = "contract_id"
        df = df.drop(columns=["sentiment_score", "sentiment_confidence"], errors="ignore")
        df = df.set_index("contract_id").join(sent_df, how="left").reset_index()

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    # 5. Model inference
    X = df[FEATURE_COLS].values
    df["p_model"] = model.predict_proba(X)[:, 1]
    df["confidence"] = (df["p_model"] - 0.5).abs() * 2

    # 6. Chronological order so balance evolves correctly
    df = df.sort_values("fetched_at").reset_index(drop=True)

    # 7. Simulation loop — identical logic to _load_dummy_trades but uses
    #    real market_price and real resolved_yes instead of synthetic values
    balance = starting_balance
    cumulative_pnl = 0.0
    trade_log = []
    open_positions: dict[str, float] = {}

    for _, row in df.iterrows():
        contract_id: str = row["contract_id"]
        p_model: float = row["p_model"]
        market_price = row.get("market_price")
        confidence: float = row["confidence"]
        resolved_yes: int = int(row["resolved_yes"])

        if market_price is None or pd.isna(market_price):
            logger.debug("Skipping %s — missing market_price", contract_id)
            continue

        risk = check_trade(
            p_model=p_model,
            market_price=market_price,
            confidence=confidence,
            open_positions=open_positions,
            account_balance=balance,
        )
        if not risk.passed:
            logger.debug("Skipping %s — %s", contract_id, risk.reason)
            continue

        edge = abs(p_model - market_price)

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

        won = (side == "YES" and resolved_yes == 1) or (side == "NO" and resolved_yes == 0)
        payout_per_contract = (1 - price) if won else 0.0
        pnl = n_contracts * payout_per_contract - (0 if won else cost)
        balance += pnl
        cumulative_pnl += pnl

        del open_positions[contract_id]

        trade_log.append({
            "contract_id": contract_id,
            "timestamp": row.get("fetched_at", ""),
            "p_model": round(p_model, 4),
            "market_price": round(market_price, 4),
            "side": side,
            "n_contracts": n_contracts,
            "cost": round(cost, 4),
            "edge": round(edge, 4),
            "resolved_yes": resolved_yes,
            "won": won,
            "pnl": round(pnl, 4),
            "balance": round(balance, 2),
            "cumulative_pnl": round(cumulative_pnl, 4),
        })

    return pd.DataFrame(trade_log)


def run_backtest(
    features_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
    model=None,
    starting_balance: float = TRADING_CFG["starting_balance"],
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
    """
    if features_path is None and model is None:
        logger.info("No features/model provided — running in dummy mode from %s", _PREDICTIONS_PATH)
        trades = _load_dummy_trades(_PREDICTIONS_PATH, starting_balance)
    else:
        if features_path is None:
            raise ValueError("features_path is required when a model is provided")
        trades = _run_real_backtest(
            features_path=Path(features_path),
            sentiment_path=Path(sentiment_path) if sentiment_path else None,
            model=model,
            starting_balance=starting_balance,
        )

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
