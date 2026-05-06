"""
execution/trader.py
─────────────────────
Main trading loop.

Reads the latest predictions (from Team 2) and live market prices
(from Team 1), applies risk filters, sizes positions with Kelly,
and submits or logs orders.

Team 3 — Execution — implement run_once() and main().

Run via: python -m execution.trader  or  bash scripts/run_bot.sh
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from execution.kelly import kelly_fraction
from execution.order_manager import OrderManager
from execution.risk import check_trade

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

TRADING_CFG = CONFIG["trading"]
PIPELINE_CFG = CONFIG["pipeline"]
DATA_CFG = CONFIG["data"]


def run_once(order_manager: OrderManager) -> int:
    """
    Execute one pass of the trading loop.

    For each prediction:
        1. Look up the current market price from live_features.parquet
        2. Run risk checks (check_trade)
        3. Compute Kelly bet size (kelly_fraction)
        4. Submit the order (order_manager.submit_order)

    Returns: number of orders placed or logged this pass.
    """
    predictions_path = Path(DATA_CFG["predictions_path"])
    if not predictions_path.exists():
        logger.warning("predictions.json not found — skipping pass")
        return 0

    with open(predictions_path) as f:
        predictions = json.load(f)

    features_path = Path(DATA_CFG["features_path"])
    if not features_path.exists():
        logger.warning("live_features.parquet not found — skipping pass")
        return 0

    df = pd.read_parquet(features_path)
    prices = dict(zip(df["contract_id"], df["market_price"]))

    balance = order_manager.account_balance
    n_placed = 0

    for contract_id, entry in predictions.items():
        market_price = prices.get(contract_id)
        if market_price is None or pd.isna(market_price):
            continue

        p_model = entry["p_model"]
        confidence = entry["confidence"]

        risk = check_trade(
            p_model=p_model,
            market_price=market_price,
            confidence=confidence,
            open_positions=order_manager.open_positions,
            account_balance=balance,
        )
        if not risk.passed:
            logger.debug("Risk check failed for %s: %s", contract_id, risk.reason)
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

        record = order_manager.submit_order(
            contract_id=contract_id,
            side=side,
            bet_dollars=bet_dollars,
            market_price=market_price,
            p_model=p_model,
        )
        if record is not None:
            n_placed += 1

    logger.info("Pass complete — %d orders placed/logged", n_placed)
    return n_placed


def main() -> None:
    """Run the trading loop indefinitely, sleeping between passes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Trader starting in %s mode", TRADING_CFG["mode"].upper())
    order_manager = OrderManager()

    while True:
        try:
            run_once(order_manager)
        except Exception:
            logger.exception("Unhandled error in trading pass — continuing")
        time.sleep(PIPELINE_CFG["poll_interval_sec"])


if __name__ == "__main__":
    main()
