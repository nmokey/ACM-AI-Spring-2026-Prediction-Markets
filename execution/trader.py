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

    TODO (Week 5):
        1. Load predictions.json — return 0 early if file doesn't exist yet
        2. Load live_features.parquet — build a dict of {contract_id: market_price}
        3. Get current balance from order_manager.account_balance
        4. Loop over predictions:
               a. Skip if no live market price found for this contract
               b. Call check_trade(...) — skip if result.passed is False
               c. Call kelly_fraction(...) — skip if bet_dollars <= 0
               d. Call order_manager.submit_order(...)
        5. Log and return the count of orders placed
    """
    raise NotImplementedError


def main() -> None:
    """
    Run the trading loop indefinitely, sleeping between passes.

    TODO (Week 5):
        1. Set up logging
        2. Log the current mode (dry_run vs live) from TRADING_CFG
        3. Create an OrderManager instance
        4. Loop forever: call run_once(), then time.sleep(PIPELINE_CFG["poll_interval_sec"])
        5. Catch exceptions so one bad pass doesn't kill the whole loop
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
