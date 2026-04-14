"""
execution/order_manager.py
───────────────────────────
Kalshi order management — the only file allowed to submit real orders.

All order placement MUST go through this module. It enforces dry_run mode
so members can't accidentally spend real money while testing.

Team 3 (Execution half) — implement all methods marked with TODO.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from data.features.schema import TradeRecord
from data.ingestion.kalshi_client import KalshiClient
from execution.dry_run import log_dry_run_trade

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)


class OrderManager:

    def __init__(self) -> None:
        self.mode = CONFIG["trading"]["mode"]
        self.kalshi = KalshiClient()
        self._open_positions: dict[str, float] = {}  # contract_id → dollars at risk
        logger.info(f"OrderManager initialized in {self.mode.upper()} mode")

    @property
    def open_positions(self) -> dict[str, float]:
        return self._open_positions

    @property
    def account_balance(self) -> float:
        """
        Return current account balance in dollars.

        TODO (Week 5):
            - In dry_run mode: return a hardcoded float (e.g. 100.0)
            - In live mode: GET /portfolio/balance from Kalshi API
              (balance is returned in cents — divide by 100)
        """
        raise NotImplementedError

    def submit_order(
        self,
        contract_id: str,
        side: str,
        bet_dollars: float,
        market_price: float,
        p_model: float,
    ) -> TradeRecord | None:
        """
        Submit an order (or log it in dry_run mode).

        Args:
            contract_id:  Kalshi ticker
            side:         "YES" or "NO"
            bet_dollars:  dollar amount to bet (from Kelly sizing)
            market_price: current YES price (0–1)
            p_model:      model's probability estimate

        Returns:
            TradeRecord if the order was placed or logged, None if skipped.

        TODO (Week 5):
            1. Compute the per-contract price:
                   price = market_price if side == "YES" else (1 - market_price)
            2. Compute n_contracts = int(bet_dollars / price) — skip if < 1
            3. Skip if contract_id is already in self._open_positions
            4. Build a TradeRecord (see data/features/schema.py)
            5. If self.mode == "dry_run": call log_dry_run_trade(record)
               If self.mode == "live":   call self.kalshi.place_order(...)
            6. Add contract_id → bet_dollars to self._open_positions
            7. Return the TradeRecord
        """
        raise NotImplementedError

    def clear_position(self, contract_id: str) -> None:
        """Remove a contract from open positions after it resolves."""
        self._open_positions.pop(contract_id, None)
