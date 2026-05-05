"""
execution/order_manager.py
───────────────────────────
Kalshi order management — the only file allowed to submit real orders.

All order placement MUST go through this module. It enforces dry_run mode
so members can't accidentally spend real money while testing.

Team 3 — Execution — implement all methods marked with TODO.
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
        """Return current account balance in dollars."""
        if self.mode == "dry_run":
            return 100.0
        resp = self.kalshi._get("/portfolio/balance")
        return resp["balance"] / 100

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
        """
        if contract_id in self._open_positions:
            logger.debug("Skipping %s — already have open position", contract_id)
            return None

        price = market_price if side == "YES" else (1 - market_price)
        if price <= 0:
            return None
        n_contracts = int(bet_dollars / price)
        if n_contracts < 1:
            return None

        limit_price_cents = int(round(price * 100))
        record = TradeRecord(
            contract_id=contract_id,
            timestamp=datetime.now(timezone.utc),
            side=side,
            size=n_contracts,
            limit_price=limit_price_cents,
            p_model=p_model,
            market_price=market_price,
            edge=abs(p_model - market_price),
            mode=self.mode,
        )

        if self.mode == "dry_run":
            log_dry_run_trade(record)
        else:
            self.kalshi.place_order(
                ticker=contract_id,
                side=side.lower(),
                count=n_contracts,
                limit_price=limit_price_cents,
            )

        self._open_positions[contract_id] = bet_dollars
        logger.info("Order logged: %s %s x%d @ %dc", side, contract_id, n_contracts, limit_price_cents)
        return record

    def clear_position(self, contract_id: str) -> None:
        """Remove a contract from open positions after it resolves."""
        self._open_positions.pop(contract_id, None)
