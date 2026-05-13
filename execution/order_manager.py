"""
execution/order_manager.py
───────────────────────────
Kalshi order management — the only file allowed to submit real orders.

All order placement MUST go through this module. It enforces dry_run mode
so members can't accidentally spend real money while testing.

Team 3 — Execution — implement all methods marked with TODO.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from data.schema import TradeRecord
from data.ingestion.kalshi_client import KalshiClient
from execution.dry_run import log_dry_run_trade

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

# How long to wait for a resting live order to fill before canceling it (seconds).
_FILL_TIMEOUT_SEC = 30


class OrderManager:

    def __init__(self) -> None:
        self.mode = CONFIG["trading"]["mode"]
        self.kalshi = KalshiClient()
        self._open_positions: dict[str, float] = {}  # contract_id → dollars at risk
        # Live mode only: order_id per open position for fill-status polling.
        self._order_ids: dict[str, str] = {}         # contract_id → kalshi order_id
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

        In live mode, waits up to _FILL_TIMEOUT_SEC for the order to fill.
        If the order doesn't fill in time it is canceled and None is returned —
        so open_positions only ever contains orders that actually executed.

        Returns:
            TradeRecord if the order was placed/logged, None if skipped or unfilled.
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
            order_id="",
        )

        if self.mode == "dry_run":
            log_dry_run_trade(record)
        else:
            order = self.kalshi.place_order(
                ticker=contract_id,
                side=side.lower(),
                count=n_contracts,
                limit_price=limit_price_cents,
            )
            order_id = order.get("order_id", "")
            status = order.get("status", "unknown")

            if status == "canceled":
                logger.warning("Order %s for %s immediately canceled — skipping", order_id, contract_id)
                return None

            if status == "resting":
                order_id, status = self._wait_for_fill(order_id, contract_id)
                if status != "executed":
                    try:
                        self.kalshi._delete(f"/portfolio/orders/{order_id}")
                    except Exception as e:
                        logger.warning("Could not cancel order %s: %s", order_id, e)
                    logger.warning(
                        "Order %s for %s did not fill within %ds (status=%s) — canceled",
                        order_id, contract_id, _FILL_TIMEOUT_SEC, status,
                    )
                    return None

            record = record.model_copy(update={"order_id": order_id})
            self._order_ids[contract_id] = order_id
            logger.info("Live order %s filled: %s %s x%d @ %dc",
                        order_id, side, contract_id, n_contracts, limit_price_cents)

        self._open_positions[contract_id] = bet_dollars
        logger.info("Position opened: %s %s x%d @ %dc", side, contract_id, n_contracts, limit_price_cents)
        return record

    def _wait_for_fill(self, order_id: str, contract_id: str) -> tuple[str, str]:
        """Poll GET /portfolio/orders/{order_id} until executed or timeout."""
        deadline = time.time() + _FILL_TIMEOUT_SEC
        while time.time() < deadline:
            time.sleep(2)
            try:
                order = self.kalshi._get(f"/portfolio/orders/{order_id}")["order"]
                status = order.get("status", "unknown")
                if status in ("executed", "canceled"):
                    return order_id, status
            except Exception as e:
                logger.warning("Error polling order %s: %s", order_id, e)
        return order_id, "resting"

    def clear_position(self, contract_id: str) -> None:
        """Remove a contract from open positions after it resolves."""
        self._open_positions.pop(contract_id, None)
        self._order_ids.pop(contract_id, None)

    def check_resolutions(self) -> list[dict]:
        """
        Poll Kalshi for each open position and return resolution events for any
        that have settled (status == "finalized").

        Each event dict:
            contract_id, side, size, entry_price_cents, result, won, pnl_dollars

        Resolved positions are removed from open_positions automatically.
        """
        resolved = []
        for contract_id in list(self._open_positions.keys()):
            try:
                market = self.kalshi.get_market(contract_id)
            except Exception as e:
                logger.warning("Could not fetch market %s: %s", contract_id, e)
                continue

            if market.get("status") != "finalized":
                continue

            result = (market.get("result") or "").lower()
            if result not in ("yes", "no"):
                continue

            entry = self._get_log_entry(contract_id)
            side = entry.get("side", "YES") if entry else "YES"
            size = int(entry.get("size", 0)) if entry else 0
            entry_price_cents = int(entry.get("limit_price", 50)) if entry else 50

            won = (side.lower() == result)
            pnl = size * (1.0 - entry_price_cents / 100) if won else -size * (entry_price_cents / 100)

            resolved.append({
                "contract_id": contract_id,
                "side": side,
                "size": size,
                "entry_price_cents": entry_price_cents,
                "result": result,
                "won": won,
                "pnl_dollars": round(pnl, 2),
            })
            self.clear_position(contract_id)
            logger.info("Resolved %s → %s (%s)  P&L: $%.2f",
                        contract_id, result.upper(), "WIN" if won else "LOSS", pnl)

        return resolved

    def _get_log_entry(self, contract_id: str) -> dict | None:
        """Return the most recent CSV row for contract_id."""
        log_path = Path(CONFIG["data"]["dry_run_log_path"])
        if not log_path.exists():
            return None
        last = None
        with open(log_path) as f:
            for row in csv.DictReader(f):
                if row.get("contract_id") == contract_id:
                    last = row
        return last
