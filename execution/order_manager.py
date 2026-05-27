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
        self._open_positions: dict[str, float] = {}
        self._order_ids: dict[str, str] = {}
        self._realized_pnl: float = 0.0
        self.kalshi = KalshiClient()  # used for resolution checks in both modes; orders are gated by self.mode
        self._restore_open_positions()
        logger.info(f"OrderManager initialized in {self.mode.upper()} mode with {len(self._open_positions)} restored positions")

    def _restore_open_positions(self) -> None:
        """
        On startup, reload open positions from the trade log so restarts don't
        re-enter the same contracts. A position is considered open if it appears
        in the entry log but not yet in the resolved log.
        """
        if self.mode == "live":
            entry_log = Path(CONFIG["data"]["live_log_path"])
        else:
            entry_log = Path(CONFIG["data"]["dry_run_log_path"])
        resolved_log = Path(CONFIG["data"]["resolved_log_path"])

        if not entry_log.exists():
            return

        entered: dict[str, float] = {}  # contract_id → dollars_at_risk (limit_price * size / 100)
        with open(entry_log) as f:
            for row in csv.DictReader(f):
                cid = row.get("contract_id", "")
                if cid:
                    # Reconstruct dollars_at_risk from size × price
                    try:
                        dollars = int(row["size"]) * int(row["limit_price"]) / 100
                    except (KeyError, ValueError):
                        dollars = 1.0
                    entered[cid] = dollars

        resolved: set[str] = set()
        if resolved_log.exists():
            with open(resolved_log) as f:
                for row in csv.DictReader(f):
                    cid = row.get("contract_id", "")
                    if cid:
                        resolved.add(cid)

        for cid, dollars in entered.items():
            if cid not in resolved:
                self._open_positions[cid] = dollars

        if resolved_log.exists():
            with open(resolved_log) as f:
                for row in csv.DictReader(f):
                    try:
                        self._realized_pnl += float(row.get("pnl_dollars", 0))
                    except ValueError:
                        pass

    @property
    def open_positions(self) -> dict[str, float]:
        return self._open_positions

    _DRY_RUN_STARTING_BALANCE: float = CONFIG["trading"]["starting_balance"]

    @property
    def account_balance(self) -> float:
        """Return current account balance in dollars."""
        if self.mode == "dry_run":
            return self._dry_run_balance
        if self.kalshi is None:
            return self._dry_run_balance
        resp = self.kalshi._get("/portfolio/balance")
        return resp["balance"] / 100

    @property
    def _dry_run_balance(self) -> float:
        """Simulate balance: starting cash adjusted by closed-trade P&L only."""
        return self._DRY_RUN_STARTING_BALANCE + self._realized_pnl

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
            pass
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

            self._open_positions[contract_id] = n_contracts * price

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
                    self._open_positions.pop(contract_id, None)
                    return None

            record = record.model_copy(update={"order_id": order_id})
            self._order_ids[contract_id] = order_id
            log_dry_run_trade(record)
            logger.info("Live order %s filled: %s %s x%d @ %dc",
                        order_id, side, contract_id, n_contracts, limit_price_cents)

        if self.mode == "dry_run":
            log_dry_run_trade(record)
            self._open_positions[contract_id] = n_contracts * price

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
                market = self.kalshi.get_market(contract_id)  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("Could not fetch market %s: %s", contract_id, e)
                continue

            status = market.get("status", "")
            # "finalized" = fully settled; "closed" = trading ended, result may already be posted
            if status not in ("finalized", "closed"):
                continue

            result = (market.get("result") or "").lower()
            if result not in ("yes", "no"):
                # closed but result not posted yet — skip until next poll
                continue

            entry = self._get_log_entry(contract_id)
            side = entry.get("side", "YES") if entry else "YES"
            size = int(entry.get("size", 0)) if entry else 0
            entry_price_cents = int(entry.get("limit_price", 50)) if entry else 50

            won = (side.lower() == result)
            pnl = size * (1.0 - entry_price_cents / 100) if won else -size * (entry_price_cents / 100)

            pnl_rounded = round(pnl, 2)
            self._realized_pnl += pnl_rounded
            resolved.append({
                "contract_id": contract_id,
                "side": side,
                "size": size,
                "entry_price_cents": entry_price_cents,
                "result": result,
                "won": won,
                "pnl_dollars": pnl_rounded,
            })
            self.clear_position(contract_id)
            logger.info("Resolved %s → %s (%s)  P&L: $%.2f",
                        contract_id, result.upper(), "WIN" if won else "LOSS", pnl)

        return resolved

    def _get_log_entry(self, contract_id: str) -> dict | None:
        """Return the most recent CSV row for contract_id from the appropriate trade log."""
        if self.mode == "live":
            log_path = Path(CONFIG["data"]["live_log_path"])
        else:
            log_path = Path(CONFIG["data"]["dry_run_log_path"])
        if not log_path.exists():
            return None
        last = None
        with open(log_path) as f:
            for row in csv.DictReader(f):
                if row.get("contract_id") == contract_id:
                    last = row
        return last
