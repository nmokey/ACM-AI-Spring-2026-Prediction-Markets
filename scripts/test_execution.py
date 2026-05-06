"""
scripts/test_execution.py
───────────────────────────
Smoke test for the Team 3 execution pipeline using dummy data.

Bypasses the real Kalshi API and live_features.parquet by injecting
mock market prices directly. Verifies kelly sizing, risk checks,
dry_run logging, and the trade record format.

Usage: python scripts/test_execution.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from data.schema import PredictionSignal, TradeRecord
from execution.kelly import kelly_fraction, dollars_to_contracts
from execution.risk import check_trade, RiskCheckResult
from execution.dry_run import log_dry_run_trade

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PREDICTIONS_PATH = Path("signals/predictions.json")

# Simulated market prices for each contract (what Kalshi shows right now).
# In production these come from live_features.parquet; here we hardcode them.
MOCK_MARKET_PRICES: dict[str, float] = {
    "KXBTC-26APR21-T9000000":  0.62,   # model=0.72 → edge=+0.10 → BUY YES
    "KXBTC-26APR21-T8500000":  0.83,   # model=0.88 → edge=+0.05 → below min_edge, SKIP
    "KXETH-26APR21-T300000":   0.55,   # model=0.41 → edge=-0.14 → BUY NO
    "KXNYRAIN-26APR21":        0.45,   # model=0.33 → edge=-0.12 → BUY NO
    "KXLAXRAIN-26APR21":       0.22,   # model=0.11 → edge=-0.11 → BUY NO
    "KXNBA-LAL-GSW-26APR21":   0.52,   # model=0.58 → edge=+0.06 → confidence=0.55 < 0.60, SKIP
}

MOCK_ACCOUNT_BALANCE = 100.0
MOCK_OPEN_POSITIONS: dict[str, float] = {}


def run_test() -> None:
    log.info("Loading predictions from %s", PREDICTIONS_PATH)
    with open(PREDICTIONS_PATH) as f:
        raw = json.load(f)

    predictions = {cid: PredictionSignal(**data) for cid, data in raw.items()}
    log.info("Loaded %d predictions", len(predictions))

    orders_placed = 0
    orders_skipped = 0

    print()
    print(f"{'CONTRACT':<35} {'p_model':>7} {'mkt_px':>7} {'edge':>7} {'conf':>6}  {'RESULT'}")
    print("-" * 85)

    for cid, pred in predictions.items():
        market_price = MOCK_MARKET_PRICES.get(cid)
        if market_price is None:
            log.warning("No mock market price for %s — skipping", cid)
            continue

        risk = check_trade(
            p_model=pred.p_model,
            market_price=market_price,
            confidence=pred.confidence,
            open_positions=MOCK_OPEN_POSITIONS,
            account_balance=MOCK_ACCOUNT_BALANCE,
        )

        edge = pred.p_model - market_price

        if not risk.passed:
            print(f"{cid:<35} {pred.p_model:>7.2f} {market_price:>7.2f} {edge:>+7.2f} {pred.confidence:>6.2f}  SKIP  ({risk.reason})")
            orders_skipped += 1
            continue

        bet_dollars, side = kelly_fraction(
            p_model=pred.p_model,
            market_price=market_price,
            bankroll=MOCK_ACCOUNT_BALANCE,
        )

        if bet_dollars <= 0:
            print(f"{cid:<35} {pred.p_model:>7.2f} {market_price:>7.2f} {edge:>+7.2f} {pred.confidence:>6.2f}  SKIP  (kelly=0)")
            orders_skipped += 1
            continue

        price = market_price if side == "YES" else (1 - market_price)
        n_contracts = dollars_to_contracts(bet_dollars, price)

        if n_contracts < 1:
            print(f"{cid:<35} {pred.p_model:>7.2f} {market_price:>7.2f} {edge:>+7.2f} {pred.confidence:>6.2f}  SKIP  (n_contracts=0)")
            orders_skipped += 1
            continue

        record = TradeRecord(
            contract_id=cid,
            timestamp=datetime.now(timezone.utc),
            side=side,
            size=n_contracts,
            limit_price=round(price * 100),
            p_model=pred.p_model,
            market_price=market_price,
            edge=abs(edge),
            mode="dry_run",
        )

        log_dry_run_trade(record)
        MOCK_OPEN_POSITIONS[cid] = bet_dollars
        orders_placed += 1

        print(f"{cid:<35} {pred.p_model:>7.2f} {market_price:>7.2f} {edge:>+7.2f} {pred.confidence:>6.2f}  ORDER {side:<3}  ${bet_dollars:.2f}  ({n_contracts} contracts @ {price:.2f})")

    print()
    log.info("Done: %d orders logged, %d skipped", orders_placed, orders_skipped)
    log.info("Trade log written to logs/dry_run_trades.csv")

    if orders_placed > 0:
        import csv
        with open("logs/dry_run_trades.csv") as f:
            rows = list(csv.DictReader(f))
        print(f"\nCSV log ({len(rows)} total rows):")
        for row in rows[-orders_placed:]:
            print(f"  {row}")


if __name__ == "__main__":
    run_test()
