"""
scripts/run_dry_run_bot.py
────────────────────────
Persistent dry-run weather bot simulator.

This script loads NOAA-style weather history, computes weather arbitrage
signals from extensions/weather_arb/strategy.py, submits dry-run orders
through execution.OrderManager, and immediately settles them using actual
weather outcomes so balance changes are visible.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Force dry_run mode for this launcher regardless of config file state.
CONFIG["trading"]["mode"] = "dry_run"

import execution.order_manager as order_manager_mod
from execution.kelly import kelly_fraction
from execution.order_manager import OrderManager
from execution.risk import check_trade
from extensions.weather_arb.strategy import compute_signals

order_manager_mod.CONFIG["trading"]["mode"] = "dry_run"

RESOLVED_FIELDS = [
    "timestamp",
    "contract_id",
    "side",
    "size",
    "entry_price_cents",
    "result",
    "won",
    "pnl_dollars",
    "mode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a persistent dry-run weather bot simulator."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "extensions" / "weather_arb" / "4312480.csv",
        help="Weather CSV with DATE and TMAX columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Temperature threshold for YES contracts.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Rolling lookback window for the climatology model.",
    )
    parser.add_argument(
        "--market-noise",
        type=float,
        default=0.05,
        help="Std dev of simulated market price noise.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge to consider for placing a trade.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds to wait between processing signals.",
    )
    return parser.parse_args()


def load_weather_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["DATE"])
    if "DATE" not in df.columns or "TMAX" not in df.columns:
        raise ValueError("Input CSV must contain DATE and TMAX columns.")

    return df[["DATE", "TMAX"]].dropna().reset_index(drop=True)


def build_contract_id(date: pd.Timestamp) -> str:
    return f"KXHIGHLAX-{date.strftime('%Y%m%d')}"


def append_resolved_event(event: dict[str, object]) -> None:
    resolved_log = Path(CONFIG["data"]["resolved_log_path"])
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    file_exists = resolved_log.exists()
    with open(resolved_log, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESOLVED_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(event)


def settle_trade(record: object, resolved_yes: int) -> float:
    side = record.side
    size = record.size
    entry_price_cents = record.limit_price
    result = "YES" if resolved_yes == 1 else "NO"
    won = side == result
    price = entry_price_cents / 100.0
    pnl = size * (1.0 - price) if won else -size * price
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract_id": record.contract_id,
        "side": side,
        "size": size,
        "entry_price_cents": entry_price_cents,
        "result": result,
        "won": won,
        "pnl_dollars": round(pnl, 2),
        "mode": "dry_run",
    }
    append_resolved_event(event)
    return round(pnl, 2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    np.random.seed(args.seed)
    weather = load_weather_history(args.input)
    logging.info("Loaded %d weather rows from %s", len(weather), args.input)

    signals = compute_signals(
        historical_temps=weather,
        threshold_f=args.threshold,
        lookback_days=args.lookback_days,
        market_noise=args.market_noise,
        min_edge=args.min_edge,
    )

    if signals.empty:
        logging.info("No weather signals generated. Exiting.")
        return

    order_manager_mod.CONFIG["trading"]["mode"] = "dry_run"
    order_manager = OrderManager()
    order_manager.mode = "dry_run"
    order_manager._open_positions = {}
    order_manager._realized_pnl = 0.0
    balance = order_manager.account_balance

    logging.info("Starting dry-run balance: $%.2f", balance)
    placed = 0
    skipped = 0
    total_pnl = 0.0

    for row in signals.itertuples(index=False):
        contract_id = build_contract_id(pd.to_datetime(row.date))
        p_model = float(row.p_model)
        market_price = float(row.market_price)
        edge = abs(p_model - market_price)
        confidence = 1.0

        risk = check_trade(
            p_model=p_model,
            market_price=market_price,
            confidence=confidence,
            open_positions=order_manager.open_positions,
            account_balance=balance,
        )
        if not risk.passed:
            logging.info("SKIP %s  edge=%.3f  reason=%s", contract_id, edge, risk.reason)
            skipped += 1
            continue

        bet_dollars, side = kelly_fraction(
            p_model=p_model,
            market_price=market_price,
            bankroll=balance,
            kelly_multiplier=CONFIG["trading"]["kelly_fraction"],
            max_position_pct=CONFIG["trading"]["max_position_pct"],
        )
        if bet_dollars <= 0:
            logging.info("SKIP %s  edge=%.3f  kelly bet=0", contract_id, edge)
            skipped += 1
            continue

        record = order_manager.submit_order(
            contract_id=contract_id,
            side=side,
            bet_dollars=bet_dollars,
            market_price=market_price,
            p_model=p_model,
        )
        if record is None:
            logging.info("SKIP %s  order not submitted", contract_id)
            skipped += 1
            continue

        placed += 1
        settlement = settle_trade(record, int(row.resolved_yes))
        total_pnl += settlement
        order_manager.clear_position(contract_id)
        order_manager._realized_pnl += settlement
        balance = order_manager.account_balance

        logging.info(
            "TRADE %s %s %d @ %.2f  edge=%.3f  pnl=%+.2f  balance=%.2f",
            contract_id,
            side,
            record.size,
            record.limit_price / 100.0,
            edge,
            settlement,
            balance,
        )

        time.sleep(args.poll_interval)

    logging.info("Finished dry-run bot: placed=%d skipped=%d total_pnl=%+.2f final_balance=%.2f", placed, skipped, total_pnl, balance)


if __name__ == "__main__":
    main()
