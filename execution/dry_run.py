"""
execution/dry_run.py
──────────────────────
Dry-run trade logger.

In dry_run mode, orders are never submitted to Kalshi — they are
written to logs/dry_run_trades.csv for analysis and backtesting.

Team 3 — Execution owns this file.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import yaml

from data.schema import TradeRecord

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.yaml"
with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DRY_RUN_LOG = Path(CONFIG["data"]["dry_run_log_path"])
DRY_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

PRETTY_LOG = DRY_RUN_LOG.parent / "trades.log"

_FIELDNAMES = [
    "contract_id", "timestamp", "side", "size",
    "limit_price", "p_model", "market_price", "edge", "mode",
]


def log_dry_run_trade(record: TradeRecord) -> None:
    if record.mode == "live":
        log_path = Path(CONFIG["data"]["live_log_path"])
    else:
        log_path = DRY_RUN_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "contract_id": record.contract_id,
            "timestamp": record.timestamp.isoformat(),
            "side": record.side,
            "size": record.size,
            "limit_price": record.limit_price,
            "p_model": record.p_model,
            "market_price": record.market_price,
            "edge": record.edge,
            "mode": record.mode,
        })


def log_pretty_entry(record: TradeRecord, trade_num: int) -> None:
    """Append a human-readable trade entry to logs/trades.log."""
    profit_per = 100 - record.limit_price
    ts = record.timestamp.strftime("%Y-%m-%d %H:%M")
    mode_tag = "DRY RUN" if record.mode == "dry_run" else "LIVE"
    with open(PRETTY_LOG, "a") as f:
        f.write(
            f"  #{trade_num:<4} {ts}  {record.side:<3}  {record.contract_id:<42}"
            f"paid {record.limit_price}¢ × {record.size}  (+{profit_per}¢ profit/ea)\n"
            f"         p_model={record.p_model:.3f}  mkt={record.market_price:.3f}  "
            f"edge={record.edge:.3f}  conf={(abs(record.p_model - 0.5) * 2):.3f}  [{mode_tag}]\n"
        )


def log_pretty_resolution(contract_id: str, result: str, won: bool, pnl: float) -> None:
    """Append a resolution line under the matching entry in logs/trades.log."""
    label = "WIN " if won else "LOSS"
    with open(PRETTY_LOG, "a") as f:
        f.write(
            f"         → [{label}]  {contract_id}  result={result.upper()}  "
            f"P&L {pnl:+.2f}\n\n"
        )
