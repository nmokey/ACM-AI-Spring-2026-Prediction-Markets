"""
execution/trader.py
─────────────────────
Main trading loop with live terminal dashboard.

Renders a persistent status bar at the bottom of the terminal showing
model metrics and trade stats, with a scrolling trade log above it.

Run via: python -m execution.trader  or  bash scripts/run_bot.sh
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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

DRY_RUN_LOG = Path(DATA_CFG["dry_run_log_path"])

# ── Terminal dashboard ────────────────────────────────────────────────────────

_log_lines: list[str] = []
_DASHBOARD_ROWS = 10  # lines reserved at bottom for status panel


def _term_width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def _ticker_category(contract_id: str) -> str:
    prefix = contract_id.split("-")[0]
    crypto = {"KXBTCD", "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M",
               "KXDOGE", "KXDOGE15M", "KXBNB15M"}
    weather = {"KXHIGHLAX", "KXHIGHNY", "KXHIGHCHI", "KXHIGHDEN", "KXHIGHMIA",
                "KXHIGHTSFO", "KXHIGHAUS", "KXHIGHTDC", "KXHIGHTPHX", "KXHIGHTSEA",
                "KXLOWTCHI", "KXLOWTDEN", "KXLOWTHOU", "KXLOWTLAX", "KXLOWTNYC",
                "KXLOWTSEA", "KXLOWTSFO"}
    sports  = {"KXMLB", "KXMLBHRR", "KXNBA", "KXNHL", "KXF1"}
    macro   = {"KXFED", "KXCPI", "KXGDP", "KXADP", "KXWTI", "KXEURUSD", "KXUSDJPY"}
    if prefix in crypto:  return "crypto"
    if prefix in weather: return "weather"
    if prefix in sports:  return "sports"
    if prefix in macro:   return "macro"
    return "other"


def _load_trade_log() -> pd.DataFrame | None:
    if not DRY_RUN_LOG.exists():
        return None
    try:
        return pd.read_csv(DRY_RUN_LOG)
    except Exception:
        return None


def _compute_stats(df: pd.DataFrame) -> dict:
    stats: dict = {
        "total_trades": len(df),
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "avg_edge": 0.0,
        "by_category": defaultdict(int),
        "by_side": defaultdict(int),
        "resolved": 0,
    }
    if df.empty:
        return stats

    stats["avg_edge"] = df["edge"].mean()
    stats["by_side"]["YES"] = int((df["side"] == "YES").sum())
    stats["by_side"]["NO"]  = int((df["side"] == "NO").sum())

    for cid in df["contract_id"]:
        stats["by_category"][_ticker_category(cid)] += 1

    # Estimate P&L: for contracts where market_price is available as proxy for outcome
    # (real P&L requires knowing resolution — use edge as a proxy for expected value)
    stats["total_pnl"] = float((df["edge"] * df["size"] * (df["limit_price"] / 100)).sum())

    return stats


def _render_dashboard(stats: dict, mode: str, next_poll: float) -> None:
    w = _term_width()
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    secs_left = max(0, int(next_poll - time.time()))

    sep = "─" * w

    cat_str = "  ".join(
        f"{cat}:{n}" for cat, n in sorted(stats["by_category"].items()) if n > 0
    ) or "—"
    side_str = f"YES:{stats['by_side']['YES']}  NO:{stats['by_side']['NO']}"

    lines = [
        sep,
        f" {'DRY RUN' if mode == 'dry_run' else 'LIVE':^8}  {now}  next poll in {secs_left:>3}s",
        sep,
        f"  Trades: {stats['total_trades']:>4}    "
        f"Avg edge: {stats['avg_edge']:>5.3f}    "
        f"Est. EV: ${stats['total_pnl']:>+7.2f}",
        f"  Sides:  {side_str:<30}",
        f"  By cat: {cat_str}",
        sep,
    ]

    # Move cursor up to overwrite dashboard area, then redraw
    sys.stdout.write(f"\033[{_DASHBOARD_ROWS}A")
    for line in lines:
        sys.stdout.write(f"\r{line:<{w}}\n")
    # Pad remaining reserved rows
    for _ in range(_DASHBOARD_ROWS - len(lines)):
        sys.stdout.write(f"\r{'':<{w}}\n")
    sys.stdout.flush()


def _print_log(msg: str) -> None:
    """Print a log line above the dashboard area."""
    # Move up past dashboard, print, move back down
    sys.stdout.write(f"\033[{_DASHBOARD_ROWS}A")
    sys.stdout.write(f"\r{msg:<{_term_width()}}\n")
    sys.stdout.write(f"\033[{_DASHBOARD_ROWS - 1}B")
    sys.stdout.flush()


def _init_display() -> None:
    """Reserve dashboard rows at bottom by printing blank lines."""
    sys.stdout.write("\n" * _DASHBOARD_ROWS)
    sys.stdout.flush()


# ── Trading logic ─────────────────────────────────────────────────────────────

def run_once(order_manager: OrderManager) -> int:
    predictions_path = Path(DATA_CFG["predictions_path"])
    if not predictions_path.exists():
        _print_log(f"[{_ts()}] WARN predictions.json not found — skipping")
        return 0

    with open(predictions_path) as f:
        predictions = json.load(f)

    features_path = Path(DATA_CFG["features_path"])
    if not features_path.exists():
        _print_log(f"[{_ts()}] WARN live_features.parquet not found — skipping")
        return 0

    df = pd.read_parquet(features_path)
    prices = dict(zip(df["contract_id"], df["market_price"]))
    balance = order_manager.account_balance
    n_placed = 0

    for contract_id, entry in predictions.items():
        market_price = prices.get(contract_id)
        if market_price is None or pd.isna(market_price):
            continue

        p_model    = entry["p_model"]
        confidence = entry["confidence"]

        risk = check_trade(
            p_model=p_model,
            market_price=market_price,
            confidence=confidence,
            open_positions=order_manager.open_positions,
            account_balance=balance,
        )
        if not risk.passed:
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
            cat = _ticker_category(contract_id)
            _print_log(
                f"[{_ts()}] {side:>3} {contract_id:<45} "
                f"p={p_model:.3f} mkt={market_price:.3f} "
                f"edge={record.edge:.3f} sz={record.size} [{cat}]"
            )

    if n_placed == 0:
        _print_log(f"[{_ts()}] pass complete — no new trades (open positions: {len(order_manager.open_positions)})")
    else:
        _print_log(f"[{_ts()}] pass complete — {n_placed} order(s) logged")

    return n_placed


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def main() -> None:
    # Plain logging to stderr so it doesn't clobber the display
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")

    mode = TRADING_CFG["mode"]
    poll = PIPELINE_CFG["poll_interval_sec"]

    sys.stdout.write(f"Trader starting in {mode.upper()} mode — poll every {poll}s\n")
    _init_display()

    order_manager = OrderManager()
    next_poll = time.time()

    while True:
        try:
            run_once(order_manager)
        except Exception as e:
            _print_log(f"[{_ts()}] ERROR {e}")

        next_poll = time.time() + poll
        df = _load_trade_log()
        stats = _compute_stats(df) if df is not None else _compute_stats(pd.DataFrame())
        _render_dashboard(stats, mode, next_poll)

        time.sleep(poll)


if __name__ == "__main__":
    main()
