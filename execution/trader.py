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
RESOLVED_LOG = Path(DATA_CFG["resolved_log_path"])
RESOLVED_LOG.parent.mkdir(parents=True, exist_ok=True)

_RESOLVED_FIELDS = [
    "timestamp", "contract_id", "side", "size",
    "entry_price_cents", "result", "won", "pnl_dollars", "mode",
]


def _log_resolved(events: list[dict], mode: str) -> None:
    if not events:
        return
    file_exists = RESOLVED_LOG.exists()
    with open(RESOLVED_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RESOLVED_FIELDS)
        if not file_exists:
            w.writeheader()
        for ev in events:
            w.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "contract_id": ev["contract_id"],
                "side": ev["side"],
                "size": ev["size"],
                "entry_price_cents": ev["entry_price_cents"],
                "result": ev["result"],
                "won": ev["won"],
                "pnl_dollars": ev["pnl_dollars"],
                "mode": mode,
            })


# ── Terminal dashboard ────────────────────────────────────────────────────────

_DASHBOARD_ROWS = 12

# In-memory position ledger: contract_id → entry metadata
_open_book: dict[str, dict] = {}
_realized: list[float] = []
_wins = 0
_losses = 0


def _term_width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def _ticker_category(contract_id: str) -> str:
    prefix = contract_id.split("-")[0]
    crypto  = {"KXBTCD", "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M",
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


def _sync_closed_positions(resolution_events: list[dict], mode: str) -> None:
    global _wins, _losses
    for ev in resolution_events:
        cid = ev["contract_id"]
        pnl = ev["pnl_dollars"]
        _realized.append(pnl)
        if ev["won"]:
            _wins += 1
        else:
            _losses += 1
        _open_book.pop(cid, None)
        label = "WIN " if ev["won"] else "LOSS"
        _print_log(
            f"[{_ts()}] {label} {ev['side']:>3} {cid:<42} "
            f"result={ev['result'].upper()}  P&L=${pnl:>+.2f}"
        )
    _log_resolved(resolution_events, mode)


def _compute_stats(order_manager: OrderManager) -> dict:
    open_ev = sum(
        e["edge"] * e["size"] * (e["limit_price"] / 100)
        for e in _open_book.values()
    )
    realized_pnl = sum(_realized)
    n_closed = _wins + _losses
    win_rate = (_wins / n_closed * 100) if n_closed else 0.0

    by_cat: dict[str, int] = defaultdict(int)
    by_side: dict[str, int] = defaultdict(int)
    for cid, e in _open_book.items():
        by_cat[_ticker_category(cid)] += 1
        by_side[e["side"]] += 1

    return {
        "open": len(_open_book),
        "closed": n_closed,
        "wins": _wins,
        "losses": _losses,
        "win_rate": win_rate,
        "open_ev": open_ev,
        "realized_pnl": realized_pnl,
        "avg_edge": (
            sum(e["edge"] for e in _open_book.values()) / len(_open_book)
            if _open_book else 0.0
        ),
        "by_cat": dict(by_cat),
        "by_side": dict(by_side),
        "balance": order_manager.account_balance,
    }


def _render_dashboard(stats: dict, mode: str, next_poll: float) -> None:
    w = _term_width()
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    secs_left = max(0, int(next_poll - time.time()))
    mode_label = "DRY RUN" if mode == "dry_run" else "  LIVE "

    sep = "─" * w

    cat_str = "  ".join(
        f"{cat}:{n}" for cat, n in sorted(stats["by_cat"].items()) if n > 0
    ) or "none"
    side_str = f"YES:{stats['by_side'].get('YES', 0)}  NO:{stats['by_side'].get('NO', 0)}"

    n_closed = stats["closed"]
    win_str = (
        f"{stats['wins']}W / {stats['losses']}L  ({stats['win_rate']:.0f}% win)"
        if n_closed else "no resolved contracts yet"
    )
    pnl_str = f"realized ${stats['realized_pnl']:>+.2f}  |  open EV ${stats['open_ev']:>+.2f}"

    lines = [
        sep,
        f" {mode_label}  {now}  next poll in {secs_left:>3}s  |  balance: ${stats['balance']:.2f}",
        sep,
        f"  Open positions : {stats['open']:>3}   Closed: {n_closed:>3}   {win_str}",
        f"  P&L            : {pnl_str}",
        f"  Avg edge (open): {stats['avg_edge']:.3f}",
        sep,
        f"  Sides  : {side_str}",
        f"  Markets: {cat_str}",
        sep,
    ]

    sys.stdout.write(f"\033[{_DASHBOARD_ROWS}A\r")
    for line in lines:
        sys.stdout.write(f"\033[2K{line}\n")
    for _ in range(_DASHBOARD_ROWS - len(lines)):
        sys.stdout.write(f"\033[2K\n")
    sys.stdout.flush()


def _print_log(msg: str) -> None:
    sys.stdout.write(f"\033[{_DASHBOARD_ROWS}A\r\033[2K{msg}\n")
    sys.stdout.write(f"\033[{_DASHBOARD_ROWS - 1}B")
    sys.stdout.flush()


def _init_display() -> None:
    sys.stdout.write("\n" * _DASHBOARD_ROWS)
    sys.stdout.flush()


# ── Trading logic ─────────────────────────────────────────────────────────────

def run_once(order_manager: OrderManager) -> int:
    predictions_path = Path(DATA_CFG["predictions_path"])
    if not predictions_path.exists():
        _print_log(f"[{_ts()}] WARN  predictions.json not found — skipping")
        return 0

    with open(predictions_path) as f:
        predictions = json.load(f)

    features_path = Path(DATA_CFG["features_path"])
    if not features_path.exists():
        _print_log(f"[{_ts()}] WARN  live_features.parquet not found — skipping")
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
            edge = record.edge
            _open_book[contract_id] = {
                "side": side,
                "size": record.size,
                "limit_price": record.limit_price,
                "p_model": p_model,
                "market_price": market_price,
                "edge": edge,
            }
            _print_log(
                f"[{_ts()}] OPEN  {side:>3} {contract_id:<42} "
                f"p={p_model:.3f} mkt={market_price:.3f} "
                f"edge={edge:.3f} sz={record.size} [{cat}]"
            )

    resolution_events = order_manager.check_resolutions()
    _sync_closed_positions(resolution_events, order_manager.mode)

    if n_placed == 0:
        _print_log(f"[{_ts()}] pass — no new trades  (open: {len(order_manager.open_positions)})")
    else:
        _print_log(f"[{_ts()}] pass — {n_placed} new order(s)  (open: {len(order_manager.open_positions)})")

    return n_placed


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")

    mode = TRADING_CFG["mode"]
    poll = PIPELINE_CFG["poll_interval_sec"]

    sys.stdout.write(f"Trader starting in {mode.upper()} mode — poll every {poll}s\n")
    _init_display()

    order_manager = OrderManager()
    next_poll = time.time()

    while True:
        if time.time() >= next_poll:
            try:
                run_once(order_manager)
            except Exception as e:
                _print_log(f"[{_ts()}] ERROR {e}")
            next_poll = time.time() + poll

        stats = _compute_stats(order_manager)
        _render_dashboard(stats, mode, next_poll)
        time.sleep(1)


if __name__ == "__main__":
    main()
