"""
execution/trader.py
─────────────────────
Main trading loop with live terminal dashboard.

Uses a terminal scroll region to keep the status panel pinned at the bottom
while trade log lines scroll naturally above it.

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
LIVE_LOG = Path(DATA_CFG["live_log_path"])
RESOLVED_LOG = Path(DATA_CFG["resolved_log_path"])
RESOLVED_LOG.parent.mkdir(parents=True, exist_ok=True)

_RESOLVED_FIELDS = [
    "timestamp", "contract_id", "side", "size",
    "entry_price_cents", "result", "won", "pnl_dollars", "mode",
]

# Number of lines the status panel occupies at the bottom.
# Must match the number of lines _build_dashboard() produces.
_PANEL_ROWS = 10

# Terminal dimensions captured once at startup and reused everywhere.
# Re-querying mid-session causes the panel to drift to different row offsets.
_TERM_W: int = 120
_TERM_H: int = 40


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


# ── Terminal display helpers ──────────────────────────────────────────────────

def _query_term_size() -> tuple[int, int]:
    """Query the real tty size, trying each fd and /dev/tty as fallbacks."""
    import os
    for fd in (sys.stdout.fileno(), sys.stderr.fileno(), sys.stdin.fileno()):
        try:
            s = os.get_terminal_size(fd)
            return s.columns, s.lines
        except OSError:
            continue
    try:
        with open("/dev/tty") as tty:
            s = os.get_terminal_size(tty.fileno())
            return s.columns, s.lines
    except OSError:
        pass
    s = shutil.get_terminal_size((120, 40))
    return s.columns, s.lines


def _term_size() -> tuple[int, int]:
    """Return the terminal dimensions captured at startup."""
    return _TERM_W, _TERM_H


def _set_scroll_region(top: int, bottom: int) -> None:
    """Set terminal scroll region to rows top..bottom (1-based)."""
    sys.stdout.write(f"\033[{top};{bottom}r")


def _move_to(row: int, col: int = 1) -> None:
    sys.stdout.write(f"\033[{row};{col}H")


def _clear_line() -> None:
    sys.stdout.write("\033[2K")


def _init_display() -> None:
    """Capture terminal size once, then set up the fixed panel and scroll region."""
    global _TERM_W, _TERM_H
    _TERM_W, _TERM_H = _query_term_size()
    w, h = _TERM_W, _TERM_H
    sys.stdout.write("\033[2J")           # clear screen
    sys.stdout.write("\033[3J")           # clear scrollback buffer
    _set_scroll_region(1, h - _PANEL_ROWS)
    _move_to(1, 1)                        # park cursor at top of scroll region
    sys.stdout.flush()


def _print_log(msg: str) -> None:
    """Write a log line; the scroll region scrolls it upward naturally."""
    w = _TERM_W
    # \r ensures we start at column 1; \n triggers the scroll region scroll.
    sys.stdout.write(f"\r{msg[:w]}\n")
    sys.stdout.flush()


def _build_dashboard(stats: dict, mode: str, next_poll: float) -> list[str]:
    w, _ = _term_size()
    now = datetime.now(timezone.utc).strftime("%m-%d %H:%M UTC")
    secs_left = max(0, int(next_poll - time.time()))
    mode_label = "DRY RUN" if mode == "dry_run" else "  LIVE "
    sep = "─" * w

    n_closed = stats["closed"]
    win_str = (
        f"{stats['wins']}W / {stats['losses']}L  ({stats['win_rate']:.0f}% win)"
        if n_closed else "awaiting resolutions"
    )
    realized = stats["realized_pnl"]
    open_ev  = stats["open_ev"]
    pnl_str  = f"realized ${realized:+.2f}  |  open EV ${open_ev:+.2f}"

    # All-time category + side counts (open + closed)
    cat_str  = "  ".join(
        f"{cat}:{n}" for cat, n in sorted(stats["all_cat"].items()) if n > 0
    ) or "none"
    side_str = (
        f"YES:{stats['all_side'].get('YES', 0)}  "
        f"NO:{stats['all_side'].get('NO', 0)}"
    )

    return [
        sep,
        f" {mode_label}  {now}  next poll in {secs_left:>3}s  |  balance: ${stats['balance']:.2f}",
        sep,
        f"  Open: {stats['open']:>3}   Closed: {n_closed:>3}   {win_str}",
        f"  P&L : {pnl_str}",
        f"  Edge: avg open {stats['avg_edge']:.3f}",
        sep,
        f"  Sides   (all-time): {side_str}",
        f"  Markets (all-time): {cat_str}",
        sep,
    ]
    # _PANEL_ROWS must equal len(lines) + 1 (the blank line between scroll region and panel)


def _render_dashboard(stats: dict, mode: str, next_poll: float) -> None:
    w, h = _TERM_W, _TERM_H
    lines = _build_dashboard(stats, mode, next_poll)
    # Panel occupies the last _PANEL_ROWS rows (outside the scroll region).
    panel_start = h - _PANEL_ROWS + 1
    for i, line in enumerate(lines):
        _move_to(panel_start + i)
        _clear_line()
        sys.stdout.write(line[:w])
    # Park cursor at top of scroll region so future _print_log calls land there.
    _move_to(1, 1)
    sys.stdout.flush()


# ── Position tracking ─────────────────────────────────────────────────────────

_open_book: dict[str, dict] = {}   # contract_id → entry metadata
_all_trades: list[dict] = []       # every trade ever placed this session
_realized: list[float] = []
_wins = 0
_losses = 0


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


def _compute_stats(order_manager: OrderManager) -> dict:
    open_ev      = sum(e["edge"] * e["size"] * (e["limit_price"] / 100) for e in _open_book.values())
    realized_pnl = sum(_realized)
    n_closed     = _wins + _losses
    win_rate     = (_wins / n_closed * 100) if n_closed else 0.0

    # All-time counts from _all_trades list
    all_cat:  dict[str, int] = defaultdict(int)
    all_side: dict[str, int] = defaultdict(int)
    for t in _all_trades:
        all_cat[_ticker_category(t["contract_id"])] += 1
        all_side[t["side"]] += 1

    return {
        "open":         len(_open_book),
        "closed":       n_closed,
        "wins":         _wins,
        "losses":       _losses,
        "win_rate":     win_rate,
        "open_ev":      open_ev,
        "realized_pnl": realized_pnl,
        "avg_edge":     (
            sum(e["edge"] for e in _open_book.values()) / len(_open_book)
            if _open_book else 0.0
        ),
        "all_cat":      dict(all_cat),
        "all_side":     dict(all_side),
        "balance":      order_manager.account_balance,
    }


def _sync_closed_positions(resolution_events: list[dict], mode: str) -> None:
    global _wins, _losses
    for ev in resolution_events:
        cid = ev["contract_id"]
        pnl = ev["pnl_dollars"]
        _realized.append(pnl)
        entry = _open_book.pop(cid, {})
        if ev["won"]:
            _wins += 1
            label = "WIN "
        else:
            _losses += 1
            label = "LOSS"
        cat = _ticker_category(cid)
        _print_log(
            f"[{_ts()}] {label} {ev['side']:>3} {cid}  "
            f"result={ev['result'].upper()}  P&L=${pnl:+.2f}  [{cat}]"
        )
        if entry:
            _print_log(
                f"          entry: p_model={entry.get('p_model', 0):.3f}  "
                f"mkt_at_entry={entry.get('market_price', 0):.3f}  "
                f"edge={entry.get('edge', 0):.3f}  sz={entry.get('size', 0)}"
            )
    _log_resolved(resolution_events, mode)


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
            cat  = _ticker_category(contract_id)
            edge = record.edge
            kelly_pct = (bet_dollars / balance * 100) if balance > 0 else 0
            book_entry = {
                "contract_id":  contract_id,
                "side":         side,
                "size":         record.size,
                "limit_price":  record.limit_price,
                "p_model":      p_model,
                "market_price": market_price,
                "edge":         edge,
            }
            _open_book[contract_id] = book_entry
            _all_trades.append(book_entry)

            # Two-line entry: what was placed + why
            _print_log(
                f"[{_ts()}] OPEN {side:>3} {contract_id}  sz={record.size}  "
                f"@ {record.limit_price}¢  [{cat}]"
            )
            _print_log(
                f"          p_model={p_model:.3f}  mkt={market_price:.3f}  "
                f"edge={edge:.3f}  conf={confidence:.3f}  "
                f"kelly=${bet_dollars:.2f} ({kelly_pct:.1f}%)"
            )

    resolution_events = order_manager.check_resolutions()
    _sync_closed_positions(resolution_events, order_manager.mode)

    if n_placed == 0:
        _print_log(
            f"[{_ts()}] poll — no new trades  "
            f"(open={len(order_manager.open_positions)}  closed={_wins + _losses})"
        )
    else:
        _print_log(f"[{_ts()}] poll — {n_placed} new order(s) placed")

    return n_placed


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


def _restore_open_book(order_manager: OrderManager) -> None:
    """On startup, rebuild _open_book and _all_trades from the CSV logs."""
    trade_log = LIVE_LOG if order_manager.mode == "live" else DRY_RUN_LOG
    if not trade_log.exists():
        return

    resolved: set[str] = set()
    if RESOLVED_LOG.exists():
        with open(RESOLVED_LOG) as f:
            for row in csv.DictReader(f):
                if row.get("contract_id"):
                    resolved.add(row["contract_id"])

    seen: set[str] = set()
    with open(trade_log) as f:
        for row in csv.DictReader(f):
            cid = row.get("contract_id", "")
            if not cid:
                continue
            try:
                entry = {
                    "contract_id":  cid,
                    "side":         row["side"],
                    "size":         int(row["size"]),
                    "limit_price":  int(row["limit_price"]),
                    "p_model":      float(row["p_model"]),
                    "market_price": float(row["market_price"]),
                    "edge":         float(row["edge"]),
                }
            except (KeyError, ValueError):
                continue
            # _all_trades gets every unique entry
            if cid not in seen:
                seen.add(cid)
                _all_trades.append(entry)
            # _open_book only gets unresolved ones
            if cid not in resolved:
                _open_book[cid] = entry

    # Also restore _wins/_losses from resolved log
    global _wins, _losses
    if RESOLVED_LOG.exists():
        with open(RESOLVED_LOG) as f:
            for row in csv.DictReader(f):
                pnl = float(row.get("pnl_dollars", 0))
                _realized.append(pnl)
                won = row.get("won", "").lower() == "true"
                if won:
                    _wins += 1
                else:
                    _losses += 1

    if _open_book or _wins or _losses:
        _print_log(
            f"[{_ts()}] restored {len(_open_book)} open, "
            f"{_wins}W/{_losses}L from logs"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")

    mode = TRADING_CFG["mode"]
    poll = PIPELINE_CFG["poll_interval_sec"]

    _init_display()
    _print_log(f"[{_ts()}] Trader starting — {mode.upper()} mode, poll every {poll}s")

    order_manager = OrderManager()
    _restore_open_book(order_manager)
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
