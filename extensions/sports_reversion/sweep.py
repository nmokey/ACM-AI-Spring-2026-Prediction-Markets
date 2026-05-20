"""
extensions/sports_reversion/sweep.py
─────────────────────────────────────
Parameter sweep for the sports mean-reversion strategy.

Runs the standard backtest loop (see extensions/SPEC.md) across every
combination of:
  - favorite_threshold ∈ {0.65, 0.70, 0.75, 0.80}
  - public_bias        ∈ {0.03, 0.05, 0.08, 0.10}
  - bet_direction      ∈ {fade, follow}
  - season             ∈ {all, regular, playoffs}
  - kelly_multiplier   ∈ {0.10, 0.25}

= 192 backtests. Reports top/bottom configs by Sharpe, marginal effects
per parameter, and go/no-go pass count.

Caveats:
  - market_price is *simulated* (elo_prob1 + public_bias), not real Kalshi data
  - fade direction has structurally low win_rate (~25%); the project's
    win_rate > 52% gate is incompatible with this strategy by design
  - playoff slices are small (~80–400 trades); Sharpe estimates are noisy
  - Sharpe is annualized by √252 in compute_metrics; this strategy runs
    closer to 50–100 trades/year, so the reported Sharpe is inflated by
    roughly √(252/75) ≈ 1.8x relative to a per-trade-frequency-matched ratio

Usage: uv run python -m extensions.sports_reversion.sweep
"""

from __future__ import annotations

import itertools
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))  # repo root

import pandas as pd

from backtest.metrics import compute_metrics
from execution.kelly import dollars_to_contracts, kelly_fraction
from extensions.sports_reversion.strategy import _load_fte_nba_elo, compute_signals

STARTING_BALANCE = 1000.0

THRESHOLDS  = [0.70, 0.75, 0.80, 0.85]
BIASES      = [0.05, 0.08, 0.10, 0.12]
DIRECTIONS  = ["fade"]           # "follow" consistently loses, skip it
SEASONS     = ["playoffs"]       # playoffs >> regular season
KELLY_MULTS = [0.10, 0.15, 0.25]


def backtest(signals: pd.DataFrame, kelly_multiplier: float) -> dict:
    """Run the SPEC.md standard backtest loop on a signals DataFrame."""
    balance = STARTING_BALANCE
    cum = 0.0
    trades = []

    for _, row in signals.iterrows():
        if balance < STARTING_BALANCE * 0.50:
            break  # stop-loss: halt if balance drops below 50%
        bet_dollars, side = kelly_fraction(
            p_model=row["p_model"],
            market_price=row["market_price"],
            bankroll=balance,
            kelly_multiplier=kelly_multiplier,
            max_position_pct=0.05,
        )
        if bet_dollars <= 0:
            continue

        price = row["market_price"] if side == "YES" else (1 - row["market_price"])
        n = dollars_to_contracts(bet_dollars, price)
        if n == 0:
            continue

        cost = n * price
        won = (side == "YES" and row["resolved_yes"] == 1) or \
              (side == "NO"  and row["resolved_yes"] == 0)
        pnl = n * (1 - price) if won else -cost
        balance += pnl
        cum += pnl

        trades.append({
            "edge": abs(row["p_model"] - row["market_price"]),
            "won": won,
            "pnl": pnl,
            "cumulative_pnl": cum,
            "resolved_yes": row["resolved_yes"],
        })

    if not trades:
        return {"n_trades": 0}
    return compute_metrics(pd.DataFrame(trades), starting_balance=STARTING_BALANCE)


def main() -> None:
    print("loading FTE NBA Elo archive...")
    elo_full = _load_fte_nba_elo()
    elo_full = elo_full[elo_full["date"] >= "2010-01-01"]
    print(f"loaded {len(elo_full):,} games (2010+ slice of the 1946–2015 archive)\n")

    rows = []
    combos = list(itertools.product(THRESHOLDS, BIASES, DIRECTIONS, SEASONS, KELLY_MULTS))
    for thr, bias, direction, season, km in combos:
        df = elo_full
        if season == "regular":
            df = df[df["is_playoffs"] == 0]
        elif season == "playoffs":
            df = df[df["is_playoffs"] == 1]

        sig = compute_signals(df, favorite_threshold=thr, public_bias=bias, bet_direction=direction)
        m = backtest(sig, kelly_multiplier=km)
        rows.append({
            "thr": thr, "bias": bias, "dir": direction, "season": season, "km": km,
            "n":      m.get("n_trades", 0),
            "pnl":    m.get("total_pnl", 0.0),
            "roi%":   m.get("roi_pct", 0.0),
            "win%":   m.get("win_rate", 0.0),
            "sharpe": m.get("sharpe_ratio", 0.0),
            "mdd%":   m.get("max_drawdown_pct", 0.0),
        })

    res = pd.DataFrame(rows)
    print(f"ran {len(res)} backtests\n")

    print("──── TOP 10 by Sharpe ─────────────────────────────────────────────")
    print(res.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    print("\n──── BOTTOM 5 by Sharpe ───────────────────────────────────────────")
    print(res.sort_values("sharpe").head(5).to_string(index=False))

    print("\n──── Marginal mean Sharpe by parameter ────────────────────────────")
    for col in ["thr", "bias", "dir", "season", "km"]:
        grp = res.groupby(col, observed=True)["sharpe"].mean().round(3)
        print(f"\n  by {col}:")
        print(grp.to_string())

    gng = res[(res["sharpe"] > 1.0) & (res["win%"] > 0.52) & (res["mdd%"] > -30)]
    print(f"\n──── Go/no-go: {len(gng)} of {len(res)} configs pass (Sharpe>1, win>52%, mdd>-30%) ────")
    if not gng.empty:
        print(gng.sort_values("sharpe", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
