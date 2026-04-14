"""
backtest/metrics.py
─────────────────────
Trading performance metrics.

Used by backtest/engine.py and notebooks/backtest_results.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, starting_balance: float) -> dict[str, float]:
    """
    Compute all key trading metrics from a completed backtest.

    Args:
        trades:            DataFrame output from backtest/engine.py
        starting_balance:  starting bankroll

    Returns:
        Dict of metric_name → value
    """
    if trades.empty:
        return {"error": "No trades to evaluate"}

    pnl = trades["pnl"]
    total_pnl = pnl.sum()
    roi = total_pnl / starting_balance

    win_rate = trades["won"].mean()
    n_trades = len(trades)
    avg_edge = trades["edge"].abs().mean()

    # Sharpe ratio (annualized, assuming each trade is independent)
    # Use daily returns proxy: group by date if available, else per-trade
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)  # annualized
    else:
        sharpe = 0.0

    # Sortino: only penalizes downside volatility
    downside = pnl[pnl < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (pnl.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = float("inf")

    # Max drawdown
    cumulative = trades["cumulative_pnl"]
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown / starting_balance

    return {
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi * 100, 2),
        "win_rate": round(win_rate, 4),
        "avg_edge": round(avg_edge, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
    }


def print_metrics(trades: pd.DataFrame, starting_balance: float) -> None:
    """Pretty-print backtest metrics to stdout."""
    m = compute_metrics(trades, starting_balance)
    if "error" in m:
        print(f"  ⚠️  {m['error']}")
        return

    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"  Trades          : {m['n_trades']}")
    print(f"  Total P&L       : ${m['total_pnl']:+.2f}")
    print(f"  ROI             : {m['roi_pct']:+.2f}%")
    print(f"  Win Rate        : {m['win_rate']:.1%}  {'✅' if m['win_rate'] > 0.52 else '⚠️'}")
    print(f"  Avg Edge        : {m['avg_edge']:.4f}")
    print(f"  Sharpe Ratio    : {m['sharpe_ratio']:.3f}  {'✅' if m['sharpe_ratio'] > 1.0 else '⚠️'}")
    print(f"  Sortino Ratio   : {m['sortino_ratio']:.3f}")
    print(f"  Max Drawdown    : ${m['max_drawdown']:.2f} ({m['max_drawdown_pct']:.1f}%)")
    print(f"{'='*50}\n")

    go = m["sharpe_ratio"] > 1.0 and m["win_rate"] > 0.52 and m["max_drawdown_pct"] > -30
    status = "✅ GO for live trading" if go else "❌ Stay in dry-run — review strategy"
    print(f"  Week 6 Decision: {status}\n")
