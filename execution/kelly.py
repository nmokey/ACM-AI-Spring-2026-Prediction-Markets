"""
execution/kelly.py
────────────────────
Fractional Kelly Criterion for position sizing.

The Kelly Criterion answers: given our edge, what fraction of our
bankroll should we bet to maximize long-run growth?

Full Kelly is too aggressive for real trading (too sensitive to model errors).
We use Fractional Kelly: f* × kelly_fraction (default 0.25 = "quarter Kelly").

Team 3 — Execution owns this file.

Reference: https://en.wikipedia.org/wiki/Kelly_criterion
"""

from __future__ import annotations

import math


def kelly_fraction(
    p_model: float,
    market_price: float,
    bankroll: float,
    kelly_multiplier: float = 0.25,
    max_position_pct: float = 0.05,
    min_position_dollars: float = 0.50,
) -> tuple[float, str]:
    """
    Compute the optimal fractional Kelly bet size.

    On Kalshi, a YES contract at price p pays $1 if it resolves YES
    and $0 if it resolves NO. So:
      - If we think p_model > market_price → buy YES (we think it's underpriced)
      - If we think p_model < market_price → buy NO (we think YES is overpriced)

    Kelly formula for a binary bet with payout b = (1 - p) / p (decimal odds):
        f* = (p * b - (1 - p)) / b  =  p - (1 - p) / b  =  (p_model - market_price) / (1 - market_price)

    Args:
        p_model:            Our model's probability estimate (0–1)
        market_price:       Current Kalshi YES price (0–1)
        bankroll:           Current account balance in dollars
        kelly_multiplier:   Fraction of full Kelly to use (0.25 = quarter Kelly)
        max_position_pct:   Hard cap as % of bankroll (e.g. 0.05 = $5 on $100)
        min_position_dollars: Don't trade if bet would be below this (transaction costs)

    Returns:
        (bet_dollars, side) where side is "YES" or "NO"
    """
    edge = p_model - market_price

    if edge >= 0:
        # We think YES is underpriced → buy YES
        side = "YES"
        p = p_model
        q = 1 - p
        # Kalshi YES pays (1 - market_price) / market_price times your stake
        if market_price <= 0 or market_price >= 1:
            return 0.0, side
        b = (1 - market_price) / market_price  # decimal odds
    else:
        # We think YES is overpriced → buy NO
        side = "NO"
        # For NO, we flip: our probability of NO winning is (1 - p_model)
        # NO price on Kalshi = 1 - market_price
        no_price = 1 - market_price
        p = 1 - p_model
        q = 1 - p
        if no_price <= 0 or no_price >= 1:
            return 0.0, side
        b = market_price / no_price  # payout for NO

    # Kelly formula
    if b <= 0:
        return 0.0, side

    f_star = (p * b - q) / b  # full Kelly fraction

    if f_star <= 0:
        # No positive edge
        return 0.0, side

    # Apply fractional Kelly and hard cap
    bet_fraction = f_star * kelly_multiplier
    bet_fraction = min(bet_fraction, max_position_pct)
    bet_dollars = bet_fraction * bankroll

    if bet_dollars < min_position_dollars:
        return 0.0, side

    return round(bet_dollars, 2), side


def dollars_to_contracts(bet_dollars: float, price: float) -> int:
    """
    Convert a dollar bet amount to a number of Kalshi contracts.

    On Kalshi, each contract costs `price` dollars (where price is 0–1).
    Rounds down to the nearest whole contract.

    Args:
        bet_dollars: amount to spend
        price:       contract price in dollars (0.0–1.0)

    Returns:
        Number of whole contracts to buy.
    """
    if price <= 0:
        return 0
    return max(0, math.floor(bet_dollars / price))
