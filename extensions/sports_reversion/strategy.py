"""
extensions/sports_reversion/strategy.py
─────────────────────────────────────────
Sports mean-reversion: fade heavy favorites using FiveThirtyEight Elo ratings.

See extensions/SPEC.md for full instructions and the backtest loop to run.

Quick start:
    import pandas as pd
    url = "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
    elo = pd.read_csv(url, parse_dates=["date"])
    elo = elo[elo["date"] >= "2018-01-01"].dropna(subset=["score1", "score2"])
    signals = compute_signals(elo)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_signals(
    elo_df: pd.DataFrame,
    favorite_threshold: float = 0.70,
    public_bias: float = 0.05,
    bet_direction: str = "fade",
) -> pd.DataFrame:
    """
    Generate signals by comparing Elo true probability vs. inflated market price.

    Args:
        elo_df:             FTE Elo DataFrame. Required columns:
                              date, team1, team2, elo_prob1, score1, score2.
        favorite_threshold: only trade when market_price >= this value.
        public_bias:        market over-pricing assumed for favorites
                            (market_price = elo_prob1 + public_bias).
        bet_direction:      "fade" = trust Elo, bet NO on over-priced favorites.
                            "follow" = bet YES (useful as a comparison baseline).

    Returns:
        DataFrame with columns: date, team1, team2, p_model, market_price, resolved_yes.

    Hints:
        1. market_price = (elo_df['elo_prob1'] + public_bias).clip(0.05, 0.95)
        2. Filter to rows where market_price >= favorite_threshold.
        3. p_model = elo_df['elo_prob1'] (Elo is our unbiased estimate).
        4. resolved_yes = 1 if score1 > score2 (home team won), else 0.
        5. For "fade": p_model < market_price, so kelly_fraction returns side="NO".
           For "follow": p_model > market_price is impossible here by construction
           (we added bias to market) — swap the sign or use a different framing.
    """
    raise NotImplementedError("Implement me! See extensions/SPEC.md for full hints.")
