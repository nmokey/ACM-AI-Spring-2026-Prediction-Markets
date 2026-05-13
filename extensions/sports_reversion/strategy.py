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

import pandas as pd


def compute_signals(
    elo_df: pd.DataFrame,
    favorite_threshold: float = 0.70,
    public_bias: float = 0.05,
    bet_direction: str = "fade",
) -> pd.DataFrame:
    """
    Generate signals by comparing Elo true probability vs. inflated market price.

    Note: market_price here is *simulated* as elo_prob1 + public_bias. We don't have
    historical Kalshi NBA prices, so this is a sensitivity study over the public_bias
    parameter rather than a clean backtest. A positive Sharpe for "fade" is largely
    baked in when public_bias > 0 and Elo is well-calibrated at the favorite tail.

    Args:
        elo_df:             FTE Elo DataFrame. Required columns:
                              date, team1, team2, elo_prob1, score1, score2.
        favorite_threshold: only trade when market_price >= this value.
        public_bias:        market over-pricing assumed for favorites
                            (market_price = elo_prob1 + public_bias).
        bet_direction:      "fade" = trust Elo, bet NO on over-priced favorites.
                            "follow" = bet YES on the same favorites.

    Returns:
        DataFrame with columns: date, team1, team2, p_model, market_price, resolved_yes.
    """
    if bet_direction not in {"fade", "follow"}:
        raise ValueError(f"bet_direction must be 'fade' or 'follow', got {bet_direction!r}")

    df = elo_df.copy()

    if "resolved_yes" not in df.columns:
        df = df.dropna(subset=["score1", "score2"])
        df["resolved_yes"] = (df["score1"] > df["score2"]).astype(int)

    df["market_price"] = (df["elo_prob1"] + public_bias).clip(0.05, 0.95)
    df = df[df["market_price"] >= favorite_threshold].copy()

    if bet_direction == "fade":
        df["p_model"] = df["elo_prob1"]
    else:
        # Mirror "fade" across market_price so kelly_fraction returns side="YES"
        df["p_model"] = (df["market_price"] + public_bias).clip(0.05, 0.95)

    return df[["date", "team1", "team2", "p_model", "market_price", "resolved_yes"]].reset_index(drop=True)


def _load_fte_nba_elo() -> pd.DataFrame:
    """
    Fetch the FiveThirtyEight NBA Elo archive and normalize it to the schema
    compute_signals expects (date, team1, team2, elo_prob1, score1, score2).

    The original projects.fivethirtyeight.com URL from the README went offline
    when FTE was shut down in 2025. The team's data repo on GitHub still hosts
    nbaallelo.csv (1946–2015, ~63k games after dedup). Each game appears twice
    in the source — once per team perspective — so we keep the home-team row
    and rename columns to match the README's interface.
    """
    import io
    import requests

    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text))
    home = raw[(raw["_iscopy"] == 0) & (raw["game_location"] == "H")].copy()

    return pd.DataFrame({
        "date":         pd.to_datetime(home["date_game"]),
        "team1":        home["team_id"].values,
        "team2":        home["opp_id"].values,
        "elo_prob1":    home["forecast"].values,
        "score1":       home["pts"].values,
        "score2":       home["opp_pts"].values,
        "is_playoffs":  home["is_playoffs"].astype(int).values,
    })


if __name__ == "__main__":
    elo = _load_fte_nba_elo()
    elo = elo[elo["date"] >= "2010-01-01"]  # ~5 most recent seasons in the archive

    signals = compute_signals(elo, favorite_threshold=0.70, public_bias=0.05, bet_direction="fade")

    print(f"games loaded     : {len(elo):,}")
    print(f"signals returned : {len(signals):,}")
    print(f"home win rate    : {signals['resolved_yes'].mean():.3f}")
    print(f"avg market_price : {signals['market_price'].mean():.3f}")
    print(f"avg p_model      : {signals['p_model'].mean():.3f}")
    print(f"avg edge         : {(signals['market_price'] - signals['p_model']).mean():+.3f}")
