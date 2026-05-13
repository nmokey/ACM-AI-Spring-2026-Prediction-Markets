# Extensions Spec — Prediction Markets Strategies

These are self-contained backtesting projects you can run locally with no API keys and no server access. Each one tests a distinct trading hypothesis on Kalshi-style binary contracts. Pick one, implement the signal logic, run the backtest, and see if the strategy has a positive edge.

You do **not** need to integrate with the live bot. The goal is to implement a signal, wire it through the shared backtest infrastructure, and measure its performance (Sharpe, win rate, max drawdown).

---

## How This Fits into the Codebase

The main bot pipeline (data → NLP → XGBoost → Kelly → execution) lives in `data/`, `nlp/`, `models/`, and `execution/`. Extensions live in `extensions/` and are **completely independent** — they share only two utilities from the main codebase:

- **`execution/kelly.kelly_fraction()`** — fractional Kelly position sizing (same formula the live bot uses)
- **`backtest/metrics.compute_metrics()` and `print_metrics()`** — standardized performance report

Your extension should produce a `signals` DataFrame with these three columns, then run it through the standard backtest loop below:

| Column | Type | Description |
|--------|------|-------------|
| `p_model` | float [0, 1] | Your strategy's estimated YES probability |
| `market_price` | float [0, 1] | The market's implied YES probability (simulated or historical) |
| `resolved_yes` | int 0 or 1 | Ground truth: did the contract resolve YES? |

### Standard backtest loop

Copy this into your notebook/script and fill in your `signals` DataFrame:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))  # repo root

import pandas as pd
from execution.kelly import kelly_fraction, dollars_to_contracts
from backtest.metrics import compute_metrics, print_metrics

# ── produce your signals here ─────────────────────────────────────────────────
# signals = compute_signals(...)   ← your function, defined below
# ─────────────────────────────────────────────────────────────────────────────

STARTING_BALANCE = 1000.0
balance = STARTING_BALANCE
cumulative_pnl = 0.0
trades = []

for _, row in signals.iterrows():
    bet_dollars, side = kelly_fraction(
        p_model=row["p_model"],
        market_price=row["market_price"],
        bankroll=balance,
        kelly_multiplier=0.25,
        max_position_pct=0.05,
    )
    if bet_dollars <= 0:
        continue

    price = row["market_price"] if side == "YES" else (1 - row["market_price"])
    n_contracts = dollars_to_contracts(bet_dollars, price)
    if n_contracts == 0:
        continue

    cost = n_contracts * price
    won = (side == "YES" and row["resolved_yes"] == 1) or \
          (side == "NO" and row["resolved_yes"] == 0)
    pnl = n_contracts * (1 - price) if won else -cost
    balance += pnl
    cumulative_pnl += pnl

    trades.append({
        "p_model": row["p_model"],
        "market_price": row["market_price"],
        "side": side,
        "n_contracts": n_contracts,
        "cost": round(cost, 4),
        "edge": round(abs(row["p_model"] - row["market_price"]), 4),
        "resolved_yes": row["resolved_yes"],
        "won": won,
        "pnl": round(pnl, 4),
        "balance": round(balance, 2),
        "cumulative_pnl": round(cumulative_pnl, 4),
    })

trades_df = pd.DataFrame(trades)
print_metrics(trades_df, starting_balance=STARTING_BALANCE)
```

**Go/no-go thresholds** (same as the main bot):
- Sharpe ratio > 1.0
- Win rate > 52%
- Max drawdown > −30%

---

## Extension 1: Crypto Momentum

**Location:** `extensions/momentum/`

### Hypothesis

Crypto prediction markets overreact to short-term price momentum. When BTC has been trending up sharply over the last few hours, the market already prices "BTC up today" contracts too high — and vice versa. A contrarian (mean-reversion) signal that fades extreme moves should find consistent edge, especially near daily open/close boundaries.

Alternatively, try the momentum direction: buy YES when BTC is trending up, buy NO when it's trending down. Compare which variant has better Sharpe on your data.

### Data

Use `ccxt` to pull free hourly OHLCV data from Binance (no API key needed for public market data):

```python
pip install ccxt

import ccxt
import pandas as pd

exchange = ccxt.binance()
# Fetch 90 days of hourly candles (limit=2160 = 90*24)
raw = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=2160)
df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df = df.set_index('timestamp')
```

Or use `yfinance` if you prefer:

```python
pip install yfinance

import yfinance as yf
df = yf.download('BTC-USD', period='90d', interval='1h')
```

### What to implement

**File:** `extensions/momentum/strategy.py`

```python
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

def compute_signals(
    df: pd.DataFrame,
    lookback_hours: int = 4,
    threshold: float = 0.0,
    p_scale: float = 3.0,
) -> pd.DataFrame:
    """
    Generate YES/NO probability signal from BTC price momentum.

    Args:
        df:             OHLCV DataFrame with DatetimeIndex and 'close' column
        lookback_hours: rolling window for momentum calculation
        threshold:      minimum |momentum| to generate a signal (filter noise)
        p_scale:        controls how aggressively z-scored momentum maps to probability
                        (higher = more extreme p_model values)

    Returns:
        DataFrame with columns: p_model, market_price, resolved_yes

    Implementation hints:
        1. Compute pct_change over lookback_hours: df['close'].pct_change(lookback_hours)
        2. Z-score it: (x - rolling_mean) / rolling_std over e.g. 48h window
        3. Convert to p_model: expit(z_score * p_scale)
           - Momentum strategy: high positive z → high p_model (predict YES)
           - Mean-reversion strategy: high positive z → low p_model (predict YES = fade the rally)
        4. Simulate Kalshi contract: resolved_yes = 1 if close > open that day
        5. Market price: use a fixed 0.50 baseline, or add small random noise ±0.05
           to simulate a noisy market. Keep it simple.
        6. Filter rows where |z_score| < threshold (no edge) before returning.
    """
    raise NotImplementedError("Implement me!")
```

### Parameters to tune

| Parameter | Default | Try |
|-----------|---------|-----|
| `lookback_hours` | 4 | 1, 2, 4, 8, 24 |
| `threshold` | 0.0 | 0.5, 1.0, 1.5 (z-score cutoff) |
| `p_scale` | 3.0 | 1.0, 2.0, 5.0 |
| Kelly `kelly_multiplier` | 0.25 | 0.10, 0.25, 0.50 |

### Stretch goals

- Try multiple assets (ETH, SOL) and combine signals
- Add a time-of-day filter (momentum near daily close may be more predictive)
- Compare momentum vs. mean-reversion on the same data

---

## Extension 2: Weather Arbitrage

**Location:** `extensions/weather_arb/`

### Hypothesis

NOAA weather forecast models (GFS, NAM) update every 6 hours with refined predictions. Kalshi weather contract prices are set by human traders who may not be watching the latest model output. A strategy that uses rolling historical temperature distributions as a simple climatology model should produce better-calibrated probabilities than a naive 50/50 baseline, especially for markets near the temperature threshold.

### Data

Download historical daily high/low temperature records from NOAA's Climate Data Online (CDO). This is free, no account needed for bulk CSV downloads.

Steps:
1. Go to [ncei.noaa.gov/cdo-web](https://www.ncei.noaa.gov/cdo-web/)
2. Click "Data Tools" → "Climate Data Online"
3. Search for a weather station near LA (e.g. "Los Angeles International Airport, CA"), NY (Central Park), or Chicago (O'Hare)
4. Select "Daily Summaries", date range 2022-01-01 to 2025-01-01
5. Download as CSV — you want the `TMAX` column (daily high in tenths of °C)

Or use NOAA's API directly (free, no key needed for public data):

```python
import requests, pandas as pd

# Get station metadata for LAX
r = requests.get("https://www.ncei.noaa.gov/cdo-web/api/v2/data", params={
    "datasetid": "GHCND",
    "stationid": "GHCND:USW00023174",  # LAX
    "datatypeid": "TMAX",
    "startdate": "2023-01-01",
    "enddate": "2024-12-31",
    "limit": 1000,
    "units": "standard",  # Fahrenheit
}, headers={"token": "your_token_here"})
# Note: CDO API requires a free token — register at ncei.noaa.gov/cdo-web/token
# Alternatively, download the CSV manually from the website above (no token needed)
```

**Easier path:** Just download the CSV file from the website. It takes 30 seconds and requires no code.

```python
# After downloading "3229895.csv" from NOAA CDO:
df = pd.read_csv("3229895.csv", parse_dates=["DATE"])
df = df[["DATE", "TMAX"]].dropna()
df["TMAX"] = df["TMAX"]  # already in °F if you selected "standard" units
```

### What to implement

**File:** `extensions/weather_arb/strategy.py`

```python
import numpy as np
import pandas as pd

def compute_signals(
    historical_temps: pd.DataFrame,
    threshold_f: float = 85.0,
    lookback_days: int = 30,
    market_noise: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate p(daily high > threshold_f) using a rolling historical window.

    Args:
        historical_temps: DataFrame with columns 'DATE' (datetime) and 'TMAX' (°F)
        threshold_f:      temperature threshold (e.g. 85°F for LA in summer)
        lookback_days:    rolling window to estimate climatology probability
        market_noise:     std of Gaussian noise added to market_price (simulates
                          market not tracking forecast perfectly)

    Returns:
        DataFrame with columns: date, p_model, market_price, resolved_yes

    Implementation hints:
        1. Sort by DATE. For each day i, look at the last lookback_days days.
        2. p_model = (number of days in window where TMAX > threshold_f) / lookback_days
           This is a rolling empirical probability — your "model."
        3. resolved_yes = 1 if that day's TMAX > threshold_f, else 0.
        4. market_price = 0.50 + random noise (mean 0, std=market_noise, clipped to [0.05, 0.95])
           This simulates a naive market. Better: use actual Kalshi prices from
           data/features/snapshots.parquet if you want a more realistic test.
        5. Return one row per day where abs(p_model - market_price) > 0.05 (min edge filter).
    """
    raise NotImplementedError("Implement me!")
```

### Parameters to tune

| Parameter | Default | Try |
|-----------|---------|-----|
| `threshold_f` | 85.0 | Try different thresholds for LA (80, 85, 90), NY (75, 80, 85), Chicago (70, 75) |
| `lookback_days` | 30 | 14, 30, 60, 90 |
| `market_noise` | 0.05 | 0.02, 0.05, 0.10 |
| Season filter | None | Test only summer months (Jun–Aug) vs. year-round |

### Stretch goals

- Use `data/ingestion/weather_client.py` to pull a live NOAA forecast and compare against the rolling climatology model
- Test all three cities and compare which one has the most consistent edge
- Replace the simulated market price with actual Kalshi prices from `data/features/snapshots.parquet`

---

## Extension 3: Sports Mean-Reversion

**Location:** `extensions/sports_reversion/`

### Hypothesis

Public prediction markets (like sports books) systematically over-price heavy favorites due to public money bias — casual bettors disproportionately back the obvious favorite, inflating the implied probability above the true probability. A strategy that bets NO on extreme favorites, guided by a simple Elo-based true win probability, should find consistent positive EV over many games.

On Kalshi, NBA "Will [Team] win tonight?" contracts often price heavy favorites at 70–85¢. If the Elo model says the true probability is 60–70¢, the NO side is underpriced.

### Data

Use FiveThirtyEight's historical NBA Elo dataset — free, no API needed, just a CSV download:

```python
import pandas as pd

# Direct URL to FTE's historical NBA Elo data (1977–2024, ~70k games)
url = "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
elo = pd.read_csv(url, parse_dates=["date"])

# Key columns:
# date          - game date
# team1         - home team abbreviation
# team2         - away team abbreviation
# elo1_pre      - home team Elo before this game
# elo2_pre      - away team Elo before this game
# elo_prob1     - home team win probability from Elo model (0–1)
# score1, score2 - final scores (NaN for future games)

# Filter to recent seasons only for faster backtesting:
elo = elo[elo["date"] >= "2018-01-01"].copy()
elo = elo.dropna(subset=["score1", "score2"])  # only completed games
elo["resolved_yes"] = (elo["score1"] > elo["score2"]).astype(int)  # home team won
```

### What to implement

**File:** `extensions/sports_reversion/strategy.py`

```python
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
        elo_df:             FTE Elo DataFrame (see data loading above)
        favorite_threshold: only consider games where market implies >= this win prob
                            (i.e. only bet when the market thinks someone is a big favorite)
        public_bias:        how much the market over-prices favorites relative to Elo
                            (market_price = elo_prob1 + public_bias for favorites)
        bet_direction:      "fade" = bet NO on heavy favorites (contrarian)
                            "follow" = bet YES on heavy favorites (momentum)

    Returns:
        DataFrame with columns: date, team1, team2, p_model, market_price, resolved_yes

    Implementation hints:
        1. market_price = elo_prob1 + public_bias (simulate public over-pricing favorites)
           Clip to [0.05, 0.95].
        2. Filter to rows where market_price >= favorite_threshold
           (we only trade when the market says someone is a heavy favorite).
        3. p_model: for "fade" direction, p_model = elo_prob1 (trust Elo over market)
           This means p_model < market_price → strategy bets NO.
        4. resolved_yes = 1 if home team won (score1 > score2)
        5. Edge = market_price - p_model (should be > 0 for "fade" direction by construction)
    """
    raise NotImplementedError("Implement me!")
```

### Parameters to tune

| Parameter | Default | Try |
|-----------|---------|-----|
| `favorite_threshold` | 0.70 | 0.65, 0.70, 0.75, 0.80 |
| `public_bias` | 0.05 | 0.03, 0.05, 0.08, 0.10 |
| `bet_direction` | `"fade"` | Try `"follow"` too — which has better Sharpe? |
| Season filter | All | Regular season only vs. playoffs only |
| Kelly `kelly_multiplier` | 0.25 | 0.10, 0.25 |

### Stretch goals

- Add per-team public bias estimation (some teams, like the Lakers, attract more public money)
- Compare NBA vs. NHL (FTE also publishes [NHL Elo](https://projects.fivethirtyeight.com/nba-model/))
- Try a simple power-ratings model instead of Elo (average point differential over last N games)

---

## Deliverables

For the meeting, aim to have:

1. **A working `compute_signals()` function** that returns the right DataFrame shape
2. **A backtest run** using the loop above, producing `print_metrics()` output
3. **A chart** — plot cumulative P&L over time (`trades_df["cumulative_pnl"].plot()`)
4. **One interesting finding** — e.g., "momentum works on 4h lookback but not 1h" or "weather arb only has edge in summer"

Share results in the meeting. If your strategy clears the go/no-go bar (Sharpe > 1.0, win rate > 52%), we can discuss whether it's worth integrating into the live pipeline.

---

## Setup

No extra server access or API keys needed. Install dependencies in your own virtual environment:

```bash
# From the repo root
uv sync  # installs base dependencies (pandas, numpy, requests, etc.)

# Per-extension extras (install only what you need):
pip install ccxt          # Extension 1: Crypto Momentum
pip install yfinance      # Extension 1: alternative data source
pip install nba_api       # Extension 3: optional live NBA data

# No extra installs needed for Extension 2 (weather) — requests is already a dependency
```

Run your backtest as a Jupyter notebook in `notebooks/` or as a standalone script:

```bash
# From the repo root, so imports resolve correctly:
python -m extensions.momentum.strategy        # if you add __main__
# or just open notebooks/ and run interactively
```
