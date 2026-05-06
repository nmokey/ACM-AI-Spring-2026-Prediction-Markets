# ACM AI — Prediction Markets Trading Bot Spring 2026 · UCLA

**Thesis:** Prediction market prices are inefficient in predictable ways. A model that fuses structured time-series price signals with unstructured NLP sentiment signals can produce better-calibrated probability estimates than the crowd — and trade on that edge automatically.

We build an end-to-end quantitative trading bot targeting Kalshi prediction markets, focusing on weather, crypto, and sports contracts that resolve multiple times per day.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/nmokey/ACM-AI-Spring-2026-Prediction-Markets.git
cd ACM-AI-Spring-2026-Prediction-Markets

# 2. Set up your environment (requires Python 3.11+)
pip install uv      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync             # installs all dependencies from pyproject.toml in ~30s

# macOS only — required for XGBoost
brew install libomp

# 3. Set up your API keys
cp .env.example .env
# Edit .env and fill in KALSHI_API_KEY, KALSHI_API_SECRET, GNEWS_API_KEY

# 4. Smoke test your API connections
uv run python -m data.ingestion.kalshi_client   # prints 5 open Kalshi markets
uv run python -m data.ingestion.weather_client  # prints today's precip probabilities
uv run python -m data.ingestion.crypto_client   # prints BTC/ETH price changes
uv run python -m nlp.news_client                # prints 5 recent headlines

# 5. Smoke test the execution pipeline (no API keys needed)
uv run python scripts/test_execution.py         # places 4 dummy dry-run orders
```

## Architecture

```
┌─────────────────────────────────────────┐
│               Club Server               │
│                                         │
│  ┌──────────┐    ┌─────────────┐        │
│  │ Kalshi   │───▶│   Team 1    │        │
│  │ NOAA     │    │  Data &     │        │
│  │ Coinbase │    │  Features   │        │
│  └──────────┘    └──────┬──────┘        │
│                         │               │
│  ┌──────────┐           │               │
│  │ GNews    │───────────┤               │
│  │ GDELT    │           │               │
│  └──────────┘           ▼               │
│                live_features.parquet    │
│                         │               │
│                         ▼               │
│               ┌──────────────────┐      │
│               │     Team 2       │      │
│               │  Modeling &      │      │
│               │  Intelligence    │      │
│               │  ┌────────────┐  │      │
│               │  │FinBERT/    │  │      │
│               │  │VADER (NLP) │  │      │
│               │  └─────┬──────┘  │      │
│               │        ▼(internal)      │
│               │  ┌────────────┐  │      │
│               │  │  XGBoost   │  │      │
│               │  │ +isotonic  │  │      │
│               │  └────────────┘  │      │
│               └────────┬─────────┘      │
│                        │                │
│                        ▼                │
│               predictions.json          │
│                        │                │
│                        ▼                │
│        ┌─────────────────────────────┐  │
│        │      Team 3 Execution       │  │
│        │  Kelly sizing → OrderMgr    │  │
│        └─────────────┬───────────────┘  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
              Kalshi API (dry_run or live)
```

## Subteams & Repo Structure

Each team owns a top-level folder. Do not edit another team's folder without a PR.
The only shared write zone is `signals/` (predictions JSON).

Status legend: ✅ done · 🚧 in progress · ⬜ not started

```
prediction-markets/
│
├── data/                   🗄️ TEAM 1 — Data & Features
│   ├── ingestion/
│   │   ├── kalshi_client.py  ✅  Kalshi REST API — RSA-PSS auth, get_markets, get_market,
│   │   │                         get_orderbook, get_resolved_markets, backfill_all_resolved
│   │   ├── weather_client.py ✅  NOAA — get_forecast, get_todays_precip_prob (NY/LA/Chicago)
│   │   └── crypto_client.py  ✅  Coinbase Advanced — get_price, get_24h_stats,
│   │                             get_klines, compute_price_changes (BTC-USD, ETH-USD)
│   ├── features/
│   │   ├── schema.py         ✅  ⭐ SHARED — Pydantic data contracts (do not modify w/o PR)
│   │   └── engineer.py       ✅  Feature engineering pipeline → live_features.parquet
│   │                             (17 columns: Kalshi + Coinbase + NOAA, refreshed every 15 min)
│   └── store/                     (gitignored) SQLite DB, raw parquet files
│
├── nlp/                    🧠 TEAM 2 — Modeling & Intelligence (NLP half)
│   ├── news_client.py        ✅  GNews + GDELT fallback, SQLite store, _extract_query
│   ├── relevance.py          ✅  Cosine similarity scorer (all-MiniLM-L6-v2)
│   └── sentiment.py          ✅  FinBERT / VADER sentiment scoring, build_sentiment_signals,
│                                 save_sentiment_signals → nlp/sentiment.json
│
├── models/                 🧠 TEAM 2 — Modeling & Intelligence (Modeling half)
│   ├── train.py              ✅  XGBoost + isotonic calibration, GroupShuffleSplit,
│   │                             class-imbalance weighting, saves xgb_v1.joblib
│   ├── predict.py            ✅  Live inference → signals/predictions.json
│   ├── evaluate.py           ✅  Brier score, log-loss, calibration curve, feature importance
│   └── trained/                   (gitignored) serialized model weights
│
├── execution/              ⚡ TEAM 3 — Execution
│   ├── kelly.py              ✅  Fractional Kelly Criterion position sizing
│   ├── risk.py               ✅  Pre-trade risk checks (edge, confidence, exposure)
│   ├── order_manager.py      ✅  Order submission — the only gateway to Kalshi orders
│   │                             (account_balance, submit_order, dry_run / live routing)
│   ├── dry_run.py            ✅  Mock order logger → logs/dry_run_trades.csv
│   └── trader.py             ✅  Main trading loop (reads predictions → places orders)
│
├── signals/                🔗 SHARED (read/write by Teams 2 & 3)
│   └── predictions.json           Team 2 writes → Team 3 reads
│
├── backtest/               📊 SHARED
│   ├── engine.py             🚧  Dummy mode working (from predictions.json + simulated outcomes)
│   │                             Real model backtest (historical features + model) — Week 6
│   └── metrics.py            ✅  Sharpe, Sortino, win rate, max drawdown, go/no-go verdict
│
├── notebooks/              📓 SHARED (visualization only — not production code)
│   ├── eda.ipynb                  Exploratory analysis of features
│   ├── model_eval.ipynb           Calibration curve, Brier score comparison
│   └── backtrack_results.ipynb    Backtest P&L, trade log analysis
│
├── config/
│   └── settings.yaml              ⚙️ Central config — trading mode, risk params, paths
│
├── scripts/
│   ├── run_pipeline.sh            Starts data + NLP loop on club server
│   ├── run_bot.sh                 Starts trader.py (prompts confirmation in live mode)
│   └── test_execution.py          Smoke test for Team 3 pipeline (no API keys needed)
│
├── logs/                          (gitignored) trade logs
├── .env.example                   API key template — copy to .env
└── pyproject.toml                 uv dependency manifest
```

## Running the Pipeline

### Step 1 — Refresh live features (Team 1)
```bash
uv run python -m data.engineer
# Writes data/features/live_features.parquet — 200 open contracts, 17 feature columns
```

### Step 2 — Fetch headlines and score sentiment (Team 2 NLP)
```bash
uv run python -m nlp.sentiment
# Reads live_features.parquet for contract titles, queries GNews/GDELT,
# scores with VADER or FinBERT, writes nlp/sentiment.json
```

### Step 3 — Train or retrain the model (Team 2 Modeling)
```bash
uv run python -m models.train
# Reads data/features/historical_features.parquet + nlp/sentiment.json
# Trains XGBoost + isotonic calibration, writes models/trained/xgb_v1.joblib
```

### Step 4 — Run live inference (Team 2 Modeling)
```bash
uv run python -m models.predict
# Reads live_features.parquet + sentiment.json + xgb_v1.joblib
# Writes signals/predictions.json
```

### Step 5 — Start the trading bot (Team 3)
```bash
bash scripts/run_bot.sh
# Reads predictions.json every 60 seconds, applies risk checks + Kelly sizing,
# logs orders to logs/dry_run_trades.csv (or submits live when mode: "live")
```

For overnight runs on the club server:
```bash
nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
nohup bash scripts/run_bot.sh > logs/bot.log 2>&1 &
```

## Data Contracts

These are the interfaces **between** teams. Defined as Pydantic models in `data/features/schema.py`. Do not change that file without a team-wide PR.

> **Note on NLP signals:** `nlp/` and `models/` are both owned by Team 2 (Modeling & Intelligence). Sentiment scores are an **internal Team 2 artifact** — they flow directly from `nlp/sentiment.py` into `models/predict.py` at runtime and are never written as a cross-team file. The only output Team 2 exposes externally is `signals/predictions.json`.

### `MarketFeatures` — Team 1 → Team 2 (`live_features.parquet`, refreshed every 15 min)

| Field | Type | Description |
|---|---|---|
| contract_id | `str` | Kalshi market ticker |
| title | `str` | Human-readable contract title |
| market_category | `str` or null | Contract category — null for MVE parlay contracts |
| market_price | `float` [0, 1] | Mid of yes_ask/yes_bid in dollars; null if illiquid |
| volume_24h | `float` | Contracts traded in last 24h |
| open_interest | `float` | Total open contracts |
| days_to_resolution | `float` | Days until expected expiration |
| btc_price | `float` | Current BTC/USD spot price |
| btc_change_1h | `float` | BTC 1h price change (e.g. 0.012 = +1.2%) |
| btc_change_6h | `float` | BTC 6h price change |
| eth_price | `float` | Current ETH/USD spot price |
| eth_change_1h | `float` | ETH 1h price change |
| eth_change_6h | `float` | ETH 6h price change |
| precip_prob_new_york | `float` | Today's max precipitation probability 0–100 |
| precip_prob_los_angeles | `float` | Today's max precipitation probability 0–100 |
| precip_prob_chicago | `float` | Today's max precipitation probability 0–100 |
| fetched_at | `str` | ISO 8601 UTC timestamp of snapshot |

### `PredictionSignal` — Team 2 → Team 3 (`signals/predictions.json`, refreshed every 15 min)

| Field | Type | Constraints | Description |
|---|---|---|---|
| contract_id | `str` | | Kalshi market ticker |
| timestamp | `datetime` (UTC) | | Inference time |
| p_model | `float` | [0.0, 1.0] | Calibrated probability of YES outcome |
| confidence | `float` | [0.0, 1.0] | Model certainty — low confidence → skip trade |

### `SentimentSignal` — Internal Team 2 (`nlp/sentiment.json`)

| Field | Type | Constraints | Description |
|---|---|---|---|
| contract_id | `str` | | Kalshi market ticker |
| timestamp | `datetime` (UTC) | | Scoring time |
| sentiment_score | `float` | [−1.0, 1.0] | Positive = bullish, negative = bearish |
| sentiment_confidence | `float` | [0.0, 1.0] | Confidence in the sentiment score |
| n_relevant_headlines | `int` | ≥ 0 | Number of headlines used |

### `TradeRecord` — Internal Team 3 (`logs/dry_run_trades.csv` / live order log)

| Field | Type | Constraints | Description |
|---|---|---|---|
| contract_id | `str` | | Kalshi market ticker |
| timestamp | `datetime` (UTC) | | Order submission time |
| side | `str` | `"YES"` / `"NO"` | Direction of the trade |
| size | `int` | ≥ 0 | Number of contracts |
| limit_price | `int` | [0, 100] | Price in Kalshi cents |
| p_model | `float` | [0.0, 1.0] | Model probability at time of trade |
| market_price | `float` | [0.0, 1.0] | Market YES price at time of trade |
| edge | `float` | | `\|p_model − market_price\|` |
| mode | `str` | `"dry_run"` / `"live"` | Trading mode |

## Key Metrics & Targets

| Metric | Target | Notes |
|---|---|---|
| Brier Score | < 0.20 | Primary model quality metric. Random = 0.25 |
| Sharpe Ratio | > 1.0 | Risk-adjusted return on backtest |
| Win Rate | > 52% | % of trades that close profitably |
| Edge per Trade | > 0.05 | Avg \|p_model − market_price\| on winning trades |
| Dry-Run Trades | > 50 | Proof the system is running autonomously |

**Week 6 go/no-go gate:** Sharpe > 1.0 AND win rate > 52% → flip to `mode: "live"` in `config/settings.yaml`.

## Trading Config (`config/settings.yaml`)

```yaml
trading:
  mode: "dry_run"          # "dry_run" | "live"
  min_edge: 0.06           # min |p_model - market_price| to trade
  kelly_fraction: 0.25     # quarter Kelly — never use full Kelly
  max_position_pct: 0.05   # max 5% of account per trade ($5 on $100)
  max_total_exposure_pct: 0.40
  min_confidence: 0.60
```

## Known Data Limitations

These are active issues that affect model quality. Understanding them is important before interpreting any backtest results or Brier scores.

### 1. Training data is all MVE parlay contracts

Kalshi's `GET /markets?status=settled` only returns MVE (multi-variate event) parlay contracts — same-day sports parlays that resolve quickly. The 3,931 rows in `historical_features.parquet` are all of this type.

**Why this matters:** These are exotic multi-leg bets that are structurally different from the single-event weather, crypto, and sports markets we actually want to trade. A model trained on parlays won't generalize.

**Mitigation:** Run `data.engineer` on a schedule to snapshot live features. When open contracts resolve, match them to their outcomes. After a week or two of collection, retrain on that data instead.

### 2. Severe class imbalance in historical data

Of 3,931 resolved contracts: 3,611 resolved NO (92%), 320 resolved YES (8%). This is expected — multi-leg parlays rarely hit all legs. `train.py` compensates with `scale_pos_weight=11.3`, but the model still has limited positive examples to learn from.

### 3. `market_price` is zero for most historical rows

75%+ of rows have `market_price = 0.0` because parlay contracts typically had no orderbook activity before resolving. This means the single most predictive feature is degenerate in the training set. The current trained model learns essentially: *zero market price → likely resolves NO*.

**Brier score is misleadingly good (0.064)** because predicting NO on 92% NO data scores well even without real predictive signal. Check feature importances — `market_price` carries 100% of the weight, all others are zero.

### 4. Kalshi does not return `market_category` for MVE contracts

`market_category` is null for all 200 live contracts and all 3,931 historical contracts. Category-based filtering or features are not currently usable.

### 5. Crypto features are null in historical data

`btc_change_1h/6h` and `eth_change_1h/6h` are all null in the historical parquet (resolved markets don't have live Coinbase data attached). `train.py` fills these with `0.0`. They contribute no signal to the current model.

## External APIs

| API | Owned by | Auth | Docs |
|---|---|---|---|
| Kalshi REST | Team 1, Team 3 | RSA-PSS key pair in `.env` | trading-api.kalshi.com/docs |
| NOAA Weather | Team 1 | None (free) | weather.gov/documentation |
| Coinbase Advanced | Team 1 | API key pair in `.env` (public endpoints used for candles) | docs.cloud.coinbase.com/advanced-trade-api |
| GNews | Team 2 | API key in `.env` | gnews.io — request free academic access from UCLA email |
| GDELT | Team 2 | None (free fallback) | gdeltproject.org |

## Project Links

- 📋 [Notion SSOT](https://www.notion.so/33640be8288a808ca693c13986e2a526) — week-by-week tasks, team specs, symposium info
- 📊 [Kalshi account](https://kalshi.com) — funded with $100 real capital for Week 7 live trading
