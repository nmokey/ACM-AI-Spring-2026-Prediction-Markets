# Repo Structure & Team Ownership

## Ownership Rules

Each team owns a top-level folder. **Do not edit another team's folder without opening a PR.**

The only shared write zones are:
- `signals/` — Team 2 writes `predictions.json`, Team 3 reads it.
- `backtest/` — shared by all teams.
- `notebooks/` — shared visualization space, not production code.

`data/schema.py` is the cross-team interface spec. **Do not modify it without a team-wide PR.**

---

## Directory Map

```
prediction-markets/
│
├── data/                   🗄️  TEAM 1 — Data & Features
│   ├── schema.py               ⭐ SHARED — Pydantic data contracts (PR required to change)
│   ├── engineer.py             Feature engineering pipeline → live_features.parquet + snapshots.parquet
│   ├── ingestion/
│   │   ├── kalshi_client.py    Kalshi REST API — RSA-PSS auth, market/orderbook/backfill
│   │   ├── weather_client.py   NOAA — precipitation probabilities (NY, LA, Chicago)
│   │   └── crypto_client.py    Coinbase Advanced — BTC/ETH prices, klines, 1h/6h changes
│   ├── features/               (gitignored) parquet artifacts
│   │   ├── live_features.parquet   current snapshot, refreshed every 15 min
│   │   └── snapshots.parquet       accumulating history for training
│   └── store/                  (gitignored) SQLite headline store
│       └── headlines.db
│
├── nlp/                    🧠  TEAM 2 — Modeling & Intelligence (NLP half)
│   ├── news_client.py          GNews + GDELT fallback, SQLite store, _extract_query
│   ├── relevance.py            Cosine similarity scorer (all-MiniLM-L6-v2)
│   ├── sentiment.py            FinBERT / VADER scoring → nlp/sentiment.json (internal artifact)
│   └── sentiment.json          (gitignored) per-contract sentiment scores, refreshed every 30 min
│
├── models/                 🧠  TEAM 2 — Modeling & Intelligence (Modeling half)
│   ├── train.py                XGBoost + isotonic calibration, GroupShuffleSplit, --dedup flag
│   ├── predict.py              Live inference → signals/predictions.json
│   ├── evaluate.py             Brier score, log-loss, feature importance → logs/model_metrics.jsonl
│   └── trained/                (gitignored) serialized model weights
│       └── xgb_v1.joblib
│
├── execution/              ⚡  TEAM 3 — Execution
│   ├── kelly.py                Fractional Kelly Criterion position sizing
│   ├── risk.py                 Pre-trade risk checks (edge, confidence, exposure)
│   ├── order_manager.py        Order submission — the only gateway to Kalshi orders
│   │                           Tracks open positions; polls resolutions; simulates balance in dry_run
│   ├── dry_run.py              Mock order logger → logs/dry_run_trades.csv
│   └── trader.py               Main trading loop with live terminal dashboard
│
├── signals/                🔗  SHARED — Team 2 writes, Team 3 reads
│   └── predictions.json
│
├── backtest/               📊  SHARED
│   ├── engine.py               Backtest runner (dummy mode working; real model backtest in progress)
│   └── metrics.py              Sharpe, Sortino, win rate, max drawdown, go/no-go verdict
│
├── notebooks/              📓  SHARED (visualization only — not production code)
│   ├── performance.ipynb       Live P&L, win rate, brier score over time, feature importance drift
│   ├── backtest_results.ipynb  Backtest P&L, trade log analysis
│   └── backtest_weather.ipynb  Weather-specific backtest analysis
│
├── scripts/                🛠️  SHARED — operational scripts
│   ├── run_pipeline.sh         Starts the data + NLP loop on the club server
│   ├── run_bot.sh              Starts trader.py (prompts confirmation in live mode)
│   ├── label_resolved.py       Daily labeling job — stamps resolved_yes on settled contracts
│   ├── validate_features.py    Pipeline health check — prints PASS/FAIL per invariant
│   └── test_execution.py       Smoke test for Team 3 pipeline (no API keys needed)
│
├── tests/                  🧪  SHARED — test suite
│   ├── smoketest_crypto.py
│   ├── test_news_client.py
│   └── test_news_client_init_db.py
│
├── logs/                       (gitignored) runtime logs
│   ├── dry_run_trades.csv      Every order placed in dry_run mode
│   ├── resolved_trades.csv     Every closed position with result and P&L
│   ├── model_metrics.jsonl     One record per retrain: brier, log-loss, feature importances
│   └── pipeline.log / bot.log  stdout from tmux sessions
│
├── config/
│   └── settings.yaml           Central config — trading mode, risk params, all file paths
│
├── .env.example                API key template — copy to .env
└── pyproject.toml              uv dependency manifest
```

---

## Data Contracts

These are the interfaces **between** teams. Defined as Pydantic models in `data/schema.py`. Do not change that file without a team-wide PR.

> **Note on NLP signals:** `nlp/` and `models/` are both owned by Team 2. Sentiment scores are an **internal Team 2 artifact** — they flow from `nlp/sentiment.py` into `data/engineer.py` (joined at snapshot write time) and `models/predict.py` at inference time. The only output Team 2 exposes externally is `signals/predictions.json`.

### `MarketFeatures` — Team 1 → Team 2 (`data/features/live_features.parquet`, refreshed every 15 min)

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
| sentiment_score | `float` | From nlp/sentiment.json, joined at engineer write time |
| sentiment_confidence | `float` | From nlp/sentiment.json, joined at engineer write time |
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

### `resolved_trades.csv` — Internal Team 3 (`logs/resolved_trades.csv`)

One row written per position when Kalshi reports the contract as settled.

| Field | Description |
|---|---|
| timestamp | UTC time resolution was detected |
| contract_id | Kalshi market ticker |
| side | `"YES"` or `"NO"` |
| size | Number of contracts held |
| entry_price_cents | Price paid, in Kalshi cents |
| result | `"yes"` or `"no"` (Kalshi's reported outcome) |
| won | `True` if `side == result` |
| pnl_dollars | `size × (1 − entry_price/100)` if won, else `−size × (entry_price/100)` |
| mode | `"dry_run"` or `"live"` |
