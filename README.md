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

# 3. Set up your API keys
cp .env.example .env
# Edit .env and fill in KALSHI_API_KEY, KALSHI_API_SECRET, GNEWS_API_KEY

# 4. Smoke test your API connections
python -m data.ingestion.kalshi_client   # prints 5 open Kalshi markets
python -m data.ingestion.weather_client  # prints today's precip probabilities
python -m data.ingestion.crypto_client   # prints BTC/ETH price changes
python -m nlp.news_client                # prints 5 recent headlines
```

## Architecture

```
┌─────────────────────────────────────────┐
│               Club Server               │
│                                         │
│  ┌──────────┐    ┌─────────────┐        │
│  │ Kalshi   │───▶│   Team 1    │        │
│  │ NOAA     │    │  Data &     │        │
│  │ Binance  │    │  Features   │        │
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

```
prediction-markets/
│
├── data/                   🗄️ TEAM 1 — Data & Features
│   ├── ingestion/
│   │   ├── kalshi_client.py       Kalshi REST API wrapper + backfill
│   │   ├── weather_client.py      NOAA forecast fetcher
│   │   └── crypto_client.py       Binance public REST (BTC/ETH prices)
│   ├── features/
│   │   ├── schema.py              ⭐ SHARED — Pydantic data contracts (do not modify w/o PR)
│   │   └── engineer.py            Feature engineering pipeline → live_features.parquet
│   └── store/                     (gitignored) SQLite DB, raw parquet files
│
├── nlp/                    🧠 TEAM 2 — Modeling & Intelligence (NLP half)
│   ├── news_client.py             GNews + GDELT fallback headline fetcher
│   ├── relevance.py               Cosine similarity relevance scorer (all-MiniLM-L6-v2)
│   └── sentiment.py               FinBERT / VADER sentiment scoring (internal to Team 2)
│
├── models/                 🧠 TEAM 2 — Modeling & Intelligence (Modeling half)
│   ├── train.py                   XGBoost + isotonic calibration training
│   ├── predict.py                 Live inference → signals/predictions.json
│   ├── evaluate.py                Brier score, calibration curve, feature importance
│   └── trained/                   (gitignored) serialized model weights
│
├── execution/              ⚡ TEAM 3 — Execution
│   ├── kelly.py                   Fractional Kelly Criterion position sizing
│   ├── risk.py                    Pre-trade risk checks (edge, confidence, exposure)
│   ├── order_manager.py           Order submission — the only gateway to Kalshi orders
│   ├── dry_run.py                 Mock order logger → logs/dry_run_trades.csv
│   └── trader.py                  Main trading loop (reads predictions → places orders)
│
├── signals/                🔗 SHARED (read/write by Teams 2 & 3)
│   └── predictions.json           Team 2 writes → Team 3 reads
│
├── backtest/               📊 SHARED
│   ├── engine.py                  Simulates full pipeline on historical resolved contracts
│   └── metrics.py                 Sharpe, Sortino, win rate, max drawdown, Brier score
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
│   └── run_bot.sh                 Starts trader.py (prompts confirmation in live mode)
│
├── logs/                          (gitignored) trade logs
├── .env.example                   API key template — copy to .env
└── pyproject.toml                 uv dependency manifest
```

## Data Contracts

These are the interfaces **between** teams. Do not change `data/features/schema.py` without a team-wide PR — it is the plug-and-play contract that makes the pipeline modular.

> **Note on NLP signals:** `nlp/` and `models/` are both owned by Team 2 (Modeling & Intelligence). Sentiment scores are an **internal Team 2 artifact** — they flow directly from `nlp/sentiment.py` into `models/predict.py` at runtime and are never written as a cross-team file. The only output Team 2 exposes externally is `signals/predictions.json`.

**live_features.parquet — Team 1 → Team 2** *(refreshed every 15 min)*

| Field | Type | Description |
|---|---|---|
| contract_id | str | Kalshi market ticker e.g. `KXBTC-25APR14-T100000` |
| timestamp | datetime UTC | Snapshot time |
| market_price | float [0–1] | Normalized from Kalshi 0–100 cents |
| volume_24h | float | Contracts traded in last 24h |
| days_to_resolution | float | Time until market closes |
| price_change_1h | float | Price delta vs. 1h ago |
| price_change_6h | float | Price delta vs. 6h ago |
| market_category | str | `"weather"` / `"crypto"` / `"sports"` |

**signals/predictions.json — Team 2 → Team 3** *(refreshed every 15 min)*

| Field | Type | Description |
|---|---|---|
| contract_id | str | Kalshi market ticker |
| timestamp | datetime UTC | Inference time |
| p_model | float [0–1] | Calibrated probability of YES outcome |
| confidence | float [0, 1] | Model uncertainty — low confidence → skip trade |

## Key Metrics & Targets

| Metric | Target | Notes |
|---|---|---|
| Brier Score | < 0.20 | Primary model quality metric. Random = 0.25 |
| Sharpe Ratio | > 1.0 | Risk-adjusted return on backtest |
| Win Rate | > 52% | % of trades that close profitably |
| Edge per Trade | > 0.05 | Avg |p_model − market_price| on winning trades |
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

## Running the Pipeline

```bash
# Start data + NLP refresh loop (runs on club server)
bash scripts/run_pipeline.sh

# Start model inference (run separately, or integrate into pipeline)
python -m models.predict

# Start the trading bot
bash scripts/run_bot.sh
```

For overnight runs:
```bash
nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
nohup bash scripts/run_bot.sh > logs/bot.log 2>&1 &
```

## External APIs

| API | Owned by | Auth | Docs |
|---|---|---|---|
| Kalshi REST | Team 1, Team 3 | API key + secret in `.env` | trading-api.kalshi.com/docs |
| NOAA Weather | Team 1 | None (free) | weather.gov/documentation |
| Binance Public REST | Team 1 | None (free) | binance-docs.github.io |
| GNews | Team 2 | API key in `.env` | gnews.io — request free academic access |
| GDELT | Team 2 | None (free fallback) | gdeltproject.org |

## Project Links

- 📋 [Notion SSOT](https://www.notion.so/33640be8288a808ca693c13986e2a526) — week-by-week tasks, team specs, symposium info
- 📊 [Kalshi account](https://kalshi.com) — funded with $100 real capital for Week 7 live trading
