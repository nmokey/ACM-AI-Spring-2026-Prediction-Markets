# Runbook — Pipeline, Training & Execution

## Prerequisites

```bash
# Clone the repo
git clone https://github.com/nmokey/ACM-AI-Spring-2026-Prediction-Markets.git
cd ACM-AI-Spring-2026-Prediction-Markets

# Requires Python 3.11+
pip install uv      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync             # installs all dependencies from pyproject.toml in ~30s
brew install libomp # macOS only — required for XGBoost

# Set up API keys
cp .env.example .env
# Edit .env and fill in: KALSHI_API_KEY, KALSHI_API_SECRET, GNEWS_API_KEY
```

---

## Architecture: Two Concurrent Processes

The live system requires two processes running simultaneously on the club server:

```
run_pipeline.sh  (every 15 min)         run_bot.sh  (every 60 sec)
  │                                        │
  ├─ data.engineer                         └─ execution.trader
  │    ├─ fetches Kalshi markets                reads predictions.json
  │    ├─ fetches BTC/ETH from Coinbase         applies risk + Kelly sizing
  │    ├─ fetches weather from NOAA             logs to dry_run_trades.csv
  │    ├─ writes live_features.parquet
  │    └─ appends to snapshots.parquet     label_resolved.py  (daily)
  │         (resolved_yes = null)            │
  ├─ nlp.sentiment  (every 30 min)          └─ queries Kalshi for settled status
  │    reads live_features.parquet               stamps resolved_yes = 0 or 1
  │    writes nlp/sentiment.json                 → snapshots become training rows
  └─ models.predict
       reads live_features.parquet
       reads nlp/sentiment.json
       writes predictions.json
```

**`run_pipeline.sh`** — the data + ML loop. Runs every 15 min. Fetches live market/crypto/weather data, scores NLP sentiment (every other cycle, ~30 min), runs model inference, and appends a snapshot row for every open contract to `data/features/snapshots.parquet`. Does not submit trades.

**`run_bot.sh`** — the execution loop. Reads `predictions.json` every 60 seconds and submits (or dry-run logs) orders. Depends on `run_pipeline.sh` to keep predictions fresh.

**`scripts/label_resolved.py`** — the labeling job. Run once daily. Checks Kalshi for settled contracts and stamps `resolved_yes` on their snapshot rows, converting raw snapshots into labeled training data.

---

## Running on the Club Server

```bash
# Install uv (user-level, no sudo needed):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies:
uv sync

# Start the pipeline in a tmux session (survives SSH disconnect):
tmux new -s pipeline
bash scripts/run_pipeline.sh 2>&1 | tee logs/pipeline.log
# Detach with Ctrl+B D

# In a second tmux window, start the bot:
tmux new -s bot
bash scripts/run_bot.sh 2>&1 | tee logs/bot.log
# Detach with Ctrl+B D

# Daily labeling — run once in the morning or cron it:
uv run python -m scripts.label_resolved
# Cron example:
# 0 9 * * * cd /path/to/repo && ~/.local/bin/uv run python -m scripts.label_resolved >> logs/label.log 2>&1
```

---

## Bot Modes

The trading mode is set in `config/settings.yaml`:

```yaml
trading:
  mode: "dry_run"          # "dry_run" | "live"
  min_edge: 0.06           # min |p_model - market_price| to trade
  kelly_fraction: 0.25     # quarter Kelly — never use full Kelly
  max_position_pct: 0.05   # max 5% of account per trade ($5 on $100)
  max_total_exposure_pct: 0.40
  min_confidence: 0.60
```

**`dry_run`** — all orders are logged to `logs/dry_run_trades.csv` via `execution/dry_run.py`. No real orders are placed. Default mode.

**`live`** — orders are submitted to Kalshi via `execution/order_manager.py`. `run_bot.sh` prompts for confirmation before starting in live mode. Only flip this after passing the Week 6 go/no-go gate (Sharpe > 1.0 AND win rate > 52%).

---

## Manual Step-by-Step (Development / Debugging)

Use these to run each pipeline stage individually:

```bash
# Step 1 — Refresh live features (Team 1)
uv run python -m data.engineer
# Writes: data/features/live_features.parquet
# Appends: data/features/snapshots.parquet (resolved_yes = null)

# Step 2 — Score sentiment (Team 2 NLP)
uv run python -m nlp.sentiment
# Reads: data/features/live_features.parquet (contract titles)
# Writes: nlp/sentiment.json

# Step 3 — Run inference (Team 2 Modeling)
uv run python -m models.predict
# Reads: data/features/live_features.parquet + nlp/sentiment.json
# Writes: signals/predictions.json

# Step 4 — Label settled contracts (run daily)
uv run python -m scripts.label_resolved
# Reads: data/features/snapshots.parquet
# Queries Kalshi for settled status
# Stamps: resolved_yes = 0 or 1

# Step 5 — Retrain (once you have 200+ labeled rows)
uv run python -m models.train
# Reads: data/features/snapshots.parquet (labeled rows only)
# Writes: models/trained/xgb_v1.joblib
```

---

## Smoke Tests

Run these to verify API connections and pipeline plumbing before starting the full system:

```bash
# Verify API connections
uv run python -m data.ingestion.kalshi_client   # prints 5 open Kalshi markets
uv run python -m data.ingestion.weather_client  # prints today's precip probabilities
uv run python -m data.ingestion.crypto_client   # prints BTC/ETH price changes
uv run python -m nlp.news_client                # prints 5 recent headlines

# Smoke test the execution pipeline (no API keys needed)
uv run python scripts/test_execution.py         # places 4 dummy dry-run orders
```

---

## Validate Pipeline Outputs

After the pipeline has completed at least one full cycle:

```bash
uv run python scripts/validate_features.py
```

Checks row counts, null fields, value ranges, sentiment coverage, and snapshot accumulation. Prints `PASS`/`FAIL` per invariant.

---

## Script Reference

| Script | How to run | What it does |
|---|---|---|
| `scripts/run_pipeline.sh` | `bash scripts/run_pipeline.sh` | Full data + ML loop, every 15 min |
| `scripts/run_bot.sh` | `bash scripts/run_bot.sh` | Execution loop, every 60 sec |
| `scripts/label_resolved.py` | `uv run python -m scripts.label_resolved` | Daily: stamps resolved_yes on settled contracts |
| `scripts/validate_features.py` | `uv run python scripts/validate_features.py` | Pipeline health check — PASS/FAIL per invariant |
| `scripts/test_execution.py` | `uv run python scripts/test_execution.py` | Smoke test for Team 3 (no API keys needed) |
| `data/engineer.py` | `uv run python -m data.engineer` | Refresh live_features.parquet + append to snapshots |
| `nlp/sentiment.py` | `uv run python -m nlp.sentiment` | Score sentiment → nlp/sentiment.json |
| `models/predict.py` | `uv run python -m models.predict` | Live inference → signals/predictions.json |
| `models/train.py` | `uv run python -m models.train` | Retrain XGBoost on labeled snapshots |
| `models/evaluate.py` | `uv run python -m models.evaluate` | Brier score, calibration curve, feature importance |
| `backtest/engine.py` | `uv run python -m backtest.engine` | Run backtest (dummy mode) |

---

## NLP Pipeline Notes

### Query deduplication

`_extract_query()` in `nlp/news_client.py` maps Kalshi contract titles to clean search queries. It handles all live contract types: crypto (BTC, ETH, SOL, DOGE, BNB, XRP), macro (Fed rate, CPI, GDP, ADP, WTI, EUR/USD, USD/JPY), weather (city name from title), sports (MLB, NBA, NHL, F1, per-player stat lines). Numbers, thresholds, and dates are stripped so that contracts on the same underlying (e.g. all 318 KXBTCD contracts) map to a single query like `"bitcoin price"` — reducing ~1,100 API calls to ~47 unique queries per sentiment cycle.

### Sentiment cycle timing

| Phase | Time |
|---|---|
| News fetch (47 queries via GDELT, with timeouts) | ~90s |
| Batch embedding + cosine similarity | ~50s |
| VADER sentiment scoring | ~1s |
| **Total** | **~2m 30s** |

The sentiment step runs every 30 min (every other 15-min pipeline cycle). Most cycles are just the fast `data.engineer` run (~10s).

### Batched embedding

`nlp/sentiment.py` embeds all headlines once into a matrix, then scores all contracts in a single batched matrix multiplication — O(1) embedding passes instead of O(contracts). This reduced scoring from hanging indefinitely (sequential per-contract embedding over 1,100 iterations) to ~50s.

### GNews rate limits

The free GNews tier has a low daily request cap. When the key is exhausted, GNews returns 403 — GDELT fallback takes over automatically. To increase limits, request free academic access from your UCLA email at gnews.io.
