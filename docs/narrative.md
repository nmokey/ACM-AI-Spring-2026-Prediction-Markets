# Project Narrative — Progress & Issues

## Overview

This document tracks the story of the project: what was built, in what order, and what problems were encountered along the way. It is a living record, not a finished report.

---

## Timeline

### Weeks 1–2 — Foundation

- Established repo structure and team ownership conventions (Teams 1, 2, 3).
- Scaffolded `data/`, `nlp/`, `models/`, `execution/`, `signals/`, `backtest/`, `scripts/`.
- Wrote Pydantic data contracts in `data/schema.py` — the cross-team interface spec.
- Implemented `kalshi_client.py` with RSA-PSS auth, market fetching, orderbook, and backfill of resolved markets.
- Implemented `weather_client.py` (NOAA) and `crypto_client.py` (Coinbase Advanced).

### Weeks 3–4 — NLP & Feature Engineering

- Built `nlp/news_client.py` with GNews primary + GDELT fallback; SQLite headline store.
- Built `nlp/relevance.py` cosine similarity scorer using `all-MiniLM-L6-v2`.
- Built `nlp/sentiment.py` with FinBERT and VADER scoring.
- Wrote `data/engineer.py` — full feature engineering pipeline producing `live_features.parquet` and appending to `snapshots.parquet`.
- Added `_extract_query()` in `news_client.py` to map ~1,100 Kalshi contract titles down to ~47 unique search queries, dramatically reducing API load.

### Week 5 — Pipeline Integration & Bugs

- Integrated the full pipeline end-to-end: `data.engineer` → `nlp.sentiment` → `models.predict` → `execution.trader`.
- Fixed 6 QA-identified bugs across data, model, NLP, and execution layers (see commit `14b877a`).
- Fixed NLP pipeline crash and documented server setup (see commit `3899c4c`).
- Added snapshot accumulation pipeline and fixed `sentiment` `__main__` bug (see commit `536b064`).
- Added VADER, FinBERT, and `score_text()` to output score and confidence intervals (see commit `e4bef9c`).
- Added series-based market fetching with category tagging (see commit `66aefef`).
- Wrote `scripts/validate_features.py` for pipeline output health checks.

### Week 6 (Current) — Execution & Training Data

- `backtest/engine.py` working in dummy mode (predictions.json + simulated outcomes); real model backtest on historical features is the current milestone.
- Training data quality issues identified (see Known Issues below). Retraining on live-accumulated snapshots is the path forward.

---

## Known Issues & Active Limitations

### 1. Training data is all MVE parlay contracts

Kalshi's `GET /markets?status=settled` only returns MVE (multi-variate event) parlay contracts — same-day sports parlays that resolve quickly. The 3,931 rows in `historical_features.parquet` are all of this type.

**Why it matters:** These are exotic multi-leg bets structurally different from the weather, crypto, and sports contracts we actually want to trade. A model trained on parlays won't generalize.

**Mitigation in progress:** `data.engineer` is running on a schedule to snapshot live features. When contracts resolve, `scripts/label_resolved.py` stamps `resolved_yes`. After ~1–2 weeks of collection we retrain on that data instead.

### 2. Severe class imbalance in historical data

Of 3,931 resolved contracts: 3,611 resolved NO (92%), 320 resolved YES (8%). `train.py` compensates with `scale_pos_weight=11.3`, but positive examples are scarce.

### 3. `market_price` is degenerate in training data

75%+ of rows have `market_price = 0.0` — parlay contracts had no orderbook activity before resolving. The trained model effectively learns: *zero market price → resolves NO*. The Brier score of 0.064 is misleadingly good because predicting NO on 92% NO data scores well regardless of real predictive signal. Feature importances confirm `market_price` carries ~100% of weight; all others are zero.

### 4. `market_category` is null everywhere

Kalshi does not return `market_category` for MVE contracts. Category-based filtering or features are not currently usable for any historical or live MVE contract.

### 5. Crypto features are null in historical data

`btc_change_1h/6h` and `eth_change_1h/6h` are null in `historical_features.parquet` (resolved markets don't carry live Coinbase data). `train.py` fills these with `0.0`; they contribute no signal to the current model.

### 6. GNews rate limits

The free GNews tier has a low daily request cap. When exhausted, all queries return 403 — GDELT fallback kicks in automatically. To increase limits, request free academic access from your UCLA email at gnews.io.

---

## Go/No-Go Gate (Week 6)

The decision to flip from `mode: "dry_run"` to `mode: "live"` in `config/settings.yaml` requires passing both thresholds on the backtest:

| Metric | Target |
|---|---|
| Sharpe Ratio | > 1.0 |
| Win Rate | > 52% |

If both pass → flip `trading.mode` to `"live"` in `config/settings.yaml`. The account is funded with $100 real capital on Kalshi.

---

## Key Metrics & Targets

| Metric | Target | Notes |
|---|---|---|
| Brier Score | < 0.20 | Primary model quality metric. Random = 0.25 |
| Sharpe Ratio | > 1.0 | Risk-adjusted return on backtest |
| Win Rate | > 52% | % of trades that close profitably |
| Edge per Trade | > 0.05 | Avg \|p_model − market_price\| on winning trades |
| Dry-Run Trades | > 50 | Proof the system is running autonomously |
