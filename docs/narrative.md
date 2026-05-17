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

- Integrated the full pipeline end-to-end: `nlp.sentiment` → `data.engineer` → `models.predict` → `execution.trader`.
- Fixed 6 QA-identified bugs across data, model, NLP, and execution layers.
- Fixed NLP pipeline crash and documented server setup.
- Added snapshot accumulation pipeline.
- Added VADER, FinBERT, and `score_text()` to output score and confidence intervals.
- Added series-based market fetching with category tagging.
- Wrote `scripts/validate_features.py` for pipeline output health checks.
- Bot started trading autonomously in dry-run mode against live Kalshi prices.

### Week 6 — Execution Hardening & Training Data Quality

- Rewrote `execution/trader.py` with a proper terminal scroll region dashboard — live countdown, per-trade context lines (p_model, edge, kelly%), all-time win/loss tracking, simulated balance.
- Implemented real resolution tracking via Kalshi API: `check_resolutions()` polls `get_market()` each cycle; handles both `"closed"` and `"finalized"` status since Kalshi posts results before finalization.
- Implemented order fill confirmation for live mode: polls `GET /portfolio/orders/{id}` every 2s up to 30s; cancels unfilled resting orders via `DELETE`.
- Fixed simulated balance: `$100 − open_position_cost + realized_pnl`, updated on every resolution.
- Fixed critical pipeline ordering bug: `nlp.sentiment` was running **after** `data.engineer`, causing sentiment to always be stale (zero) in snapshot rows. Reordering increased sentiment coverage from 24% to 47% overall, 74% for weather contracts.
- Retrained model on 32k unique live-accumulated contracts (deduplicated with `--dedup`). Sentiment now registers non-zero feature importance (2.4% combined) for the first time.
- Added `logs/model_metrics.jsonl` — one record per retrain with timestamp, brier score, log-loss, n_train, n_test, and full feature importance dict. Consumed by `notebooks/performance.ipynb`.
- Added `logs/resolved_trades.csv` — one row per resolved position with side, size, entry price, result, win/loss, and P&L.

---

## Current State (May 2026)

The bot has been running in dry-run mode for several weeks, trading exclusively **weather contracts**. Crypto contracts are too efficiently priced — the model cannot find sufficient edge (`|p_model − market_price| > 0.06`) to size a bet. Weather contracts have lower market efficiency because NOAA forecasts give the model a real informational edge.

**Live dry-run stats:**
- 89 orders placed, all weather (high/low temperature contracts across 10 US cities)
- 38 resolved: 24W / 14L — 63.2% win rate
- Realized P&L: +$3.18 on a $100 simulated balance
- Brier score: 0.0162 (well under 0.20 target)

**Sentiment signal status:**
- Weather contracts have 74% sentiment coverage; the model uses it but the marginal edge over market_price is tiny (Brier improvement ~0.00003 for weather).
- Sentiment would matter most for macro contracts (KXFED, KXCPI, KXEURUSD) where headlines directly predict the outcome — these are not currently in the trading universe.

---

## Known Issues & Active Limitations

### 1. `market_price` dominates feature importances (85%)

This is structurally expected: the model is trained on 32k contracts spanning crypto (majority of rows), weather, sports, and macro. For crypto contracts, the market price IS the signal — these are highly efficient markets and the model learns to echo the crowd. For weather, the model has more room to deviate from market price, and that's where trades actually get placed.

Sentiment contributes 2.4% (score + confidence combined) and is expected to grow as more labeled rows accumulate with correct pipeline ordering.

### 2. Sentiment adds negligible edge on weather contracts specifically

Correlation of sentiment_score with weather contract outcomes: r = +0.013. Adding sentiment to a logistic regression on top of market_price improves Brier score by 0.00003. This is because financial/news headlines (GNews/GDELT) are not about daily temperature in Chicago — they're about macro events. Sentiment would be more predictive for macro and political contracts.

### 3. Bot only trades weather — crypto edge is too small

The `min_edge: 0.06` threshold in `settings.yaml` filters out nearly all crypto contracts because the market price is already well-calibrated. The model essentially agrees with the crowd on crypto. This is correct behavior — the bot should not trade when it has no edge — but it limits diversification.

### 4. No early exit / mark-to-market tracking

Positions are held to resolution. There is no stop-loss or take-profit logic. The bot buys contracts and waits for them to settle. This is appropriate for binary prediction markets where the payout structure is binary, but it means the bot cannot react to a market moving sharply against a position.

### 5. GNews rate limits

The free GNews tier has a low daily request cap. When exhausted, all queries return 403 — GDELT fallback kicks in automatically. To increase limits, request free academic access from your UCLA email at gnews.io.

---

## Go/No-Go Gate (Week 7)

The decision to flip from `mode: "dry_run"` to `mode: "live"` in `config/settings.yaml` requires passing both thresholds:

| Metric | Target | Current |
|---|---|---|
| Win Rate | > 52% | **63.2%** ✓ |
| Sharpe Ratio | > 1.0 | backtest pending |

Win rate clears the bar. Sharpe ratio requires completing the backtest in `backtest/engine.py` against historical snapshots. If both pass → flip `trading.mode` to `"live"`. The account is funded with $100 real capital on Kalshi.

---

## Key Metrics & Targets

| Metric | Target | Notes |
|---|---|---|
| Brier Score | < 0.20 | Primary model quality metric. Random = 0.25. Current: **0.0162** |
| Sharpe Ratio | > 1.0 | Risk-adjusted return on backtest — pending |
| Win Rate | > 52% | % of trades that close profitably. Current: **63.2%** |
| Edge per Trade | > 0.05 | Avg \|p_model − market_price\| on placed trades |
| Dry-Run Trades | > 50 | Proof the system is running autonomously. Current: **89** |
| Sentiment coverage | — | % of labeled rows with non-zero sentiment. Current: **47%** overall, **74%** weather |
