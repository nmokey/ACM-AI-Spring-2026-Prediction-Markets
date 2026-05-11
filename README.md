# ACM AI — Prediction Markets Trading Bot · UCLA Spring 2026

**Thesis:** Prediction market prices are inefficient in predictable ways. A model that fuses structured time-series price signals with unstructured NLP sentiment signals can produce better-calibrated probability estimates than the crowd — and trade on that edge automatically.

We build an end-to-end quantitative trading bot targeting [Kalshi](https://kalshi.com) prediction markets, focusing on weather, crypto, and sports contracts that resolve multiple times per day.

---

## System Architecture

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
│               │  │FinBERT /   │  │      │
│               │  │VADER (NLP) │  │      │
│               │  └─────┬──────┘  │      │
│               │        ▼         │      │
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

---

## APIs & Models

| Component | Technology | Notes |
|---|---|---|
| Market data | [Kalshi REST API](https://trading-api.kalshi.com/docs) | RSA-PSS auth; prices, orderbook, resolved markets |
| Weather data | [NOAA](https://weather.gov/documentation) | Precipitation probabilities for NY, LA, Chicago |
| Crypto data | [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api) | BTC/ETH spot + 1h/6h candles |
| News headlines | [GNews](https://gnews.io) + [GDELT](https://gdeltproject.org) | GNews primary; GDELT free fallback on 403/429 |
| NLP sentiment | FinBERT + VADER | FinBERT for financial tone; VADER for speed |
| Relevance scoring | `all-MiniLM-L6-v2` | Cosine similarity — headline-to-contract relevance |
| Prediction model | XGBoost + isotonic calibration | GroupShuffleSplit; class-imbalance weighting |
| Position sizing | Fractional Kelly Criterion | 0.25× Kelly; 5% max per trade |

---

## Results

> Live results pending Week 6 backtest. Current status: pipeline is running in dry-run mode, accumulating labeled snapshots for retraining.

| Metric | Target | Current |
|---|---|---|
| Brier Score | < 0.20 | 0.064 (⚠️ misleading — see [narrative](docs/narrative.md)) |
| Sharpe Ratio | > 1.0 | Backtest in progress |
| Win Rate | > 52% | Backtest in progress |
| Dry-Run Trades | > 50 | Accumulating |

---

## Quick Start

```bash
git clone https://github.com/nmokey/ACM-AI-Spring-2026-Prediction-Markets.git
cd ACM-AI-Spring-2026-Prediction-Markets

pip install uv
uv sync
cp .env.example .env   # fill in KALSHI_API_KEY, KALSHI_API_SECRET, GNEWS_API_KEY
```

See [docs/runbook.md](docs/runbook.md) for full setup, server deployment, and script reference.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/runbook.md](docs/runbook.md) | How to run the pipeline, bot modes, manual steps, script reference, NLP notes |
| [docs/repo_structure.md](docs/repo_structure.md) | Directory map, team ownership, data contracts between teams |
| [docs/narrative.md](docs/narrative.md) | Project progress, known data limitations, go/no-go gate |

---

## Project Links

- [Notion SSOT](https://www.notion.so/33640be8288a808ca693c13986e2a526) — week-by-week tasks, team specs, symposium info
- [Kalshi account](https://kalshi.com) — funded with $100 real capital for Week 7 live trading
