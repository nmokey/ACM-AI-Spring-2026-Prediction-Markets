# Extension: Crypto Momentum

Implement `compute_signals()` in [strategy.py](strategy.py), then run it through the standard backtest loop in [extensions/SPEC.md](../SPEC.md#extension-1-crypto-momentum).

**Data:** `ccxt` (Binance public OHLCV, no API key) or `yfinance`
**Key reuse:** `execution/kelly.kelly_fraction()`, `backtest/metrics.print_metrics()`
