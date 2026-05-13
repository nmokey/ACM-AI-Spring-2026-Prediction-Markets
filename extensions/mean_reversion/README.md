# Extension: Crypto Mean Reversion

Implement `compute_signals()` in [strategy.py](strategy.py), then run it through the standard backtest loop in [extensions/SPEC.md](../SPEC.md#extension-1-crypto-mean-reversion).

**Data:** `ccxt` (Binance public OHLCV, no API key) or `yfinance`
**Key reuse:** `execution/kelly.kelly_fraction()`, `backtest/metrics.print_metrics()`
