# Extension: Sports Mean-Reversion

Implement `compute_signals()` in [strategy.py](strategy.py), then run it through the standard backtest loop in [extensions/SPEC.md](../SPEC.md#extension-3-sports-mean-reversion).

**Data:** FiveThirtyEight NBA Elo CSV (free direct download, ~70k games, 1977–2024) — URL in SPEC.md.
**Key reuse:** `execution/kelly.kelly_fraction()`, `backtest/metrics.print_metrics()`
