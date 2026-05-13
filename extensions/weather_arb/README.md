# Extension: Weather Arbitrage

Implement `compute_signals()` in [strategy.py](strategy.py), then run it through the standard backtest loop in [extensions/SPEC.md](../SPEC.md#extension-2-weather-arbitrage).

**Data:** NOAA Climate Data Online CSV download (free, no account) — see SPEC.md for the download link and steps.
**Key reuse:** `data/ingestion/weather_client.py` (for live forecasts), `backtest/metrics.print_metrics()`
