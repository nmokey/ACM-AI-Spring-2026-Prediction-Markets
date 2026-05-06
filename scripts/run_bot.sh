#!/usr/bin/env bash
# scripts/run_bot.sh
# ────────────────────
# Starts the trading bot (execution/trader.py).
# Make sure run_pipeline.sh is already running so predictions.json stays fresh.
#
# Usage: bash scripts/run_bot.sh
# Run in background: nohup bash scripts/run_bot.sh > logs/bot.log 2>&1 &

set -euo pipefail

MODE=$(uv run python -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['trading']['mode'])")
echo "[bot] Starting trader in ${MODE^^} mode"

if [ "$MODE" = "live" ]; then
    echo "[bot] ⚠️  LIVE MODE — real orders will be placed on Kalshi"
    read -p "[bot] Type 'yes' to confirm: " confirm
    if [ "$confirm" != "yes" ]; then
        echo "[bot] Aborted."
        exit 1
    fi
fi

uv run python -m execution.trader
