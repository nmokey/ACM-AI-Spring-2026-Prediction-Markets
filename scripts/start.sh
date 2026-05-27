#!/usr/bin/env bash
# scripts/start.sh
# ──────────────────
# Single entry point — starts the full bot stack:
#   1. Data + inference pipeline loop  (background, logs to logs/pipeline.log)
#   2. Trading execution loop          (foreground, terminal dashboard)
#
# Usage:
#   bash scripts/start.sh             # dry_run mode (default)
#
# To stop: Ctrl+C — kills both processes cleanly.
#
# Daily labeling cron (run once to install):
#   (crontab -l 2>/dev/null; echo "0 9 * * * cd $(pwd) && $HOME/.local/bin/uv run python -m scripts.label_resolved >> logs/label.log 2>&1") | crontab -

set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Create required directories
mkdir -p logs signals

PIPELINE_PID=""

cleanup() {
    echo ""
    echo "[start] Shutting down..."
    if [ -n "$PIPELINE_PID" ] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
        kill "$PIPELINE_PID"
        echo "[start] Pipeline loop stopped (PID $PIPELINE_PID)"
    fi
    exit 0
}

trap cleanup INT TERM

# Check required files
if [ ! -f "models/trained/xgb_v1.joblib" ]; then
    echo "[start] ERROR: models/trained/xgb_v1.joblib not found — run training first"
    exit 1
fi

MODE=$(uv run python -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['trading']['mode'])")
echo "[start] ──────────────────────────────────────"
echo "[start] Prediction Markets Bot — $(echo "$MODE" | tr '[:lower:]' '[:upper:]') mode"
echo "[start] ──────────────────────────────────────"

if [ "$MODE" = "live" ]; then
    echo "[start] ⚠️  LIVE MODE — real orders will be placed on Kalshi"
    read -p "[start] Type 'yes' to confirm: " confirm
    if [ "$confirm" != "yes" ]; then
        echo "[start] Aborted."
        exit 1
    fi
fi

# Step 1: run an initial feature + inference pass so predictions.json exists
# before the trader starts (avoids the "predictions.json not found" warning)
echo "[start] Running initial feature fetch + inference..."
uv run python -m data.engineer || echo "[start] WARN: initial feature fetch failed"
uv run python -m models.predict || echo "[start] WARN: initial inference failed"

# Step 2: start pipeline loop in background
echo "[start] Starting pipeline loop (logs → logs/pipeline.log)..."
bash scripts/run_pipeline.sh >> logs/pipeline.log 2>&1 &
PIPELINE_PID=$!
echo "[start] Pipeline PID: $PIPELINE_PID"

# Step 3: start trader in foreground (shows terminal dashboard)
echo "[start] Starting trader..."
uv run python -m execution.trader
