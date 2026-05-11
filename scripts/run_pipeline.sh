#!/usr/bin/env bash
# scripts/run_pipeline.sh
# ─────────────────────────
# Data + NLP pipeline loop — runs on the club server overnight.
# Refreshes live_features.parquet (Team 1) and NLP sentiment cache (Team 2) on schedule.
#
# Usage: bash scripts/run_pipeline.sh
# Run in background: nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

FEATURE_INTERVAL=900   # 15 min in seconds
SENTIMENT_INTERVAL=1800 # 30 min in seconds

echo "[pipeline] Starting data + NLP pipeline loop"
echo "[pipeline] Feature refresh: every ${FEATURE_INTERVAL}s"
echo "[pipeline] Sentiment refresh: every ${SENTIMENT_INTERVAL}s"

last_sentiment=0

while true; do
    now=$(date +%s)

    echo "[pipeline] $(date -u '+%Y-%m-%d %H:%M:%S UTC') — refreshing features..."
    uv run python -m data.engineer || echo "[pipeline] Feature refresh failed — continuing"

    # Refresh sentiment every SENTIMENT_INTERVAL
    if (( now - last_sentiment >= SENTIMENT_INTERVAL )); then
        echo "[pipeline] $(date -u '+%Y-%m-%d %H:%M:%S UTC') — refreshing NLP sentiment..."
        uv run python -m nlp.sentiment || echo "[pipeline] Sentiment refresh failed — continuing"
        last_sentiment=$now
    fi

    echo "[pipeline] $(date -u '+%Y-%m-%d %H:%M:%S UTC') — running inference..."
    uv run python -m models.predict || echo "[pipeline] Inference failed — continuing"

    echo "[pipeline] Sleeping ${FEATURE_INTERVAL}s..."
    sleep $FEATURE_INTERVAL
done
