"""
scripts/label_resolved.py
──────────────────────────
Fills in resolved_yes on snapshot rows for contracts that have now settled.

Run this daily (or manually) to grow the labeled training set:

    uv run python -m scripts.label_resolved

How it works:
    1. Load snapshots.parquet and find all contract_ids with resolved_yes=NaN
    2. Group unlabeled tickers by their series prefix (the part before the first '-S2026')
    3. For each series, fetch settled markets from Kalshi and build a ticker→result map
    4. Fall back to direct get_market() for non-series tickers (standard format)
    5. Write resolved_yes labels back to snapshots.parquet

Why series-based lookup:
    KXMV* (parlay/multi-leg) contracts disappear from /markets/{ticker} within hours
    of settling. They remain accessible via /markets?series_ticker=...&status=settled,
    so we must query the series endpoint rather than individual tickers.

After labeling, retrain with:
    uv run python -m models.train --features data/features/snapshots.parquet
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from data.ingestion.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
with open(ROOT / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

SNAPSHOTS_PATH = ROOT / CONFIG["data"]["snapshots_path"]

# Kalshi uses "finalized" as well as "settled" for resolved markets
_RESOLVED_STATUSES = {"settled", "finalized"}
_VALID_RESULTS = {"yes", "no"}


def _series_prefix(ticker: str) -> str | None:
    """
    Extract the series prefix from a ticker, or None if it's a standard ticker.

    KXMVESPORTSMULTIGAMEEXTENDED-S20267193E5278F5-7A6B500C4C7 → KXMVESPORTSMULTIGAMEEXTENDED
    KXBTCD-24DEC31-T95000 → None  (standard format, use direct lookup)
    """
    parts = ticker.split("-")
    if len(parts) >= 2 and parts[1].startswith("S2026"):
        return parts[0]
    return None


def _fetch_series_labels(
    kalshi: KalshiClient,
    series: str,
    wanted: set[str],
) -> dict[str, int]:
    """
    Query settled markets for one series and return {ticker: resolved_yes} for
    tickers in `wanted`. Paginates until all wanted tickers are found or exhausted.
    """
    labels: dict[str, int] = {}
    cursor: str | None = None

    for _ in range(50):  # safety cap — each page = 100 markets
        try:
            params: dict = {"series_ticker": series, "status": "settled", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            resp = kalshi._get("/markets", params=params)
        except Exception as e:
            logger.warning("Series fetch failed for %s: %s", series, e)
            break

        for m in resp.get("markets", []):
            ticker = m.get("ticker", "")
            result = m.get("result", "")
            status = m.get("status", "")
            if ticker in wanted and status in _RESOLVED_STATUSES and result in _VALID_RESULTS:
                labels[ticker] = 1 if result == "yes" else 0

        cursor = resp.get("cursor") or None
        # Stop early if we've found everything we're looking for
        if not cursor or wanted.issubset(labels):
            break
        time.sleep(0.25)

    return labels


def label_resolved(dry_run: bool = False) -> None:
    if not SNAPSHOTS_PATH.exists():
        logger.error("No snapshots file at %s — run data.engineer first", SNAPSHOTS_PATH)
        return

    df = pd.read_parquet(SNAPSHOTS_PATH)
    unlabeled_ids = df[df["resolved_yes"].isna()]["contract_id"].unique().tolist()
    logger.info("%d unique contracts need labeling", len(unlabeled_ids))

    if not unlabeled_ids:
        logger.info("Nothing to label.")
        return

    kalshi = KalshiClient()
    labels: dict[str, int] = {}

    # Split into series-based vs direct-lookup tickers
    by_series: dict[str, set[str]] = defaultdict(set)
    direct: list[str] = []

    for ticker in unlabeled_ids:
        prefix = _series_prefix(ticker)
        if prefix:
            by_series[prefix].add(ticker)
        else:
            direct.append(ticker)

    # Series-based batch lookup (covers KXMV* and similar)
    for series, wanted in by_series.items():
        logger.info("Querying series %s (%d tickers)...", series, len(wanted))
        found = _fetch_series_labels(kalshi, series, wanted)
        labels.update(found)
        logger.info("  → %d/%d resolved", len(found), len(wanted))
        time.sleep(0.25)

    # Direct per-ticker lookup for standard tickers
    for ticker in direct:
        try:
            market = kalshi.get_market(ticker)
            status = market.get("status", "")
            result = market.get("result", "")
            if status in _RESOLVED_STATUSES and result in _VALID_RESULTS:
                labels[ticker] = 1 if result == "yes" else 0
                logger.info("  %s → resolved_yes=%d", ticker, labels[ticker])
            else:
                logger.debug("  %s still open (status=%s)", ticker, status)
        except Exception as e:
            logger.warning("  Failed to fetch %s: %s", ticker, e)
        time.sleep(0.15)

    if not labels:
        logger.info("No newly settled contracts found.")
        return

    logger.info("Labeling %d contracts", len(labels))
    if not dry_run:
        df["resolved_yes"] = df.apply(
            lambda row: labels[row["contract_id"]]
            if row["contract_id"] in labels and pd.isna(row["resolved_yes"])
            else row["resolved_yes"],
            axis=1,
        )
        df.to_parquet(SNAPSHOTS_PATH, index=False)

    labeled_total = df["resolved_yes"].notna().sum()
    logger.info(
        "Done. %d/%d rows now labeled in %s",
        labeled_total, len(df), SNAPSHOTS_PATH,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be labeled without writing to disk",
    )
    args = parser.parse_args()
    label_resolved(dry_run=args.dry_run)
