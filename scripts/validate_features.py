"""
scripts/validate_features.py
─────────────────────────────
Validates that all pipeline outputs are well-formed and non-null after data
has been accumulated. Run this after at least one full pipeline cycle.

Usage:
    uv run python scripts/validate_features.py

Exit code 0 = all checks passed. Exit code 1 = one or more failures.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
FAILURES = []


def check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not passed:
        FAILURES.append(name)


# ── 1. live_features.parquet ─────────────────────────────────────────────────

print("\n=== live_features.parquet ===")
live_path = ROOT / "data/features/live_features.parquet"
if not live_path.exists():
    print("  [FAIL] file not found — has the pipeline run yet?")
    FAILURES.append("live_features.parquet missing")
else:
    df = pd.read_parquet(live_path)
    check("row count in [1, 2000]", 1 <= len(df) <= 2000, f"{len(df)} rows")
    check("contract_id non-null", df["contract_id"].notna().all(),
          f"{df['contract_id'].isna().sum()} nulls")
    check("title non-null", df["title"].notna().all(),
          f"{df['title'].isna().sum()} nulls")
    check("market_price in [0,1] where non-null",
          df["market_price"].dropna().between(0, 1).all(),
          f"out-of-range: {(~df['market_price'].dropna().between(0,1)).sum()}")
    check("btc_change_1h non-null", df["btc_change_1h"].notna().all(),
          f"{df['btc_change_1h'].isna().sum()} nulls")
    check("btc_change_6h non-null", df["btc_change_6h"].notna().all(),
          f"{df['btc_change_6h'].isna().sum()} nulls")
    check("eth_change_1h non-null", df["eth_change_1h"].notna().all(),
          f"{df['eth_change_1h'].isna().sum()} nulls")
    check("eth_change_6h non-null", df["eth_change_6h"].notna().all(),
          f"{df['eth_change_6h'].isna().sum()} nulls")
    check("precip_prob_new_york non-null", df["precip_prob_new_york"].notna().all(),
          f"{df['precip_prob_new_york'].isna().sum()} nulls")
    check("precip_prob_los_angeles non-null", df["precip_prob_los_angeles"].notna().all(),
          f"{df['precip_prob_los_angeles'].isna().sum()} nulls")
    check("precip_prob_chicago non-null", df["precip_prob_chicago"].notna().all(),
          f"{df['precip_prob_chicago'].isna().sum()} nulls")
    check("precip values in [0,100]",
          df[["precip_prob_new_york","precip_prob_los_angeles","precip_prob_chicago"]]
          .dropna().apply(lambda s: s.between(0,100)).all(axis=None),
          "")
    check("fetched_at non-null", df["fetched_at"].notna().all(),
          f"{df['fetched_at'].isna().sum()} nulls")
    check("no duplicate contract_ids", not df["contract_id"].duplicated().any(),
          f"{df['contract_id'].duplicated().sum()} duplicates")

    null_counts = df[["market_price","btc_change_1h","precip_prob_new_york"]].isnull().sum()
    print(f"\n  null summary (key cols): {null_counts.to_dict()}")

# ── 2. nlp/sentiment.json ─────────────────────────────────────────────────────

print("\n=== nlp/sentiment.json ===")
sent_path = ROOT / "nlp/sentiment.json"
if not sent_path.exists():
    print("  [FAIL] file not found")
    FAILURES.append("sentiment.json missing")
elif not live_path.exists():
    print("  [SKIP] live_features.parquet missing — can't check coverage")
else:
    with open(sent_path) as f:
        s = json.load(f)
    live = pd.read_parquet(live_path)
    coverage = len(s) / max(len(live), 1)
    nonzero = sum(1 for v in s.values() if v.get("sentiment_score", 0.0) != 0.0)

    check("coverage > 90%", coverage > 0.9, f"{coverage:.0%} ({len(s)}/{len(live)})")
    check("some non-zero scores", nonzero > 0, f"{nonzero}/{len(s)} non-zero")

    for cid, entry in s.items():
        score = entry.get("sentiment_score", 0.0)
        conf = entry.get("sentiment_confidence", 0.0)
        if not (-1.0 <= score <= 1.0):
            FAILURES.append(f"sentiment_score out of range for {cid}")
        if not (0.0 <= conf <= 1.0):
            FAILURES.append(f"sentiment_confidence out of range for {cid}")

    bad_scores = sum(1 for v in s.values()
                     if not (-1.0 <= v.get("sentiment_score", 0.0) <= 1.0))
    check("all sentiment_scores in [-1,1]", bad_scores == 0, f"{bad_scores} out of range")
    bad_conf = sum(1 for v in s.values()
                   if not (0.0 <= v.get("sentiment_confidence", 0.0) <= 1.0))
    check("all sentiment_confidences in [0,1]", bad_conf == 0, f"{bad_conf} out of range")
    print(f"\n  coverage: {coverage:.0%}  non-zero scores: {nonzero}/{len(s)}")

# ── 3. signals/predictions.json ───────────────────────────────────────────────

print("\n=== signals/predictions.json ===")
pred_path = ROOT / "signals/predictions.json"
if not pred_path.exists():
    print("  [FAIL] file not found")
    FAILURES.append("predictions.json missing")
else:
    with open(pred_path) as f:
        p = json.load(f)
    check("non-empty", len(p) > 0, f"{len(p)} predictions")
    vals_p = [v["p_model"] for v in p.values()]
    vals_c = [v["confidence"] for v in p.values()]
    check("all p_model in [0,1]", all(0.0 <= v <= 1.0 for v in vals_p),
          f"{sum(1 for v in vals_p if not 0<=v<=1)} out of range")
    check("all confidence in [0,1]", all(0.0 <= v <= 1.0 for v in vals_c),
          f"{sum(1 for v in vals_c if not 0<=v<=1)} out of range")
    if vals_p:
        print(f"\n  {len(p)} predictions  "
              f"p_model: min={min(vals_p):.3f} max={max(vals_p):.3f} mean={sum(vals_p)/len(vals_p):.3f}")

# ── 4. snapshots.parquet ──────────────────────────────────────────────────────

print("\n=== data/features/snapshots.parquet ===")
snap_path = ROOT / "data/features/snapshots.parquet"
if not snap_path.exists():
    print("  [FAIL] file not found")
    FAILURES.append("snapshots.parquet missing")
else:
    df = pd.read_parquet(snap_path)
    labeled = df["resolved_yes"].notna().sum()
    timestamps = df["fetched_at"].nunique()
    check("row count > 0", len(df) > 0, f"{len(df)} rows")
    check("contract_id non-null", df["contract_id"].notna().all(),
          f"{df['contract_id'].isna().sum()} nulls")
    check("fetched_at non-null", df["fetched_at"].notna().all(),
          f"{df['fetched_at'].isna().sum()} nulls")
    check("multiple snapshots accumulated", timestamps > 1,
          f"{timestamps} unique fetched_at timestamps")
    if labeled > 0:
        check("resolved_yes is 0 or 1 where labeled",
              df["resolved_yes"].dropna().isin([0, 1, 0.0, 1.0]).all(),
              f"{labeled} labeled rows")
    print(f"\n  {len(df)} total rows | {labeled} labeled | {len(df)-labeled} pending")
    print(f"  {timestamps} unique fetched_at timestamps")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
if FAILURES:
    print(f"FAILED — {len(FAILURES)} check(s) did not pass:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
