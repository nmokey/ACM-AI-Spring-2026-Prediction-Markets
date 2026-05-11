"""
models/train.py
─────────────────
XGBoost model training script — Team 2's main Week 4 deliverable.

Reads:
    data/features/live_features.parquet   (from Team 1 via data contract)
    nlp/sentiment.json                    (internal Team 2 sentiment cache — run nlp/sentiment.py first)

Trains an XGBoost classifier to predict whether a contract resolves YES,
calibrates it with isotonic regression, and saves the model.

Team 2 owns this file.

Usage: python -m models.train
"""

from __future__ import annotations

import json
import logging
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path

import pandas as pd
import yaml
import joblib

from models import evaluate

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

FEATURES_PATH = Path(CONFIG["data"]["features_path"])
SENTIMENT_PATH = Path(CONFIG["data"]["sentiment_path"])
MODEL_PATH = Path("models/trained/xgb_v1.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Feature columns fed to the model — add new ones here as you engineer them
FEATURE_COLS = [
    "market_price",
    "volume_24h",
    "days_to_resolution",
    "btc_change_1h",
    "btc_change_6h",
    "eth_change_1h",
    "eth_change_6h",
    "precip_prob_new_york",
    "precip_prob_los_angeles",
    "precip_prob_chicago",
    "sentiment_score",
    "sentiment_confidence",
]

TARGET_COL = "resolved_yes"  # 1 = contract resolved YES, 0 = resolved NO


def load_data(features_path: Path | None = None) -> pd.DataFrame:
    """
    Load historical features and join sentiment signals.

    The features parquet must include a `resolved_yes` column.

    Args:
        features_path: override the default path from settings.yaml.
                       Pass data/features/historical_features.parquet for backfill data.
    """
    path = features_path or FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_parquet(path)
    df = df.dropna(subset=[TARGET_COL])

    if SENTIMENT_PATH.exists():
        with open(SENTIMENT_PATH) as f:
            import json
            raw = json.load(f)
        sentiment_df = pd.DataFrame.from_dict(raw, orient="index")[
            ["sentiment_score", "sentiment_confidence"]
        ]
        sentiment_df.index.name = "contract_id"
        df = df.drop(columns=["sentiment_score", "sentiment_confidence"], errors="ignore")
        df = df.set_index("contract_id").join(sentiment_df, how="left").reset_index()
    else:
        df["sentiment_score"] = 0.0
        df["sentiment_confidence"] = 0.0

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    logger.info("Loaded %d labeled rows from %s", len(df), path)
    return df


def train(df: pd.DataFrame):
    """
    Train XGBoost + isotonic calibration on the feature matrix.

    Split strategy: split by contract_id groups, NOT randomly by row.
    Why? Different snapshots of the same contract are correlated — if you
    split randomly, the same contract appears in both train and test,
    which inflates your metrics (data leakage).

    Use sklearn's GroupShuffleSplit with groups=df["contract_id"].

    TODO (Week 4):
        1. Extract X = df[FEATURE_COLS].values and y = df[TARGET_COL].astype(int).values
        2. Use GroupShuffleSplit(n_splits=1, test_size=0.2) to get train/test indices
        3. Train xgb.XGBClassifier on X_train, y_train
           Suggested params: n_estimators=200, max_depth=4, learning_rate=0.05
        4. Wrap with CalibratedClassifierCV(base_model, method="isotonic", cv=3)
           and fit on X_train, y_train
        5. Call evaluate(model, X_test, y_test) and print the Brier score
        6. Return (model, X_test, y_test) for further analysis
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].astype(int).values

    # Handle class imbalance: weight the minority class proportionally
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info("Class balance: %d NO / %d YES — scale_pos_weight=%.1f", n_neg, n_pos, scale_pos_weight)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
    train_idx, test_index = next(splitter.split(X, y, groups=df["contract_id"]))
    X_train, X_test = X[train_idx], X[test_index]
    y_train, y_test = y[train_idx], y[test_index]

    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
    )
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    evaluate.evaluate_model(model, X_test, y_test, feature_names=FEATURE_COLS)
    return model, X_test, y_test


def save_model(model) -> None:
    """
    Serialize the trained model to disk.

    TODO (Week 4): import joblib; joblib.dump(model, MODEL_PATH)
    """
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/features/snapshots.parquet",
        help="Path to labeled features parquet (default: snapshots.parquet)",
    )
    args = parser.parse_args()
    df = load_data(Path(args.features))
    model, X_test, y_test = train(df)
    save_model(model)
