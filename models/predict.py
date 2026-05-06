"""
models/predict.py
──────────────────
Live inference — loads the trained model and writes predictions for all
active contracts to signals/predictions.json.

Reads:   data/features/live_features.parquet
         nlp/sentiment.json   (internal Team 2 cache — produced by nlp/sentiment.py)
Writes:  signals/predictions.json

Team 2 owns this file.

Usage: python -m models.predict
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from data.features.schema import PredictionSignal

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

FEATURES_PATH = Path(CONFIG["data"]["features_path"])
SENTIMENT_PATH = Path(CONFIG["data"]["sentiment_path"])
PREDICTIONS_PATH = Path(CONFIG["data"]["predictions_path"])
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("models/trained/xgb_v1.joblib")

FEATURE_COLS = [
    "market_price", "volume_24h", "days_to_resolution",
    "btc_change_1h", "btc_change_6h",
    "eth_change_1h", "eth_change_6h",
    "precip_prob_new_york", "precip_prob_los_angeles", "precip_prob_chicago",
    "sentiment_score", "sentiment_confidence",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict() -> dict[str, PredictionSignal]:
    model = load_model()

    df = pd.read_parquet(FEATURES_PATH)
    # Live features have no resolved_yes — keep all rows
    df = df[df["market_price"].notna()].copy()

    if SENTIMENT_PATH.exists():
        with open(SENTIMENT_PATH) as f:
            raw = json.load(f)
        sentiment_df = pd.DataFrame.from_dict(raw, orient="index")[
            ["sentiment_score", "sentiment_confidence"]
        ]
        sentiment_df.index.name = "contract_id"
        df = df.set_index("contract_id").join(sentiment_df, how="left").reset_index()
    else:
        df["sentiment_score"] = 0.0
        df["sentiment_confidence"] = 0.0

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    X = df[FEATURE_COLS].values
    p_yes = model.predict_proba(X)[:, 1]
    confidence = abs(p_yes - 0.5) * 2

    now = datetime.now(timezone.utc)
    signals: dict[str, PredictionSignal] = {}
    for i, row in df.iterrows():
        cid = row["contract_id"]
        signals[cid] = PredictionSignal(
            contract_id=cid,
            timestamp=now,
            p_model=float(p_yes[i]),
            confidence=float(confidence[i]),
        )

    save_predictions(signals)
    return signals


def save_predictions(signals: dict[str, PredictionSignal]) -> None:
    payload = {cid: sig.model_dump(mode="json") for cid, sig in signals.items()}
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preds = predict()
    for cid, p in list(preds.items())[:5]:
        print(f"{cid:50s}  p_model={p.p_model:.3f}  conf={p.confidence:.3f}")
