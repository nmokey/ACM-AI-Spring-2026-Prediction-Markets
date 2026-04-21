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
    "price_change_1h", "price_change_6h",
    "sentiment_score", "sentiment_confidence",
]


def load_model():
    """
    Load the trained model from disk.

    TODO (Week 5):
        import joblib
        if not MODEL_PATH.exists(): raise FileNotFoundError(...)
        return joblib.load(MODEL_PATH)
    """
    raise NotImplementedError


def predict() -> dict[str, PredictionSignal]:
    """
    Run live inference on all active (unresolved) contracts.

    TODO (Week 5):
        1. Load model with load_model()
        2. Read live_features.parquet — filter OUT rows with a resolved_yes label
           (we only want to predict on contracts still open)
        3. Join sentiment_score and sentiment_confidence from nlp/sentiment.json — internal Team 2 cache (default 0.0)
        4. Fill NaNs in FEATURE_COLS with 0.0
        5. Call model.predict_proba(X)[:, 1] to get P(YES) for each row
        6. Compute confidence as a proxy for certainty:
               confidence = abs(p_model - 0.5) * 2
               (scales distance from 0.5 into [0, 1])
        7. Build a PredictionSignal per contract and store in a dict
        8. Call save_predictions() and return the dict
    """
    raise NotImplementedError


def save_predictions(signals: dict[str, PredictionSignal]) -> None:
    """
    Write predictions dict to signals/predictions.json.

    TODO (Week 5):
        payload = {cid: sig.model_dump(mode="json") for cid, sig in signals.items()}
        json.dump(payload, open(PREDICTIONS_PATH, "w"), indent=2, default=str)
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preds = predict()
    for cid, p in list(preds.items())[:5]:
        print(f"{cid:50s}  p_model={p.p_model:.3f}  conf={p.confidence:.3f}")
