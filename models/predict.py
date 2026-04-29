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
    import joblib 

    if not MODEL_PATH().exists():
        raise FileNotFoundError("Model not found at {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict() -> dict[str, PredictionSignal]:
    import pandas as pd
    import json

    model = load_model()
    df = pd.read_parquet("data/live_features.parquet")
    df = df[df["resolved_yes"].isna()]

    with open("nlp/sentiment.json") as f:
        sentiment = json.load(f)
        sentiment_df = pd.DataFrame(sentiment).set_index("market_id")

        df = df.join(sentiment_df[["sentiment_score", "sentiment_confidence"]], how="left")

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    X = df[FEATURE_COLS]
    p_yes = model.predict_proba(X)[:, 1]

    confidence = abs(p_yes - 0.5) * 2

    signals = {}
    for i, row in enumerate(df.itertuples()):
        signals[row.Index] = PredictionSignal(
            market_id=row.Index,
            p_yes=float(p_yes[i]),
            confidence=float(confidence[i]),
        )
    save_predictions(signals)
    return signals



def save_predictions(signals: dict[str, PredictionSignal]) -> None:
    import json
    
    payload = {cid: sig.model_dump(mode="json") for cid, sig in signals.items()}
    json.dump(payload, open(PREDICTIONS_PATH, "w"), indent=2, default=str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preds = predict()
    for cid, p in list(preds.items())[:5]:
        print(f"{cid:50s}  p_model={p.p_model:.3f}  conf={p.confidence:.3f}")
