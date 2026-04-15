"""
nlp/sentiment.py
──────────────────
Sentiment scoring pipeline — Team 2 (Modeling & Intelligence) NLP deliverable.

Takes relevant headlines (from nlp/relevance.py) and produces a
per-contract SentimentSignal. This is an internal Team 2 artifact —
sentiment flows directly into models/predict.py and is NOT a cross-team contract.

Team 2 — Modeling & Intelligence (NLP half) — implement all functions marked with TODO.

Models to try:
    VADER (rule-based, zero setup):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        compound score is already in [-1, 1]

    FinBERT (transformer, finance-tuned):
        from transformers import pipeline
        pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        result = pipe(text[:512])  # returns list of {label, score}
        labels: "positive", "negative", "neutral"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from data.features.schema import SentimentSignal
from nlp.news_client import NewsClient
from nlp.relevance import score_relevance

logger = logging.getLogger(__name__)

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

SENTIMENT_PATH = Path(CONFIG["data"]["sentiment_path"])
SENTIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)


def score_text(text: str, use_finbert: bool = True) -> tuple[float, float]:
    """
    Score a single piece of text for sentiment.

    Returns:
        (sentiment_score, confidence)
        score in [-1, 1]:  positive → bullish/favorable, negative → bearish/unfavorable
        confidence in [0, 1]

    TODO (Week 3 — start with VADER, upgrade to FinBERT in Week 4):

    VADER approach (easier):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores["compound"], abs(scores["compound"])

    FinBERT approach (more accurate for financial text):
        result = pipe(text[:512])[0]   # list of {label, score}
        Convert to a single [-1, 1] score:
            score = P(positive) - P(negative)
            confidence = 1 - P(neutral)
    """
    raise NotImplementedError


def build_sentiment_signals(
    contracts: list[dict[str, Any]],
    use_finbert: bool = False,
) -> dict[str, SentimentSignal]:
    """
    For each contract, fetch relevant headlines and produce a SentimentSignal.

    Args:
        contracts:   list of dicts with "contract_id" and "title" keys
        use_finbert: if True, use FinBERT; if False, use VADER (start here)

    Returns:
        Dict mapping contract_id → SentimentSignal

    TODO (Week 3):
        1. Instantiate NewsClient() and call get_recent_headlines() to load from SQLite
        2. For each contract:
               a. Call score_relevance(title, all_headlines) to get relevant headlines
               b. If no relevant headlines: return a neutral SentimentSignal (score=0.0, conf=0.0, n=0)
               c. For each relevant headline, call score_text(headline["text"])
               d. Aggregate scores across headlines — think about how to weight them
                  (hint: weight by relevance_score so more relevant headlines matter more)
               e. Clamp final score to [-1, 1] and confidence to [0, 1]
               f. Build and store a SentimentSignal
        3. Return the dict
    """
    raise NotImplementedError


def save_sentiment_signals(signals: dict[str, SentimentSignal]) -> None:
    """
    Serialize signals dict to nlp/sentiment.json (internal Team 2 cache).

    TODO (Week 3):
        - Call sig.model_dump(mode="json") on each signal
        - json.dump the result to SENTIMENT_PATH
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test — replace with a real contract title from Kalshi
    test_contracts = [
        {"contract_id": "TEST-001", "title": "Will BTC close above $90k today?"},
    ]
    signals = build_sentiment_signals(test_contracts)
    for cid, sig in signals.items():
        print(f"{cid}: score={sig.sentiment_score:.3f}  conf={sig.sentiment_confidence:.3f}  n={sig.n_relevant_headlines}")
