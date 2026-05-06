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

from data.schema import SentimentSignal
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
    """
    if use_finbert:
        from transformers import pipeline
        pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        result = pipe(text[:512])[0]  # list of {label, score}
        scores = {r["label"]: r["score"] for r in result}
        score = scores.get("positive", 0.0) - scores.get("negative", 0.0)
        confidence = 1 - scores.get("neutral", 0.0)
    else:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        score = scores["compound"]
        confidence = abs(scores["compound"])
    return score, confidence


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
    
    client = NewsClient()
    all_headlines = client.get_recent_headlines()

    signals: dict[str, SentimentSignal] = {}
    now = datetime.now(timezone.utc)

    for contract in contracts:
        cid = contract["contract_id"]
        relevant = score_relevance(contract["title"], all_headlines)

        if not relevant:
            signals[cid] = SentimentSignal(
                contract_id=cid,
                timestamp=now,
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                n_relevant_headlines=0,
            )
            continue

        total_weight = sum(h["relevance_score"] for h in relevant)
        weighted_score = 0.0
        weighted_conf = 0.0
        for h in relevant:
            w = h["relevance_score"] / total_weight
            s, c = score_text(h["text"], use_finbert=use_finbert)
            weighted_score += w * s
            weighted_conf += w * c

        signals[cid] = SentimentSignal(
            contract_id=cid,
            timestamp=now,
            sentiment_score=max(-1.0, min(1.0, weighted_score)),
            sentiment_confidence=max(0.0, min(1.0, weighted_conf)),
            n_relevant_headlines=len(relevant),
        )

    return signals


def save_sentiment_signals(signals: dict[str, SentimentSignal]) -> None:
    """
    Serialize signals dict to nlp/sentiment.json (internal Team 2 cache).

    TODO (Week 3):
        - Call sig.model_dump(mode="json") on each signal
        - json.dump the result to SENTIMENT_PATH
    """
    out = {cid: sig.model_dump(mode="json") for cid, sig in signals.items()}
    SENTIMENT_PATH.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test — replace with a real contract title from Kalshi
    test_contracts = [
        {"contract_id": "TEST-001", "title": "Will BTC close above $90k today?"},
    ]
    signals = build_sentiment_signals(test_contracts)
    for cid, sig in signals.items():
        print(f"{cid}: score={sig.sentiment_score:.3f}  conf={sig.sentiment_confidence:.3f}  n={sig.n_relevant_headlines}")
