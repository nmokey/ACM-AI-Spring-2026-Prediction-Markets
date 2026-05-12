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


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

# Load once at module level — not inside the function
_vader = SentimentIntensityAnalyzer()
_finbert = None  # Lazy-load on first use to avoid startup cost if not needed

def _get_finbert():
    global _finbert
    if _finbert is None:
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            device=-1  # CPU; set to 0 if you have a GPU
        )
    return _finbert

def score_text(
    text: str,
    use_finbert: bool = True,
    finbert_weight: float = 0.7,
) -> tuple[float, float]:
    """
    Score a piece of text for sentiment using VADER, FinBERT, or an ensemble.

    Returns:
        (sentiment_score, confidence)
        sentiment_score in [-1, 1]: positive → bullish/favorable
        confidence in [0, 1]: how non-neutral the signal is
    """
    text = text.strip()
    if not text:
        return 0.0, 0.0

    vader_scores = _vader.polarity_scores(text[:1000])
    vader_score = vader_scores["compound"]  
    vader_confidence = abs(vader_score)

    if not use_finbert:
        return vader_score, vader_confidence

    finbert = _get_finbert()
    result = finbert(text[:512])[0]  
    fb = {r["label"]: r["score"] for r in result}

    finbert_score = fb.get("positive", 0.0) - fb.get("negative", 0.0)
    finbert_confidence = 1.0 - fb.get("neutral", 0.0)  

    vader_weight = 1.0 - finbert_weight
    score = finbert_weight * finbert_score + vader_weight * vader_score
    confidence = finbert_weight * finbert_confidence + vader_weight * vader_confidence

    return float(score), float(confidence)


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
    client.poll_for_contracts([c["title"] for c in contracts])
    all_headlines = client.get_recent_headlines()

    signals: dict[str, SentimentSignal] = {}
    now = datetime.now(timezone.utc)

    if not all_headlines:
        for contract in contracts:
            signals[contract["contract_id"]] = SentimentSignal(
                contract_id=contract["contract_id"],
                timestamp=now,
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                n_relevant_headlines=0,
            )
        return signals

    # Pre-score all headline text once so it's cached for per-contract weighting
    headline_scores: list[tuple[float, float]] = [
        score_text(h["text"], use_finbert=use_finbert) for h in all_headlines
    ]

    # Embed all headlines once, then batch-score all contracts against that matrix
    from nlp.relevance import _get_model
    import numpy as np
    model = _get_model()
    headline_texts = [h["text"] for h in all_headlines]
    headline_embs = model.encode(headline_texts, batch_size=64, show_progress_bar=False)
    # Normalize for cosine similarity
    headline_norms = np.linalg.norm(headline_embs, axis=1, keepdims=True)
    headline_embs_norm = headline_embs / np.maximum(headline_norms, 1e-9)

    contract_titles = [c["title"] for c in contracts]
    contract_embs = model.encode(contract_titles, batch_size=64, show_progress_bar=False)
    contract_norms = np.linalg.norm(contract_embs, axis=1, keepdims=True)
    contract_embs_norm = contract_embs / np.maximum(contract_norms, 1e-9)

    # Shape: (n_contracts, n_headlines)
    sim_matrix = contract_embs_norm @ headline_embs_norm.T
    MIN_SCORE = 0.25
    TOP_K = 5

    for i, contract in enumerate(contracts):
        cid = contract["contract_id"]
        sims = sim_matrix[i]
        top_idx = np.where(sims >= MIN_SCORE)[0]

        if len(top_idx) == 0:
            signals[cid] = SentimentSignal(
                contract_id=cid,
                timestamp=now,
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                n_relevant_headlines=0,
            )
            continue

        top_idx = top_idx[np.argsort(-sims[top_idx])][:TOP_K]
        weights = sims[top_idx]
        total_weight = float(weights.sum())

        weighted_score = 0.0
        weighted_conf = 0.0
        for j, w in zip(top_idx, weights):
            s, c = headline_scores[j]
            weight = float(w) / total_weight
            weighted_score += weight * s
            weighted_conf += weight * c

        signals[cid] = SentimentSignal(
            contract_id=cid,
            timestamp=now,
            sentiment_score=max(-1.0, min(1.0, weighted_score)),
            sentiment_confidence=max(0.0, min(1.0, weighted_conf)),
            n_relevant_headlines=len(top_idx),
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
    import yaml
    logging.basicConfig(level=logging.INFO)

    with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
        _cfg = yaml.safe_load(f)

    features_path = Path(_cfg["data"]["features_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"Run data.engineer first — {features_path} not found")

    import pandas as pd
    df = pd.read_parquet(features_path)
    contracts = df[["contract_id", "title"]].dropna().to_dict("records")
    logger.info("Scoring sentiment for %d live contracts", len(contracts))

    signals = build_sentiment_signals(contracts)
    save_sentiment_signals(signals)

    nonzero = sum(1 for s in signals.values() if s.sentiment_score != 0.0)
    print(f"Saved {len(signals)} signals ({nonzero} non-zero) to {SENTIMENT_PATH}")
