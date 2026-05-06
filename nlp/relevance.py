"""
nlp/relevance.py
──────────────────
Headline relevance scorer.

Given a Kalshi contract title and a list of headlines, return the headlines
most semantically related to that contract using cosine similarity between
sentence embeddings.

Team 2 — Modeling & Intelligence (NLP half) — implement score_relevance().

Model: sentence-transformers/all-MiniLM-L6-v2
Docs:  https://www.sbert.net/docs/pretrained_models.html
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the sentence embedding model (only on first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded all-MiniLM-L6-v2")
    return _model


def score_relevance(
    contract_title: str,
    headlines: list[dict[str, Any]],
    top_k: int = 5,
    min_score: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Score and filter headlines by relevance to a contract title.

    Args:
        contract_title: e.g. "Will BTC close above $90k on April 20?"
        headlines:      list of dicts, each with at least {"id": str, "text": str}
        top_k:          max headlines to return
        min_score:      minimum cosine similarity to include (0.0–1.0)

    Returns:
        Filtered, ranked list of headline dicts with an added "relevance_score" field.

    TODO (Week 3):
        1. Return [] if headlines is empty
        2. Load the model with _get_model()
        3. Encode contract_title into a query embedding vector
        4. Encode all headline["text"] values into a matrix of embeddings
        5. Compute cosine similarity between query and each headline embedding
           Hint: cosine_sim = dot(query, headline) / (|query| * |headline|)
           Or use: scores = model.similarity(query_emb, headline_embs)
        6. Filter to scores >= min_score, sort descending, return top_k
        7. Add "relevance_score": float to each returned headline dict
    """

    if not headlines:
        return []
    model = _get_model()

    query_embedding = model.encode(contract_title)
    headline_embeddings = model.encode([h["text"] for h in headlines])

    scores = model.similarity(query_embedding, headline_embeddings)[0]

    for h, s in zip(headlines, scores):
        h["relevance_score"] = float(s)

    filtered = [h for h in headlines if h["relevance_score"] >= min_score]
    filtered.sort(key=lambda x: x["relevance_score"], reverse=True)
    return filtered[:top_k]
