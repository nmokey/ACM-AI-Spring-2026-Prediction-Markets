"""
data/schema.py
────────────────
Pydantic data contracts shared across all teams.

DO NOT modify without a team-wide PR — this is the cross-team interface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal  # used by TradeRecord

from pydantic import BaseModel, Field


class MarketFeatures(BaseModel):
    """Team 1 → Team 2: one row of live_features.parquet."""
    contract_id: str
    title: str
    market_category: str | None
    market_price: float | None = Field(default=None, ge=0.0, le=1.0)
    volume_24h: float = Field(ge=0.0)
    open_interest: float = Field(ge=0.0)
    days_to_resolution: float | None
    btc_price: float | None = None
    btc_change_1h: float | None = None
    btc_change_6h: float | None = None
    eth_price: float | None = None
    eth_change_1h: float | None = None
    eth_change_6h: float | None = None
    precip_prob_new_york: float | None = None
    precip_prob_los_angeles: float | None = None
    precip_prob_chicago: float | None = None
    fetched_at: str


class SentimentSignal(BaseModel):
    """Internal Team 2 artifact: per-contract sentiment from NLP pipeline."""
    contract_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(ge=0.0, le=1.0)
    n_relevant_headlines: int = Field(ge=0)


class PredictionSignal(BaseModel):
    """Team 2 → Team 3: calibrated probability for a single contract."""
    contract_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    p_model: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class TradeRecord(BaseModel):
    """Team 3 internal: one submitted or dry-run order."""
    contract_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    side: Literal["YES", "NO"]
    size: int = Field(ge=0)
    limit_price: int = Field(ge=0, le=100, description="Kalshi cents (0–100)")
    p_model: float = Field(ge=0.0, le=1.0)
    market_price: float = Field(ge=0.0, le=1.0)
    edge: float
    mode: Literal["dry_run", "live"]
