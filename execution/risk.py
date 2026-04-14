"""
execution/risk.py
───────────────────
Risk management filters.

All trade candidates must pass through these checks before being sized
or submitted. If any check fails, the trade is skipped and logged.

Team 3 Execution owns this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.yaml"
with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)["trading"]


@dataclass
class RiskCheckResult:
    passed: bool
    reason: str = ""


def check_trade(
    p_model: float,
    market_price: float,
    confidence: float,
    open_positions: dict[str, float],
    account_balance: float,
) -> RiskCheckResult:
    """
    Run all pre-trade risk checks.

    Args:
        p_model:         Model's probability estimate
        market_price:    Current market YES price (0–1)
        confidence:      Model confidence (0–1)
        open_positions:  Dict of {contract_id: dollars_at_risk} for open trades
        account_balance: Current account balance

    Returns:
        RiskCheckResult with passed=True if all checks clear.
    """
    edge = abs(p_model - market_price)

    # 1. Minimum edge
    if edge < CONFIG["min_edge"]:
        return RiskCheckResult(False, f"Edge {edge:.3f} < min_edge {CONFIG['min_edge']}")

    # 2. Minimum confidence
    if confidence < CONFIG["min_confidence"]:
        return RiskCheckResult(False, f"Confidence {confidence:.3f} < min_confidence {CONFIG['min_confidence']}")

    # 3. Total exposure cap
    total_exposure = sum(open_positions.values())
    max_exposure = CONFIG["max_total_exposure_pct"] * account_balance
    if total_exposure >= max_exposure:
        return RiskCheckResult(False, f"Total exposure ${total_exposure:.2f} at cap (${max_exposure:.2f})")

    return RiskCheckResult(True)
