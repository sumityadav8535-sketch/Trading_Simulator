"""
Position sizing based on capital and max risk % per trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class PositionSizeResult:
    quantity: int
    risk_amount: float
    capital_deployed: float
    risk_per_share: float


def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
) -> PositionSizeResult:
    """
    Shares = (capital * risk_pct/100) / (entry - stop_loss)
    Returns 0 quantity if stop is at/above entry.
    """
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return PositionSizeResult(0, 0.0, 0.0, risk_per_share)

    risk_amount = capital * (risk_pct / 100.0)
    qty = int(risk_amount / risk_per_share)
    capital_deployed = qty * entry
    return PositionSizeResult(qty, risk_amount, capital_deployed, risk_per_share)


def decimal_or_none(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(round(value, 2)))