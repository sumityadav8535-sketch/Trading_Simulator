"""
Smart entry suggestions with stop-loss and risk-reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from stage_analysis_v2.services.breakout_detector import BreakoutResult


@dataclass
class EntryPlan:
    primary_entry: float
    pullback_entry: float
    stop_loss: float
    target_price: float
    risk_reward: float
    primary_label: str
    pullback_label: str
    action: str


def compute_entries(
    price: float,
    ma_30w: float,
    weekly_stage: int,
    breakout: BreakoutResult,
    quality_score: int,
) -> EntryPlan:
    primary = breakout.breakout_level if breakout.breakout_level > 0 else price
    pullback = round(ma_30w * 1.01, 2)

    if weekly_stage == 2:
        stop = round(ma_30w * 0.95, 2)
        target = round(price + (price - stop) * 2.5, 2)
    elif weekly_stage == 1:
        stop = round(min(ma_30w, primary) * 0.96, 2)
        target = round(primary + (primary - stop) * 2.0, 2)
    else:
        stop = round(price * 0.93, 2)
        target = round(price * 1.05, 2)

    risk = primary - stop
    reward = target - primary
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    if weekly_stage == 2 and quality_score >= 60:
        action = "Buy on pullback to 30-week MA or breakout continuation"
    elif weekly_stage == 2:
        action = "Watch — wait for higher quality setup or pullback entry"
    elif weekly_stage == 1:
        action = "Prepare — buy on Stage 2 breakout above base"
    else:
        action = "Avoid new long entries"

    return EntryPlan(
        primary_entry=round(primary, 2),
        pullback_entry=pullback,
        stop_loss=stop,
        target_price=target,
        risk_reward=rr,
        primary_label="Breakout entry above base high",
        pullback_label="High-probability pullback to rising 30-week MA",
        action=action,
    )