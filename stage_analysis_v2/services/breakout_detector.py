"""
Clean vs Weak breakout detection on weekly bars.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BASE_LOOKBACK = 20
VOL_SURGE_MIN = 1.5
FOLLOW_THROUGH_DAYS = 3


@dataclass
class BreakoutResult:
    breakout_type: str
    breakout_level: float
    volume_ratio: float
    close_strength: float
    follow_through: bool
    rejection: bool
    reasons: list[str]


def detect_breakout(weekly: pd.DataFrame, daily: pd.DataFrame) -> BreakoutResult:
    """
    Classify breakout quality using volume, close strength, follow-through, rejection.
    """
    if len(weekly) < BASE_LOOKBACK + 2:
        return BreakoutResult("none", 0.0, 0.0, 0.0, False, False, ["Insufficient data"])

    base = weekly.iloc[-(BASE_LOOKBACK + 1):-1]
    current = weekly.iloc[-1]
    base_high = float(base["high"].max())
    base_vol_avg = float(base["volume"].mean()) or 1.0
    vol_ratio = float(current["volume"]) / base_vol_avg

    price = float(current["close"])
    high = float(current["high"])
    low = float(current["low"])
    range_size = high - low if high > low else 0.01
    close_strength = (price - low) / range_size

    breakout_level = base_high
    is_breakout = price > base_high * 0.998

    follow_through = False
    rejection = False
    if not daily.empty and len(daily) >= FOLLOW_THROUGH_DAYS + 1:
        recent = daily.tail(FOLLOW_THROUGH_DAYS + 1)
        post = recent.iloc[1:]
        follow_through = float(post["close"].iloc[-1]) > float(recent.iloc[0]["close"])
        rejection = float(post["low"].min()) < breakout_level * 0.97

    reasons: list[str] = []
    if not is_breakout:
        reasons.append("Price has not broken above base high")
        return BreakoutResult("none", breakout_level, vol_ratio, close_strength, follow_through, rejection, reasons)

    reasons.append(f"Breakout above base high {breakout_level:.2f}")
    reasons.append(f"Volume {vol_ratio:.1f}x average ({'OK' if vol_ratio >= VOL_SURGE_MIN else 'weak'})")
    reasons.append(f"Close strength {close_strength:.0%} of candle range")

    score = 0
    if vol_ratio >= VOL_SURGE_MIN:
        score += 2
    elif vol_ratio >= 1.2:
        score += 1
    if close_strength >= 0.65:
        score += 2
    elif close_strength >= 0.5:
        score += 1
    if follow_through:
        score += 2
        reasons.append("Follow-through confirmed on daily")
    else:
        reasons.append("No clear daily follow-through yet")
    if rejection:
        score -= 2
        reasons.append("Immediate rejection below breakout — bearish")
    else:
        reasons.append("No immediate rejection")

    if score >= 5:
        btype = "clean"
        reasons.append("Classified: Clean Breakout")
    elif score >= 2:
        btype = "weak"
        reasons.append("Classified: Weak Breakout")
    else:
        btype = "weak"
        reasons.append("Classified: Weak / failed breakout")

    return BreakoutResult(
        breakout_type=btype,
        breakout_level=round(breakout_level, 2),
        volume_ratio=round(vol_ratio, 2),
        close_strength=round(close_strength, 2),
        follow_through=follow_through,
        rejection=rejection,
        reasons=reasons,
    )