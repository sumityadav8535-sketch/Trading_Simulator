"""
Stage 2 Quality Score (0-100) — composite confluence rating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stage_analysis_v2.services.breakout_detector import BreakoutResult
from stage_analysis_v2.services.relative_strength import RSResult


@dataclass
class QualityScore:
    total: int
    breakdown: dict[str, int]
    reasons: list[str]


def _pullback_score(weekly: pd.DataFrame, ma: float) -> tuple[int, str]:
    """Healthy pullback: shallow retracement, holding MA."""
    if len(weekly) < 10:
        return 5, "Limited pullback data"
    recent = weekly.tail(8)
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    close = float(recent["close"].iloc[-1])
    if high <= 0:
        return 5, "N/A"
    depth = (high - low) / high * 100
    holding_ma = close > ma * 0.98
    if depth <= 12 and holding_ma:
        return 15, f"Shallow pullback ({depth:.1f}%), holding MA"
    if depth <= 20 and holding_ma:
        return 10, f"Moderate pullback ({depth:.1f}%), above MA"
    if holding_ma:
        return 6, f"Deep pullback ({depth:.1f}%) but above MA"
    return 2, f"Deep pullback ({depth:.1f}%), below MA"


def _sector_score(sector: str, sector_stage_avg: float | None) -> tuple[int, str]:
    if not sector or sector_stage_avg is None:
        return 5, "Sector data unavailable"
    if sector_stage_avg >= 2.2:
        return 10, f"Strong sector ({sector}, avg stage {sector_stage_avg:.1f})"
    if sector_stage_avg >= 1.8:
        return 7, f"Neutral-positive sector ({sector})"
    return 3, f"Weak sector ({sector}, avg stage {sector_stage_avg:.1f})"


def compute_quality_score(
    weekly_stage: int,
    daily_stage: int,
    weekly_metrics: dict[str, Any],
    breakout: BreakoutResult,
    rs: RSResult,
    weekly: pd.DataFrame,
    sector: str = "",
    sector_stage_avg: float | None = None,
    market_favorable: bool = True,
) -> QualityScore:
    breakdown: dict[str, int] = {}
    reasons: list[str] = []

    if weekly_stage != 2:
        return QualityScore(0, {"note": 0}, ["Not in weekly Stage 2 — quality score applies to Stage 2 only"])

    b_score = 0
    if breakout.breakout_type == "clean":
        b_score = 20
        reasons.append("Clean breakout from Stage 1 base (+20)")
    elif breakout.breakout_type == "weak":
        b_score = 10
        reasons.append("Weak breakout (+10)")
    else:
        b_score = 5
        reasons.append("No recent breakout, trend continuation (+5)")
    breakdown["breakout_base"] = b_score

    v_score = 0
    if breakout.volume_ratio >= 2.0:
        v_score = 15
        reasons.append(f"Strong volume surge {breakout.volume_ratio}x (+15)")
    elif breakout.volume_ratio >= 1.5:
        v_score = 12
        reasons.append(f"Volume surge {breakout.volume_ratio}x (+12)")
    elif breakout.volume_ratio >= 1.2:
        v_score = 6
        reasons.append(f"Moderate volume {breakout.volume_ratio}x (+6)")
    breakdown["volume"] = v_score

    ma = weekly_metrics.get("ma", 0)
    ma_score = 0
    if weekly_metrics.get("ma_rising") and weekly_metrics.get("price_vs_ma_pct", 0) > 2:
        ma_score = 20
        reasons.append("Price firmly above rising 30-week MA (+20)")
    elif weekly_metrics.get("ma_rising"):
        ma_score = 14
        reasons.append("Above rising 30-week MA (+14)")
    else:
        ma_score = 6
    breakdown["ma_trend"] = ma_score

    rs_score = 0
    if rs.rating >= 70 and rs.improving:
        rs_score = 20
        reasons.append(f"Strong improving RS {rs.rating:.0f} (+20)")
    elif rs.rating >= 60:
        rs_score = 14
        reasons.append(f"Good RS {rs.rating:.0f} (+14)")
    elif rs.rating >= 50:
        rs_score = 8
        reasons.append(f"Neutral RS {rs.rating:.0f} (+8)")
    else:
        rs_score = 3
        reasons.append(f"Weak RS {rs.rating:.0f} (+3)")
    breakdown["relative_strength"] = rs_score

    pb_score, pb_reason = _pullback_score(weekly, ma)
    breakdown["pullback"] = pb_score
    reasons.append(f"Pullback: {pb_reason} (+{pb_score})")

    sec_score, sec_reason = _sector_score(sector, sector_stage_avg)
    breakdown["sector"] = sec_score
    reasons.append(f"Sector: {sec_reason} (+{sec_score})")

    mtf_bonus = 0
    if daily_stage == 2:
        mtf_bonus = 8
        reasons.append("Daily confirms Stage 2 (+8)")
    elif daily_stage == 1:
        mtf_bonus = 4
        reasons.append("Daily Stage 1 — early confirmation (+4)")
    breakdown["daily_confirm"] = mtf_bonus

    market_penalty = 0
    if not market_favorable:
        market_penalty = -10
        reasons.append("Unfavorable market context (-10)")
    breakdown["market_context"] = market_penalty

    total = sum(breakdown.values())
    total = max(0, min(100, total))

    return QualityScore(total=total, breakdown=breakdown, reasons=reasons)