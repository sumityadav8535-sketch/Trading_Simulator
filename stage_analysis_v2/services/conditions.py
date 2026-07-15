"""
Transparent condition checklist — shows which confluence factors are met.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stage_analysis_v2.services.breakout_detector import BreakoutResult
from stage_analysis_v2.services.relative_strength import RSResult

STRONG_RS_THRESHOLD = 60


@dataclass
class ConditionCheck:
    name: str
    met: bool
    detail: str
    weight: str = ""


def build_condition_checks(
    weekly_stage: int,
    daily_stage: int,
    weekly_metrics: dict[str, Any],
    breakout: BreakoutResult,
    rs: RSResult,
    quality_score: int,
    market_favorable: bool,
    sector: str = "",
) -> list[ConditionCheck]:
    checks: list[ConditionCheck] = []

    checks.append(ConditionCheck(
        "Weekly Stage 2 (Advancing)",
        weekly_stage == 2,
        f"Weekly stage is {weekly_stage}" + (" ✓" if weekly_stage == 2 else " — required for high-probability long"),
        "Required",
    ))
    checks.append(ConditionCheck(
        "Daily Confirms Uptrend",
        daily_stage in (1, 2),
        f"Daily stage {daily_stage} (150-day MA framework)",
        "Confirm",
    ))
    checks.append(ConditionCheck(
        "Price Above Rising 30W MA",
        weekly_metrics.get("ma_rising", False) and weekly_metrics.get("price_vs_ma_pct", 0) > 0,
        f"vs MA: {weekly_metrics.get('price_vs_ma_pct', 0):+.1f}%, slope {weekly_metrics.get('ma_slope_pct', 0):+.1f}%",
        "Core",
    ))
    checks.append(ConditionCheck(
        "Strong Relative Strength",
        rs.rating >= STRONG_RS_THRESHOLD and (rs.improving or rs.rating >= 65),
        f"RS {rs.rating:.0f} ({rs.trend}), vs Nifty {rs.vs_benchmark_pct:+.1f}%",
        "Core",
    ))
    checks.append(ConditionCheck(
        "Clean Breakout",
        breakout.breakout_type == "clean",
        breakout.reasons[-1] if breakout.reasons else "No breakout",
        "Core",
    ))
    checks.append(ConditionCheck(
        "Volume Surge (≥1.5x)",
        breakout.volume_ratio >= 1.5,
        f"Volume {breakout.volume_ratio:.1f}x average",
        "Volume",
    ))
    checks.append(ConditionCheck(
        "No Immediate Rejection",
        not breakout.rejection,
        "Rejected below breakout" if breakout.rejection else "Holding above breakout zone",
        "Breakout",
    ))
    checks.append(ConditionCheck(
        "Bullish Structure (HH/HL)",
        weekly_metrics.get("swings", {}).get("higher_highs") or weekly_metrics.get("swings", {}).get("higher_lows"),
        "Higher highs/lows in recent swings",
        "Structure",
    ))
    checks.append(ConditionCheck(
        "Market Context Favorable",
        market_favorable,
        "Nifty 50 in Stage 1 or 2" if market_favorable else "Nifty in Stage 3/4 — caution",
        "Market",
    ))
    checks.append(ConditionCheck(
        "Quality Score ≥ 75",
        quality_score >= 75,
        f"Score: {quality_score}/100",
        "Score",
    ))
    if sector:
        checks.append(ConditionCheck(
            f"Sector: {sector}",
            True,
            "Sector strength factored into quality score",
            "Sector",
        ))

    return checks


def why_scored_high(checks: list[ConditionCheck], score_reasons: list[str]) -> list[str]:
    """Human-readable bullets for high-scoring stocks."""
    bullets = [c.detail for c in checks if c.met and c.weight in ("Core", "Required", "Volume")]
    bullets.extend(score_reasons[:6])
    return bullets[:10]