"""
Parameterized stage detection for weekly and daily timeframes.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from stage_analysis.services.stage_detector import (
    FLAT_MA_THRESHOLD,
    NEAR_MA_THRESHOLD,
    STAGE_ACTIONS,
    STAGE_DESCRIPTIONS,
    _compute_ma_slope,
    _swing_structure,
)

WEEKLY_MA = 30
WEEKLY_SLOPE_LB = 8
WEEKLY_SWING_LB = 20

DAILY_MA = 150
DAILY_SLOPE_LB = 40
DAILY_SWING_LB = 100


def detect_stage_tf(
    df: pd.DataFrame,
    *,
    ma_period: int,
    slope_lookback: int,
    swing_lookback: int,
    label: str = "MA",
) -> tuple[int, list[str], dict[str, Any]]:
    """Weinstein stage rules with configurable MA period and lookbacks."""
    min_bars = ma_period + slope_lookback
    if len(df) < min_bars:
        raise ValueError(f"Insufficient {label} data ({len(df)} bars, need {min_bars})")

    work = df.copy()
    work["ma"] = work["close"].rolling(ma_period).mean()
    work = work.dropna(subset=["ma"])
    if work.empty:
        raise ValueError(f"Could not compute {label} moving average.")

    row = work.iloc[-1]
    price = float(row["close"])
    ma = float(row["ma"])
    price_vs_ma_pct = ((price - ma) / ma) * 100.0 if ma else 0.0
    ma_slope_pct = _compute_ma_slope(work["ma"], slope_lookback)

    swings = _swing_structure(work, swing_lookback)
    ma_rising = ma_slope_pct > FLAT_MA_THRESHOLD
    ma_falling = ma_slope_pct < -FLAT_MA_THRESHOLD
    ma_flat = not ma_rising and not ma_falling
    price_above_ma = price > ma
    price_below_ma = price < ma
    price_near_ma = abs(price_vs_ma_pct) <= NEAR_MA_THRESHOLD

    reasons: list[str] = []
    reasons.append(
        f"Price {'above' if price_above_ma else 'below'} {label} "
        f"({price_vs_ma_pct:+.1f}%)"
    )
    reasons.append(
        f"{label} slope: {ma_slope_pct:+.1f}% "
        f"({'rising' if ma_rising else 'falling' if ma_falling else 'flat'})"
    )

    structure_parts = []
    for key, label_s in (
        ("higher_highs", "higher highs"),
        ("higher_lows", "higher lows"),
        ("lower_highs", "lower highs"),
        ("lower_lows", "lower lows"),
    ):
        if swings[key]:
            structure_parts.append(label_s)
    reasons.append(
        f"Structure: {', '.join(structure_parts)}" if structure_parts else "Structure: inconclusive"
    )

    metrics = {
        "price": price,
        "ma": ma,
        "price_vs_ma_pct": price_vs_ma_pct,
        "ma_slope_pct": ma_slope_pct,
        "swings": swings,
        "ma_rising": ma_rising,
        "ma_falling": ma_falling,
        "ma_flat": ma_flat,
    }

    if price_below_ma and (ma_falling or swings["lower_lows"]):
        reasons.append("Stage 4: below falling MA, weak structure")
        return 4, reasons, metrics
    if price_above_ma and ma_rising and (swings["higher_highs"] or swings["higher_lows"]):
        reasons.append("Stage 2: above rising MA, bullish structure")
        return 2, reasons, metrics
    if price_near_ma and ma_flat and not swings["lower_highs"]:
        reasons.append("Stage 1: basing near flat MA")
        return 1, reasons, metrics
    if price_above_ma and (ma_flat or ma_falling or swings["lower_highs"]):
        reasons.append("Stage 3: topping, fading momentum")
        return 3, reasons, metrics
    if price_below_ma:
        reasons.append("Fallback Stage 4")
        return 4, reasons, metrics
    if ma_rising and price_above_ma:
        reasons.append("Fallback Stage 2")
        return 2, reasons, metrics
    if price_near_ma:
        reasons.append("Fallback Stage 1")
        return 1, reasons, metrics
    reasons.append("Fallback Stage 3")
    return 3, reasons, metrics


def detect_weekly_stage(df: pd.DataFrame) -> tuple[int, list[str], dict[str, Any]]:
    return detect_stage_tf(
        df,
        ma_period=WEEKLY_MA,
        slope_lookback=WEEKLY_SLOPE_LB,
        swing_lookback=WEEKLY_SWING_LB,
        label="30-week MA",
    )


def detect_daily_stage(df: pd.DataFrame) -> tuple[int, list[str], dict[str, Any]]:
    return detect_stage_tf(
        df,
        ma_period=DAILY_MA,
        slope_lookback=DAILY_SLOPE_LB,
        swing_lookback=DAILY_SWING_LB,
        label="150-day MA",
    )


def stage_action(stage: int) -> str:
    return STAGE_ACTIONS.get(stage, "")


def stage_description(stage: int) -> str:
    return STAGE_DESCRIPTIONS.get(stage, "")