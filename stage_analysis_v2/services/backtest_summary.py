"""
Simple forward-return summary for historical high-quality Stage 2 setups.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stage_analysis.services.stage_detector import daily_to_weekly
from stage_analysis_v2.services.stage_engine import detect_weekly_stage
from trading.services.market_data import load_price_dataframe


FORWARD_WEEKS = 13
MIN_QUALITY_PROXY = 4  # min met conditions out of 6 for historical label


@dataclass
class BacktestSummary:
    sample_size: int
    avg_return_pct: float
    win_rate_pct: float
    median_return_pct: float
    note: str


def _historical_setup_score(weekly: pd.DataFrame, idx: int) -> int:
    """Lightweight proxy for quality at a historical bar."""
    window = weekly.iloc[: idx + 1]
    if len(window) < 40:
        return 0
    try:
        stage, _, metrics = detect_weekly_stage(window)
    except ValueError:
        return 0
    if stage != 2:
        return 0
    score = 0
    if metrics.get("ma_rising"):
        score += 1
    if metrics.get("price_vs_ma_pct", 0) > 3:
        score += 1
    if metrics.get("swings", {}).get("higher_lows"):
        score += 1
    if metrics.get("swings", {}).get("higher_highs"):
        score += 1
    vol = weekly.iloc[idx]["volume"]
    vol_avg = weekly.iloc[max(0, idx - 20):idx]["volume"].mean()
    if vol_avg and vol / vol_avg >= 1.5:
        score += 1
    if metrics.get("price_vs_ma_pct", 0) > 0:
        score += 1
    return score


def compute_backtest_summary(symbol: str) -> BacktestSummary:
    from trading.services.nse_price_sync import nse_symbol_from_ticker
    daily = load_price_dataframe(nse_symbol_from_ticker(symbol))
    if daily.empty or len(daily) < 300:
        return BacktestSummary(0, 0.0, 0.0, 0.0, "Insufficient history for backtest.")

    weekly = daily_to_weekly(daily)
    returns: list[float] = []

    for i in range(40, len(weekly) - FORWARD_WEEKS):
        if _historical_setup_score(weekly, i) < MIN_QUALITY_PROXY:
            continue
        entry = float(weekly.iloc[i]["close"])
        exit_p = float(weekly.iloc[i + FORWARD_WEEKS]["close"])
        if entry > 0:
            returns.append((exit_p - entry) / entry * 100)

    if not returns:
        return BacktestSummary(0, 0.0, 0.0, 0.0, "No similar historical setups found.")

    wins = sum(1 for r in returns if r > 0)
    return BacktestSummary(
        sample_size=len(returns),
        avg_return_pct=round(sum(returns) / len(returns), 2),
        win_rate_pct=round(wins / len(returns) * 100, 1),
        median_return_pct=round(float(pd.Series(returns).median()), 2),
        note=f"Based on {len(returns)} similar Stage 2 setups over {FORWARD_WEEKS}-week hold.",
    )