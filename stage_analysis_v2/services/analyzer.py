"""
Stage Analysis 2.0 — full confluence pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from stage_analysis.services.stage_detector import MA_PERIOD, normalize_ticker
from stage_analysis_v2.services.backtest_summary import BacktestSummary, compute_backtest_summary
from stage_analysis_v2.services.breakout_detector import detect_breakout
from stage_analysis_v2.services.conditions import (
    ConditionCheck,
    build_condition_checks,
    why_scored_high,
)
from stage_analysis_v2.services.data_loader import load_daily, load_weekly, stock_meta
from stage_analysis_v2.services.entry_suggestions import compute_entries
from stage_analysis_v2.services.indicators import DAILY_MA_PERIOD, add_daily_indicators, add_weekly_indicators
from stage_analysis_v2.services.market_context import MarketContext, analyze_market_context
from stage_analysis_v2.services.quality_score import QualityScore, compute_quality_score
from stage_analysis_v2.services.relative_strength import compute_relative_strength
from stage_analysis_v2.services.stage_engine import (
    detect_daily_stage,
    detect_weekly_stage,
    stage_action,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.nse_price_sync import yfinance_ticker


@dataclass
class V2AnalysisResult:
    ticker: str
    company_name: str
    sector: str

    weekly_stage: int
    daily_stage: int
    weekly_reasons: list[str]
    daily_reasons: list[str]
    weekly_metrics: dict[str, Any]
    daily_metrics: dict[str, Any]

    quality_score: int
    score_breakdown: dict[str, int]
    score_reasons: list[str]
    quality_tier: str

    rs_rating: float
    rs_trend: str
    rs_vs_benchmark_pct: float
    rs_improving: bool
    benchmark: str

    breakout_type: str
    breakout_details: dict[str, Any]

    market_context: MarketContext
    market_highlight: bool

    current_price: float
    ma_30w: float
    ma_150d: float

    primary_entry: float
    pullback_entry: float
    stop_loss: float
    target_price: float
    risk_reward: float
    suggested_action: str

    condition_checks: list[ConditionCheck] = field(default_factory=list)
    why_high: list[str] = field(default_factory=list)
    backtest: BacktestSummary | None = None
    chart_payload: dict[str, Any] = field(default_factory=dict)


def _quality_tier(weekly_stage: int, quality_score: int) -> str:
    if weekly_stage == 2 and quality_score >= 75:
        return "high"
    if weekly_stage == 2 and quality_score >= 50:
        return "average"
    if weekly_stage == 1:
        return "base"
    if weekly_stage == 3:
        return "top"
    return "decline"


def _serialize_ohlc(df: pd.DataFrame, ma_col: str, tail: int) -> dict[str, Any]:
    plot = df.tail(tail).copy()
    ohlc = [
        {
            "x": idx.strftime("%Y-%m-%d"),
            "o": round(float(r["open"]), 2),
            "h": round(float(r["high"]), 2),
            "l": round(float(r["low"]), 2),
            "c": round(float(r["close"]), 2),
        }
        for idx, r in plot.iterrows()
    ]
    ma_vals = [round(float(v), 2) if pd.notna(v) else None for v in plot[ma_col]]
    return {
        "ohlc": ohlc,
        "ma": ma_vals,
        "ma_label": ma_col,
        "volume": [int(v) for v in plot["volume"]],
    }


def _build_chart_payload(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    rs_line: list[dict],
    weekly_stage: int,
    daily_stage: int,
) -> dict[str, Any]:
    stage_colors = {1: "#60a5fa", 2: "#22c55e", 3: "#f59e0b", 4: "#ef4444"}
    w = add_weekly_indicators(weekly)
    d = add_daily_indicators(daily) if not daily.empty else pd.DataFrame()

    payload: dict[str, Any] = {
        "weekly": _serialize_ohlc(w, "ma_30w", 104),
        "rs_line": rs_line,
        "weekly_stage": weekly_stage,
        "daily_stage": daily_stage,
        "stage_color": stage_colors.get(weekly_stage, "#94a3b8"),
    }
    if not d.empty:
        payload["daily"] = _serialize_ohlc(d, "ma_150d", 120)
    return payload


def analyze_ticker_v2(
    ticker: str,
    *,
    market: Optional[MarketContext] = None,
    sector_stage_avg: float | None = None,
    benchmark: str = NIFTY50_SYMBOL,
    include_backtest: bool = True,
) -> V2AnalysisResult:
    """Run full Stage Analysis 2.0 pipeline for one ticker."""
    symbol = normalize_ticker(ticker)
    if not symbol.endswith(".NS") and not symbol.startswith("^"):
        base = symbol.replace(".NS", "")
        if not any(c in symbol for c in (".", "^")):
            symbol = yfinance_ticker(base)

    company_name, sector = stock_meta(symbol)
    weekly = add_weekly_indicators(load_weekly(symbol))
    daily = add_daily_indicators(load_daily(symbol))
    bench_weekly = load_weekly(benchmark)

    if weekly.empty:
        raise ValueError(f"No weekly data for {symbol}")

    w_stage, w_reasons, w_metrics = detect_weekly_stage(weekly)
    d_stage, d_reasons, d_metrics = detect_daily_stage(daily) if not daily.empty else (0, ["No daily data"], {})

    rs = compute_relative_strength(weekly, bench_weekly, benchmark)
    breakout = detect_breakout(weekly, daily)
    market_ctx = market or analyze_market_context(benchmark)

    quality = compute_quality_score(
        weekly_stage=w_stage,
        daily_stage=d_stage,
        weekly_metrics=w_metrics,
        breakout=breakout,
        rs=rs,
        weekly=weekly,
        sector=sector,
        sector_stage_avg=sector_stage_avg,
        market_favorable=market_ctx.favorable,
    )

    entries = compute_entries(
        price=w_metrics["price"],
        ma_30w=w_metrics["ma"],
        weekly_stage=w_stage,
        breakout=breakout,
        quality_score=quality.total,
    )

    tier = _quality_tier(w_stage, quality.total)
    highlight = w_stage == 2 and market_ctx.favorable and quality.total >= 75

    checks = build_condition_checks(
        w_stage, d_stage, w_metrics, breakout, rs,
        quality.total, market_ctx.favorable, sector,
    )
    why = why_scored_high(checks, quality.reasons) if quality.total >= 50 else []

    backtest = None
    if include_backtest and w_stage == 2:
        backtest = compute_backtest_summary(symbol)

    return V2AnalysisResult(
        ticker=symbol,
        company_name=company_name,
        sector=sector,
        weekly_stage=w_stage,
        daily_stage=d_stage,
        weekly_reasons=w_reasons,
        daily_reasons=d_reasons,
        weekly_metrics=w_metrics,
        daily_metrics=d_metrics,
        quality_score=quality.total,
        score_breakdown=quality.breakdown,
        score_reasons=quality.reasons,
        quality_tier=tier,
        rs_rating=rs.rating,
        rs_trend=rs.trend,
        rs_vs_benchmark_pct=rs.vs_benchmark_pct,
        rs_improving=rs.improving,
        benchmark=benchmark,
        breakout_type=breakout.breakout_type,
        breakout_details={
            "level": breakout.breakout_level,
            "volume_ratio": breakout.volume_ratio,
            "close_strength": breakout.close_strength,
            "follow_through": breakout.follow_through,
            "rejection": breakout.rejection,
            "reasons": breakout.reasons,
        },
        market_context=market_ctx,
        market_highlight=highlight,
        current_price=w_metrics["price"],
        ma_30w=w_metrics["ma"],
        ma_150d=d_metrics.get("ma", 0.0),
        primary_entry=entries.primary_entry,
        pullback_entry=entries.pullback_entry,
        stop_loss=entries.stop_loss,
        target_price=entries.target_price,
        risk_reward=entries.risk_reward,
        suggested_action=entries.action if w_stage == 2 else stage_action(w_stage),
        condition_checks=checks,
        why_high=why,
        backtest=backtest,
        chart_payload=_build_chart_payload(weekly, daily, rs.rs_line, w_stage, d_stage),
    )