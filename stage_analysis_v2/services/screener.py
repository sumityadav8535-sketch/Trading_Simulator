"""
V2 universe screener with quality ranking and filters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from stage_analysis.services.stage_detector import (
    MA_PERIOD,
    MA_SLOPE_LOOKBACK,
    daily_to_weekly,
    result_from_weekly_df,
)
from stage_analysis_v2.services.analyzer import V2AnalysisResult, analyze_ticker_v2
from stage_analysis_v2.services.market_context import MarketContext, analyze_market_context
from trading.models import Stock
from trading.services.market_data import get_universe_symbols, load_price_dataframe
from trading.services.nse_price_sync import yfinance_ticker


@dataclass
class ScreenerFilters:
    min_quality_score: int = 75
    weekly_stage: int = 2
    daily_stage: int | None = None
    min_rs_rating: float = 60.0
    breakout_type: str = "clean"
    market_favorable_only: bool = False
    highlight_only: bool = False
    strong_rs_only: bool = True
    sector: str = ""
    nifty200_only: bool = True


@dataclass
class ScreenerResult:
    items: list[V2AnalysisResult] = field(default_factory=list)
    by_sector: dict[str, list[V2AnalysisResult]] = field(default_factory=dict)
    market: MarketContext | None = None
    scanned: int = 0
    stage2_candidates: int = 0
    skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _quick_weekly_stage(symbol: str) -> int | None:
    """Fast weekly stage from local DB without full V2 pipeline."""
    daily = load_price_dataframe(symbol)
    if daily.empty:
        return None
    weekly = daily_to_weekly(daily)
    if len(weekly) < MA_PERIOD + MA_SLOPE_LOOKBACK:
        return None
    try:
        r = result_from_weekly_df(
            yfinance_ticker(symbol), symbol, weekly, include_chart=False
        )
        return r.stage
    except ValueError:
        return None


def _passes_filters(r: V2AnalysisResult, f: ScreenerFilters) -> bool:
    if r.weekly_stage != f.weekly_stage:
        return False
    if f.daily_stage is not None and r.daily_stage != f.daily_stage:
        return False
    if r.quality_score < f.min_quality_score:
        return False
    if r.rs_rating < f.min_rs_rating:
        return False
    if f.breakout_type and r.breakout_type != f.breakout_type:
        return False
    if f.market_favorable_only and not r.market_context.favorable:
        return False
    if f.highlight_only and not r.market_highlight:
        return False
    if f.strong_rs_only and not (r.rs_rating >= 60 and (r.rs_improving or r.rs_rating >= 65)):
        return False
    if f.sector and r.sector != f.sector:
        return False
    return True


def scan_universe(filters: ScreenerFilters | None = None) -> ScreenerResult:
    f = filters or ScreenerFilters()
    market = analyze_market_context()
    symbols = get_universe_symbols(nifty200_only=f.nifty200_only)
    result = ScreenerResult(market=market)

    sector_stages: dict[str, list[int]] = {}
    candidates: list[str] = []

    for sym in symbols:
        result.scanned += 1
        stage = _quick_weekly_stage(sym)
        if stage is None:
            candidates.append(sym)
            continue
        stock = Stock.objects.filter(pk=sym).first()
        sec = (stock.sector if stock else "") or "Unknown"
        sector_stages.setdefault(sec, []).append(stage)
        if stage == f.weekly_stage:
            candidates.append(sym)

    sector_avgs = {s: sum(v) / len(v) for s, v in sector_stages.items() if v}
    result.stage2_candidates = len(candidates)

    for sym in candidates:
        ticker = yfinance_ticker(sym)
        stock = Stock.objects.filter(pk=sym).first()
        sec = (stock.sector if stock else "") or "Unknown"
        try:
            analysis = analyze_ticker_v2(
                ticker,
                market=market,
                sector_stage_avg=sector_avgs.get(sec),
                include_backtest=False,
            )
            if _passes_filters(analysis, f):
                result.items.append(analysis)
        except ValueError as exc:
            result.errors.append((sym, str(exc)))
            result.skipped += 1

    result.items.sort(key=lambda x: x.quality_score, reverse=True)
    for item in result.items:
        sec = item.sector or "Unknown"
        result.by_sector.setdefault(sec, []).append(item)
    return result