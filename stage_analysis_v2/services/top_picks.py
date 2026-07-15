"""
Top Stage 2 picks — strict preset, DB-cached for fast dashboard load.
"""
from __future__ import annotations

from datetime import timedelta

from django.utils import timezone

from stage_analysis_v2.models import ScanCache, StockAnalysis
from stage_analysis_v2.services.screener import ScreenerFilters, scan_universe

STRICT_FILTERS = ScreenerFilters(
    min_quality_score=75,
    weekly_stage=2,
    min_rs_rating=60,
    breakout_type="clean",
    market_favorable_only=False,
    highlight_only=False,
)

CACHE_KEY = "top_stage2_picks"
CACHE_MAX_AGE = timedelta(hours=6)


def _serialize_pick(r) -> dict:
    return {
        "ticker": r.ticker,
        "company_name": r.company_name,
        "sector": r.sector,
        "quality_score": r.quality_score,
        "weekly_stage": r.weekly_stage,
        "daily_stage": r.daily_stage,
        "rs_rating": r.rs_rating,
        "rs_trend": r.rs_trend,
        "breakout_type": r.breakout_type,
        "current_price": r.current_price,
        "pullback_entry": r.pullback_entry,
        "stop_loss": r.stop_loss,
        "risk_reward": r.risk_reward,
        "market_highlight": r.market_highlight,
        "why_high": (r.why_high or r.score_reasons)[:5],
    }


def save_analysis_global(r) -> StockAnalysis:
    StockAnalysis.objects.filter(ticker=r.ticker, user=None).delete()
    return StockAnalysis.objects.create(
        user=None,
        ticker=r.ticker,
        company_name=r.company_name,
        sector=r.sector,
        weekly_stage=r.weekly_stage,
        daily_stage=r.daily_stage,
        quality_score=r.quality_score,
        rs_rating=r.rs_rating,
        rs_trend=r.rs_trend,
        benchmark=r.benchmark,
        breakout_type=r.breakout_type,
        market_favorable=r.market_context.favorable,
        current_price=r.current_price,
        ma_30w=r.ma_30w,
        ma_150d=r.ma_150d or None,
        primary_entry=r.primary_entry,
        pullback_entry=r.pullback_entry,
        stop_loss=r.stop_loss,
        target_price=r.target_price,
        risk_reward=r.risk_reward,
        suggested_action=r.suggested_action,
        score_breakdown=r.score_breakdown,
        analysis_details={
            "score_reasons": r.score_reasons,
            "conditions": [
                {"name": c.name, "met": c.met, "detail": c.detail}
                for c in getattr(r, "condition_checks", [])
            ],
            "why_high": getattr(r, "why_high", []),
        },
        chart_payload=r.chart_payload,
    )


def refresh_top_picks(sector: str = "") -> list[dict]:
    filters = ScreenerFilters(
        min_quality_score=STRICT_FILTERS.min_quality_score,
        weekly_stage=STRICT_FILTERS.weekly_stage,
        min_rs_rating=STRICT_FILTERS.min_rs_rating,
        breakout_type=STRICT_FILTERS.breakout_type,
        sector=sector,
    )
    result = scan_universe(filters)
    picks = []
    for r in result.items[:25]:
        save_analysis_global(r)
        picks.append(_serialize_pick(r))

    ScanCache.objects.update_or_create(
        key=CACHE_KEY if not sector else f"{CACHE_KEY}:{sector}",
        defaults={
            "payload": picks,
            "item_count": len(picks),
            "filters": {"sector": sector, **filters.__dict__},
        },
    )
    return picks


def get_top_picks(sector: str = "", force: bool = False) -> tuple[list[dict], bool]:
    """Return (picks, from_cache)."""
    cache_key = CACHE_KEY if not sector else f"{CACHE_KEY}:{sector}"
    if not force:
        cached = ScanCache.objects.filter(key=cache_key).first()
        if cached and cached.updated_at > timezone.now() - CACHE_MAX_AGE and cached.payload:
            return cached.payload, True

    return refresh_top_picks(sector), False


def get_sector_list() -> list[str]:
    from trading.models import Stock
    return sorted(
        s for s in Stock.objects.filter(is_nifty200=True, is_active=True)
        .exclude(sector="").values_list("sector", flat=True).distinct()
        if s
    )