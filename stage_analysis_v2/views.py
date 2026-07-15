"""
Views for Stage Analysis 2.0.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from stage_analysis.services.stage_detector import normalize_ticker
from stage_analysis_v2.forms import AnalyzeForm, ScreenerForm, StageV2BacktestForm, WatchlistForm
from stage_analysis_v2.models import StockAnalysis, Watchlist
from stage_analysis_v2.services.analyzer import V2AnalysisResult, analyze_ticker_v2
from stage_analysis_v2.services.market_context import analyze_market_context
from stage_analysis_v2.services.screener import ScreenerFilters, scan_universe
from stage_analysis_v2.services.backtester import run_stage_v2_backtest
from stage_analysis_v2.services.top_picks import get_top_picks, get_sector_list
from trading.services.charts import (
    build_equity_curve,
    build_exit_breakdown_chart,
    build_monthly_returns_chart,
    build_win_loss_pie,
)
from trading.services.market_data import get_universe_symbols
from trading.services.nse_price_sync import nse_symbol_from_ticker


def _user(request: HttpRequest):
    return request.user if request.user.is_authenticated else None


def _parse_backtest_symbols(universe: str, symbol: str, symbols_raw: str) -> list[str]:
    if universe == "nifty200":
        return get_universe_symbols(nifty200_only=True)

    raw = symbols_raw if universe == "custom" else symbol
    parts = [s.strip().upper() for s in raw.replace("\n", ",").split(",") if s.strip()]
    out: list[str] = []
    for part in parts:
        sym = nse_symbol_from_ticker(part) if "." in part else part
        if sym and sym not in out:
            out.append(sym)
    return out


def _watchlist_qs(request: HttpRequest):
    return Watchlist.objects.filter(user=_user(request))


def _save_analysis(request: HttpRequest, r: V2AnalysisResult) -> StockAnalysis:
    user = _user(request)
    if user is None:
        StockAnalysis.objects.filter(ticker=r.ticker, user=None).delete()
    obj, _ = StockAnalysis.objects.update_or_create(
        user=user,
        ticker=r.ticker,
        defaults={
            "company_name": r.company_name,
            "sector": r.sector,
            "weekly_stage": r.weekly_stage,
            "daily_stage": r.daily_stage,
            "quality_score": r.quality_score,
            "rs_rating": r.rs_rating,
            "rs_trend": r.rs_trend,
            "benchmark": r.benchmark,
            "breakout_type": r.breakout_type,
            "market_favorable": r.market_context.favorable,
            "current_price": r.current_price,
            "ma_30w": r.ma_30w,
            "ma_150d": r.ma_150d or None,
            "primary_entry": r.primary_entry,
            "pullback_entry": r.pullback_entry,
            "stop_loss": r.stop_loss,
            "target_price": r.target_price,
            "risk_reward": r.risk_reward,
            "suggested_action": r.suggested_action,
            "score_breakdown": r.score_breakdown,
            "analysis_details": {
                "weekly_reasons": r.weekly_reasons,
                "daily_reasons": r.daily_reasons,
                "score_reasons": r.score_reasons,
                "why_high": r.why_high,
                "conditions": [
                    {"name": c.name, "met": c.met, "detail": c.detail, "weight": c.weight}
                    for c in r.condition_checks
                ],
                "breakout": r.breakout_details,
                "market": {
                    "stage": r.market_context.stage,
                    "label": r.market_context.stage_label,
                    "warning": r.market_context.warning,
                    "favorable": r.market_context.favorable,
                },
                "backtest": {
                    "sample_size": r.backtest.sample_size,
                    "avg_return_pct": r.backtest.avg_return_pct,
                    "win_rate_pct": r.backtest.win_rate_pct,
                    "median_return_pct": r.backtest.median_return_pct,
                    "note": r.backtest.note,
                } if r.backtest else None,
                "quality_tier": r.quality_tier,
                "market_highlight": r.market_highlight,
                "rs_vs_benchmark_pct": r.rs_vs_benchmark_pct,
            },
            "chart_payload": r.chart_payload,
        },
    )
    return obj


def dashboard(request: HttpRequest) -> HttpResponse:
    market = analyze_market_context()
    form = AnalyzeForm(request.GET if request.GET.get("ticker") else None)
    result = None

    if request.GET.get("ticker"):
        form = AnalyzeForm(request.GET)
        if form.is_valid():
            try:
                result = analyze_ticker_v2(form.cleaned_data["ticker"])
                _save_analysis(request, result)
            except ValueError as exc:
                messages.error(request, str(exc))

    force_refresh = request.GET.get("refresh_picks") == "1"
    top_picks, from_cache = get_top_picks(force=force_refresh)
    cache_note = "cached" if from_cache else "fresh scan"

    return render(request, "stage_analysis_v2/dashboard.html", {
        "form": form,
        "result": result,
        "market": market,
        "top_picks": top_picks,
        "cache_note": cache_note,
        "watchlist_count": _watchlist_qs(request).count(),
        "sectors": get_sector_list(),
    })


def stock_detail(request: HttpRequest, ticker: str) -> HttpResponse:
    symbol = normalize_ticker(ticker)
    user = _user(request)
    cached = StockAnalysis.objects.filter(ticker=symbol, user=user).order_by("-updated_at").first()

    try:
        result = analyze_ticker_v2(symbol)
        analysis = _save_analysis(request, result)
    except ValueError as exc:
        if cached and cached.chart_payload:
            messages.warning(request, f"Live refresh failed; showing cached data. ({exc})")
            return render(request, "stage_analysis_v2/stock_detail.html", {
                "result": None,
                "analysis": cached,
                "cached_only": True,
                "on_watchlist": _watchlist_qs(request).filter(ticker=symbol).exists(),
                "chart_json": json.dumps(cached.chart_payload),
            })
        messages.error(request, str(exc))
        return redirect("stage_analysis_v2:dashboard")

    on_watchlist = _watchlist_qs(request).filter(ticker=result.ticker).exists()
    return render(request, "stage_analysis_v2/stock_detail.html", {
        "result": result,
        "analysis": analysis,
        "cached_only": False,
        "on_watchlist": on_watchlist,
        "chart_json": json.dumps(result.chart_payload),
    })


def watchlist_view(request: HttpRequest) -> HttpResponse:
    user = _user(request)
    if request.method == "POST":
        form = WatchlistForm(request.POST)
        if form.is_valid():
            ticker = normalize_ticker(form.cleaned_data["ticker"])
            if not ticker.endswith(".NS") and "." not in ticker:
                from trading.services.nse_price_sync import yfinance_ticker
                ticker = yfinance_ticker(ticker)
            try:
                r = analyze_ticker_v2(ticker)
                if user is None:
                    Watchlist.objects.filter(ticker=r.ticker, user=None).delete()
                Watchlist.objects.update_or_create(
                    user=user, ticker=r.ticker,
                    defaults={"company_name": r.company_name, "sector": r.sector,
                              "notes": form.cleaned_data.get("notes", "")},
                )
                _save_analysis(request, r)
                messages.success(request, f"{r.ticker} added.")
            except ValueError as exc:
                messages.error(request, str(exc))
        return redirect("stage_analysis_v2:watchlist")

    items = []
    for w in _watchlist_qs(request):
        a = StockAnalysis.objects.filter(ticker=w.ticker, user=user).order_by("-updated_at").first()
        if not a:
            try:
                r = analyze_ticker_v2(w.ticker)
                a = _save_analysis(request, r)
            except ValueError:
                a = None
        items.append({"watch": w, "analysis": a})

    return render(request, "stage_analysis_v2/watchlist.html", {"form": WatchlistForm(), "items": items})


@require_POST
def watchlist_remove(request: HttpRequest, pk: int) -> HttpResponse:
    item = get_object_or_404(Watchlist, pk=pk, user=_user(request))
    item.delete()
    return redirect("stage_analysis_v2:watchlist")


def screener(request: HttpRequest) -> HttpResponse:
    form = ScreenerForm(request.GET or None)
    scan_result = None

    if request.GET.get("run_scan"):
        form = ScreenerForm(request.GET)
        if form.is_valid():
            daily = form.cleaned_data.get("daily_stage")
            filters = ScreenerFilters(
                min_quality_score=form.cleaned_data["min_quality_score"],
                min_rs_rating=form.cleaned_data["min_rs_rating"],
                daily_stage=int(daily) if daily else None,
                breakout_type=form.cleaned_data.get("breakout_type") or "",
                sector=form.cleaned_data.get("sector") or "",
                market_favorable_only=form.cleaned_data.get("market_favorable_only", False),
                highlight_only=form.cleaned_data.get("highlight_only", False),
                strong_rs_only=form.cleaned_data.get("strong_rs_only", True),
            )
            scan_result = scan_universe(filters)
            messages.success(
                request,
                f"Found {len(scan_result.items)} stocks matching strict Stage 2 filters.",
            )

    return render(request, "stage_analysis_v2/screener.html", {
        "form": form,
        "scan_result": scan_result,
        "market": analyze_market_context() if not scan_result else scan_result.market,
    })


def backtest(request: HttpRequest) -> HttpResponse:
    default_end = date.today()
    default_start = default_end - timedelta(days=120)
    run_requested = request.GET.get("run") == "1"

    form = StageV2BacktestForm(
        request.GET if run_requested else None,
        initial={"start_date": default_start, "end_date": default_end},
    )
    bt_result = None
    charts: dict[str, str] = {}
    symbol_list: list[str] = []

    if run_requested and form.is_valid():
        cd = form.cleaned_data
        symbol_list = _parse_backtest_symbols(
            cd["universe"],
            cd.get("symbol") or "",
            cd.get("symbols") or "",
        )
        if not symbol_list:
            messages.error(request, "No valid symbols to backtest.")
        else:
            bt_result = run_stage_v2_backtest(
                symbols=symbol_list,
                start_date=cd["start_date"],
                end_date=cd["end_date"],
                capital=float(cd["capital"]),
                min_quality_score=cd["min_quality_score"],
                market_filter=cd["market_filter"],
            )
            if bt_result.equity_curve:
                charts["equity"] = build_equity_curve(bt_result.equity_curve)
            if bt_result.monthly_returns:
                charts["monthly"] = build_monthly_returns_chart(bt_result.monthly_returns)
            if bt_result.exit_breakdown:
                charts["exits"] = build_exit_breakdown_chart(bt_result.exit_breakdown)
            if bt_result.trades:
                wins = sum(1 for t in bt_result.trades if t.pnl > 0)
                charts["win_loss"] = build_win_loss_pie(wins, len(bt_result.trades) - wins)

    return render(request, "stage_analysis_v2/backtest.html", {
        "form": form,
        "bt_result": bt_result,
        "charts": charts,
        "symbol_list": symbol_list,
        "run_requested": run_requested,
    })


@require_POST
def add_to_watchlist(request: HttpRequest, ticker: str) -> HttpResponse:
    user = _user(request)
    symbol = normalize_ticker(ticker)
    try:
        r = analyze_ticker_v2(symbol)
        if user is None:
            Watchlist.objects.filter(ticker=r.ticker, user=None).delete()
        Watchlist.objects.update_or_create(
            user=user, ticker=r.ticker,
            defaults={"company_name": r.company_name, "sector": r.sector},
        )
        _save_analysis(request, r)
        messages.success(request, f"{r.ticker} saved.")
    except ValueError as exc:
        messages.error(request, str(exc))
    return redirect("stage_analysis_v2:stock_detail", ticker=symbol)