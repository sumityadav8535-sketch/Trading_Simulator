"""
Views for Stan Weinstein Stage Analysis section.
"""
from __future__ import annotations

import json

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from stage_analysis.forms import AnalyzeTickerForm, WatchlistForm
from stage_analysis.models import Stage, StockAnalysis, Watchlist
from stage_analysis.services.nifty200_scanner import scan_nifty200_stage2
from stage_analysis.services.stage_detector import analyze_ticker, normalize_ticker


def _current_user(request: HttpRequest):
    if request.user.is_authenticated:
        return request.user
    return None


def _save_analysis(request: HttpRequest, result) -> StockAnalysis:
    user = _current_user(request)
    if user is None:
        StockAnalysis.objects.filter(ticker=result.ticker, user=None).delete()
    obj, _ = StockAnalysis.objects.update_or_create(
        user=user,
        ticker=result.ticker,
        defaults={
            "company_name": result.company_name,
            "stage": result.stage,
            "current_price": result.current_price,
            "ma_30w": result.ma_30w,
            "price_vs_ma_pct": result.price_vs_ma_pct,
            "ma_slope_pct": result.ma_slope_pct,
            "suggested_action": result.suggested_action,
            "breakout_level": result.breakout_level,
            "support_level": result.support_level,
            "stop_loss": result.stop_loss,
            "stage_reasons": result.reasons,
            "chart_payload": result.chart_payload,
        },
    )
    return obj


def _analyze_and_render(request: HttpRequest, ticker: str, template: str, extra: dict | None = None):
    try:
        result = analyze_ticker(ticker)
        analysis = _save_analysis(request, result)
        context = {
            "result": result,
            "analysis": analysis,
            "stage_info": _stage_education(),
            "chart_json": json.dumps(result.chart_payload),
        }
        if extra:
            context.update(extra)
        return render(request, template, context)
    except ValueError as exc:
        messages.error(request, str(exc))
        return None


def _stage_education() -> list[dict]:
    from stage_analysis.services.stage_detector import STAGE_ACTIONS, STAGE_DESCRIPTIONS

    colors = {1: "slate", 2: "emerald", 3: "amber", 4: "red"}
    return [
        {
            "stage": n,
            "label": Stage(n).label,
            "description": STAGE_DESCRIPTIONS[n],
            "action": STAGE_ACTIONS[n],
            "color": colors[n],
        }
        for n in (1, 2, 3, 4)
    ]


def _watchlist_queryset(request: HttpRequest):
    user = _current_user(request)
    return Watchlist.objects.filter(user=user)


def _add_watchlist_item(user, ticker: str, company_name: str = "", notes: str = "") -> Watchlist:
    """Create or update watchlist entry; avoids duplicate anonymous rows."""
    if user is None:
        Watchlist.objects.filter(ticker=ticker, user=None).delete()
    item, _ = Watchlist.objects.update_or_create(
        user=user,
        ticker=ticker,
        defaults={"company_name": company_name, "notes": notes},
    )
    return item


def _is_on_watchlist(request: HttpRequest, ticker: str) -> bool:
    return _watchlist_queryset(request).filter(ticker=normalize_ticker(ticker)).exists()


def dashboard(request: HttpRequest) -> HttpResponse:
    user = _current_user(request)
    form = AnalyzeTickerForm(request.GET if request.GET.get("ticker") else None)
    recent = StockAnalysis.objects.filter(user=user).order_by("-updated_at")[:8]

    analyze_result = None
    if request.GET.get("ticker"):
        form = AnalyzeTickerForm(request.GET)
        if form.is_valid():
            ticker = form.cleaned_data["ticker"]
            try:
                result = analyze_ticker(ticker)
                _save_analysis(request, result)
                analyze_result = result
            except ValueError as exc:
                messages.error(request, str(exc))

    stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for item in _watchlist_queryset(request):
        latest = (
            StockAnalysis.objects.filter(ticker=item.ticker, user=user)
            .order_by("-updated_at")
            .first()
        )
        if latest:
            stage_counts[latest.stage] = stage_counts.get(latest.stage, 0) + 1

    return render(request, "stage_analysis/dashboard.html", {
        "form": form,
        "recent_analyses": recent,
        "analyze_result": analyze_result,
        "stage_info": _stage_education(),
        "stage_counts": stage_counts,
        "watchlist_count": _watchlist_queryset(request).count(),
    })


def stock_detail(request: HttpRequest, ticker: str) -> HttpResponse:
    symbol = normalize_ticker(ticker)
    response = _analyze_and_render(
        request,
        symbol,
        "stage_analysis/stock_detail.html",
        extra={"on_watchlist": _is_on_watchlist(request, symbol)},
    )
    if response:
        return response
    return redirect("stage_analysis:dashboard")


def watchlist_view(request: HttpRequest) -> HttpResponse:
    user = _current_user(request)

    if request.method == "POST":
        form = WatchlistForm(request.POST)
        if form.is_valid():
            ticker = normalize_ticker(form.cleaned_data["ticker"])
            try:
                result = analyze_ticker(ticker)
                _add_watchlist_item(
                    user,
                    ticker,
                    result.company_name,
                    form.cleaned_data.get("notes", ""),
                )
                _save_analysis(request, result)
                messages.success(request, f"{ticker} added to Stage Analysis watchlist.")
            except ValueError as exc:
                messages.error(request, str(exc))
        return redirect("stage_analysis:watchlist")

    items = []
    for item in _watchlist_queryset(request):
        latest = (
            StockAnalysis.objects.filter(ticker=item.ticker, user=user)
            .order_by("-updated_at")
            .first()
        )
        if latest is None:
            try:
                result = analyze_ticker(item.ticker)
                latest = _save_analysis(request, result)
            except ValueError:
                latest = None
        items.append({"item": item, "analysis": latest})

    return render(request, "stage_analysis/watchlist.html", {
        "form": WatchlistForm(),
        "items": items,
        "stage_info": _stage_education(),
    })


@require_POST
def watchlist_remove(request: HttpRequest, pk: int) -> HttpResponse:
    user = _current_user(request)
    item = get_object_or_404(Watchlist, pk=pk, user=user)
    item.delete()
    messages.success(request, f"{item.ticker} removed from watchlist.")
    return redirect("stage_analysis:watchlist")


def screener(request: HttpRequest) -> HttpResponse:
    """Show watchlist and/or Nifty 200 stocks in Stage 2 (Advancing)."""
    user = _current_user(request)
    stage2_items = []
    nifty200_scan = None

    if request.GET.get("nifty200") == "1":
        nifty200_scan = scan_nifty200_stage2()
        if nifty200_scan.stage2_count:
            messages.success(
                request,
                f"Found {nifty200_scan.stage2_count} Stage 2 stocks "
                f"out of {nifty200_scan.scanned} Nifty 200 scanned.",
            )
        else:
            messages.info(request, "No Stage 2 stocks found in Nifty 200 right now.")

    for item in _watchlist_queryset(request):
        latest = (
            StockAnalysis.objects.filter(ticker=item.ticker, user=user)
            .order_by("-updated_at")
            .first()
        )
        if latest is None:
            try:
                result = analyze_ticker(item.ticker)
                latest = _save_analysis(request, result)
            except ValueError:
                continue
        if latest.stage == Stage.ADVANCING:
            stage2_items.append({"item": item, "analysis": latest})

    return render(request, "stage_analysis/screener.html", {
        "stage2_items": stage2_items,
        "nifty200_scan": nifty200_scan,
        "watchlist_count": _watchlist_queryset(request).count(),
        "stage_info": _stage_education(),
    })


@require_POST
def add_to_watchlist(request: HttpRequest, ticker: str) -> HttpResponse:
    user = _current_user(request)
    symbol = normalize_ticker(ticker)
    try:
        result = analyze_ticker(symbol)
        _add_watchlist_item(user, symbol, result.company_name)
        _save_analysis(request, result)
        messages.success(request, f"{symbol} saved to watchlist.")
    except ValueError as exc:
        messages.error(request, str(exc))
    return redirect("stage_analysis:stock_detail", ticker=symbol)