"""
Views for Confluence Trend Pullback Swing Strategy dashboard.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from decimal import Decimal

from django.contrib import messages
from django.db.models import Count, Q
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from trading.forms import (
    ChartForm,
    FnoForm,
    IntradayForm,
    JournalForm,
    RiskCalculatorForm,
    ScannerForm,
    ScreenerForm,
    SignalsForm,
    WatchlistForm,
)
from trading.models import (
    BacktestRun,
    Signal,
    Stock,
    StrategyConfig,
    TradeJournalEntry,
    WatchlistItem,
)
from trading.services.charts import (
    build_drawdown_chart,
    build_equity_curve,
    build_exit_breakdown_chart,
    build_monthly_returns_chart,
    build_scanner_signal_bars,
    build_stock_chart,
    build_win_loss_pie,
)
from trading.services.elite_scanner_history import run_elite_scanner_history
from trading.services.market_data import get_universe_symbols
from trading.services.signal_backtester import build_outcome_detail, run_signal_backtest
from trading.services.swing_strategy import STRATEGY_NAME
from trading.services.market_regime import get_market_regime_status
from trading.services.tournament_results import (
    ANNUAL_RESULTS,
    CHAMPION_DESCRIPTION,
    TOURNAMENT_META,
    TOURNAMENT_RANKINGS,
)
from trading.services.walk_forward_results import (
    WALK_FORWARD_META,
    WALK_FORWARD_RESULTS,
    WALK_FORWARD_VERDICT,
)
from trading.services.indicators import compute_indicators, get_indicator_frame
from trading.services.fno_live import build_fno_chart_json, get_fno_dashboard
from trading.services.fno_results import load_model_meta
from trading.services.intraday_data import (
    build_intraday_chart_json,
    fetch_index_snapshot,
    fetch_nifty100_quotes,
)
from trading.services.intraday_history_sync import (
    get_history_coverage,
    start_intraday_history_sync_async,
)
from trading.services.fno_backtest_runner import (
    get_backtest_status,
    start_fno_backtest_async,
)

logger = logging.getLogger(__name__)
from trading.services.market_data import load_price_dataframe
from trading.services.position_sizing import calculate_position_size
from trading.services.strategy import evaluate_stock
from trading.services.swing_strategy import (
    SCANNER_STRATEGY_LABEL,
    scan_swing_universe,
)


def _stock_choices():
    return Stock.objects.filter(is_active=True).order_by("symbol")


def dashboard(request: HttpRequest) -> HttpResponse:
    config = StrategyConfig.get_active()
    nifty_count = Stock.objects.filter(is_nifty200=True, is_active=True).count()
    market_bias = _estimate_market_bias()

    return render(request, "trading/dashboard.html", {
        "config": config,
        "nifty_count": nifty_count,
        "market_bias": market_bias,
        "price_bars": _total_price_bars(),
    })


def _estimate_market_bias() -> dict:
    """Nifty 50 index trend vs 50 EMA and 200 EMA."""
    import pandas as pd

    from trading.services.nifty50_index import load_nifty50_frame

    df = load_nifty50_frame()
    if df.empty:
        return {"trend": "Unknown", "above_200ema": 0, "below_200ema": 0, "avg_adx": None}
    row = df.iloc[-1]
    close = float(row["close"])
    above_50 = bool(row.get("ema_50") and close > float(row["ema_50"]))
    above_200 = bool(row.get("ema_200") and close > float(row["ema_200"]))
    trend = "Bullish" if above_50 and above_200 else ("Neutral" if above_50 else "Bearish")
    adx = float(row["adx_14"]) if row.get("adx_14") and pd.notna(row["adx_14"]) else None
    return {
        "trend": trend,
        "above_200ema": 1 if above_200 else 0,
        "below_200ema": 0 if above_200 else 1,
        "avg_adx": round(adx, 1) if adx is not None else None,
    }


def _total_price_bars() -> int:
    from trading.models import DailyPrice
    return DailyPrice.objects.count()


def screener(request: HttpRequest) -> HttpResponse:
    config = StrategyConfig.get_active()
    form = ScreenerForm(request.GET or None)
    results = []

    if form.is_valid():
        if form.cleaned_data["enable_fundamentals"]:
            config.enable_fundamental_filters = True
        qs = Stock.objects.filter(is_active=True)
        if form.cleaned_data["nifty200_only"]:
            qs = qs.filter(is_nifty200=True)

        for stock in qs:
            if form.cleaned_data["enable_fundamentals"] and not stock.passes_fundamental_filters(config):
                continue
            ev = evaluate_stock(stock.symbol, config=config)
            if ev.confluence_score < form.cleaned_data["min_confluence"]:
                continue
            snap = ev.indicator_snapshot
            if form.cleaned_data["above_200ema"]:
                if not snap.get("ema_200") or snap.get("close", 0) <= snap["ema_200"]:
                    continue
            if snap.get("adx_14") and snap["adx_14"] < form.cleaned_data["min_adx"]:
                continue
            results.append({"stock": stock, "eval": ev})

        results.sort(key=lambda x: -x["eval"].confluence_score)

    return render(request, "trading/screener.html", {"form": form, "results": results})


def scanner(request: HttpRequest) -> HttpResponse:
    config = StrategyConfig.get_active()
    form = ScannerForm(request.GET or None)
    scan_results = []

    if form.is_valid():
        watchlist = form.cleaned_data["scope"] == "watchlist"
        symbols = get_universe_symbols(nifty200_only=not watchlist, watchlist_only=watchlist)
        capital = float(form.cleaned_data["capital"])
        scan_results = scan_swing_universe(
            symbols,
            config=config,
            capital=capital,
            min_score=form.cleaned_data["min_score"],
            elite_only=True,
        )
        for r in scan_results:
            if r.is_valid:
                Signal.objects.update_or_create(
                    stock_id=r.symbol,
                    date=r.eval_date,
                    defaults={
                        "confluence_score": r.confluence_score,
                        "is_valid": True,
                        "entry_price": r.entry_price,
                        "stop_loss": r.stop_loss,
                        "target_1r": r.target_1r,
                        "target_2r": r.target_2r,
                        "target_3r": r.target_3r,
                        "risk_reward": r.risk_reward,
                        "position_size": r.position_size,
                        "capital_used": r.capital_used,
                        "reasons": r.reasons,
                        "rejection_reasons": r.rejection_reasons,
                        "indicator_snapshot": {
                            **r.indicator_snapshot,
                            "entry_path": r.entry_path,
                            "strategy": SCANNER_STRATEGY_LABEL,
                        },
                    },
                )

    history_default_end = date.today()
    history_default_start = history_default_end - timedelta(days=274)

    return render(request, "trading/scanner.html", {
        "form": form,
        "scan_results": scan_results,
        "config": config,
        "strategy_name": SCANNER_STRATEGY_LABEL,
        "history_default_start": history_default_start.isoformat(),
        "history_default_end": history_default_end.isoformat(),
    })


def _resolve_scanner_history_range(
    period: str,
    start_str: str | None,
    end_str: str | None,
) -> tuple[date, date]:
    end = date.today()
    if end_str:
        try:
            end = date.fromisoformat(end_str)
        except ValueError:
            pass

    if period == "custom" and start_str:
        try:
            return date.fromisoformat(start_str), end
        except ValueError:
            pass

    days_map = {"9m": 274, "1y": 365, "2y": 730, "3y": 1095}
    days = days_map.get(period, 274)
    return end - timedelta(days=days), end


@require_GET
def scanner_history_api(request: HttpRequest) -> JsonResponse:
    """JSON: daily signal bars + per-date trade details for scanner history chart."""
    from django.core.cache import cache

    period = request.GET.get("period", "9m")
    start_str = request.GET.get("start_date")
    end_str = request.GET.get("end_date")
    capital = float(request.GET.get("capital", 100_000))

    start_date, end_date = _resolve_scanner_history_range(period, start_str, end_str)
    if start_date >= end_date:
        return JsonResponse({"error": "Start date must be before end date"}, status=400)

    cache_key = f"scanner_history:v2:{start_date}:{end_date}:{capital}"
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)

    config = StrategyConfig.get_active()
    symbols = [
        s for s in get_universe_symbols(nifty200_only=True) if s != "NIFTY50"
    ]
    history = run_elite_scanner_history(
        symbols, start_date, end_date, capital=capital, config=config,
    )

    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "period": period,
        "summary": history.summary,
        "daily": history.daily,
        "by_date": history.by_date,
        "chart": json.loads(build_scanner_signal_bars(history.daily)),
    }
    cache.set(cache_key, payload, 3600)
    return JsonResponse(payload)


def stock_chart(request: HttpRequest) -> HttpResponse:
    symbols = _stock_choices()
    initial = request.GET.get("symbol") or (symbols.first().symbol if symbols.exists() else "")
    form = ChartForm(request.GET or None, initial={"symbol": initial})
    form.fields["symbol"].choices = [(s.symbol, s.symbol) for s in symbols]

    chart_json = None
    eval_result = None
    if form.is_valid():
        symbol = form.cleaned_data["symbol"]
        chart_json = build_stock_chart(symbol)
        eval_result = evaluate_stock(symbol)

    return render(request, "trading/chart.html", {
        "form": form,
        "chart_json": chart_json,
        "eval_result": eval_result,
    })


def risk_calculator(request: HttpRequest) -> HttpResponse:
    form = RiskCalculatorForm(request.GET or None)
    result = None
    if form.is_valid():
        pos = calculate_position_size(
            float(form.cleaned_data["capital"]),
            float(form.cleaned_data["risk_pct"]),
            float(form.cleaned_data["entry_price"]),
            float(form.cleaned_data["stop_loss"]),
        )
        entry = float(form.cleaned_data["entry_price"])
        sl = float(form.cleaned_data["stop_loss"])
        risk = entry - sl
        result = {
            "position": pos,
            "target_1r": entry + risk,
            "target_2r": entry + risk * 2,
            "target_3r": entry + risk * 3,
            "rr_2r": 2.0 if risk > 0 else 0,
        }
    return render(request, "trading/risk_calculator.html", {"form": form, "result": result})


def signals_view(request: HttpRequest) -> HttpResponse:
    """EMA20 Elite — full signal history + backtest dashboard."""
    config = StrategyConfig.get_active()
    default_capital = float(config.capital_default)
    default_end = date(2025, 6, 1)
    default_start = default_end - timedelta(days=3 * 365)
    form = SignalsForm(
        request.GET or None,
        initial={
            "start_date": default_start,
            "end_date": default_end,
            "capital": default_capital,
        },
    )
    bt_result = None
    charts = {}

    run_backtest = False
    start_date = default_start
    end_date = default_end
    capital = default_capital

    force_refresh = request.GET.get("refresh") == "1"

    if not request.GET:
        form = SignalsForm(initial={
            "start_date": default_start,
            "end_date": default_end,
            "capital": default_capital,
        })
    elif form.is_valid():
        run_backtest = True
        start_date = form.cleaned_data["start_date"]
        end_date = form.cleaned_data["end_date"]
        capital = float(form.cleaned_data["capital"])

    cached = None
    if run_backtest:
        cached = BacktestRun.objects.filter(
            name__startswith=STRATEGY_NAME,
            start_date=start_date,
            end_date=end_date,
            capital=Decimal(str(capital)),
        ).first()

        if cached and not force_refresh:
            from trading.services.signal_backtester import (
                HistoricalSignal,
                SignalBacktestResult,
                SignalBacktestTrade,
            )
            bt_result = SignalBacktestResult(
                strategy_name=STRATEGY_NAME,
                symbols=["NIFTY200"],
                start_date=start_date,
                end_date=end_date,
                capital=capital,
                total_trades=cached.total_trades,
                win_rate=cached.win_rate,
                profit_factor=cached.profit_factor,
                max_drawdown_pct=cached.max_drawdown_pct,
                avg_rr=cached.avg_rr,
                total_return_pct=cached.total_return_pct,
                equity_curve=cached.equity_curve,
            )
            bt_result.trades = []
            bt_result.signal_history = []
            for t in cached.trades:
                detail = t.get("outcome_detail") or build_outcome_detail(
                    outcome="win" if t["pnl"] > 0 else "loss",
                    exit_reason=t.get("exit_reason", ""),
                    signal_close=t.get("signal_close", t["entry_price"]),
                    actual_entry=t["entry_price"],
                    stop=t["stop_loss"],
                    target=t["target"],
                    exit_price=t["exit_price"],
                    exit_date=t["exit_date"],
                    days_held=t.get("days_held", 0),
                    max_high=t["exit_price"],
                    min_low=t["exit_price"],
                    entry_path="",
                    reasons=t.get("reasons", []),
                )
                bt_result.trades.append(
                    SignalBacktestTrade(
                        symbol=t["symbol"],
                        signal_date=t.get("signal_date", ""),
                        entry_date=t.get("entry_date", ""),
                        exit_date=t.get("exit_date", ""),
                        entry_price=t["entry_price"],
                        exit_price=t["exit_price"],
                        stop_loss=t["stop_loss"],
                        target=t["target"],
                        quantity=t.get("quantity", 0),
                        pnl=t["pnl"],
                        pnl_pct=t.get("pnl_pct", 0),
                        rr_achieved=t.get("rr_achieved", 0),
                        exit_reason=t.get("exit_reason", ""),
                        confluence_score=t.get("confluence_score", 0),
                        reasons=t.get("reasons", []),
                        outcome_detail=detail,
                        signal_close=t.get("signal_close", t["entry_price"]),
                        days_held=t.get("days_held", 0),
                    )
                )
                bt_result.signal_history.append(
                    HistoricalSignal(
                        symbol=t["symbol"],
                        signal_date=t.get("signal_date", t.get("entry_date", "")),
                        entry_date=t.get("entry_date", ""),
                        entry_price=t.get("signal_close", t["entry_price"]),
                        stop_loss=t["stop_loss"],
                        target_2r=t["target"],
                        risk_reward=2.0,
                        confluence_score=t.get("confluence_score", 0),
                        reasons=t.get("reasons", []),
                        outcome="win" if t["pnl"] > 0 else "loss",
                        exit_date=t["exit_date"],
                        exit_price=t["exit_price"],
                        pnl=t["pnl"],
                        pnl_pct=t.get("pnl_pct", 0),
                        rr_achieved=t.get("rr_achieved", 0),
                        exit_reason=t.get("exit_reason", ""),
                        actual_entry_price=t["entry_price"],
                        outcome_detail=detail,
                    )
                )
            bt_result.total_signals = len(bt_result.signal_history)
            bt_result.exit_breakdown = {}
            from collections import Counter
            bt_result.exit_breakdown = dict(Counter(t.exit_reason for t in bt_result.trades))
            bt_result.monthly_returns = []
            bt_result.expectancy_r = cached.avg_rr
        else:
            symbols = get_universe_symbols(nifty200_only=True)
            bt_result = run_signal_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                capital=capital,
            )
        wins = sum(1 for t in bt_result.trades if t.pnl > 0)
        losses = bt_result.total_trades - wins
        charts = {
            "equity": build_equity_curve(bt_result.equity_curve),
            "win_loss": build_win_loss_pie(wins, losses),
            "exits": build_exit_breakdown_chart(bt_result.exit_breakdown),
            "monthly": build_monthly_returns_chart(bt_result.monthly_returns),
            "drawdown": build_drawdown_chart(bt_result.equity_curve),
        }
        if not cached or force_refresh:
            BacktestRun.objects.create(
                name=f"{STRATEGY_NAME} {start_date}",
                symbols="NIFTY200",
                start_date=start_date,
                end_date=end_date,
                capital=Decimal(str(capital)),
                total_trades=bt_result.total_trades,
                win_rate=bt_result.win_rate,
                profit_factor=bt_result.profit_factor,
                max_drawdown_pct=bt_result.max_drawdown_pct,
                avg_rr=bt_result.avg_rr,
                total_return_pct=bt_result.total_return_pct,
                equity_curve=bt_result.equity_curve,
                trades=[{**t.__dict__} for t in bt_result.trades],
            )

    market_regime = get_market_regime_status()

    return render(request, "trading/signals.html", {
        "form": form,
        "bt_result": bt_result,
        "charts": charts,
        "strategy_name": STRATEGY_NAME,
        "tournament_meta": TOURNAMENT_META,
        "tournament_rankings": TOURNAMENT_RANKINGS,
        "annual_results": ANNUAL_RESULTS,
        "champion_description": CHAMPION_DESCRIPTION,
        "market_regime": market_regime,
        "walk_forward_meta": WALK_FORWARD_META,
        "walk_forward_results": WALK_FORWARD_RESULTS,
        "walk_forward_verdict": WALK_FORWARD_VERDICT,
    })


def backtest_view(request: HttpRequest) -> HttpResponse:
    """Redirect legacy /backtest/ URL to Stage Analysis 2.0 backtest."""
    from django.urls import reverse

    params = request.GET.copy()
    params["run"] = "1"
    if params.get("symbols") and not params.get("universe"):
        params["universe"] = "custom"
    elif not params.get("universe"):
        params["universe"] = "single"
    return redirect(f"{reverse('stage_analysis_v2:backtest')}?{params.urlencode()}")


def watchlist(request: HttpRequest) -> HttpResponse:
    add_form = WatchlistForm(request.POST or None)
    if request.method == "POST" and add_form.is_valid():
        sym = add_form.cleaned_data["symbol"].upper()
        stock = get_object_or_404(Stock, symbol=sym)
        WatchlistItem.objects.get_or_create(
            stock=stock,
            defaults={"notes": add_form.cleaned_data.get("notes", "")},
        )
        messages.success(request, f"Added {sym} to watchlist.")
        return redirect("watchlist")

    items = WatchlistItem.objects.select_related("stock")
    statuses = []
    for item in items:
        ev = evaluate_stock(item.stock.symbol)
        statuses.append({"item": item, "eval": ev})

    return render(request, "trading/watchlist.html", {
        "add_form": add_form,
        "statuses": statuses,
    })


@require_POST
def watchlist_remove(request: HttpRequest, pk: int) -> HttpResponse:
    item = get_object_or_404(WatchlistItem, pk=pk)
    item.delete()
    messages.info(request, f"Removed {item.stock_id} from watchlist.")
    return redirect("watchlist")


def journal(request: HttpRequest) -> HttpResponse:
    form = JournalForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        entry = form.save(commit=False)
        entry.compute_pnl()
        entry.save()
        messages.success(request, "Trade logged.")
        return redirect("journal")

    entries = TradeJournalEntry.objects.select_related("stock")[:50]
    closed = TradeJournalEntry.objects.filter(status=TradeJournalEntry.STATUS_CLOSED)
    stats = {
        "total": entries.count(),
        "open": TradeJournalEntry.objects.filter(status=TradeJournalEntry.STATUS_OPEN).count(),
        "total_pnl": sum(e.pnl or 0 for e in closed),
        "winners": closed.filter(pnl__gt=0).count(),
    }
    return render(request, "trading/journal.html", {
        "form": form,
        "entries": entries,
        "stats": stats,
    })


def intraday(request: HttpRequest) -> HttpResponse:
    """Live Nifty 100 intraday dashboard."""
    form = IntradayForm(request.GET or None)
    interval = "5m"
    sort_key = "change_pct"
    selected_symbol = (request.GET.get("symbol") or "").strip().upper()

    if form.is_valid():
        interval = form.cleaned_data["interval"]
        sort_key = form.cleaned_data["sort"]
        if form.cleaned_data.get("symbol"):
            selected_symbol = form.cleaned_data["symbol"].strip().upper()

    snapshot = fetch_index_snapshot()
    data = fetch_nifty100_quotes(interval=interval)
    quotes = data.get("quotes", [])

    if sort_key == "symbol":
        quotes = sorted(quotes, key=lambda q: q["symbol"])
    elif sort_key == "ltp":
        quotes = sorted(quotes, key=lambda q: q["ltp"], reverse=True)
    elif sort_key == "volume":
        quotes = sorted(quotes, key=lambda q: q["volume"], reverse=True)
    else:
        quotes = sorted(quotes, key=lambda q: q["change_pct"], reverse=True)

    chart_json = None
    if selected_symbol:
        chart_json = build_intraday_chart_json(selected_symbol, interval)

    gainers = sorted(data.get("quotes", []), key=lambda q: q["change_pct"], reverse=True)[:5]
    losers = sorted(data.get("quotes", []), key=lambda q: q["change_pct"])[:5]

    return render(request, "trading/intraday.html", {
        "form": form,
        "interval": interval,
        "sort_key": sort_key,
        "selected_symbol": selected_symbol,
        "quotes": quotes,
        "gainers": gainers,
        "losers": losers,
        "market": data.get("market", snapshot.get("market", {})),
        "index": snapshot.get("nifty100", {}),
        "updated_at": data.get("updated_at", snapshot.get("updated_at")),
        "symbol_count": data.get("symbol_count", len(quotes)),
        "advancers": data.get("advancers", 0),
        "decliners": data.get("decliners", 0),
        "unchanged": data.get("unchanged", 0),
        "chart_json": chart_json,
        "data_error": data.get("error"),
        "history_coverage": get_history_coverage(),
    })


@require_GET
def intraday_quotes_api(request: HttpRequest) -> JsonResponse:
    interval = request.GET.get("interval", "5m")
    force = request.GET.get("refresh") == "1"
    snapshot = fetch_index_snapshot()
    data = fetch_nifty100_quotes(interval=interval, force_refresh=force)
    data["index"] = snapshot.get("nifty100", {})
    return JsonResponse(data)


@require_GET
def intraday_chart_api(request: HttpRequest, symbol: str) -> JsonResponse:
    interval = request.GET.get("interval", "5m")
    chart = json.loads(build_intraday_chart_json(symbol.upper(), interval))
    return JsonResponse({"symbol": symbol.upper(), "interval": interval, "chart": chart})


@require_GET
def intraday_history_status_api(request: HttpRequest) -> JsonResponse:
    """Coverage of stored 5m pickles + live sync job status."""
    return JsonResponse(get_history_coverage())


@require_POST
def intraday_history_sync_api(request: HttpRequest) -> JsonResponse:
    """
    Fetch remaining 5m bars for Nifty 100 equities + F&O indices through today (IST).
    Runs in a background thread; poll GET /api/intraday/history/ for progress.
    """
    result = start_intraday_history_sync_async()
    status = 200 if result.get("ok") or result.get("started") else 409
    return JsonResponse(result, status=status)


@require_GET
def fno_backtest_status_api(request: HttpRequest) -> JsonResponse:
    """Frozen-model status (retrain disabled)."""
    return JsonResponse(get_backtest_status())


@require_POST
def fno_backtest_run_api(request: HttpRequest) -> JsonResponse:
    """
    Retrain is disabled — model is frozen so live signals stay stable.
    Returns 403 with explanation.
    """
    result = start_fno_backtest_async()
    return JsonResponse(result, status=403)


def _paper_tick_from_fno(force: bool = False) -> dict | None:
    """Open/manage paper positions whenever F&O Live evaluates signals."""
    try:
        from trading.services.paper_trading import run_tick

        return run_tick(force_refresh=force, ensure_auto=True)
    except Exception:
        logger.exception("Paper tick from F&O Live failed")
        return None


def fno_live(request: HttpRequest) -> HttpResponse:
    """Live F&O ML short strategy dashboard."""
    form = FnoForm(request.GET or None)
    instrument = "NIFTY"
    trade_period = request.GET.get("trades", "full")
    if trade_period not in ("full", "oos"):
        trade_period = "full"

    if form.is_valid():
        instrument = form.cleaned_data["instrument"]

    # Auto paper-fill on page load (same path as 30s poll)
    paper_tick = _paper_tick_from_fno(force=False)

    dashboard = get_fno_dashboard(instrument, trade_period=trade_period)
    chart_json = build_fno_chart_json(instrument)
    model_meta = load_model_meta()

    return render(request, "trading/fno.html", {
        "form": form,
        "instrument": instrument,
        "trade_period": trade_period,
        "signal": dashboard["signal"],
        "instruments": dashboard["instruments"],
        "results": dashboard["results"],
        "trades_data": dashboard["trades_data"],
        "strategy_trades": dashboard["strategy_trades"],
        "trade_summary": dashboard["trade_summary"],
        "recent_days": dashboard.get("recent_days"),
        "strategy_name": dashboard["strategy_name"],
        "strategy_filters": dashboard["strategy_filters"],
        "top_features": dashboard["top_features"],
        "chart_json": chart_json,
        "updated_at": dashboard["updated_at"],
        "history_coverage": get_history_coverage(),
        "model_meta": model_meta,
        "model_frozen": True,
        "paper_tick": paper_tick,
    })


@require_GET
def fno_signal_api(request: HttpRequest) -> JsonResponse:
    instrument = request.GET.get("instrument", "NIFTY").upper()
    force = request.GET.get("refresh") == "1"
    trade_period = request.GET.get("trades", "full")
    if trade_period not in ("full", "oos"):
        trade_period = "full"
    paper_tick = _paper_tick_from_fno(force=force)
    payload = get_fno_dashboard(instrument, force=force, trade_period=trade_period)
    payload["paper_tick"] = paper_tick
    payload["model_frozen"] = True
    return JsonResponse(payload)


@require_GET
def fno_chart_api(request: HttpRequest, instrument: str) -> JsonResponse:
    force = request.GET.get("refresh") == "1"
    chart = json.loads(build_fno_chart_json(instrument.upper(), force=force))
    return JsonResponse({"instrument": instrument.upper(), "chart": chart})


def paper_trading(request: HttpRequest) -> HttpResponse:
    """Paper trading dashboard — fake money, auto F&O orders."""
    from trading.services.paper_trading import get_dashboard, run_tick

    # Keep positions/signals fresh when page is opened
    run_tick(force_refresh=False)
    dash = get_dashboard()
    return render(request, "trading/paper.html", {"dash": dash})


@require_GET
def paper_status_api(request: HttpRequest) -> JsonResponse:
    from trading.services.paper_trading import get_dashboard, run_tick

    tick = request.GET.get("tick", "1") == "1"
    force = request.GET.get("refresh") == "1"
    tick_result = None
    if tick:
        tick_result = run_tick(force_refresh=force)
    dash = get_dashboard()
    return JsonResponse({"ok": True, "tick": tick_result, "dashboard": dash})


@require_POST
def paper_action_api(request: HttpRequest) -> JsonResponse:
    """JSON body: {action, ...} — auto_on, auto_off, capital, deposit, reset, close, manual_entry."""
    import json as _json

    from trading.models import PaperAccount, PaperPosition, PaperTrade
    from trading.services.fno_live import evaluate_signal
    from trading.services.paper_trading import (
        _latest_bar,
        close_position,
        deposit,
        get_dashboard,
        open_position_from_signal,
        reset_account,
        set_capital,
    )

    try:
        body = _json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        body = {}
    action = (body.get("action") or request.POST.get("action") or "").strip().lower()
    account = PaperAccount.get_active()

    try:
        if action == "auto_on":
            account.auto_trade = True
            account.save(update_fields=["auto_trade", "updated_at"])
        elif action == "auto_off":
            account.auto_trade = False
            account.save(update_fields=["auto_trade", "updated_at"])
        elif action == "capital":
            set_capital(account, float(body.get("amount", account.starting_capital)))
        elif action == "deposit":
            deposit(account, float(body.get("amount", 0)))
        elif action == "reset":
            cap = body.get("amount")
            reset_account(account, float(cap) if cap is not None else None)
        elif action == "risk":
            account.risk_pct = float(body.get("risk_pct", account.risk_pct))
            account.max_trades_per_day = int(
                body.get("max_trades_per_day", account.max_trades_per_day)
            )
            account.save(update_fields=["risk_pct", "max_trades_per_day", "updated_at"])
        elif action == "close":
            pos_id = int(body.get("position_id", 0))
            pos = get_object_or_404(
                PaperPosition, pk=pos_id, account=account, status=PaperPosition.STATUS_OPEN
            )
            bar = _latest_bar(pos.instrument)
            ltp = float(bar["close"]) if bar else float(pos.entry_price)
            close_position(pos, ltp, PaperTrade.EXIT_MANUAL)
        elif action == "manual_entry":
            instrument = str(body.get("instrument", "NIFTY")).upper()
            signal = evaluate_signal(instrument, force=True)
            if signal.get("status") != "active" and not body.get("force"):
                return JsonResponse(
                    {
                        "ok": False,
                        "error": f"No active signal on {instrument} (status={signal.get('status')})",
                    },
                    status=400,
                )
            pos = open_position_from_signal(account, signal, force=bool(body.get("force")))
            if not pos:
                return JsonResponse(
                    {"ok": False, "error": "Could not open position (margin / limits)"},
                    status=400,
                )
        else:
            return JsonResponse({"ok": False, "error": f"Unknown action: {action}"}, status=400)
    except ValueError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)

    return JsonResponse({"ok": True, "dashboard": get_dashboard()})


@require_GET
def api_signal(request: HttpRequest, symbol: str) -> JsonResponse:
    """HTMX/JSON endpoint for live signal check."""
    result = evaluate_stock(symbol.upper())
    return JsonResponse({
        "symbol": result.symbol,
        "date": str(result.eval_date),
        "is_valid": result.is_valid,
        "score": result.confluence_score,
        "entry": result.entry_price,
        "stop_loss": result.stop_loss,
        "target_2r": result.target_2r,
        "reasons": result.reasons,
        "rejections": result.rejection_reasons,
    })