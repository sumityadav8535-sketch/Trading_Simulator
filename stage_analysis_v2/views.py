"""
Views for Stage Analysis 2.0.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

from django.contrib import messages
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from stage_analysis.services.stage_detector import normalize_ticker
from stage_analysis_v2.forms import AnalyzeForm, ScreenerForm, StageV2BacktestForm, WatchlistForm
from stage_analysis_v2.models import StockAnalysis, Watchlist
from stage_analysis_v2.services.analyzer import V2AnalysisResult, analyze_ticker_v2
from stage_analysis_v2.services.market_context import analyze_market_context
from stage_analysis_v2.services.screener import ScreenerFilters, scan_universe
from stage_analysis_v2.services.backtester import (
    ensure_signal_log,
    fill_performance_metrics,
    run_stage_v2_backtest,
    trades_as_json,
)
from stage_analysis_v2.services.signal_history import collect_stage_v2_signal_history
from stage_analysis_v2.services.cup_breakout import CupParams, run_cup_breakout_backtest
from stage_analysis_v2.services.strategy_catalog import (
    BACKTEST_STRATEGY_CHOICES,
    CUP_100_END,
    CUP_100_START,
    DEFAULT_STAGE_MAX_POS_PCT,
    DEFAULT_STRATEGY,
    STRATEGY_BLURBS,
    STRATEGY_CUP,
    STRATEGY_STAGE_V2,
    coerce_cup_params,
    coerce_supertrend_params,
    is_cup_strategy,
    is_supertrend_strategy,
    is_union_strategy,
    normalize_strategy,
    strategy_defaults,
)
from stage_analysis_v2.services.st_union_swing import run_st_union_backtest
from stage_analysis_v2.services.supertrend_swing import run_supertrend_swing_backtest
from stage_analysis_v2.services.tech_filters import DEFAULT_TECH_FILTER, TECH_FILTER_CHOICES
from stage_analysis_v2.services.top_picks import get_top_picks, get_sector_list
from trading.services.charts import (
    build_equity_curve,
    build_exit_breakdown_chart,
    build_monthly_returns_chart,
    build_signal_review_chart,
    build_stage_v2_signal_bars,
    build_win_loss_pie,
)
from trading.services.market_data import resolve_universe_symbols
from trading.services.nse_price_sync import nse_symbol_from_ticker


def _user(request: HttpRequest):
    return request.user if request.user.is_authenticated else None


def _parse_backtest_symbols(universe: str, symbol: str, symbols_raw: str) -> list[str]:
    if universe in ("nifty200", "nifty100", "nifty_smallcap250", "smallcap250"):
        return resolve_universe_symbols(universe)

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


def _form_date_value(value) -> str:
    """Render date field value for HTML input type=date (handles date or str)."""
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip()
    return s[:10] if len(s) >= 10 else s


def _parse_date_arg(raw: str | None, fallback: date) -> date:
    if not raw:
        return fallback
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return fallback


def _resolve_signals_range(
    period: str,
    start_str: str | None,
    end_str: str | None,
) -> tuple[date, date]:
    end = date.today()
    if end_str:
        end = _parse_date_arg(end_str, end)

    if period == "custom" and start_str:
        start = _parse_date_arg(start_str, end - timedelta(days=183))
        return start, end

    days_map = {
        "3m": 92,
        "6m": 183,
        "1y": 365,
        "2y": 730,
    }
    days = days_map.get(period, 183)  # default 6 months
    return end - timedelta(days=days), end


def signals(request: HttpRequest) -> HttpResponse:
    """Stage Analysis 2.0 historical signals chart page (default 6 months)."""
    default_end = date.today()
    default_start = default_end - timedelta(days=183)
    return render(
        request,
        "stage_analysis_v2/signals.html",
        {
            "default_period": "6m",
            "history_default_start": default_start.isoformat(),
            "history_default_end": default_end.isoformat(),
            "tech_filter_choices": TECH_FILTER_CHOICES,
            "default_tech_filter": DEFAULT_TECH_FILTER,
        },
    )


@require_GET
def signals_api(request: HttpRequest) -> JsonResponse:
    """JSON: daily Stage 2 signal bars + per-date stock cards (same engine as Backtest)."""
    from stage_analysis_v2.services.backtester import DEFAULT_EXIT_MODE

    period = (request.GET.get("period") or "6m").strip().lower()
    start_str = request.GET.get("start_date")
    end_str = request.GET.get("end_date")
    start_date, end_date = _resolve_signals_range(period, start_str, end_str)
    if start_date >= end_date:
        return JsonResponse({"error": "Start date must be before end date"}, status=400)

    try:
        min_quality = int(float(request.GET.get("min_quality") or 0))
    except (TypeError, ValueError):
        min_quality = 0
    min_quality = max(0, min(100, min_quality))
    market_filter = request.GET.get("market_filter") in ("1", "true", "on", "yes")
    tech_filter = (request.GET.get("tech_filter") or DEFAULT_TECH_FILTER).strip()
    exit_mode = (request.GET.get("exit_mode") or DEFAULT_EXIT_MODE).strip()
    try:
        capital = float(request.GET.get("capital") or 1_000_000)
    except (TypeError, ValueError):
        capital = 1_000_000.0
    capital = max(1000.0, capital)

    cache_key = (
        f"stage_v2_signals:v2:{start_date}:{end_date}:"
        f"{min_quality}:{int(market_filter)}:{tech_filter}:{exit_mode}:{int(capital)}"
    )
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)

    history = collect_stage_v2_signal_history(
        start_date=start_date,
        end_date=end_date,
        min_quality_score=min_quality,
        market_filter=market_filter,
        tech_filter=tech_filter,
        capital=capital,
        exit_mode=exit_mode,
    )
    backtest_url = (
        f"/stage-analysis-v2/backtest/?run=1&universe=nifty200"
        f"&strategy={STRATEGY_STAGE_V2}"
        f"&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        f"&capital={int(capital)}&min_quality_score={min_quality}"
        f"&tech_filter={history.tech_filter}&exit_mode={history.exit_mode}"
    )
    if market_filter:
        backtest_url += "&market_filter=on"

    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "period": period,
        "tech_filter": history.tech_filter,
        "tech_filter_label": history.tech_filter_label,
        "exit_mode": history.exit_mode,
        "exit_mode_label": history.exit_mode_label,
        "min_quality_score": history.min_quality_score,
        "market_filter": history.market_filter,
        "capital": history.capital,
        "summary": history.summary,
        "daily": history.daily,
        "by_date": history.by_date,
        "chart": json.loads(build_stage_v2_signal_bars(history.daily)),
        "backtest_url": backtest_url,
        "note": (
            "Signals = all Stage 2 entries. Trades = signals the shared-capital "
            "backtest actually took (cash / cooldown can skip some)."
        ),
    }
    cache.set(cache_key, payload, 1800)  # 30 min
    return JsonResponse(payload)


def _opt_float(raw: str | None):
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@require_GET
def signal_chart_api(request: HttpRequest) -> JsonResponse:
    """Plotly JSON for one backtest signal: candles + signal/entry/stop/exit marks."""
    symbol = (request.GET.get("symbol") or "").strip().upper()
    signal_date = (request.GET.get("signal_date") or "").strip()[:10]
    if not symbol or not signal_date:
        return JsonResponse(
            {"error": "symbol and signal_date are required"},
            status=400,
        )
    try:
        chart = json.loads(build_signal_review_chart(
            symbol=symbol,
            signal_date=signal_date,
            entry_date=(request.GET.get("entry_date") or "").strip()[:10] or None,
            exit_date=(request.GET.get("exit_date") or "").strip()[:10] or None,
            stop_loss=_opt_float(request.GET.get("stop_loss")),
            target=_opt_float(request.GET.get("target")),
            entry_price=_opt_float(request.GET.get("entry_price")),
            exit_price=_opt_float(request.GET.get("exit_price")),
        ))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"symbol": symbol, "signal_date": signal_date, "chart": chart})


def backtest(request: HttpRequest) -> HttpResponse:
    from stage_analysis_v2.services.backtester import (
        DEFAULT_COOLDOWN_DAYS,
        DEFAULT_ENTRY_ON,
        DEFAULT_ENTRY_STAGE,
        DEFAULT_EXIT_MODE,
        DEFAULT_MA_PERIOD,
        DEFAULT_MA_TYPE,
        DEFAULT_MAX_HOLD_DAYS,
        DEFAULT_RISK_PCT,
        DEFAULT_STOP_MA_MULT,
        DEFAULT_TARGET_RR,
        DEFAULT_TRAIL_MA_MULT,
        EntryFilters,
        MA_COND_ABOVE,
        MA_COND_NONE,
    )
    from stage_analysis_v2.services.tech_filters import DEFAULT_TECH_FILTER

    default_end = date.today()
    default_start = default_end - timedelta(days=365)  # last 1 year
    run_requested = request.GET.get("run") == "1"

    # Merge defaults so old bookmarks / missing fields still get new filters
    if run_requested:
        data = request.GET.copy()
        data.setdefault("strategy", DEFAULT_STRATEGY)
        data.setdefault("universe", "nifty200")
        data.setdefault("exit_mode", DEFAULT_EXIT_MODE)
        data.setdefault("tech_filter", DEFAULT_TECH_FILTER)
        data.setdefault("capital", "1000000")
        data.setdefault("min_quality_score", "0")
        data.setdefault("min_rs_rating", "0")
        data.setdefault("entry_stage", str(DEFAULT_ENTRY_STAGE))
        data.setdefault("entry_on", DEFAULT_ENTRY_ON)
        data.setdefault("target_rr", str(DEFAULT_TARGET_RR))
        data.setdefault("max_hold_days", str(DEFAULT_MAX_HOLD_DAYS))
        data.setdefault("stop_ma_mult", str(DEFAULT_STOP_MA_MULT))
        data.setdefault("trail_ma_mult", str(DEFAULT_TRAIL_MA_MULT))
        data.setdefault("risk_pct", str(DEFAULT_RISK_PCT))
        data.setdefault("cooldown_days", str(DEFAULT_COOLDOWN_DAYS))
        data.setdefault("min_price", "0")
        data.setdefault("max_price", "0")
        data.setdefault("min_volume", "0")
        data.setdefault("min_volume_ratio", "0")
        data.setdefault("ma_condition", MA_COND_NONE)
        data.setdefault("ma_period", str(DEFAULT_MA_PERIOD))
        data.setdefault("ma_type", DEFAULT_MA_TYPE)
        data.setdefault("max_pos_pct", str(int(DEFAULT_STAGE_MAX_POS_PCT)))
        if not data.get("start_date"):
            data["start_date"] = default_start.isoformat()
        if not data.get("end_date"):
            data["end_date"] = default_end.isoformat()
        # Supertrend: the form always posts hold/risk/cooldown/max-pos, which
        # start as Stage 2.0 leftovers (2% / 65d / 40d / 100%). Coerce those
        # to the researched ST pack unless the user actually customized them.
        sid = normalize_strategy(data.get("strategy"))
        if is_supertrend_strategy(sid):
            coerced = coerce_supertrend_params(
                sid,
                risk_pct=data.get("risk_pct"),
                max_hold_days=data.get("max_hold_days"),
                cooldown_days=data.get("cooldown_days"),
                max_pos_pct=data.get("max_pos_pct"),
            )
            for key, val in coerced.items():
                data[key] = val
        elif is_cup_strategy(sid):
            coerced = coerce_cup_params(
                sid,
                risk_pct=data.get("risk_pct"),
                max_hold_days=data.get("max_hold_days"),
                cooldown_days=data.get("cooldown_days"),
                max_pos_pct=data.get("max_pos_pct"),
                target_rr=data.get("target_rr"),
            )
            for key, val in coerced.items():
                data[key] = val
        form = StageV2BacktestForm(data)
    else:
        form = StageV2BacktestForm(
            initial={
                "strategy": DEFAULT_STRATEGY,
                "start_date": default_start,
                "end_date": default_end,
                "universe": "nifty200",
                "exit_mode": DEFAULT_EXIT_MODE,
                "tech_filter": DEFAULT_TECH_FILTER,
                "capital": 1_000_000,
                "min_quality_score": 0,
                "min_rs_rating": 0,
                "shared_capital": True,
                "entry_stage": DEFAULT_ENTRY_STAGE,
                "entry_on": DEFAULT_ENTRY_ON,
                "target_rr": DEFAULT_TARGET_RR,
                "max_hold_days": DEFAULT_MAX_HOLD_DAYS,
                "stop_ma_mult": DEFAULT_STOP_MA_MULT,
                "trail_ma_mult": DEFAULT_TRAIL_MA_MULT,
                "risk_pct": DEFAULT_RISK_PCT,
                "cooldown_days": DEFAULT_COOLDOWN_DAYS,
                "min_price": 0,
                "max_price": 0,
                "min_volume": 0,
                "min_volume_ratio": 0,
                "ma_condition": MA_COND_NONE,
                "ma_period": DEFAULT_MA_PERIOD,
                "ma_type": DEFAULT_MA_TYPE,
                "max_pos_pct": DEFAULT_STAGE_MAX_POS_PCT,
            },
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
            strategy_id = normalize_strategy(cd.get("strategy"))
            exit_mode = cd.get("exit_mode") or DEFAULT_EXIT_MODE
            tech_filter = cd.get("tech_filter") or DEFAULT_TECH_FILTER
            entry_stage = int(cd.get("entry_stage") or DEFAULT_ENTRY_STAGE)
            entry_on = cd.get("entry_on") or DEFAULT_ENTRY_ON
            entry_filters = EntryFilters(
                min_price=float(cd.get("min_price") or 0),
                max_price=float(cd.get("max_price") or 0),
                min_volume=float(cd.get("min_volume") or 0),
                min_volume_ratio=float(cd.get("min_volume_ratio") or 0),
                ma_period=int(cd.get("ma_period") or 0),
                ma_type=(cd.get("ma_type") or DEFAULT_MA_TYPE),
                ma_condition=(cd.get("ma_condition") or MA_COND_NONE),
            )
            if is_cup_strategy(strategy_id):
                cup_defs = strategy_defaults(strategy_id)
                cup_params = CupParams(
                    cup_min_days=int(cd.get("cup_min_days") or 20),
                    cup_max_days=int(cd.get("cup_max_days") or 180),
                    min_depth_pct=float(cd.get("min_depth_pct") or 12),
                    max_depth_pct=float(cd.get("max_depth_pct") or 45),
                    recovery_pct=float(cd.get("recovery_pct") or 90),
                    handle_min_days=int(cd.get("handle_min_days") or 5),
                    handle_max_days=int(cd.get("handle_max_days") or 30),
                    handle_max_depth_pct=float(cd.get("handle_max_depth_pct") or 15),
                    require_handle=bool(cd.get("require_handle")),
                    breakout_buffer_pct=float(cd.get("breakout_buffer_pct") or 0.5),
                    vol_mult=float(cd.get("vol_mult") or 1.1),
                    rsi_min=float(cd.get("rsi_min") or 40),
                    rsi_max=float(cd.get("rsi_max") or 85),
                    max_gap_pct=float(cd.get("max_gap_pct") or 5),
                    min_close_loc=float(cd.get("min_close_loc") or 0.70),
                    require_close_strength=bool(cd.get("require_close_strength")),
                    require_sma200_rising=bool(cd.get("require_sma200_rising")),
                    require_rs_vs_nifty=bool(cd.get("require_rs_vs_nifty")),
                    require_nifty_sma200=bool(cd.get("require_nifty_sma200")),
                    require_trend_stack=bool(cd.get("require_trend_stack", True)),
                    stop_atr_mult=float(cd.get("stop_atr_mult") or 0.5),
                    entry_mode=cd.get("entry_mode") or "next_open",
                    retest_tol_pct=float(cd.get("retest_tol_pct") or 2),
                    retest_max_days=int(cd.get("retest_max_days") or 10),
                    cup_exit_mode=cd.get("cup_exit_mode") or "ema20_trail",
                    target_rr=float(cd.get("target_rr") or cup_defs.get("target_rr") or 2),
                    min_bottom_days=int(cd.get("min_bottom_days") or 5),
                    max_new_per_day=int(cd.get("max_new_per_day") or 10),
                    min_left_days=7,
                    min_recovery_days=5,
                    pivot_width=5,
                    nifty_ema_period=int(
                        cd["nifty_ema_period"]
                        if cd.get("nifty_ema_period") is not None
                        else 20
                    ),
                    loss_streak=int(
                        cd["loss_streak"] if cd.get("loss_streak") is not None else 3
                    ),
                    loss_streak_cooloff_days=int(
                        cd["loss_streak_cooloff_days"]
                        if cd.get("loss_streak_cooloff_days") is not None
                        else 10
                    ),
                )
                bt_result = run_cup_breakout_backtest(
                    symbols=symbol_list,
                    start_date=cd["start_date"],
                    end_date=cd["end_date"],
                    capital=float(cd["capital"]),
                    strategy_id=strategy_id,
                    risk_pct=float(cd.get("risk_pct") or cup_defs["risk_pct"]),
                    max_hold_days=int(cd.get("max_hold_days") or cup_defs["max_hold_days"]),
                    cooldown_days=int(
                        cd.get("cooldown_days")
                        if cd.get("cooldown_days") is not None
                        else cup_defs["cooldown_days"]
                    ),
                    max_pos_pct=float(cd.get("max_pos_pct") or cup_defs["max_pos_pct"]),
                    entry_filters=entry_filters,
                    params=cup_params,
                )
            elif is_union_strategy(strategy_id):
                st_defs = strategy_defaults(strategy_id)
                bt_result = run_st_union_backtest(
                    symbols=symbol_list,
                    start_date=cd["start_date"],
                    end_date=cd["end_date"],
                    capital=float(cd["capital"]),
                    strategy_id=strategy_id,
                    risk_pct=float(cd.get("risk_pct") or st_defs["risk_pct"]),
                    max_hold_days=int(cd.get("max_hold_days") or st_defs["max_hold_days"]),
                    cooldown_days=int(
                        cd.get("cooldown_days")
                        if cd.get("cooldown_days") is not None
                        else st_defs["cooldown_days"]
                    ),
                    max_pos_pct=float(cd.get("max_pos_pct") or st_defs["max_pos_pct"]),
                    entry_filters=entry_filters,
                )
            elif is_supertrend_strategy(strategy_id):
                st_defs = strategy_defaults(strategy_id)
                bt_result = run_supertrend_swing_backtest(
                    symbols=symbol_list,
                    start_date=cd["start_date"],
                    end_date=cd["end_date"],
                    capital=float(cd["capital"]),
                    strategy_id=strategy_id,
                    risk_pct=float(cd.get("risk_pct") or st_defs["risk_pct"]),
                    max_hold_days=int(cd.get("max_hold_days") or st_defs["max_hold_days"]),
                    cooldown_days=int(
                        cd.get("cooldown_days")
                        if cd.get("cooldown_days") is not None
                        else st_defs["cooldown_days"]
                    ),
                    max_pos_pct=float(cd.get("max_pos_pct") or st_defs["max_pos_pct"]),
                    entry_filters=entry_filters,
                )
            else:
                bt_result = run_stage_v2_backtest(
                    symbols=symbol_list,
                    start_date=cd["start_date"],
                    end_date=cd["end_date"],
                    capital=float(cd["capital"]),
                    min_quality_score=int(cd.get("min_quality_score") or 0),
                    min_rs_rating=float(cd.get("min_rs_rating") or 0),
                    market_filter=bool(cd.get("market_filter")),
                    exit_mode=exit_mode,
                    tech_filter=tech_filter,
                    entry_stage=entry_stage,
                    entry_on=entry_on,
                    entry_filters=entry_filters,
                    target_rr=float(cd.get("target_rr") or DEFAULT_TARGET_RR),
                    max_hold_days=int(cd.get("max_hold_days") or DEFAULT_MAX_HOLD_DAYS),
                    stop_ma_mult=float(cd.get("stop_ma_mult") or DEFAULT_STOP_MA_MULT),
                    trail_ma_mult=float(cd.get("trail_ma_mult") or DEFAULT_TRAIL_MA_MULT),
                    risk_pct=float(cd.get("risk_pct") or DEFAULT_RISK_PCT),
                    cooldown_days=int(cd.get("cooldown_days") if cd.get("cooldown_days") is not None else DEFAULT_COOLDOWN_DAYS),
                    shared_capital=(
                        True if cd.get("shared_capital") is None else bool(cd.get("shared_capital"))
                    ),
                )
            fill_performance_metrics(bt_result)
            ensure_signal_log(bt_result)
            if bt_result.equity_curve:
                charts["equity"] = build_equity_curve(bt_result.equity_curve)
            if bt_result.monthly_returns:
                charts["monthly"] = build_monthly_returns_chart(bt_result.monthly_returns)
            if bt_result.exit_breakdown:
                charts["exits"] = build_exit_breakdown_chart(bt_result.exit_breakdown)
            if bt_result.trades:
                wins = sum(1 for t in bt_result.trades if t.pnl > 0)
                charts["win_loss"] = build_win_loss_pie(wins, len(bt_result.trades) - wins)
    elif run_requested and form.errors:
        messages.error(
            request,
            "Backtest form has errors — check dates, universe, and filters below.",
        )

    # Defaults shared with Stage Analysis 2.0 Signals (same engine) for "Reset to default"
    form_defaults = {
        "strategy": DEFAULT_STRATEGY,
        "universe": "nifty200",
        "symbol": "RELIANCE",
        "symbols": "",
        "start_date": default_start.isoformat(),
        "end_date": default_end.isoformat(),
        "capital": "1000000",
        "entry_stage": str(DEFAULT_ENTRY_STAGE),
        "entry_on": DEFAULT_ENTRY_ON,
        "tech_filter": DEFAULT_TECH_FILTER,
        "market_filter": False,
        "exit_mode": DEFAULT_EXIT_MODE,
        "target_rr": str(DEFAULT_TARGET_RR),
        "max_hold_days": str(DEFAULT_MAX_HOLD_DAYS),
        "stop_ma_mult": str(DEFAULT_STOP_MA_MULT),
        "trail_ma_mult": str(DEFAULT_TRAIL_MA_MULT),
        "risk_pct": str(DEFAULT_RISK_PCT),
        "cooldown_days": str(DEFAULT_COOLDOWN_DAYS),
        "min_price": "0",
        "max_price": "0",
        "min_volume": "0",
        "min_volume_ratio": "0",
        "ma_condition": MA_COND_NONE,
        "ma_period": str(DEFAULT_MA_PERIOD),
        "ma_type": DEFAULT_MA_TYPE,
        "min_quality_score": "0",
        "min_rs_rating": "0",
        "shared_capital": True,
        "max_pos_pct": str(int(DEFAULT_STAGE_MAX_POS_PCT)),
    }
    strategy_presets = {
        sid: strategy_defaults(sid) for sid, _ in BACKTEST_STRATEGY_CHOICES
    }
    # Researched pack: ≥300% on 2023-01-01 → 2026-02-28 Nifty 200 (shared capital)
    aggressive_300_defaults = {
        **form_defaults,
        "exit_mode": "no_stage",
        "target_rr": "4",
        "max_hold_days": "90",
        "risk_pct": "5",
        "cooldown_days": "5",
        "tech_filter": DEFAULT_TECH_FILTER,
        "entry_stage": "2",
        "entry_on": DEFAULT_ENTRY_ON,
        "start_date": "2023-01-01",
        "end_date": "2026-02-28",
    }
    cup_100_defaults = {
        **form_defaults,
        **strategy_defaults(STRATEGY_CUP),
        "strategy": STRATEGY_CUP,
        "universe": "nifty200",
        "start_date": CUP_100_START,
        "end_date": CUP_100_END,
        "capital": "1000000",
    }
    # Last-1y Nifty 200: RS≥70 + price > SMA150 → WR 49%→59%, return 12%→20%, DD 8%→5%.
    rs70_pack_defaults = {
        **form_defaults,
        "strategy": DEFAULT_STRATEGY,
        "universe": "nifty200",
        "min_rs_rating": "70",
        "ma_condition": MA_COND_ABOVE,
        "ma_period": "150",
        "ma_type": "sma",
        "min_quality_score": "0",
        "tech_filter": DEFAULT_TECH_FILTER,
    }

    return render(request, "stage_analysis_v2/backtest.html", {
        "form": form,
        "bt_result": bt_result,
        "signal_log": list(bt_result.signal_log) if bt_result else [],
        "trade_log": trades_as_json(bt_result) if bt_result else [],
        "charts": charts,
        "symbol_list": symbol_list,
        "run_requested": run_requested,
        "form_defaults": form_defaults,
        "aggressive_300_defaults": aggressive_300_defaults,
        "cup_100_defaults": cup_100_defaults,
        "rs70_pack_defaults": rs70_pack_defaults,
        "strategy_presets": strategy_presets,
        "strategy_blurbs": STRATEGY_BLURBS,
        "start_date_str": _form_date_value(
            form["start_date"].value() if form.is_bound else form.initial.get("start_date")
        ),
        "end_date_str": _form_date_value(
            form["end_date"].value() if form.is_bound else form.initial.get("end_date")
        ),
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