"""Strategy Builder — single-page UI + JSON APIs."""
from __future__ import annotations

import json
from datetime import date, timedelta

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from strategy_builder.models import SavedStrategy, StrategyVersion
from strategy_builder.services.backtester import result_to_dict, run_strategy_backtest
from strategy_builder.services.catalog import catalog_payload, default_strategy, presets
from strategy_builder.services.expression import strategy_explanation, strategy_expression
from strategy_builder.services.optimizer import run_simple_grid
from strategy_builder.services.validate import StrategyValidationError, validate_strategy
from trading.services.charts import build_drawdown_chart, build_equity_curve, build_monthly_returns_chart


def _user(request: HttpRequest):
    return request.user if request.user.is_authenticated else None


def _parse_json(request: HttpRequest) -> dict:
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def builder_page(request: HttpRequest) -> HttpResponse:
    end = date.today()
    start = end - timedelta(days=365)
    return render(request, "strategy_builder/builder.html", {
        "default_start": start.isoformat(),
        "default_end": end.isoformat(),
        "catalog_json": json.dumps(catalog_payload()),
        "default_strategy_json": json.dumps(default_strategy()),
        "presets_json": json.dumps(presets()),
    })


@require_GET
def catalog_api(request: HttpRequest) -> JsonResponse:
    return JsonResponse(catalog_payload())


@require_GET
def presets_api(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"presets": presets()})


@require_POST
def explain_api(request: HttpRequest) -> JsonResponse:
    body = _parse_json(request)
    try:
        definition = validate_strategy(body.get("definition") or body)
    except StrategyValidationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({
        "expression": strategy_expression(definition),
        "explanation": strategy_explanation(definition),
        "definition": definition,
    })


@require_POST
def backtest_api(request: HttpRequest) -> JsonResponse:
    body = _parse_json(request)
    try:
        definition = validate_strategy(body.get("definition") or body)
    except StrategyValidationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    end = date.today()
    start = end - timedelta(days=365)
    if body.get("start_date"):
        try:
            start = date.fromisoformat(str(body["start_date"])[:10])
        except ValueError:
            pass
    if body.get("end_date"):
        try:
            end = date.fromisoformat(str(body["end_date"])[:10])
        except ValueError:
            pass
    if end < start:
        start, end = end, start

    try:
        result = run_strategy_backtest(definition, start_date=start, end_date=end)
    except Exception as exc:
        return JsonResponse({"error": f"Backtest failed: {exc}"}, status=500)

    payload = result_to_dict(result)
    charts = {}
    if result.equity_curve:
        charts["equity"] = json.loads(build_equity_curve(result.equity_curve))
    if result.drawdown_curve:
        # reuse drawdown builder from charts if available shape
        try:
            from trading.services.charts import build_drawdown_chart as bdd
            # stage uses equity curve based dd — we have drawdown_curve already
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p["date"] for p in result.drawdown_curve],
                y=[-p["dd"] for p in result.drawdown_curve],
                fill="tozeroy", mode="lines",
                line=dict(color="#ef4444"), name="Drawdown",
            ))
            fig.update_layout(title="Drawdown %", template="plotly_white", height=320,
                              margin=dict(l=40, r=20, t=50, b=40))
            charts["drawdown"] = json.loads(fig.to_json())
        except Exception:
            pass
    if result.monthly_returns:
        charts["monthly"] = json.loads(build_monthly_returns_chart(result.monthly_returns))

    return JsonResponse({
        "result": payload,
        "charts": charts,
        "expression": strategy_expression(definition),
        "explanation": strategy_explanation(definition),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    })


@require_POST
def optimize_api(request: HttpRequest) -> JsonResponse:
    body = _parse_json(request)
    try:
        definition = validate_strategy(body.get("definition") or body)
    except StrategyValidationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    end = date.today()
    start = end - timedelta(days=365)
    if body.get("start_date"):
        try:
            start = date.fromisoformat(str(body["start_date"])[:10])
        except ValueError:
            pass
    if body.get("end_date"):
        try:
            end = date.fromisoformat(str(body["end_date"])[:10])
        except ValueError:
            pass
    grid = run_simple_grid(definition, start, end)
    return JsonResponse(grid)


@require_http_methods(["GET", "POST"])
def strategies_api(request: HttpRequest) -> JsonResponse:
    user = _user(request)
    if request.method == "GET":
        qs = SavedStrategy.objects.filter(user=user) if user else SavedStrategy.objects.filter(user=None)
        return JsonResponse({
            "strategies": [
                {
                    "id": s.id,
                    "name": s.name,
                    "updated_at": s.updated_at.isoformat(),
                    "is_favorite": s.is_favorite,
                    "definition": s.definition,
                }
                for s in qs[:100]
            ]
        })

    body = _parse_json(request)
    try:
        definition = validate_strategy(body.get("definition") or body)
    except StrategyValidationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    name = (body.get("name") or definition.get("name") or "My Strategy")[:120]
    strategy_id = body.get("id")
    label = (body.get("version_label") or "")[:200]

    if strategy_id:
        qs = SavedStrategy.objects.filter(pk=strategy_id)
        if user:
            qs = qs.filter(user=user)
        else:
            qs = qs.filter(user=None)
        strat = get_object_or_404(qs)
        strat.name = name
        strat.definition = definition
        strat.save()
    else:
        strat = SavedStrategy.objects.create(
            user=user, name=name, definition=definition,
        )

    last_v = strat.versions.order_by("-version").first()
    ver_num = (last_v.version + 1) if last_v else 1
    StrategyVersion.objects.create(
        strategy=strat,
        version=ver_num,
        definition=definition,
        label=label or f"v{ver_num}",
    )
    return JsonResponse({
        "id": strat.id,
        "name": strat.name,
        "version": ver_num,
        "definition": definition,
    })


@require_http_methods(["GET", "DELETE"])
def strategy_detail_api(request: HttpRequest, pk: int) -> JsonResponse:
    user = _user(request)
    qs = SavedStrategy.objects.filter(pk=pk)
    if user:
        qs = qs.filter(user=user)
    else:
        qs = qs.filter(user=None)
    strat = get_object_or_404(qs)
    if request.method == "DELETE":
        strat.delete()
        return JsonResponse({"ok": True})
    versions = [
        {
            "version": v.version,
            "label": v.label,
            "created_at": v.created_at.isoformat(),
            "definition": v.definition,
        }
        for v in strat.versions.all()[:30]
    ]
    return JsonResponse({
        "id": strat.id,
        "name": strat.name,
        "definition": strat.definition,
        "versions": versions,
    })
