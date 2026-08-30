from django.urls import path

from strategy_builder import views

app_name = "strategy_builder"

urlpatterns = [
    path("", views.builder_page, name="builder"),
    path("api/catalog/", views.catalog_api, name="catalog_api"),
    path("api/presets/", views.presets_api, name="presets_api"),
    path("api/explain/", views.explain_api, name="explain_api"),
    path("api/backtest/", views.backtest_api, name="backtest_api"),
    path("api/optimize/", views.optimize_api, name="optimize_api"),
    path("api/strategies/", views.strategies_api, name="strategies_api"),
    path("api/strategies/<int:pk>/", views.strategy_detail_api, name="strategy_detail_api"),
]
