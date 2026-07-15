from django.urls import path

from trading import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("screener/", views.screener, name="screener"),
    path("scanner/", views.scanner, name="scanner"),
    path("api/scanner/history/", views.scanner_history_api, name="scanner_history_api"),
    path("chart/", views.stock_chart, name="chart"),
    path("risk/", views.risk_calculator, name="risk_calculator"),
    path("backtest/", views.backtest_view, name="backtest"),
    path("signals/", views.signals_view, name="signals"),
    path("watchlist/", views.watchlist, name="watchlist"),
    path("watchlist/<int:pk>/remove/", views.watchlist_remove, name="watchlist_remove"),
    path("journal/", views.journal, name="journal"),
    path("api/signal/<str:symbol>/", views.api_signal, name="api_signal"),
    path("intraday/", views.intraday, name="intraday"),
    path("api/intraday/quotes/", views.intraday_quotes_api, name="intraday_quotes_api"),
    path("api/intraday/chart/<str:symbol>/", views.intraday_chart_api, name="intraday_chart_api"),
    path("api/intraday/history/", views.intraday_history_status_api, name="intraday_history_status_api"),
    path("api/intraday/history/sync/", views.intraday_history_sync_api, name="intraday_history_sync_api"),
    path("fno/", views.fno_live, name="fno_live"),
    path("api/fno/signal/", views.fno_signal_api, name="fno_signal_api"),
    path("api/fno/chart/<str:instrument>/", views.fno_chart_api, name="fno_chart_api"),
    path("api/fno/backtest/", views.fno_backtest_status_api, name="fno_backtest_status_api"),
    path("api/fno/backtest/run/", views.fno_backtest_run_api, name="fno_backtest_run_api"),
    path("paper/", views.paper_trading, name="paper_trading"),
    path("api/paper/status/", views.paper_status_api, name="paper_status_api"),
    path("api/paper/action/", views.paper_action_api, name="paper_action_api"),
]