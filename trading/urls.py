from django.urls import path

from trading import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    # F&O Live (+ history APIs used by the F&O page)
    path("fno/", views.fno_live, name="fno_live"),
    path("api/fno/signal/", views.fno_signal_api, name="fno_signal_api"),
    path("api/fno/chart/<str:instrument>/", views.fno_chart_api, name="fno_chart_api"),
    path("api/fno/backtest/", views.fno_backtest_status_api, name="fno_backtest_status_api"),
    path("api/fno/backtest/run/", views.fno_backtest_run_api, name="fno_backtest_run_api"),
    path("api/intraday/history/", views.intraday_history_status_api, name="intraday_history_status_api"),
    path("api/intraday/history/sync/", views.intraday_history_sync_api, name="intraday_history_sync_api"),
    # Paper trading
    path("paper/", views.paper_trading, name="paper_trading"),
    path("api/paper/status/", views.paper_status_api, name="paper_status_api"),
    path("api/paper/action/", views.paper_action_api, name="paper_action_api"),
]
