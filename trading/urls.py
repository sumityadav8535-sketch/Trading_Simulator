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
    path("api/fno/today/", views.fno_today_check_api, name="fno_today_check_api"),
    # F&O Long (Elite ML Long v1)
    path("fno/long/", views.fno_long_live, name="fno_long_live"),
    path("api/fno/long/signal/", views.fno_long_signal_api, name="fno_long_signal_api"),
    path("api/fno/long/chart/<str:instrument>/", views.fno_long_chart_api, name="fno_long_chart_api"),
    path("api/fno/long/backtest/", views.fno_long_backtest_status_api, name="fno_long_backtest_status_api"),
    path("api/fno/long/backtest/run/", views.fno_long_backtest_run_api, name="fno_long_backtest_run_api"),
    path("api/fno/long/today/", views.fno_long_today_check_api, name="fno_long_today_check_api"),
    path("intraday/15m/", views.intraday_15m_fade, name="intraday_15m_fade"),
    path("intraday/5m/", views.intraday_5m_hunt, name="intraday_5m_hunt"),
    path("intraday/gap/", views.intraday_gap, name="intraday_gap"),
    path("api/intraday/gap/live/", views.intraday_gap_live_api, name="intraday_gap_live_api"),
    path("api/intraday/history/", views.intraday_history_status_api, name="intraday_history_status_api"),
    path("api/intraday/history/sync/", views.intraday_history_sync_api, name="intraday_history_sync_api"),
    # Paper trading
    path("paper/", views.paper_trading, name="paper_trading"),
    path("api/paper/status/", views.paper_status_api, name="paper_status_api"),
    path("api/paper/action/", views.paper_action_api, name="paper_action_api"),
]
