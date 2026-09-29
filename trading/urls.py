from django.urls import path

from trading import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("intraday/15m/", views.intraday_15m_fade, name="intraday_15m_fade"),
    path("intraday/5m/", views.intraday_5m_hunt, name="intraday_5m_hunt"),
    path("intraday/gap/", views.intraday_gap, name="intraday_gap"),
    path("fundamentals/", views.fundamental_swing, name="fundamental_swing"),
    path("swing/", views.short_swing, name="short_swing"),
    path("api/intraday/gap/live/", views.intraday_gap_live_api, name="intraday_gap_live_api"),
    path("api/intraday/history/", views.intraday_history_status_api, name="intraday_history_status_api"),
    path("api/intraday/history/sync/", views.intraday_history_sync_api, name="intraday_history_sync_api"),
    path("api/intraday/startup/", views.localhost_auto_refresh_status_api, name="localhost_auto_refresh_status_api"),
    path("api/intraday/startup/run/", views.localhost_auto_refresh_run_api, name="localhost_auto_refresh_run_api"),
]
