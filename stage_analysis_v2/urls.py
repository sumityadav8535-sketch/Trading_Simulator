from django.urls import path

from stage_analysis_v2 import views

app_name = "stage_analysis_v2"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("stock/<str:ticker>/", views.stock_detail, name="stock_detail"),
    path("watchlist/", views.watchlist_view, name="watchlist"),
    path("watchlist/<int:pk>/remove/", views.watchlist_remove, name="watchlist_remove"),
    path("watchlist/add/<str:ticker>/", views.add_to_watchlist, name="add_to_watchlist"),
    path("screener/", views.screener, name="screener"),
    path("backtest/", views.backtest, name="backtest"),
]