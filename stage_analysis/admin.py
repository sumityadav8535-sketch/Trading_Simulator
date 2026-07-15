from django.contrib import admin

from stage_analysis.models import StockAnalysis, Watchlist


@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = ("ticker", "company_name", "user", "added_at", "notes")
    search_fields = ("ticker", "company_name")
    list_filter = ("added_at",)


@admin.register(StockAnalysis)
class StockAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        "ticker",
        "stage",
        "current_price",
        "ma_30w",
        "suggested_action",
        "updated_at",
    )
    list_filter = ("stage", "updated_at")
    search_fields = ("ticker", "company_name")