from django.contrib import admin

from stage_analysis_v2.models import ScanCache, StockAnalysis, Watchlist


@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = ("ticker", "company_name", "sector", "user", "added_at")
    search_fields = ("ticker", "company_name")


@admin.register(ScanCache)
class ScanCacheAdmin(admin.ModelAdmin):
    list_display = ("key", "item_count", "updated_at")


@admin.register(StockAnalysis)
class StockAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        "ticker", "weekly_stage", "daily_stage", "quality_score",
        "rs_rating", "breakout_type", "updated_at",
    )
    list_filter = ("weekly_stage", "daily_stage", "breakout_type", "market_favorable")
    search_fields = ("ticker", "company_name", "sector")