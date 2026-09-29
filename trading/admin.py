from django.contrib import admin

from trading.models import (
    BacktestRun,
    DailyPrice,
    Signal,
    Stock,
    StrategyConfig,
    TradeJournalEntry,
    WatchlistItem,
)


@admin.register(StrategyConfig)
class StrategyConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "is_active", "risk_pct", "adx_min", "updated_at")


@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ("symbol", "name", "is_nifty200", "is_nifty500", "is_nifty_smallcap250", "sector", "last_price")
    list_filter = ("is_nifty200", "is_nifty100", "is_nifty500", "is_nifty_smallcap250", "is_active", "sector")
    search_fields = ("symbol", "name")


@admin.register(DailyPrice)
class DailyPriceAdmin(admin.ModelAdmin):
    list_display = ("stock", "date", "close", "volume")
    list_filter = ("date",)
    search_fields = ("stock__symbol",)


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ("stock", "date", "confluence_score", "is_valid", "risk_reward")
    list_filter = ("is_valid", "date")


@admin.register(WatchlistItem)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = ("stock", "added_at", "notes")


@admin.register(TradeJournalEntry)
class JournalAdmin(admin.ModelAdmin):
    list_display = ("stock", "entry_date", "status", "pnl", "quantity")


@admin.register(BacktestRun)
class BacktestRunAdmin(admin.ModelAdmin):
    list_display = ("name", "start_date", "end_date", "win_rate", "profit_factor", "created_at")