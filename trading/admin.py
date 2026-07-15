from django.contrib import admin

from trading.models import (
    BacktestRun,
    DailyPrice,
    PaperAccount,
    PaperEvent,
    PaperPosition,
    PaperTrade,
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
    list_display = ("symbol", "name", "is_nifty200", "sector", "last_price")
    list_filter = ("is_nifty200", "is_active", "sector")
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


@admin.register(PaperAccount)
class PaperAccountAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "is_active",
        "auto_trade",
        "cash",
        "margin_blocked",
        "realized_pnl",
        "risk_pct",
        "updated_at",
    )


@admin.register(PaperPosition)
class PaperPositionAdmin(admin.ModelAdmin):
    list_display = ("instrument", "side", "lots", "entry_price", "status", "entry_time")
    list_filter = ("status", "instrument", "side")


@admin.register(PaperTrade)
class PaperTradeAdmin(admin.ModelAdmin):
    list_display = (
        "instrument",
        "side",
        "lots",
        "entry_price",
        "exit_price",
        "pnl",
        "exit_reason",
        "session_date",
    )
    list_filter = ("exit_reason", "instrument", "session_date")


@admin.register(PaperEvent)
class PaperEventAdmin(admin.ModelAdmin):
    list_display = ("level", "message", "created_at", "account")
    list_filter = ("level",)