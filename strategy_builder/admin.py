from django.contrib import admin

from strategy_builder.models import SavedStrategy, StrategyVersion


@admin.register(SavedStrategy)
class SavedStrategyAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "updated_at", "is_favorite")
    search_fields = ("name",)


@admin.register(StrategyVersion)
class StrategyVersionAdmin(admin.ModelAdmin):
    list_display = ("strategy", "version", "label", "created_at")
