"""
Models for Stage Analysis 2.0 — confluence-based Weinstein staging.
"""
from django.conf import settings
from django.db import models


class Stage(models.IntegerChoices):
    ACCUMULATION = 1, "Stage 1 — Accumulation"
    ADVANCING = 2, "Stage 2 — Advancing"
    DISTRIBUTION = 3, "Stage 3 — Distribution"
    DECLINING = 4, "Stage 4 — Declining"


class BreakoutType(models.TextChoices):
    NONE = "none", "No Breakout"
    WEAK = "weak", "Weak Breakout"
    CLEAN = "clean", "Clean Breakout"


class Watchlist(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="stage_v2_watchlist",
    )
    ticker = models.CharField(max_length=20, db_index=True)
    company_name = models.CharField(max_length=200, blank=True, default="")
    sector = models.CharField(max_length=100, blank=True, default="")
    notes = models.CharField(max_length=255, blank=True, default="")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-added_at"]
        unique_together = [["user", "ticker"]]
        verbose_name_plural = "watchlist items"

    def __str__(self) -> str:
        return self.ticker


class StockAnalysis(models.Model):
    """Cached V2 analysis with scalar fields for screener queries."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="stage_v2_analyses",
    )
    ticker = models.CharField(max_length=20, db_index=True)
    company_name = models.CharField(max_length=200, blank=True, default="")
    sector = models.CharField(max_length=100, blank=True, default="")

    weekly_stage = models.PositiveSmallIntegerField(choices=Stage.choices)
    daily_stage = models.PositiveSmallIntegerField(choices=Stage.choices)
    quality_score = models.PositiveSmallIntegerField(default=0, db_index=True)

    rs_rating = models.FloatField(default=0.0, help_text="Relative strength score 0-100")
    rs_trend = models.CharField(max_length=20, default="flat")
    benchmark = models.CharField(max_length=20, default="NIFTY50")

    breakout_type = models.CharField(
        max_length=10, choices=BreakoutType.choices, default=BreakoutType.NONE
    )
    market_favorable = models.BooleanField(default=False)

    current_price = models.DecimalField(max_digits=14, decimal_places=4)
    ma_30w = models.DecimalField(max_digits=14, decimal_places=4)
    ma_150d = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)

    primary_entry = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    pullback_entry = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    stop_loss = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    target_price = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    risk_reward = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True)

    suggested_action = models.CharField(max_length=100, blank=True, default="")
    score_breakdown = models.JSONField(default=dict)
    analysis_details = models.JSONField(default=dict)
    chart_payload = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-quality_score", "-updated_at"]
        unique_together = [["user", "ticker"]]

    def __str__(self) -> str:
        return f"{self.ticker} W{self.weekly_stage}/D{self.daily_stage} Q{self.quality_score}"

    @property
    def quality_tier(self) -> str:
        if self.weekly_stage == 2 and self.quality_score >= 75:
            return "high"
        if self.weekly_stage == 2 and self.quality_score >= 50:
            return "average"
        if self.weekly_stage == 1:
            return "base"
        if self.weekly_stage == 3:
            return "top"
        return "decline"


class ScanCache(models.Model):
    """Cached screener / top-picks results for fast dashboard load."""

    key = models.CharField(max_length=64, primary_key=True)
    payload = models.JSONField(default=list)
    item_count = models.PositiveIntegerField(default=0)
    filters = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.key} ({self.item_count} items)"