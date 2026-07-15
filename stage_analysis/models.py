"""
Models for Stan Weinstein Stage Analysis feature.
"""
from django.conf import settings
from django.db import models


class Stage(models.IntegerChoices):
    ACCUMULATION = 1, "Stage 1 — Accumulation"
    ADVANCING = 2, "Stage 2 — Advancing"
    DISTRIBUTION = 3, "Stage 3 — Distribution"
    DECLINING = 4, "Stage 4 — Declining"


class Watchlist(models.Model):
    """User-tracked tickers for stage analysis (US or Indian symbols)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="stage_watchlist",
    )
    ticker = models.CharField(max_length=20, db_index=True)
    company_name = models.CharField(max_length=200, blank=True, default="")
    notes = models.CharField(max_length=255, blank=True, default="")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-added_at"]
        unique_together = [["user", "ticker"]]
        verbose_name_plural = "watchlist items"

    def __str__(self) -> str:
        return self.ticker


class StockAnalysis(models.Model):
    """Cached stage analysis result for a ticker."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="stage_analyses",
    )
    ticker = models.CharField(max_length=20, db_index=True)
    company_name = models.CharField(max_length=200, blank=True, default="")
    stage = models.PositiveSmallIntegerField(choices=Stage.choices)
    current_price = models.DecimalField(max_digits=14, decimal_places=4)
    ma_30w = models.DecimalField(max_digits=14, decimal_places=4)
    price_vs_ma_pct = models.FloatField(
        help_text="Percent distance of price above/below 30-week MA"
    )
    ma_slope_pct = models.FloatField(
        help_text="Percent change in 30-week MA over lookback period"
    )
    suggested_action = models.CharField(max_length=50)
    breakout_level = models.DecimalField(
        max_digits=14, decimal_places=4, null=True, blank=True
    )
    support_level = models.DecimalField(
        max_digits=14, decimal_places=4, null=True, blank=True
    )
    stop_loss = models.DecimalField(
        max_digits=14, decimal_places=4, null=True, blank=True
    )
    stage_reasons = models.JSONField(
        default=list,
        help_text="Transparent list of factors used in stage detection",
    )
    chart_payload = models.JSONField(
        default=dict,
        help_text="Serialized weekly OHLCV + MA for Chart.js",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        unique_together = [["user", "ticker"]]

    def __str__(self) -> str:
        return f"{self.ticker} — Stage {self.stage}"

    @property
    def stage_label(self) -> str:
        return Stage(self.stage).label