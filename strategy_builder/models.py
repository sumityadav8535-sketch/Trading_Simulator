"""Persist strategy definitions as validated JSON (no executable code)."""
from __future__ import annotations

from django.conf import settings
from django.db import models


class SavedStrategy(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="saved_strategies",
    )
    name = models.CharField(max_length=120)
    definition = models.JSONField(default=dict)
    notes = models.TextField(blank=True, default="")
    is_favorite = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [models.Index(fields=["user", "name"])]

    def __str__(self) -> str:
        return self.name


class StrategyVersion(models.Model):
    strategy = models.ForeignKey(
        SavedStrategy, on_delete=models.CASCADE, related_name="versions"
    )
    version = models.PositiveIntegerField()
    definition = models.JSONField(default=dict)
    label = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version"]
        unique_together = [("strategy", "version")]

    def __str__(self) -> str:
        return f"{self.strategy.name} v{self.version}"
