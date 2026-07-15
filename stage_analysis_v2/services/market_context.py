"""
Market context — Nifty 50 stage drives stock highlight eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stage_analysis_v2.services.data_loader import load_weekly
from stage_analysis_v2.services.stage_engine import detect_weekly_stage, stage_action
from trading.constants import NIFTY50_SYMBOL


@dataclass
class MarketContext:
    benchmark: str
    stage: int
    stage_label: str
    action: str
    favorable: bool
    warning: str
    reasons: list[str]
    price: float
    ma_30w: float


def analyze_market_context(benchmark: str = NIFTY50_SYMBOL) -> MarketContext:
    weekly = load_weekly(benchmark)
    if weekly.empty:
        return MarketContext(
            benchmark=benchmark,
            stage=0,
            stage_label="Unknown",
            action="",
            favorable=True,
            warning="",
            reasons=["Could not load Nifty 50 data"],
            price=0.0,
            ma_30w=0.0,
        )

    stage, reasons, metrics = detect_weekly_stage(weekly)
    labels = {1: "Accumulation", 2: "Advancing", 3: "Distribution", 4: "Declining"}

    favorable = stage in (1, 2)
    warning = ""
    if stage == 3:
        warning = "Market in Stage 3 (Distribution) — reduce new long exposure, tighten stops."
    elif stage == 4:
        warning = "Market in Stage 4 (Declining) — avoid new long positions until Nifty recovers."
    elif stage == 1:
        warning = "Market in early Stage 1 — watch for Nifty Stage 2 confirmation before aggressive buying."

    return MarketContext(
        benchmark=benchmark,
        stage=stage,
        stage_label=f"Stage {stage} — {labels.get(stage, '')}",
        action=stage_action(stage),
        favorable=favorable,
        warning=warning,
        reasons=reasons,
        price=metrics["price"],
        ma_30w=metrics["ma"],
    )