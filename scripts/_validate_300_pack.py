"""Validate the ≥300% aggressive pack with the official backtester."""
from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.backtester import run_stage_v2_backtest
from trading.services.market_data import get_universe_symbols

START = date(2023, 1, 1)
END = date(2026, 2, 28)
SYMS = get_universe_symbols(nifty200_only=True)

configs = [
    {
        "label": "BASELINE default",
        "risk_pct": 2.0,
        "cooldown_days": 40,
        "exit_mode": "stage_4_only",
        "target_rr": 2.5,
        "max_hold_days": 65,
        "tech_filter": "daily_mtf",
    },
    {
        "label": "300% PACK A risk5 no_stage 4R hold90 cd5",
        "risk_pct": 5.0,
        "cooldown_days": 5,
        "exit_mode": "no_stage",
        "target_rr": 4.0,
        "max_hold_days": 90,
        "tech_filter": "daily_mtf",
    },
    {
        "label": "300% PACK B risk5 no_stage 4R hold90 cd0",
        "risk_pct": 5.0,
        "cooldown_days": 0,
        "exit_mode": "no_stage",
        "target_rr": 4.0,
        "max_hold_days": 90,
        "tech_filter": "daily_mtf",
    },
    {
        "label": "300% PACK C risk4 no_stage 4R hold90 cd5",
        "risk_pct": 4.0,
        "cooldown_days": 5,
        "exit_mode": "no_stage",
        "target_rr": 4.0,
        "max_hold_days": 90,
        "tech_filter": "daily_mtf",
    },
    {
        "label": "ALT risk6 stage4 3R hold90 cd40",
        "risk_pct": 6.0,
        "cooldown_days": 40,
        "exit_mode": "stage_4_only",
        "target_rr": 3.0,
        "max_hold_days": 90,
        "tech_filter": "daily_mtf",
    },
]

print(f"Validate {START} → {END} | {len(SYMS)} symbols")
for c in configs:
    label = c.pop("label")
    r = run_stage_v2_backtest(
        symbols=SYMS,
        start_date=START,
        end_date=END,
        capital=1_000_000.0,
        min_quality_score=0,
        market_filter=False,
        entry_stage=2,
        entry_on="transition",
        **c,
    )
    flag = " *** ≥300 ***" if r.total_return_pct >= 300 else ""
    print(
        f"{label}\n"
        f"  ret={r.total_return_pct:+.2f}% WR={r.win_rate}% PF={r.profit_factor} "
        f"DD={r.max_drawdown_pct}% n={r.total_trades} signals={r.stage2_entries} "
        f"peak={r.peak_parallel} skip_cash={r.signals_skipped_cash} "
        f"final=₹{r.final_cash:,.0f}{flag}"
    )
    print(f"  exits={r.exit_breakdown}")
    c["label"] = label  # restore if reused
