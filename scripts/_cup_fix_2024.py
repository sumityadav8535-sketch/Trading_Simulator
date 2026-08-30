"""Test 2024 fixes without giving up the 2023 +100% year."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from stage_analysis_v2.services.cup_breakout import (
    CUP_EXIT_EMA20,
    CUP_EXIT_MEASURED,
    CupParams,
    run_cup_breakout_backtest,
)
from trading.constants import NIFTY50_SYMBOL
from trading.services.market_data import get_universe_symbols

BASE = CupParams(
    cup_min_days=20,
    cup_max_days=180,
    min_depth_pct=12.0,
    max_depth_pct=45.0,
    min_bottom_days=5,
    min_left_days=7,
    min_recovery_days=5,
    pivot_width=5,
    require_sma200_rising=False,
    require_rs_vs_nifty=False,
    require_close_strength=False,
    require_trend_stack=True,
    vol_mult=1.1,
    rsi_min=40.0,
    rsi_max=85.0,
    max_new_per_day=10,
    cup_exit_mode=CUP_EXIT_EMA20,
)

WINDOWS = [
    ("2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("2025", date(2025, 1, 1), date(2025, 12, 31)),
]

FIXES = [
    ("baseline", BASE, 10.0, 40, 0, 100.0),
    ("arm1R", replace(BASE, trail_arm_r=1.0), 10.0, 40, 0, 100.0),
    ("skip2% gap-down fill", replace(BASE, skip_entry_gap_down_pct=2.0), 10.0, 40, 0, 100.0),
    ("arm1R + skip2%", replace(BASE, trail_arm_r=1.0, skip_entry_gap_down_pct=2.0), 10.0, 40, 0, 100.0),
    ("pos cap 50%", BASE, 10.0, 40, 0, 50.0),
    ("arm1R + cap50 + skip2%", replace(BASE, trail_arm_r=1.0, skip_entry_gap_down_pct=2.0), 10.0, 40, 0, 50.0),
    ("meas + skip2%", replace(BASE, cup_exit_mode=CUP_EXIT_MEASURED, skip_entry_gap_down_pct=2.0), 8.0, 45, 0, 100.0),
    ("arm0.5R + skip2%", replace(BASE, trail_arm_r=0.5, skip_entry_gap_down_pct=2.0), 10.0, 40, 0, 100.0),
]


def main():
    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    for label, params, risk, hold, cd, pos in FIXES:
        bits = []
        for wname, start, end in WINDOWS:
            r = run_cup_breakout_backtest(
                symbols=symbols,
                start_date=start,
                end_date=end,
                capital=1_000_000.0,
                risk_pct=risk,
                max_hold_days=hold,
                cooldown_days=cd,
                max_pos_pct=pos,
                params=params,
            )
            bits.append(
                f"{wname}={r.total_return_pct:+6.1f}% n={r.total_trades:2d} "
                f"WR={r.win_rate:4.1f}% DD={r.max_drawdown_pct:4.1f}% PF={r.profit_factor:.2f}"
            )
        print(f"{label:28s} " + " | ".join(bits), flush=True)


if __name__ == "__main__":
    main()
