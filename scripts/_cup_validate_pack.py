"""Validate a candidate Cup 100% pack on every 1-year window via the real engine."""
from __future__ import annotations

import os
import sys
import time
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
    CUP_EXIT_TARGET_R,
    CupParams,
    run_cup_breakout_backtest,
)
from trading.services.market_data import get_universe_symbols

WINDOWS = [
    ("last_1y", date(2025, 8, 24), date(2026, 8, 24)),
    ("y2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("y2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("y2025", date(2025, 1, 1), date(2025, 12, 31)),
    ("full_5y", date(2021, 8, 25), date(2026, 8, 24)),
]


def wide_cup(**kw) -> CupParams:
    return CupParams(
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
        **kw,
    )


CONFIGS = [
    ("ema20 r10 h40", 10.0, 40, 0, 100.0, wide_cup(cup_exit_mode=CUP_EXIT_EMA20)),
    ("meas r8 h45", 8.0, 45, 0, 100.0, wide_cup(cup_exit_mode=CUP_EXIT_MEASURED)),
    ("2R r10 h90", 10.0, 90, 0, 100.0, wide_cup(cup_exit_mode=CUP_EXIT_TARGET_R, target_rr=2.0)),
]


def main():
    from trading.constants import NIFTY50_SYMBOL

    symbols = [s for s in get_universe_symbols(nifty200_only=True) if s != NIFTY50_SYMBOL]
    names = sys.argv[1:]
    configs = CONFIGS
    if names:
        configs = [c for c in CONFIGS if any(n.lower() in c[0].lower() for n in names)] or CONFIGS
    for label, risk, hold, cooldown, pos, params in configs:
        print(f"\n===== {label}  risk={risk:g} hold={hold} cd={cooldown} pos={pos:g} =====", flush=True)
        for name, start, end in WINDOWS:
            t0 = time.time()
            r = run_cup_breakout_backtest(
                symbols=symbols,
                start_date=start,
                end_date=end,
                capital=1_000_000.0,
                risk_pct=risk,
                max_hold_days=hold,
                cooldown_days=cooldown,
                max_pos_pct=pos,
                params=params,
            )
            print(
                f"{name:10s} {start}→{end}  ret={r.total_return_pct:+7.1f}%  "
                f"CAGR={r.cagr_pct:+6.1f}%  n={r.total_trades:3d} sig={r.stage2_entries:3d}  "
                f"WR={r.win_rate:5.1f}%  PF={r.profit_factor:5.2f}  DD={r.max_drawdown_pct:5.1f}%  "
                f"Sharpe={r.sharpe:5.2f}  avg={r.avg_trade:,.0f}  hold={r.avg_hold_days}d  "
                f"final={r.final_capital:,.0f}  ({time.time()-t0:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
